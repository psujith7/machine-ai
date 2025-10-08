import os, glob, argparse, textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector DB
import chromadb
from chromadb.config import Settings

# LLM (local, CPU) via llama.cpp
from llama_cpp import Llama

# Optional PDF support
from pypdf import PdfReader


# ---------------------------
# Config (change if you want)
# ---------------------------
EMBED_MODEL_ID = "intfloat/e5-small-v2"    # small, fast, good quality
CHROMA_DIR     = ".chroma"                 # where the DB persists
COLLECTION     = "rag_collection"

# A good small CPU model for testing; switch to 7B if you have RAM
# (Q4_K_M ~4–8GB depending on model).
# See: https://huggingface.co/Qwen
LLAMA_REPO     = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLAMA_FILE     = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"  # good CPU-friendly start
# For stronger model (needs more RAM/CPU):
# LLAMA_REPO   = "Qwen/Qwen2.5-7B-Instruct-GGUF"
# LLAMA_FILE   = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"


# ---------------------------
# Simple document struct
# ---------------------------
@dataclass
class DocChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


# ---------------------------
# Utilities
# ---------------------------
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def load_documents(folder: str) -> List[Tuple[str, str]]:
    """Return list of (source_path, text)."""
    out = []
    for path in glob.glob(os.path.join(folder, "**/*"), recursive=True):
        if os.path.isdir(path): 
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in [".txt", ".md", ".markdown"]:
                text = read_txt(path)
            elif ext in [".pdf"]:
                text = read_pdf(path)
            else:
                continue
            text = (text or "").strip()
            if text:
                out.append((path, text))
        except Exception:
            pass
    return out

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Simple character-based chunking (works for any language)."""
    text = " ".join(text.split())
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunk = text[i:j]
        chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


# ---------------------------
# Index building
# ---------------------------
def build_index(docs_dir: str):
    # init embeddings
    print("Loading embedding model:", EMBED_MODEL_ID)
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    # init vector DB
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_DIR,
        is_persistent=True
    ))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    # read and chunk
    sources = load_documents(docs_dir)
    print(f"Found {len(sources)} source file(s).")
    ids, texts, metas = [], [], []
    idx = 0
    for src_path, raw in sources:
        chunks = chunk_text(raw)
        for k, chunk in enumerate(chunks):
            cid = f"{os.path.basename(src_path)}::{k}"
            ids.append(cid)
            texts.append(chunk)
            metas.append({"source": src_path, "chunk": k})
            idx += 1
    if not ids:
        print("No text to index; add files to docs/ and retry.")
        return

    # embed & add to vector DB (Chroma will embed for us if we pass embeddings separately)
    print(f"Embedding {len(texts)} chunk(s)…")
    vectors = embedder.encode([f"passage: {t}" for t in texts], show_progress_bar=True, normalize_embeddings=True)

    print("Writing to Chroma…")
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    client.persist()
    print(f"✅ Index built with {len(ids)} chunks.")


# ---------------------------
# Retrieval + Generation
# ---------------------------
def load_llm() -> Llama:
    """
    Downloads the GGUF from Hugging Face on first run and memory-maps it.
    Works on CPU; set n_ctx to a comfy context length.
    """
    print("Loading local LLM (this may take a minute the first time)…")
    llm = Llama.from_pretrained(
        repo_id=LLAMA_REPO,
        filename=LLAMA_FILE,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        # You can tweak:
        # n_gpu_layers=0,  # for pure CPU; set >0 if you have Metal/CUDA via llama.cpp build
    )
    return llm

def retrieve(query: str, top_k: int = 5):
    client = chromadb.Client(Settings(persist_directory=CHROMA_DIR, is_persistent=True))
    collection = client.get_collection(COLLECTION)

    # E5 requires "query: " prefix for query embedding if we were embedding ourselves.
    # Chroma can use stored embeddings directly; we’ll embed the query just like we did passages:
    embedder = SentenceTransformer(EMBED_MODEL_ID)
    q_vec = embedder.encode([f"query: {query}"], normalize_embeddings=True)

    res = collection.query(query_embeddings=q_vec, n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
    # Normalize shapes
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    ids   = res["ids"][0]
    return list(zip(ids, docs, metas, dists))

def make_prompt(query: str, ctx: List[Tuple[str, str, Dict[str, Any], float]]) -> str:
    # Build a compact context with sources
    context_blocks = []
    for _id, doc, meta, dist in ctx:
        src = meta.get("source", "unknown")
        context_blocks.append(f"[Source: {src}] {doc}")
    context_text = "\n\n".join(context_blocks)

    system = (
        "You are a helpful assistant that answers strictly using the provided context. "
        "Cite sources inline like [source: filename.ext] after the sentence that uses them. "
        "If the answer is not in the context, say you don't know."
    )
    user = f"""Question: {query}

Context:
{context_text}
"""
    # Simple chat template for instruct models
    prompt = f"<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def answer(query: str, top_k: int = 5, temperature: float = 0.2, max_tokens: int = 512) -> str:
    ctx = retrieve(query, top_k=top_k)
    if not ctx:
        return "No results in your knowledge base yet. Add files to docs/ and rebuild the index."

    llm = load_llm()
    prompt = make_prompt(query, ctx)

    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    text = out["choices"][0]["text"].strip()
    return text


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Local, free RAG with Chroma + E5 + Qwen2.5 (llama.cpp)")
    ap.add_argument("--build", action="store_true", help="(Re)build the vector index from docs/")
    ap.add_argument("--ask", type=str, help="Ask a question against your local knowledge")
    ap.add_argument("--docs", type=str, default="docs", help="Docs folder")
    ap.add_argument("--topk", type=int, default=5, help="Retriever top-k")
    args = ap.parse_args()

    if args.build:
        build_index(args.docs)

    if args.ask:
        print("\nQ:", args.ask)
        print("\nA:", answer(args.ask, top_k=args.topk))

if __name__ == "__main__":
    main()
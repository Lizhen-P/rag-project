"""Task 4 — Retrieve.

Question in, the TOP_K most relevant chunks out. This is the "R" in RAG:
everything Task 5 will say is only as good as what this step hands it.

The flow is Task 3's smoke test grown into a real tool:
    load index + sidecar -> embed the question -> cosine search -> ranked chunks

Run it with the question in quotes (a bare `?` would be grabbed by the shell
as a filename wildcard before Python ever saw it):
    uv run python local/04_retrieve.py "How do I enable versioning on an S3 bucket?"
"""

import json
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import faiss
import numpy as np

# --- Configuration -----------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"   # MUST be the model the index was built with — same space!
TOP_K = 5          # how many chunks to return (guide's starting point)

INDEX_PATH = Path(__file__).parent / "index.faiss"
META_PATH = Path(__file__).parent / "index_meta.jsonl"


def embed(text: str) -> np.ndarray:
    """Embed ONE text; return it as a (1, 1024) float32 matrix.

    Same call as Tasks 1/3. FAISS's search() expects a MATRIX of queries
    (even for a single one), so we keep the outer list-of-one instead of
    grabbing [0] like Task 1 did — the shape is (1, 1024), one row.
    """
    payload = json.dumps({"model": MODEL, "input": [text]}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return np.array(data["embeddings"], dtype=np.float32)


def gauge(score: float) -> str:
    """Rule-of-thumb label for a cosine score — calibrated on THIS corpus.

    These thresholds are not universal constants: they come from watching
    real queries against these docs with this model (see the explained doc).
    On a different corpus or embedding model you'd re-calibrate. Task 5 will
    lean on the same idea to decide when to say "I don't know".
    """
    if score >= 0.65:
        return "strong"
    if score >= 0.55:
        return "okay  "  # padded so the output columns line up
    return "weak  "


def main() -> None:
    # The question = every command-line word after the script name, re-joined.
    # sys.argv[1:] is a list like ["How", "do", "I", ...]; " ".join glues it
    # back together — so the user doesn't have to remember quotes.
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: uv run python local/04_retrieve.py "your question"')
        sys.exit(1)

    if not INDEX_PATH.exists():
        print("no index found — run `uv run python local/03_index.py` first")
        sys.exit(1)

    # Load the artifact PAIR from Task 3.
    index = faiss.read_index(str(INDEX_PATH))
    records = [json.loads(line) for line in META_PATH.open()]

    # The tripwire for the classic RAG bug: index and metadata out of sync.
    # If someone rebuilt one file but not the other, row ids would point at
    # the wrong text — fail here, loudly, instead of retrieving nonsense.
    assert index.ntotal == len(records), (
        f"index has {index.ntotal} vectors but sidecar has {len(records)} "
        "records — re-run 03_index.py to rebuild the pair together"
    )

    # Embed the question into the SAME space as the chunks: same model,
    # same normalization. Break either half and the scores mean nothing.
    try:
        q = embed(question)
    except OSError as err:
        print(f"Ollama call failed: {err}\nIs `ollama serve` running?")
        sys.exit(1)
    faiss.normalize_L2(q)

    # The search itself — and proof of Task 3's "brute force is fast" claim.
    # perf_counter is the right clock for timing short things (monotonic,
    # high resolution); time.time() can jump if the system clock adjusts.
    t0 = time.perf_counter()
    scores, ids = index.search(q, TOP_K)
    ms = (time.perf_counter() - t0) * 1000

    print(f"question : {question}")
    print(f"searched : {index.ntotal} chunks in {ms:.2f} ms\n")

    for rank, (score, i) in enumerate(zip(scores[0], ids[0]), start=1):
        r = records[i]
        # textwrap.shorten squeezes runs of whitespace and cuts at a word
        # boundary, appending the placeholder — a tidy one-line preview.
        preview = textwrap.shorten(r["text"], width=200, placeholder=" …")
        print(f"{rank}. [{gauge(score)} {score:.3f}] "
              f"{r['chunk_id']}  ({r['n_tokens_est']} tokens)")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()

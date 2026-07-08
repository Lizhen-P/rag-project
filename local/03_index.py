"""Task 3 — Index.

Turn all 893 chunks into vectors and load them into a searchable FAISS index.
This is the moment the corpus stops being "a pile of text" and becomes
"a database you can query by MEANING".

Three jobs, in order:
  1. EMBED   — every chunk's text -> 1024-dim vector, in batches (not one by one)
  2. INDEX   — pack the vectors into a FAISS index built for cosine search
  3. PERSIST — write index.faiss + index_meta.jsonl, the artifact pair Task 4 loads

Ends with a smoke-test query so we can SEE retrieval working before Task 4
makes it a proper tool.

Run it with:
    uv run python local/03_index.py
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import faiss  # Facebook AI Similarity Search — the vector index
import numpy as np

# --- Configuration -----------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"
DIM = 1024        # bge-m3's vector size; also Titan Embeddings v2's — deliberate
BATCH_SIZE = 32   # chunks per HTTP request (see embed_batch for why we batch)

CHUNKS_PATH = Path(__file__).parent / "chunks.jsonl"
INDEX_PATH = Path(__file__).parent / "index.faiss"
META_PATH = Path(__file__).parent / "index_meta.jsonl"


# --- Embedding ---------------------------------------------------------------

def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a LIST of texts in one Ollama call; return a (len(texts), 1024) array.

    Same endpoint as Task 1 — the only news is that "input" can be a list.
    893 one-at-a-time HTTP calls would pay connection + model-warmup overhead
    893 times; batching pays it ~28 times. (Think one trip to the post office
    with a mailbag, not 893 trips with single letters.)
    """
    payload = json.dumps({"model": MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    # Big batches can take a while on first call (model loads into RAM) —
    # give it a generous timeout rather than a false failure.
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)

    # np.array turns the list-of-lists into a matrix: one row per text.
    # float32 matters: FAISS only speaks float32, and it halves memory vs
    # Python's default float64 with zero practical loss for similarity math.
    return np.array(data["embeddings"], dtype=np.float32)


# --- Main --------------------------------------------------------------------

def main() -> None:
    # 1. Load every chunk record. One json.loads per line — the whole point of
    #    the JSONL format (JSON Lines: independent objects, one per line).
    records = [json.loads(line) for line in CHUNKS_PATH.open()]
    texts = [r["text"] for r in records]
    print(f"loaded {len(records)} chunks from {CHUNKS_PATH.name}")

    # 2. The index. "Flat" = no clever shelving: every query is compared against
    #    EVERY stored vector, exactly. At 893 vectors that's a sub-millisecond
    #    matrix multiplication — approximate indexes (IVF, HNSW) only earn their
    #    complexity at millions of vectors. "IP" = inner product (dot product),
    #    which becomes cosine similarity once vectors are normalized (step 3).
    index = faiss.IndexFlatIP(DIM)

    # 3. Embed in batches and add to the index.
    t0 = time.time()
    for start in range(0, len(texts), BATCH_SIZE):  # 0, 32, 64, ... (step=32)
        batch = texts[start : start + BATCH_SIZE]   # last batch may be shorter
        try:
            vectors = embed_batch(batch)
        except OSError as err:
            print(f"\nOllama call failed: {err}\nIs `ollama serve` running?")
            sys.exit(1)

        # Safety net: if the model ever returns the wrong shape, stop loudly
        # now rather than discover a corrupt index in Task 4.
        assert vectors.shape == (len(batch), DIM), vectors.shape

        # Normalize each row to length 1 (in place). After this, dot product
        # == cosine similarity: length is gone, only DIRECTION (meaning) left.
        # Task 1 did the same thing by dividing by the norms — this is that
        # division done once at index time instead of at every search.
        faiss.normalize_L2(vectors)

        index.add(vectors)  # rows get ids 0,1,2,... in insertion order

        done = min(start + BATCH_SIZE, len(texts))
        print(f"  embedded {done:3d}/{len(texts)}  ({time.time() - t0:5.1f}s)")

    secs = time.time() - t0
    print(f"\nindex built: {index.ntotal} vectors x {DIM} dims "
          f"in {secs:.1f}s ({index.ntotal / secs:.0f} chunks/s)")

    # 4. Persist the artifact PAIR. FAISS stores vectors + integer row ids and
    #    nothing else — it has no idea what text row 412 came from. The sidecar
    #    holds the records in exactly insertion order, so:
    #        search hit id=412  ->  line 412 of index_meta.jsonl  ->  chunk.
    #    We copy the records (rather than pointing at chunks.jsonl) so the pair
    #    is self-contained: rebuilding chunks.jsonl later can never silently
    #    desynchronize an existing index from its metadata.
    faiss.write_index(index, str(INDEX_PATH))  # faiss wants a str, not a Path
    with META_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {INDEX_PATH.name} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB) "
          f"+ {META_PATH.name}")

    # 5. Smoke test: one real question, straight against the fresh index.
    #    (Task 4 turns this into a proper tool — this is just proof of life.)
    question = "How do I enable versioning on an S3 bucket?"
    q = embed_batch([question])       # embed the QUESTION with the SAME model...
    faiss.normalize_L2(q)             # ...and the SAME normalization as the chunks

    # search() takes a MATRIX of queries (ours has 1 row) and k = how many
    # neighbors to return. It gives back two matrices: scores and row ids,
    # one row per query — hence the [0] to grab our single query's results.
    scores, ids = index.search(q, k=3)
    print(f"\nsmoke test: {question!r}")
    for rank, (score, i) in enumerate(zip(scores[0], ids[0]), start=1):
        r = records[i]  # row id -> record, same order guarantee as the sidecar
        print(f"  {rank}. score {score:.3f}  {r['chunk_id']}"
              f"  [{r['heading']}]")


if __name__ == "__main__":
    main()

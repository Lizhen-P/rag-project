# Task 3 explained — `03_index.py`

> Embed all 893 chunks with bge-m3, pack the vectors into a FAISS index built
> for cosine search, save the index + metadata sidecar pair, and prove it works
> with one real query.

Run it with:

```bash
uv run python local/03_index.py
```

---

## Part 1 — Concepts

### What is an index, and why "Flat" is the honest choice here

After Task 1, each piece of text can become a point in 1024-dimensional
"meaning space". Retrieval = *given a question's point, find the nearest stored
points*. An **index** is the data structure that answers that question fast.

Here's the part most tutorials get backwards: at our scale, no cleverness is
needed. `IndexFlatIP` is **brute force** — every query is compared against
every stored vector, exactly, no approximation. That sounds slow and isn't:
comparing one query against 893 vectors is a single 893×1024 matrix
multiplication, which NumPy/FAISS dispatch to your CPU's vector units in
well under a millisecond.

Think of a library. With 893 books you just walk the one shelf and read every
spine — instant. Clever shelving systems (FAISS's IVF, HNSW — "approximate
nearest neighbor" indexes) exist for the Library of Congress: millions of
books, where scanning every spine is genuinely too slow, and you accept a
small chance of missing the true best match in exchange for speed. Reaching
for them at 893 vectors is engineering theater. **Flat = exact, simple,
correct — until scale forces a trade.**

### Normalization: how a dot product becomes cosine similarity

Task 1 computed `cos(a, b) = a·b / (||a||·||b||)` — dot product divided by both
lengths. The division is what makes the score ignore *length* and measure only
*angle* (direction = meaning).

`faiss.normalize_L2` stretches or shrinks every vector to length exactly 1
("projects it onto the unit sphere"). Once every vector has length 1, the
denominator is 1×1 — so **plain dot product IS cosine similarity**. That's the
whole trick behind pairing `normalize_L2` with `IndexFlatIP` (IP = inner
product = dot product):

```
normalize at index time (once per chunk)
normalize at query time (once per question)
→ every search score is a cosine similarity, computed as a bare dot product
```

Same math as Task 1, just reorganized: do the dividing once up front instead of
at every comparison.

### Batching: 28 trips to the post office, not 893

Ollama's `/api/embed` accepts a **list** of inputs. Every HTTP call pays fixed
overhead (connection, request parsing, model dispatch) before any real work
happens. One-at-a-time embedding pays that toll 893 times; batches of 32 pay it
28 times. Same total letters, one mailbag per trip instead of hand-delivering
each envelope. Our run: 893 chunks in 22.6s (~39 chunks/s), and the very first
batch is the slowest (2.0s) because that's when Ollama loads the model into
memory.

### The sidecar: FAISS remembers vectors, not text

A FAISS index stores exactly two things: the vectors, and their **integer row
ids** (0, 1, 2, … in insertion order). It has no idea row 412 came from
`s3_bucket_versioning.md`. So a search answers *"rows 621, 598 and 626 are
nearest"* — and something must translate row ids back into chunks.

That something is the **metadata sidecar** (`index_meta.jsonl`): the chunk
records written in *exactly* insertion order, so row id = line number. Theater
analogy: the index is the seating chart (seat numbers only); the sidecar is the
guest list saying who sits in each seat.

Why copy the records instead of just re-reading `chunks.jsonl`? Because the two
files can *drift*: re-run the chunker with new settings and `chunks.jsonl`
changes while the already-built index doesn't — now row 412 silently points at
the wrong text. That desync is a classic real-world RAG bug. Copying makes
`index.faiss` + `index_meta.jsonl` a **self-contained artifact pair**: they were
born together and stay consistent together.

### float32

FAISS only accepts `float32`. Python floats are 64-bit; the conversion halves
memory and loses nothing that matters for similarity. It's also why the index
file size is exactly predictable: 893 vectors × 1024 dims × 4 bytes ≈ 3.7 MB —
matching what the run printed.

---

## Part 2 — Walkthrough

**`embed_batch(texts)`** — Task 1's `embed()` with two upgrades: `"input"` is
now a list (Ollama embeds each element and returns a list of vectors in the
same order), and the result goes through `np.array(..., dtype=np.float32)`,
turning a list-of-lists into a proper matrix — one row per text, shape
`(len(texts), 1024)`.

**The batch loop** —

```python
for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start : start + BATCH_SIZE]
```

`range`'s third argument is the step: `start` = 0, 32, 64, … The slice grabs 32
texts; Python slices are forgiving at the end, so the final batch is just
whatever remains (29 here — hence "893/893" on the last line, not 896).

**`assert vectors.shape == (len(batch), DIM)`** — a tripwire. If Ollama ever
returns the wrong number of vectors or dimensions, the script dies *here* with
a clear message, instead of quietly building a corrupt index we'd only notice
when Task 4 returns nonsense.

**`faiss.normalize_L2(vectors)`** — note there's no `vectors = ...`: it
modifies the array **in place** (works directly on the memory it was handed).

**`index.add(vectors)`** — appends all 32 rows; FAISS assigns them the next
free row ids. Insertion order is the contract the sidecar depends on.

**`faiss.write_index(index, str(INDEX_PATH))`** — FAISS is a C++ library with
Python bindings; its file API predates `pathlib`, so it wants a plain string —
that's all `str(...)` is doing.

**The smoke test** —

```python
scores, ids = index.search(q, k=3)
```

`search` is built for many queries at once, so it takes a *matrix* (our `q` has
one row) and returns two matrices: `scores[0]` and `ids[0]` are our single
query's top-3 similarities and row ids. `zip` pairs them up; `enumerate(...,
start=1)` numbers the ranks. Each id indexes straight into `records` — same
order guarantee as the sidecar file.

The question is embedded with the **same model and same normalization** as the
chunks. That symmetry is non-negotiable: mix models (or forget to normalize one
side) and the two sides live in different meaning-spaces — scores become
garbage. In the AWS port this means: the embedding model is pipeline
configuration, pinned like everything else.

---

## Part 3 — Actual run output

```
loaded 893 chunks from chunks.jsonl
  embedded  32/893  (  2.0s)
  embedded  64/893  (  3.0s)
  embedded  96/893  (  3.7s)
  ...
  embedded 893/893  ( 22.6s)

index built: 893 vectors x 1024 dims in 22.6s (39 chunks/s)
wrote index.faiss (3.7 MB) + index_meta.jsonl

smoke test: 'How do I enable versioning on an S3 bucket?'
  1. score 0.728  s3_bucket_versioning-000  [Resource: aws_s3_bucket_versioning]
  2. score 0.705  s3_bucket-030  [Versioning]
  3. score 0.703  s3_bucket_versioning-005  [versioning_configuration]
```

Reading the smoke test like a retrieval engineer:

- **All three hits are exactly right** — the dedicated versioning resource
  doc's intro chunk, the "Versioning" section of the main `s3_bucket` doc, and
  the `versioning_configuration` argument block. Three relevant chunks from two
  different documents: the index found *meaning*, not filenames.
- **Scores ≈ 0.70–0.73 are strong.** Calibrate against Task 1: two clearly
  related sentences scored 0.471 there. A question phrased in plain English
  ("How do I enable…") landing at 0.728 against documentation prose is a
  confident match. Watch these magnitudes — you'll develop a feel for what's
  "confident" vs "grasping" in your corpus, which Task 4 exploits.
- **The gap between rank 1 and rank 3 is small** (0.728 → 0.703). The top
  results agree with each other — a healthy sign. When the top-k scores
  *plummet* after rank 1, it usually means only one chunk in the corpus
  addresses the question.

---

## Part 4 — AWS mapping

| This script (local) | AWS port |
|---|---|
| `IndexFlatIP` + `normalize_L2` (exact cosine) | S3 Vectors index with `cosine` distance metric |
| `embed_batch` loop over Ollama bge-m3 (1024-dim) | Lambda batching calls to Bedrock Titan Embeddings v2 (same 1024 dims — the schema ports unchanged) |
| `index.faiss` + `index_meta.jsonl` sidecar pair | Not needed as a pair: S3 Vectors stores **metadata with each vector** (filterable keys like `source_uri`), so the "guest list" travels with the seats |
| Row-id ↔ record order contract | Vector `key` (we'd use `chunk_id`) — explicit keys instead of positional ids, one less way to desync |
| Full rebuild each run (fine at 893 chunks) | Incremental `PutVectors` upserts, keyed by `chunk_id` — content hashes (Task 2!) tell you which chunks actually changed |
| 22.6s, $0 | Titan v2 pricing per input token; batch + cache accordingly |

The big design difference to remember for the port: **S3 Vectors eliminates the
sidecar** by attaching metadata to vectors directly — but it introduces
per-vector *keys*, which is where our `chunk_id` (stable, content-derived)
slots in perfectly. Decisions made in Task 2 are already paying rent.

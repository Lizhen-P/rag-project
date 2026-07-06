# Task 1 Explained — The Embed Probe (`01_embed_probe.py`)

> **Goal:** *feel* what an embedding is. Turn text into a 1024-dim vector via the
> local `bge-m3` model (through Ollama), then show that semantically similar text
> lands "closer" in vector space than unrelated text.
>
> This is the single most important intuition in the whole RAG project.
> Everything downstream (chunking, indexing, retrieval) is plumbing around it.

Run it with:

```bash
uv run python local/01_embed_probe.py
```

---

## 1. The big picture: why is there a network call at all?

Your Python script **does not know how to turn text into a vector**. That
"brain" — the bge-m3 model, 1.2 GB of numbers — lives inside a *separate
program* called **Ollama**, running quietly in the background on your Mac.

Two separate programs need to talk. How do programs talk? The same way your
browser talks to a website: over **HTTP** — even though both programs are on
the same laptop.

> **Analogy:** Ollama is a **kitchen**, your script is a **customer**. You can't
> cook the meal (make the vector) yourself. You write an order ("embed this text
> with bge-m3"), hand it to the kitchen, wait, and get a plate back. The `embed`
> function is the whole ritual of placing that order and receiving the plate.

The kitchen's address:

```python
OLLAMA_URL = "http://localhost:11434/api/embed"
```

| Part | Meaning |
|---|---|
| `http://` | we speak the HTTP language |
| `localhost` | "this same computer" (as opposed to `google.com`) |
| `11434` | the **port** — an apartment number so the message reaches Ollama and not some other program on the machine |
| `/api/embed` | the specific "counter" that does embedding (Ollama has others, e.g. `/api/chat` for generation — used in Task 5) |

---

## 2. `embed()` line by line — the four-step API call

```python
def embed(text: str) -> list[float]:
```

Takes a string, returns a list of floats (the 1024 numbers). The `-> list[float]`
is a *type hint* — documentation for humans and tools; Python doesn't enforce it.

### Step 1 — Write the order in the format the kitchen expects

```python
payload = json.dumps({"model": MODEL, "input": text}).encode("utf-8")
```

Three things happen, inside-out:

1. **`{"model": MODEL, "input": text}`** — a Python dict: *which* model, *what*
   text. Ollama's `/api/embed` requires exactly these two fields.
2. **`json.dumps(...)`** — dict → **JSON string**. JSON is the universal text
   format programs agree on — the *lingua franca* between programs. Why not send
   the dict directly? Ollama isn't a Python program; a Python dict is a
   Python-only concept. JSON is the neutral form both sides understand.
   (`dumps` = "dump to string".)
3. **`.encode("utf-8")`** — string → **bytes**. Networks only move raw bytes
   (numbers 0–255), not characters. UTF-8 is the rulebook for turning each
   character into byte numbers. **The network layer refuses anything that isn't
   bytes** — forgetting `.encode()` is the single most common beginner error here.

### Step 2 — Address the envelope (describe the request)

```python
req = urllib.request.Request(
    OLLAMA_URL,
    data=payload,
    headers={"Content-Type": "application/json"},
)
```

`urllib` is Python's built-in HTTP library (no install). A `Request` object
bundles the order *before* sending:

- `OLLAMA_URL` — **where** it goes.
- `data=payload` — **what's inside**. Key detail: **supplying `data=`
  automatically makes this a POST** instead of a GET. (GET = "give me
  something"; POST = "here's data, do something with it." We're *sending* text,
  so POST.)
- `headers=...` — a **note on the envelope**: "the contents are JSON, parse them
  as such." Headers are metadata *about* the request, separate from the body.

Creating the `Request` sends nothing — it just fills out the form.

### Step 3 — Mail it and wait for the reply

```python
with urllib.request.urlopen(req) as resp:
    data = json.load(resp)
```

- **`urlopen(req)`** — *this* is the moment the request flies to Ollama and
  blocks (waits) until the reply arrives. Behind the scenes Ollama runs your
  text through the 1.2 GB model and produces the vector.
- **`json.load(resp)`** — parses the JSON reply back into a Python object — the
  mirror image of `json.dumps`. (`load` reads from a stream like `resp`;
  `loads` — with an s — parses a plain string.)
- **`with ... as resp:`** — Python's "**borrow-and-return**" pattern. A network
  connection must be closed when done; `with` guarantees it closes the instant
  the block ends, *even if something errors*. You'll see `with` everywhere:
  files, connections, locks.

After this step, `data` is a normal dict:

```python
{"model": "bge-m3", "embeddings": [[-0.0402, 0.0369, ...1024 numbers...]]}
```

### Step 4 — Unwrap the answer

```python
return data["embeddings"][0]
```

`embeddings` is a **list of vectors** — because this endpoint can embed *many*
strings in one call (`"input": ["text1", "text2", ...]`) and always returns a
list for consistency. We sent one string, so we take element `[0]`.

> **Not a throwaway detail:** "send many texts in one call" — **batching** — is
> *the* main cost-and-speed lever when embedding a whole corpus (Task 3 locally,
> Bedrock on AWS). Network round-trips cost time and money; batching amortizes
> them.

---

## 3. `cosine()` — how "close" are two vectors?

```python
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

The formula:

```
cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

- `np.dot` multiplies the two 1024-number lists pairwise and sums → measures how
  much they point the same direction.
- Dividing by both `norm`s (vector lengths) cancels magnitude, leaving pure
  **angle**. That's why it's called *cosine* — it is literally the cosine of the
  angle between two arrows in 1024-dimensional space.
- Range: −1 (opposite) to 1 (identical direction). For text embeddings the
  useful range is roughly 0 (unrelated) to 1 (same meaning).

---

## 4. What the run showed (actual output, 2026-07-02)

Test sentences — two about the same topic sharing almost no words, one unrelated:

- A: `"Kinesis Data Streams ingests real-time event data."`
- B: `"Firehose delivers streaming records into an S3 bucket."`
- C: `"The recipe calls for two cups of flour and a pinch of salt."`

```
vector dimension: 1024

cosine similarity (higher = more similar in MEANING)
  A vs B  (both streaming)  = 0.471   <- winner
  A vs C  (stream vs recipe)= 0.356
  B vs C  (stream vs recipe)= 0.347
```

**The lesson:** A and B share essentially zero words, yet score clearly higher
than either does against the recipe. A keyword search would have found *nothing*
linking A and B. The embedding model found the concept — *streaming data
ingestion*. That gap (~0.47 vs ~0.35) **is** retrieval.

### Why isn't the related pair ~0.9?

Absolute cosine values are model-specific and often cluster in a narrow band.
**What matters is the *ranking*, not the raw number.** In retrieval (Task 4) we
never threshold on "is this > 0.8?" — we ask "**which chunks are the top-k
closest to the question?**" Relative order drives RAG.

---

## 5. The one sentence to remember

**`embed()` mails your text to a separate model-serving program over HTTP and
unwraps the vector it mails back** — and every API in this project (Bedrock
embeddings, Bedrock Claude) is the same four steps:

> build a JSON body → encode to bytes → send → parse the JSON reply.

Learn it once here, reuse it forever.

---

## AWS mapping (for the port later)

| Local piece | AWS twin |
|---|---|
| Ollama `/api/embed` + bge-m3 (1024-dim) | Bedrock **Titan Text Embeddings v2** (1024-dim — same!) |
| `urllib` hand-rolled HTTP | `boto3` client (same four steps, wrapped) |
| free, on-laptop | pay-per-call, batching = cost control |

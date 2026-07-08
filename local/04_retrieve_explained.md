# Task 4 explained — `04_retrieve.py`

> Turn Task 3's smoke test into a real tool: any question from the command
> line → the 5 most relevant chunks, scored, labeled, and previewed. Then use
> it to answer the question we deferred in Task 2: are 174-token chunks
> actually hurting retrieval?

Run it with the question in quotes:

```bash
uv run python local/04_retrieve.py "How do I enable versioning on an S3 bucket?"
```

*(Why quotes? `?` and `*` are filename wildcards to the shell — an unquoted
`bucket?` makes zsh hunt for files named `bucket` + one character and error
out with "no matches found" before Python ever runs. First lesson of Task 4,
learned the honest way.)*

---

## Part 1 — Concepts

### Retrieval's real job: get the right chunk IN the net

Retrieval is measured by **recall**: *is the right chunk somewhere in the
top-k?* — not "is it rank 1". Think of k as the width of a fishing net. Task 5
hands all k chunks to the LLM, which reads them and uses the relevant ones —
so a right answer at rank 5 is a win, and a plausible-looking wrong chunk at
rank 1 is survivable. The division of labor in RAG:

- **retrieval = recall** (cast a net that contains the answer)
- **generation = precision** (pick the answer out of the net)

You'll see below that this isn't theoretical — one of our four probe questions
only survives because of it.

### Scores are relative, not absolute — calibrate on YOUR corpus

Four probe questions produced a scoring map of this corpus:

| Question | Top-5 score range | Result quality |
|---|---|---|
| S3 versioning (well-covered topic) | 0.677 – 0.728 | all five relevant |
| Lambda triggered by S3 upload | 0.588 – 0.657 | right chunk at rank 5 |
| Glue crawler on a schedule | 0.573 – 0.604 | right chunks at 1, 2, 4 |
| **sourdough bread (out of scope)** | **0.378 – 0.385** | noise, as it should be |

Two lessons. First: in-scope questions land ~0.57–0.73 on this corpus while
nonsense bottoms out around **0.38 — the noise floor** — with a wide empty gap
between. A cutoff near ~0.50 would cleanly separate "we have material" from
"we don't", which is exactly what Task 5 needs for honest *"I don't know"*
answers. Second: the sourdough scores are *flat* (0.385, 0.380, 0.378…) —
when nothing fits, everything is equally mediocre. A **flat profile** is as
much a warning sign as a low one; real hits come with a peak.

These numbers belong to *this* corpus + *this* model. Different embedding
model or different documents → re-calibrate. That's why `gauge()` in the code
carries a comment saying so.

### Semantic closeness ≠ correctness (our first real miss)

The Lambda question is the star of this task:

> "How can a Lambda function be triggered when a file lands in S3?"

The correct doc is `s3_bucket_notification` (S3 events invoking Lambda). It
appeared — at **rank 5**, score 0.588. Ranks 1–4 (up to 0.657!) were chunks
about Lambda functions *using* S3: mounting S3 as a file system, fetching
deployment packages from a bucket. Same nouns, wrong direction.

That's the honest limit of pure vector search: embeddings measure *"is this
text about the same things?"*, which is not quite *"does this text answer the
question?"*. A chunk dense in Lambda-and-S3 vocabulary sits close to any
Lambda-and-S3 question, whichever way the causality runs. Note the trap:
0.657 labeled "strong" was wrong; 0.604 labeled "okay" (Glue crawler) was
right. **The score measures closeness, not correctness.**

Why we don't panic: k=5 kept the right chunk in the net (recall did its job);
Task 5's LLM reads all five and can tell "trigger Lambda FROM S3" apart from
"Lambda mounting S3". And when this class of miss becomes a real problem, the
known upgrades are hybrid search (vectors + keyword) and re-ranking — both
already listed as stretch goals in the project guide. Baseline first, measure,
then upgrade — not the other way around.

### The Task 2 checkpoint: are 174-token chunks hurting us?

Verdict from the probes: **mostly no — with one visible wart.**

- The small chunks retrieve *precise sections*, not whole documents: the exact
  `versioning_configuration` block, the exact "Scheduled Trigger" example with
  its cron line. Precision like that makes Task 5's citations sharp.
- The wart: `glue_crawler-000` — a **48-token** intro stub ("Manages a Glue
  Crawler" + a docs link) — took rank 1 for the crawler question while the
  chunks with the actual answer sat below it. Tiny chunks can win the
  similarity contest on topic-match while carrying almost no payload.
- The economics still favor us: five ~200-token chunks ≈ 1,100 tokens of
  context — tiny. If thin chunks starve Task 5's answers, the cheap first fix
  is raising k to 8–10, before touching the chunker (merging sub-100-token
  sections into neighbors — the fix planned in Task 2 — stays in reserve).

Decision: **defer to Task 5.** Retrieval quality is good; whether the payload
is *enough* is a generation-quality question, so we judge it there.

---

## Part 2 — Walkthrough

**`question = " ".join(sys.argv[1:])`** — `sys.argv[1:]` is every word after
the script name, as a list; `" ".join(...)` glues them back into one string.
(The shell already did the splitting — we're just undoing it.)

**`if not INDEX_PATH.exists()`** — a friendlier failure than FAISS's C++
stack trace if you forgot to run Task 3. Fail early, say what to do.

**The `assert index.ntotal == len(records)`** — Task 3's sidecar contract,
now *enforced* at load time: if the index and the metadata file were ever
rebuilt separately, row ids would silently point at the wrong text. This is
the desync tripwire, and its error message says exactly how to fix it.

**`embed()` returns shape `(1, 1024)`** — unlike Task 1, we keep the outer
dimension because `index.search()` takes a *matrix* of queries. One question =
a one-row matrix, and the results come back as one-row matrices too — hence
`scores[0]`, `ids[0]`.

**`time.perf_counter()`** — the right clock for timing short spans: monotonic
(never jumps backwards) and high-resolution. `time.time()` is wall-clock time
and can be adjusted under you (NTP, DST). The measured search: **~1 ms** over
893 chunks — Task 3's "brute force is fast" claim, now with a receipt.

**`gauge(score)`** — three rule-of-thumb labels with deliberately humble
comments. It exists to build intuition while reading results, not to make
decisions; Task 5 will turn the same idea into an actual answer/abstain
threshold.

**`textwrap.shorten(text, width=200, placeholder=" …")`** — stdlib one-liner
that collapses whitespace runs and cuts at a word boundary. Tidier than
`text[:200]`, which happily slices mid-wor.

---

## Part 3 — Actual run output (all four probes)

The good case — every hit relevant, scores high and clustered:

```
question : How do I enable versioning on an S3 bucket?
searched : 893 chunks in 6.53 ms

1. [strong 0.728] s3_bucket_versioning-000  (181 tokens)
2. [strong 0.705] s3_bucket-030  (274 tokens)          [s3_bucket > Versioning]
3. [strong 0.703] s3_bucket_versioning-005  (216 tokens)
4. [strong 0.685] s3_bucket_versioning-003  (303 tokens)
5. [strong 0.677] s3_bucket_lifecycle_configuration-009  (469 tokens)
```

The instructive miss — right answer in the net, but at rank 5:

```
question : How can a Lambda function be triggered when a file lands in S3?
searched : 893 chunks in 1.32 ms

1. [strong 0.657] lambda_function-008   — S3 Files *file system* (wrong direction)
2. [strong 0.656] lambda_function-019   — deployment package from S3 (wrong)
3. [okay   0.591] lambda_function-007   — file system again (wrong)
4. [okay   0.591] lambda_function-010   — logging resources (wrong)
5. [okay   0.588] s3_bucket_notification-004 — "Trigger multiple Lambda
                                               functions" ← THE answer
```

Mid-scores can be right answers (and a 48-token stub can outrank the payload):

```
question : How do I run a Glue crawler on a schedule?
searched : 893 chunks in 1.11 ms

1. [okay   0.604] glue_crawler-000  (48 tokens)  — intro stub, thin payload
2. [okay   0.602] glue_trigger-003  (58 tokens)  — "Scheduled Trigger" + cron ✓
3. [okay   0.594] glue_crawler-008  (625 tokens) — argument reference (schedule arg) ✓
4. [okay   0.578] glue_crawler-006  (166 tokens) — config example with cron ✓
5. [okay   0.573] glue_trigger-007  (197 tokens)
```

And the noise floor — out-of-scope question, flat weak scores:

```
question : How do I bake sourdough bread?
searched : 893 chunks in 0.95 ms

1. [weak   0.385] kinesis_firehose_delivery_stream-030
2. [weak   0.380] dynamodb_table-024
3. [weak   0.378] kinesis_firehose_delivery_stream-011
4. [weak   0.378] s3_bucket_acl-013
5. [weak   0.378] lambda_function-019
```

---

## Part 4 — AWS mapping

| This script (local) | AWS port |
|---|---|
| `index.search(q, TOP_K)` | S3 Vectors `QueryVectors` API (`topK`, cosine metric) — inside the query Lambda |
| Sidecar lookup (`records[i]`) | Not needed: S3 Vectors returns each hit's **metadata with the result** |
| `gauge()` thresholds / noise floor | The same calibrated cutoff in the query Lambda decides *answer vs "I don't know"* — and gets logged as a CloudWatch metric |
| ~1 ms local search | Sub-second S3 Vectors query — latency budget goes to the network + Bedrock calls, so measure per-stage (retrieval latency is a Phase 8 custom metric) |
| k=5, ~1,100 context tokens | k × chunk size = **prompt budget = per-query Bedrock cost** — the retrieval knob is also a cost knob |
| Rank-5 recall saves the day | Same argument for `topK ≥ 5` in production; revisit with the Phase 8 eval harness (hit-rate@k) |

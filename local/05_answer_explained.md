# Task 5 explained — `05_answer.py`

> The finale: question → embed → top-5 chunks → prompt → local LLM → cited
> answer. Everything Tasks 1–4 built, plus the one new craft: the prompt
> contract. Also: the qwen3.5:9b vs gpt-oss:20b bake-off, with a verdict.

Run it with:

```bash
uv run python local/05_answer.py "How do I enable versioning on an S3 bucket?"
uv run python local/05_answer.py --model gpt-oss "same question, other model"
```

---

## Part 1 — Concepts

### The prompt contract: the model's world is what we hand it

The LLM never sees the corpus, the index, or the internet. It sees exactly two
messages: a **system message** (its standing orders) and a **user message**
(the five retrieved excerpts + the question). Within that world we impose a
contract:

1. answer ONLY from the excerpts,
2. cite the excerpt id after every claim,
3. if the excerpts don't contain the answer, say so — don't guess.

This is the mechanism that converts "a language model that will cheerfully
improvise" into "a system whose answers can be checked". The facts come from
retrieval; the model contributes reading comprehension and grammar. When
people say RAG reduces hallucination, this contract — not magic — is what
they mean.

### Two honesty layers, one cheap and one smart

- **Hard floor (before the LLM):** Task 4 calibrated this corpus — real hits
  score ~0.57–0.73, noise ~0.38. If the best retrieved score is under 0.50,
  there's nothing worth reading, so we refuse *without generating*. In the
  sourdough test the LLM was never invoked: instant, free, honest. In the AWS
  port this is a real cost control — unanswerable queries cost $0 of Bedrock.
- **Soft rule (inside the prompt):** retrieval can fetch related-but-wrong
  chunks while still clearing the floor. The system prompt's "say it's not
  covered" rule lets the model be the second filter when the net is full of
  the wrong fish.

### The division of labor, proven live

Task 4 found that "trigger Lambda from S3" retrieves four wrong-direction
chunks at ranks 1–4 (scores up to 0.657) with the correct
`s3_bucket_notification` chunk trailing at rank 5 (0.588). Task 5 is where
that design bet pays off: **both models ignored the four distractors and
answered from the rank-5 chunk, citing it precisely.** Retrieval's job was
recall (get it in the net); the generator's job was precision (pick it out).
Both did their jobs. This one example is the best interview story in the
repo.

### Messages and roles: the industry-standard shape

We use Ollama's `/api/chat` — a list of `{role, content}` messages — rather
than the bare-prompt `/api/generate`. Every hosted LLM API speaks this shape
(Bedrock Converse, Anthropic, OpenAI), with the **system** role carrying
instructions that outrank the **user** content. Porting to Bedrock later
means rewriting one function (`chat()`) and nothing else.

### Small dials that matter

- **`temperature: 0.2`** — temperature scales how adventurous token choices
  are. Creative writing wants variety; grounded Q&A wants the boring,
  faithful answer, every time.
- **`num_ctx: 8192`** — the context window budget. Five chunks can reach
  ~4k tokens at our max chunk size; the default 4096 window could silently
  truncate excerpts (the model would never know what it didn't see). Silent
  truncation is a classic RAG bug — we budget explicitly.
- **Thinking channels** — both 2026 models "reason" before answering.
  qwen3.5 returns an *empty response* unless passed `"think": false`;
  gpt-oss always thinks but reliably puts the answer in `response`. The
  `MODELS` dict records each model's quirk — the per-model glue every LLM
  integration needs, and exactly the kind of difference Bedrock's Converse
  API exists to standardize away.

### Trust, but verify the citations

Models can invent plausible-looking citation ids. `check_citations()` regex-
extracts every `[chunk_id]` from the answer and compares against the set we
actually provided — a five-line preview of Phase 8's evaluation harness.
It caught a real issue immediately: gpt-oss writes `[ id ]` with padding
spaces, which the first strict regex refused to count ("citations: NONE").
Lesson: verification code gets calibrated against real outputs too.

---

## Part 2 — The bake-off

Same questions, same retrieved chunks, same prompt. Differences:

| | qwen3.5:9b | gpt-oss:20b |
|---|---|---|
| Size on disk | 6.6 GB | 14 GB |
| Speed (measured) | ~43 tok/s | ~59 tok/s |
| Answer style | concise (127 tokens) | thorough (348–492 tokens) |
| Citation discipline | **inline, per claim — as instructed** | correct ids, but lumped at the end, `[ padded ]` |
| Notable | followed format rules exactly | surfaced a buried real detail ("wait ~15 minutes" — genuinely in the doc) |
| Rank-5 rescue | ✓ | ✓ |

**Verdict: qwen3.5:9b as daily driver.** For this job — short, precisely
cited answers — instruction adherence beats eloquence, and qwen followed the
contract to the letter at half the disk size. gpt-oss:20b stays installed as
the richer-answer alternative (`--model gpt-oss`), and its habit of surfacing
extra true details makes it worth consulting on gnarly questions. Both
models' discipline came through where it counts: neither invented a fact or
a citation in any test.

---

## Part 3 — Actual run output

Baseline (qwen3.5:9b): correct resource, correct example, warns about the
legacy inline `versioning` block, one valid citation:

```
retrieved :
    0.728  s3_bucket_versioning-000
    ...
--- answer (127 tokens, 3.0s, 43 tok/s) ---

To enable versioning on an S3 bucket, use the `aws_s3_bucket_versioning`
resource with a `versioning_configuration` block where `status = "Enabled"`.
Do not mix this with the inline `versioning` configuration in
`aws_s3_bucket` ... [s3_bucket-030]

resource "aws_s3_bucket_versioning" "example" {
  bucket = aws_s3_bucket.example.id
  versioning_configuration {
    status = "Enabled"
  }
}

citations: 1 cited, all valid
```

The rank-5 rescue (both models; qwen shown):

```
retrieved :
    0.657  lambda_function-008        <- distractor
    0.656  lambda_function-019        <- distractor
    0.591  lambda_function-007        <- distractor
    0.591  lambda_function-010        <- distractor
    0.588  s3_bucket_notification-004 <- the answer, rank 5

--- answer (173 tokens, 4.1s, 43 tok/s) ---

To trigger a Lambda function when a file is uploaded to an S3 bucket,
configure `aws_s3_bucket_notification` with the target Lambda ARN and
specific events. [s3_bucket_notification-004]
...
sources:
    lambda_function-008        (unused)
    ...
  * s3_bucket_notification-004  <- the only chunk it used
```

The honesty floor (no generation, no cost):

```
question  : How do I bake sourdough bread?
retrieved :
    0.385  kinesis_firehose_delivery_stream-030
    ...
best score 0.385 < floor 0.5 — the corpus has nothing relevant;
refusing without calling the LLM.
```

---

## Part 4 — AWS mapping

| This script (local) | AWS port |
|---|---|
| `chat()` → Ollama `/api/chat` (system + messages) | Query Lambda → **Bedrock Converse API** (same message shape; per-model quirks handled by AWS) |
| qwen3.5:9b / gpt-oss:20b | Claude Haiku (cheap, fast — the generator is the cheap seat) |
| `SCORE_FLOOR` short-circuit | Same check before calling Bedrock: unanswerable queries cost $0; refusal rate → CloudWatch metric |
| `SYSTEM_PROMPT` contract | Same prompt, stored as configuration; "grounded + cited + abstains" is the interview line |
| `check_citations()` | Seed of the `/eval` harness (Phase 8): citation validity, retrieval hit-rate |
| `num_ctx` budget | Token budget = Bedrock **cost per query**; k × chunk size is the lever |
| `temperature: 0.2` | Same param in Converse `inferenceConfig` |

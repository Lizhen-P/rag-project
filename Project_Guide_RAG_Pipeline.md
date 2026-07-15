# Build Guide — RAG Document-Q&A Pipeline
### Your AI/ML-flavoured flagship portfolio project

A production-style **data-engineering pipeline** that ingests a document corpus, processes and embeds it, stores it in a vector index, and serves an LLM-powered Q&A endpoint with citations. The engineering *is* the showcase — the LLM is just an API call at the end.

> **Why this project:** it sits in the hottest area (GenAI data engineering), it's fully generic (works on any corpus, not insurance), it exercises the entire DEA-C01 skill set, and it uses 2026-current services — which quietly tells a hiring manager you keep up.

> **Status (updated 2026-07-15): the local prototype is complete.** Tasks 0–5 in `local/` run this exact flow end-to-end for $0 (Ollama `bge-m3` + FAISS + `qwen3.5:9b`), with a calibrated refusal floor and machine-verified citations. What remains is the main event below: Phases 0–9, the AWS port — one short-lived branch per phase.

---

## What it demonstrates

| DEA-C01 domain | Where it shows up here |
|---|---|
| Data Ingestion & Transformation (34%) | S3 ingestion, text extraction, chunking, batch embedding |
| Data Store Management (26%) | S3 Vectors index design, curated Parquet, metadata schema |
| Data Operations & Support (22%) | Step Functions orchestration, idempotency, CloudWatch, eval harness |
| Data Security & Governance (18%) | Least-privilege IAM, KMS encryption, no public buckets |

Plus the signal you actually want: **"I can build the data plumbing behind an AI product."**

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
  corpus (73 md docs) ──► │ S3 raw bucket (SSE-KMS, versioned)           │
                          └──────────────────────┬───────────────────────┘
                                                 │  S3 event / schedule
                                                 ▼
                          ┌──────────────────────────────────────────────┐
  Step Functions ───────► │ Ingest pipeline (Map over new/changed docs)  │
                          │   1. extract text                            │
                          │   2. chunk (structure-aware)                 │
                          │   3. embed → Bedrock Titan Embeddings v2     │
                          │   4. upsert → S3 Vectors (+ metadata)        │
                          │   5. write curated chunks → S3 Parquet       │
                          └──────────────────────┬───────────────────────┘
                                                 │  index queryable
                                                 ▼
                          ┌──────────────────────────────────────────────┐
  user question ────────► │ Query Lambda (API Gateway / Function URL)    │
                          │   1. embed the question                      │
                          │   2. query S3 Vectors → top-k chunks         │
                          │   3. best score < floor? refuse for $0       │
                          │   4. prompt contract → Bedrock Claude        │
                          │      (Converse API)                          │
                          └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                                 cited answer + verified source list
```

All infra defined in **Terraform**; deployed via **GitHub Actions**. Every box above already has a **proven local twin** — the complete prototype in `local/` runs the same flow with Ollama + FAISS, and the README's mapping table pairs each stage with its AWS service.

---

## Tech stack & why

| Layer | Choice | Why |
|---|---|---|
| Object storage | **S3** | Raw + curated zones; the DEA-C01 backbone |
| Text extraction | **Lambda** (pypdf / unstructured) or **Glue** for scale | Lambda is simpler for a portfolio; mention Glue as the scale path |
| Embeddings | **Amazon Bedrock — Titan Text Embeddings v2** | Managed, cheap, 256/512/1024-dim configurable |
| Vector store | **Amazon S3 Vectors** (GA early 2026) | **Zero idle cost**, sub-second queries, up to 90% cheaper than alternatives |
| Generation | **Bedrock — Claude Haiku** via the **Converse API** | Cheap, fast; Converse keeps you model-agnostic |
| Orchestration | **Step Functions** (Map state) | Visual, serverless, easy idempotency |
| Serving | **Lambda + API Gateway** (or a Lambda Function URL) | Serverless endpoint |
| IaC / CI-CD | **Terraform** + **GitHub Actions** | The combo AU employers ask for |
| Observability | **CloudWatch** logs + custom metrics | Ops maturity |

> **Cost landmine to call out in your README (and avoid):** the obvious vector-store choice, **OpenSearch Serverless, has a ~$700/month minimum floor** even when idle. **S3 Vectors has zero idle cost** — you pay only for storage + queries. Choosing S3 Vectors *and explaining why* is itself a signal of cost-awareness. (Exactly what this project did: FAISS for the $0 local dev loop — see `local/03_index.py` — with S3 Vectors reserved for the deployed version.)

---

## Cost guardrails (set these up first)

1. **Billing alarm on day one** — CloudWatch billing alarm at e.g. US$5, plus a Budgets alert.
2. **S3 Vectors** — pennies at portfolio scale; no idle cost.
3. **Bedrock** — embeddings and Haiku are cheap per-call; the only way to overspend is re-embedding a huge corpus in a loop. **Keep the corpus small (50–500 docs)** and **idempotent** (don't re-embed unchanged files).
4. **Never** wire this to OpenSearch Serverless "just to try it."
5. Tag everything `project=rag-portfolio` so you can see spend in Cost Explorer.

Realistic total to build and demo: **a few US dollars.**

---

## Prerequisites

- AWS account; **enable Bedrock model access** for Titan Embeddings v2 and a Claude model (console → Bedrock → Model access — approval is usually instant).
- AWS CLI configured; **Terraform** and **Python 3.12** installed locally; Docker optional.
- **Corpus: settled.** The 73 Terraform-AWS-provider markdown docs, fetched reproducibly at pinned tag `v6.53.0` by `local/00_fetch_corpus.py` — the port re-ingests the same bytes the local prototype indexed.
- **Repo: this one.** `infra/` (Terraform), `src/` (Lambda code) and `eval/` land beside the finished `local/`, so one history tells the whole story — prototyped locally for $0, then ported.
- **Git workflow for the port:** one short-lived branch per phase (`aws/phase-N-<name>`) → PR → self-review the diff → merge commit → delete the branch. `main` only ever holds completed, working phases.

---

## The build, phase by phase

### Phase 0 — Repo & foundations *(½ day)*
- Repo, README and architecture diagram already exist from the local build — add the `infra/` / `src/` / `eval/` skeleton beside `local/`.
- Write the **Terraform backend** (S3 state bucket + DynamoDB lock) and a `providers.tf`.
- Create the **billing alarm** and a **KMS key** for encryption.
- *First port branch: `aws/phase-0-foundations`. Commit early; this is also your CI/CD starting point.*

### Phase 1 — Storage layer *(½ day)*
- Terraform: **two S3 buckets** — `raw` and `curated` — both **SSE-KMS encrypted**, **versioned**, **public access blocked**.
- Define a **least-privilege IAM role** per Lambda (read raw, write curated, call Bedrock, write S3 Vectors). Resist `*` permissions — interviewers notice.
- *Deliverable: `terraform apply` creates the buckets cleanly.*

### Phase 2 — Ingestion *(½ day)*
- Upload your corpus to the raw bucket under a sensible prefix (`raw/source=docs/…`).
- Add an **S3 event notification** (or an EventBridge schedule) that will kick off the pipeline on new/changed objects.
- *Decision to document: event-driven vs scheduled batch — explain the trade-off in your README.*

### Phase 3 — Processing: extract → chunk → embed *(1–2 days, the core)*
This is where the data-engineering substance lives.
1. **Extract** text from each document (pypdf / `unstructured` for mixed formats). Handle failures gracefully (dead-letter the bad files).
2. **Chunk**: port `local/02_chunk.py`'s **structure-aware** splitter as-is. *(Settled locally: on this corpus it averages ~174 tokens — well under the classic 500–800 guidance — and Task 4/5 probes showed retrieval stays precise and answers don't run thin. No overlap needed when chunks follow section boundaries.)* Store per chunk: `chunk_id`, `source_uri`, `position`, and a **content hash**.
3. **Embed**: call **Bedrock Titan Embeddings v2** in **batches**. Dimension is **locked at 1024** — the local build chose `bge-m3` to match Titan v2 exactly so the index schema ports unchanged. Batching is your main cost control.
4. **Idempotency**: skip any chunk whose content hash already exists — this is what stops you re-embedding (and re-paying) on every run.
- *Deliverable: a Lambda (or Glue job) that turns a document into `(chunk, embedding, metadata)` records.*

### Phase 4 — Vector store *(½ day)*
- Create an **S3 Vectors** vector bucket + **index** with the **matching dimension** and **cosine** distance.
- **Upsert** vectors with their metadata (`source_uri`, `position`, etc.) so you can filter at query time.
- Also write the **curated chunks to S3 as Parquet** (queryable via Athena) — shows you keep a structured copy, not just vectors.
- *Deliverable: corpus fully indexed; a quick script proves a similarity query returns sensible neighbours.*

### Phase 5 — Retrieval + generation *(1 day)*
- **Query Lambda**: embed the incoming question → **S3 Vectors top-k** (start k=5) with optional metadata filter → assemble retrieved chunks into a context block.
- **Hard refusal floor before generating** (from `local/05_answer.py`): if the best retrieval score is below the floor, refuse without calling Bedrock — unanswerable questions cost $0. ⚠️ **Recalibrate the threshold**: the local 0.50 belongs to `bge-m3`'s score distribution (hits ≈ 0.57–0.73, noise ≈ 0.38); Titan v2 will have its own — re-run the Task 4 calibration probes against the new index first.
- **Prompt template**: already written and battle-tested as `05_answer.py`'s `SYSTEM_PROMPT` — answer **only from the provided excerpts**, **cite the chunk id** per claim, exact refusal string when context is insufficient (reduces hallucination — good talking point). Port the text as-is.
- Call **Bedrock Claude Haiku** via the **Converse API** (same messages shape as the local Ollama `/api/chat` — only the `chat()` function changes); return the answer **plus the source list**.
- *Deliverable: ask a question in the terminal, get a grounded, cited answer.*

### Phase 6 — API + a thin UI *(½ day)*
- Expose the query Lambda via **API Gateway** or a **Lambda Function URL** (auth: an API key or IAM).
- Optional but high-impact: a tiny **Streamlit** front-end (a chat box + the cited sources) so reviewers can *try it* from your README link. A working demo beats a diagram.

### Phase 7 — Orchestration & incremental loads *(1 day)*
- Wrap Phases 3–4 in a **Step Functions** state machine: *list new/changed objects → **Map** → extract → chunk → embed → upsert*. 
- Make the whole run **idempotent and incremental** (hash-based skip), with a **dead-letter queue** for failures and retries with backoff.
- *Deliverable: drop a new file in the raw bucket → it's searchable minutes later, with no duplicates.*

### Phase 8 — Production polish *(1–2 days, what separates you)*
- **CI/CD**: GitHub Actions → `terraform plan` on PR, `apply` on merge; lint + unit tests for the chunker/retriever.
- **Observability**: CloudWatch dashboards + **custom metrics** (docs processed, embedding latency, retrieval latency, **tokens/cost per query**).
- **Evaluation harness** (`/eval`): a small hand-built Q&A set; measure **retrieval hit-rate** and answer quality (RAGAS or a simple scoring script). Seeds already exist: `check_citations()` in `05_answer.py` and the calibration probes from Task 4. *Showing you measure RAG quality is rare in junior portfolios and lands well.*

### Phase 9 — README & packaging *(½ day)*
Your README is half the project's value. Include:
- One-paragraph **problem statement** + the **architecture diagram**.
- A **"production considerations & cost at scale"** section: the OpenSearch-vs-S3-Vectors cost decision, how you'd scale embedding (Glue), how you'd add re-ranking, chunking trade-offs.
- A **live demo link** (or a 60-second GIF), a **cost note** ("runs for ~$X"), and clear **setup steps**.
- A short **eval results** table.

---

## Stretch goals (pick one if you want extra polish)
- **Hybrid search** + a **re-ranker** (retrieve broad, re-rank top candidates) — meaningfully better answers.
- **Quantized embeddings** (int8) to cut storage ~4× — a sharp cost-engineering detail.
- Swap the custom pipeline for a **Bedrock Knowledge Base** in a branch and write up the **build-vs-managed trade-off** — shows architectural judgement.
- Multi-tenant metadata filtering (scope retrieval by `source`/`tenant`).

---

## Interview talking points (rehearse these)
- **Why S3 Vectors over OpenSearch Serverless?** Cost (zero idle vs ~$700/mo floor) at portfolio/small scale; when you'd flip to OpenSearch (high QPS, hybrid search).
- **Chunking strategy** and why overlap matters for retrieval recall.
- **How you keep it idempotent / incremental** and why that's both a correctness and a cost issue.
- **How you measure RAG quality**, and what you'd improve first (usually retrieval, not the LLM).
- **Security**: least-privilege IAM, KMS, no public surface.

---

## Teardown checklist (run after each work session)
- [ ] No Glue/EMR jobs left running; no idle endpoints.
- [ ] S3 Vectors + S3 are fine to leave (cheap, no idle compute).
- [ ] Confirm **no OpenSearch Serverless collection** was created by any "quick-create" wizard.
- [ ] Glance at Cost Explorer (filter tag `project=rag-portfolio`).

---

*Realistic effort: ~7–10 focused days for a strong v1, less if you reuse Terraform modules. Build Phases 0–6 first for a working demo, then 7–9 to make it look senior.*

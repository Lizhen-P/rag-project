# Task 0 explained — `00_fetch_corpus.py`

*(Numbered 0 because it was added after Tasks 1–2, when the repo went public —
it's the step that happens before everything else: getting the raw data.)*

## Part 1 — Concepts

### Why the corpus isn't in git

The 73 docs are HashiCorp's property (MPL-2.0 license). Publishing them in this
repo would be *redistribution* — legal under MPL-2.0 but only with license
notices carried along. Cleaner: don't ship the data, ship **the recipe for
getting the data**. Anyone cloning the repo runs one script and ends up with
byte-identical inputs.

This is the standard shape of real data pipelines: **code in git, data in a
store, and a reproducible path from one to the other.** Our "store" is
HashiCorp's GitHub repo; in the AWS port it becomes an S3 raw bucket.

### Why pin `v6.53.0` instead of "latest"

The provider releases roughly weekly and docs change with every release. If the
script fetched whatever is newest, my `chunks.jsonl` (built 2026-07-02) and a
clone's `chunks.jsonl` (built whenever) would silently differ — same code,
different data, different retrieval results. Pinning the git tag makes the
corpus **deterministic**: verified byte-for-byte identical to what the pipeline
was built from. Data engineers call this idea *reproducibility*; you'll meet it
again as "pin your dependencies" (uv.lock does the same job for packages).

### Idempotence (safe re-runs)

The script skips files that already exist, so it can be interrupted and re-run
without re-downloading everything — handy on a slow connection. A job you can
re-run without side effects is called **idempotent**; Airflow/Lambda jobs in the
AWS port will be designed the same way.

## Part 2 — Walkthrough

- **`BASE_URL`** — `raw.githubusercontent.com` serves the raw bytes of any file
  in a public GitHub repo at `/<owner>/<repo>/<git-ref>/<path>`. The git ref in
  our URL is the release tag, which is what pins the version.
- **`DOC_NAMES`** — an explicit manifest instead of "download the whole docs
  folder". The provider documents ~1,500 resources; we want exactly the 73 that
  belong to our 8 services. A manifest also *documents* the corpus: the repo
  states precisely which inputs the pipeline uses.
- **`fetch()`** — same `urllib` pattern as Task 1's `embed()`, plus two new
  ideas: `timeout=30` (a dead connection errors instead of hanging forever) and
  a retry loop over `(1, 2)` — transient network blips get one second chance,
  real problems still fail loudly.
- **`sys.argv`** — the command line, as a list. `argv[0]` is the script's own
  name; `argv[1]` is the first argument if given. We use it as an optional
  destination override, which is also how the version pin was verified (fetch
  into a scratch dir, `diff -r` against `local/corpus/`).
- **`dest.write_bytes(body)`** — writes exactly the bytes the server sent, no
  decode/encode round-trip, so checksums match upstream.
- **`sys.exit(1)`** — a non-zero exit code is how a script tells the shell "I
  failed". Anything that later automates this script (cron, Airflow, CI) keys
  off that code.

## Part 3 — Actual run output

Fresh clone (first run):

```
Fetching 73 docs @ terraform-provider-aws v6.53.0
     into local/corpus

[ 1/73] athena_capacity_reservation.md — 2,812 bytes
[ 2/73] athena_data_catalog.md — 4,072 bytes
[ 3/73] athena_database.md — 3,696 bytes
  ...
[73/73] sfn_state_machine.md — 9,693 bytes

done: 73 downloaded, 0 skipped, 0 failed
```

Second run (everything already on disk — the idempotence in action):

```
[ 1/73] athena_capacity_reservation.md — already present, skipped
  ...
done: 0 downloaded, 73 skipped, 0 failed
```

## Part 4 — AWS mapping

| This script (local) | AWS port |
|---|---|
| Download docs from a pinned upstream ref | Ingest job landing raw files in S3 (`raw/source=docs/...`) |
| `DOC_NAMES` manifest in code | Manifest/config in the repo, or a Glue crawler inventorying the raw prefix |
| Pin `PROVIDER_TAG` | Version prefix or S3 object versioning on the raw bucket |
| Skip-if-exists re-runs | Idempotent ingest (overwrite-same-key writes, or checks against the bucket) |
| `sys.exit(1)` on failure | Job status → CloudWatch alarm / Step Functions `Fail` state |

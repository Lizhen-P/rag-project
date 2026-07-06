"""Task 0 — Fetch the corpus.

The 73 corpus documents are pages from HashiCorp's terraform-provider-aws
documentation (MPL-2.0 licensed). They are deliberately NOT committed to this
repo — instead of redistributing someone else's docs, we download them straight
from the source, pinned to one exact release so every clone of this project
chunks and indexes the *same bytes*.

Why pin a version? The provider ships a new release almost every week and its
docs change with it. "Download whatever is newest" would mean my chunks.jsonl
and yours disagree — the classic reproducibility bug in data pipelines. Pinning
the git tag makes the corpus a pure function of this script.

Run it with:
    uv run python local/00_fetch_corpus.py

Re-running is safe: files already on disk are skipped, so a broken connection
can be resumed by just running it again.
"""

import sys
import urllib.request
from pathlib import Path

# The exact provider release the corpus (and chunks.jsonl) was built from.
PROVIDER_TAG = "v6.53.0"

# raw.githubusercontent.com serves any file from any git ref of a public repo:
#   https://raw.githubusercontent.com/<owner>/<repo>/<tag>/<path-in-repo>
# In the provider repo, resource docs live at website/docs/r/<name>.html.markdown.
BASE_URL = (
    "https://raw.githubusercontent.com/hashicorp/terraform-provider-aws/"
    f"{PROVIDER_TAG}/website/docs/r"
)

# The manifest: every doc the pipeline uses, grouped by AWS service. Each name
# maps to the resource `aws_<name>` upstream and is saved as corpus/<name>.md.
DOC_NAMES = [
    # Athena (6)
    "athena_capacity_reservation",
    "athena_data_catalog",
    "athena_database",
    "athena_named_query",
    "athena_prepared_statement",
    "athena_workgroup",
    # DynamoDB (4)
    "dynamodb_table",
    "dynamodb_table_export",
    "dynamodb_table_item",
    "dynamodb_table_replica",
    # Glue (21)
    "glue_catalog",
    "glue_catalog_database",
    "glue_catalog_table",
    "glue_catalog_table_optimizer",
    "glue_classifier",
    "glue_connection",
    "glue_crawler",
    "glue_data_catalog_encryption_settings",
    "glue_data_quality_ruleset",
    "glue_dev_endpoint",
    "glue_job",
    "glue_ml_transform",
    "glue_partition",
    "glue_partition_index",
    "glue_registry",
    "glue_resource_policy",
    "glue_schema",
    "glue_security_configuration",
    "glue_trigger",
    "glue_user_defined_function",
    "glue_workflow",
    # Kinesis (7)
    "kinesis_account_settings",
    "kinesis_analytics_application",
    "kinesis_firehose_delivery_stream",
    "kinesis_resource_policy",
    "kinesis_stream",
    "kinesis_stream_consumer",
    "kinesis_video_stream",
    # KMS (2)
    "kms_key",
    "kms_key_policy",
    # Lambda (5)
    "lambda_function",
    "lambda_function_event_invoke_config",
    "lambda_function_recursion_config",
    "lambda_function_url",
    "lambda_permission",
    # S3 (25)
    "s3_bucket",
    "s3_bucket_abac",
    "s3_bucket_accelerate_configuration",
    "s3_bucket_acl",
    "s3_bucket_analytics_configuration",
    "s3_bucket_cors_configuration",
    "s3_bucket_intelligent_tiering_configuration",
    "s3_bucket_inventory",
    "s3_bucket_lifecycle_configuration",
    "s3_bucket_logging",
    "s3_bucket_metadata_configuration",
    "s3_bucket_metric",
    "s3_bucket_notification",
    "s3_bucket_object",
    "s3_bucket_object_lock_configuration",
    "s3_bucket_ownership_controls",
    "s3_bucket_policy",
    "s3_bucket_public_access_block",
    "s3_bucket_replication_configuration",
    "s3_bucket_request_payment_configuration",
    "s3_bucket_server_side_encryption_configuration",
    "s3_bucket_versioning",
    "s3_bucket_website_configuration",
    "s3_object",
    "s3_object_copy",
    # Step Functions (3)
    "sfn_activity",
    "sfn_alias",
    "sfn_state_machine",
]


def fetch(url: str) -> bytes:
    """Download `url` and return the raw bytes, retrying once on failure.

    One retry is enough to shrug off a flaky connection without hiding a real
    problem (a typo'd name would fail twice and surface loudly).
    """
    last_error: Exception | None = None
    for attempt in (1, 2):  # at most two tries
        try:
            # timeout= makes a dead connection fail in 30s instead of hanging.
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()  # .read() gives the body as bytes
        except OSError as err:  # covers URLError, HTTPError, socket timeouts
            last_error = err
            if attempt == 1:
                print(f"      retrying after error: {err}")
    # Both attempts failed — re-raise so the caller can record the failure.
    raise last_error  # type: ignore[misc]


def main() -> None:
    # Where to save. Default: the corpus/ folder next to this script — computed
    # from __file__ (this script's own path), so it works from any directory.
    # An optional command-line argument overrides it (handy for testing):
    #     uv run python local/00_fetch_corpus.py /some/other/dir
    if len(sys.argv) > 1:
        dest_dir = Path(sys.argv[1])
    else:
        dest_dir = Path(__file__).parent / "corpus"

    # Create the folder if needed. parents=True builds intermediate dirs too;
    # exist_ok=True means "don't error if it's already there".
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(DOC_NAMES)} docs @ terraform-provider-aws {PROVIDER_TAG}")
    print(f"     into {dest_dir}\n")

    downloaded, skipped, failed = 0, 0, []

    # enumerate(..., start=1) numbers the loop 1..73 for the progress display.
    for i, name in enumerate(DOC_NAMES, start=1):
        dest = dest_dir / f"{name}.md"

        # Already fetched on a previous run? Skip it — this is what makes the
        # script resumable after an interrupted download.
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{i:2d}/{len(DOC_NAMES)}] {name}.md — already present, skipped")
            skipped += 1
            continue

        url = f"{BASE_URL}/{name}.html.markdown"
        try:
            body = fetch(url)
        except OSError as err:
            print(f"[{i:2d}/{len(DOC_NAMES)}] {name}.md — FAILED: {err}")
            failed.append(name)
            continue

        dest.write_bytes(body)  # save exactly what the server sent, no re-encoding
        print(f"[{i:2d}/{len(DOC_NAMES)}] {name}.md — {len(body):,} bytes")
        downloaded += 1

    print(f"\ndone: {downloaded} downloaded, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("failed docs:", ", ".join(failed))
        sys.exit(1)  # non-zero exit code = "something went wrong" to the shell


if __name__ == "__main__":
    main()

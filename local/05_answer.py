"""Task 5 — Answer. The full RAG loop, end to end.

    question -> embed -> top-k chunks (Task 4) -> PROMPT -> local LLM -> cited answer

The new idea here is the PROMPT CONTRACT: we hand the model the retrieved
excerpts plus strict rules — answer only from these, cite every claim, admit
when the excerpts don't cover it. The model brings grammar and synthesis;
the FACTS come from retrieval. That's the whole trick of RAG.

Two honesty layers:
  1. HARD FLOOR — if the best retrieval score is below SCORE_FLOOR, we refuse
     without even calling the LLM (Task 4's calibration: real hits ~0.57-0.73,
     noise ~0.38). No context worth reading = nothing to generate = no cost.
  2. SOFT RULE — the system prompt tells the model to say "not covered" when
     the excerpts don't actually answer the question (retrieval can fetch
     related-but-wrong chunks; the model is the second filter).

Run it with (model defaults to qwen3.5:9b; --model gpt-oss for the challenger):
    uv run python local/05_answer.py "How do I enable versioning on an S3 bucket?"
    uv run python local/05_answer.py --model gpt-oss "same question, other model"
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

import faiss
import numpy as np

# --- Configuration -----------------------------------------------------------

OLLAMA_EMBED = "http://localhost:11434/api/embed"
OLLAMA_CHAT = "http://localhost:11434/api/chat"
EMBED_MODEL = "bge-m3"

# The bake-off contenders. Each entry records the model's tag AND its
# thinking-channel quirk — the per-model glue every LLM integration ends up
# needing (exactly what Bedrock's Converse API exists to hide).
#   qwen3.5 : reasons by default and returns an EMPTY response unless told
#             "think": false (measured the hard way).
#   gpt-oss : always reasons (separate "thinking" field) but the answer
#             reliably lands in the response — no switch needed (None).
MODELS = {
    "qwen": {"tag": "qwen3.5:9b", "think": False},
    "gpt-oss": {"tag": "gpt-oss:20b", "think": None},
}

TOP_K = 5          # chunks handed to the model (Task 4's net width)
SCORE_FLOOR = 0.50  # below this best-score, refuse without generating

INDEX_PATH = Path(__file__).parent / "index.faiss"
META_PATH = Path(__file__).parent / "index_meta.jsonl"

# The standing orders. A system message is instructions that outrank the
# user message — the model treats it as "who I am / my rules" rather than
# "the thing to respond to".
SYSTEM_PROMPT = """\
You answer questions about AWS infrastructure managed with Terraform, using
ONLY the documentation excerpts provided in the user message.

Rules:
- After every factual claim, cite the excerpt it came from in square
  brackets, e.g. [s3_bucket_versioning-000]. Cite only ids that exist.
- If the excerpts do not contain the answer, reply exactly:
  "The provided documentation does not cover this." Do not guess.
- Be concise: a short direct answer first, then a minimal Terraform example
  only if one appears in the excerpts."""


# --- Ollama calls ------------------------------------------------------------

def embed(text: str) -> np.ndarray:
    """Question -> (1, 1024) float32 matrix. Same as Task 4."""
    payload = json.dumps({"model": EMBED_MODEL, "input": [text]}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_EMBED, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return np.array(json.load(resp)["embeddings"], dtype=np.float32)


def chat(model_cfg: dict, system: str, user: str) -> dict:
    """Send a system+user conversation to Ollama's chat API; return its reply.

    /api/chat mirrors how hosted LLM APIs work (Bedrock Converse, OpenAI,
    Anthropic): a MESSAGES list with roles, not one bare prompt string.
    Porting to AWS later means changing this function and nothing else.
    """
    body = {
        "model": model_cfg["tag"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            # temperature 0.2: near-deterministic. High temperature = creative
            # variety; grounded Q&A wants the boring, faithful behavior.
            "temperature": 0.2,
            # Context window budget: 5 chunks can reach ~4k tokens at our max
            # chunk size, plus rules + question + answer. The default window
            # (4096) could silently TRUNCATE the excerpts — raise it.
            "num_ctx": 8192,
            "num_predict": 700,
        },
    }
    if model_cfg["think"] is not None:  # only send the switch if the model has one
        body["think"] = model_cfg["think"]

    req = urllib.request.Request(
        OLLAMA_CHAT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


# --- The RAG glue -------------------------------------------------------------

def build_user_message(hits: list[dict], question: str) -> str:
    """Format retrieved chunks + question into one user message.

    Each excerpt is labeled with its chunk_id (for citing) and source file
    (for the human reading the answer later). The model sees exactly what we
    retrieved — nothing more. This message IS the model's entire world.
    """
    blocks = []
    for r in hits:
        blocks.append(f"[{r['chunk_id']}] (from {r['source_uri']})\n{r['text']}")
    excerpts = "\n\n---\n\n".join(blocks)
    return f"Documentation excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def check_citations(answer: str, hits: list[dict]) -> str:
    """Trust but verify: does every [citation] point at a chunk we provided?

    LLMs can invent plausible-looking ids. A cheap regex catches it: find
    every [...] that looks like a chunk_id, compare against the set we
    actually sent. This is a tiny taste of Phase 8's evaluation harness.
    """
    provided = {r["chunk_id"] for r in hits}
    # \s* tolerates "[ id ]" — gpt-oss pads its brackets with spaces.
    cited = set(re.findall(r"\[\s*([a-z0-9_]+-\d{3})\s*\]", answer))
    bogus = cited - provided
    if not cited:
        return "citations: NONE found in answer"
    if bogus:
        return f"citations: {len(cited)} cited, INVALID ids: {sorted(bogus)}"
    return f"citations: {len(cited)} cited, all valid"


def main() -> None:
    # Manual flag parsing, the minimal kind: an optional "--model X" prefix,
    # then everything else re-joined is the question (same trick as Task 4).
    args = sys.argv[1:]
    model_key = "qwen"
    if args[:1] == ["--model"]:
        model_key, args = args[1], args[2:]
    question = " ".join(args).strip()
    if not question or model_key not in MODELS:
        print(f'usage: uv run python local/05_answer.py [--model {"|".join(MODELS)}] "question"')
        sys.exit(1)
    model_cfg = MODELS[model_key]

    # -- Retrieve (Task 4, compressed) ----------------------------------------
    index = faiss.read_index(str(INDEX_PATH))
    records = [json.loads(line) for line in META_PATH.open()]
    assert index.ntotal == len(records), "index/sidecar desync — re-run 03_index.py"

    q = embed(question)
    faiss.normalize_L2(q)
    scores, ids = index.search(q, TOP_K)
    hits = [records[i] for i in ids[0]]

    print(f"question  : {question}")
    print(f"model     : {model_cfg['tag']}")
    print("retrieved :")
    for score, r in zip(scores[0], hits):
        print(f"    {score:.3f}  {r['chunk_id']}")

    # -- Honesty layer 1: the hard floor --------------------------------------
    best = float(scores[0][0])
    if best < SCORE_FLOOR:
        print(f"\nbest score {best:.3f} < floor {SCORE_FLOOR} — the corpus has "
              "nothing relevant; refusing without calling the LLM.")
        sys.exit(0)

    # -- Generate --------------------------------------------------------------
    reply = chat(model_cfg, SYSTEM_PROMPT, build_user_message(hits, question))
    answer = reply["message"]["content"].strip()

    gen_s = reply.get("eval_duration", 1) / 1e9
    tok_s = reply.get("eval_count", 0) / gen_s

    print(f"\n--- answer ({reply.get('eval_count', 0)} tokens, "
          f"{gen_s:.1f}s, {tok_s:.0f} tok/s) ---\n")
    print(answer)

    # -- Verify + provenance legend --------------------------------------------
    print(f"\n{check_citations(answer, hits)}")
    print("sources:")
    for r in hits:
        # same space-tolerant match as check_citations, for the * marker
        mark = "*" if re.search(rf"\[\s*{re.escape(r['chunk_id'])}\s*\]", answer) else " "
        print(f"  {mark} {r['chunk_id']:44s} {r['source_uri']}")


if __name__ == "__main__":
    main()

"""Task 1 — Embed probe.

Goal: FEEL what an embedding is. Turn text into a 1024-dim vector via the local
bge-m3 model (through Ollama), then show that semantically similar text lands
"closer" in vector space than unrelated text.

This is the single most important intuition in the whole RAG project. Everything
downstream (chunking, indexing, retrieval) is plumbing around this one idea.

Run it with:
    uv run python local/01_embed_probe.py
"""

import json
import urllib.request

import numpy as np

# Ollama's embedding endpoint + the model we pulled.
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"


def embed(text: str) -> list[float]:
    """Return the embedding vector for `text` by calling Ollama.

    The endpoint contract:
        POST http://localhost:11434/api/embed
        body:    {"model": "bge-m3", "input": "<text>"}
        returns: {"embeddings": [[...1024 floats...]]}
    """
    # 1. Build the JSON payload and encode it to BYTES (urllib requires bytes for
    #    the request body, not a str -- that's what .encode() is for).
    payload = json.dumps({"model": MODEL, "input": text}).encode("utf-8")

    # 2. Describe the HTTP request: URL + body + content-type header.
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,  # having data= makes this a POST automatically
        headers={"Content-Type": "application/json"},
    )

    # 3. Send it and parse the JSON that comes back.``
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)

    # 4. The API can embed many strings at once, so it returns a LIST of vectors.
    #    We sent one string, so we want the first (and only) vector.
    return data["embeddings"][0]


def cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

        cos(a, b) = (a . b) / (||a|| * ||b||)

    Ranges from -1 (opposite) to 1 (identical direction). For text embeddings
    the useful range is roughly 0 (unrelated) to 1 (same meaning). Cosine looks
    at the ANGLE between vectors, ignoring their length -- so two texts count as
    "similar" when they point the same way, regardless of magnitude.
    """
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    # Two sentences about the SAME topic (streaming ingestion) that share almost
    # no words, plus one about something totally UNRELATED.
    a = "Kinesis Data Streams ingests real-time event data."
    b = "Firehose delivers streaming records into an S3 bucket."
    c = "The recipe calls for two cups of flour and a pinch of salt."

    # Turn each sentence into a 1024-dim vector.
    va = embed(a)
    vb = embed(b)
    vc = embed(c)

    # Sanity check: bge-m3 should give us 1024 numbers per sentence.
    print(f"vector dimension: {len(va)}\n")

    # Compare every pair. Watch the numbers:
    print("cosine similarity (higher = more similar in MEANING)")
    print(f"  A vs B  (both streaming)  = {cosine(va, vb):.3f}   <- expect HIGHEST")
    print(f"  A vs C  (stream vs recipe)= {cosine(va, vc):.3f}")
    print(f"  B vs C  (stream vs recipe)= {cosine(vb, vc):.3f}")

    print(
        "\nNote: A and B share almost no words, yet score high -- retrieval works "
        "on meaning, not keyword overlap."
    )


if __name__ == "__main__":
    main()

"""Task 2 — Chunker.

Read every markdown doc in local/corpus/, split each one into structure-aware
chunks of roughly 500-800 tokens with a little overlap, attach metadata
(chunk_id, source_uri, position, content hash), and write everything to
local/chunks.jsonl — one JSON object per line, ready for embedding in Task 3.

Pure standard library — nothing to install.

Run it with:
    uv run --no-project --python 3.12 python local/02_chunk.py
"""

import hashlib
import json
from pathlib import Path

# --- Configuration -----------------------------------------------------------

# Path(__file__) is THIS file; .parent is the folder it lives in (local/).
# The "/" operator joins paths — pathlib's nicer version of "folder + filename".
CORPUS_DIR = Path(__file__).parent / "corpus"
OUT_PATH = Path(__file__).parent / "chunks.jsonl"

# Sizing. Rule of thumb: 1 token ~= 4 characters of English text.
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 600          # aim for chunks around this size...
MAX_TOKENS = 800             # ...and never exceed this.


def estimate_tokens(text: str) -> int:
    """Rough token count. Good enough for sizing; no tokenizer install needed."""
    return len(text) // CHARS_PER_TOKEN


# --- Step 1: strip frontmatter ------------------------------------------------

def strip_frontmatter(text: str) -> str:
    """Remove the YAML block between the two '---' lines at the very top.

    Our corpus files start with metadata like `subcategory: "Kinesis"` that is
    for the website's benefit, not the reader's — noise for embedding purposes.
    """
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        # Walk forward until we find the closing '---', then keep what follows.
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                # lines[i + 1:] means "everything after line i" (list slicing).
                return "\n".join(lines[i + 1:])
    return text  # no frontmatter found — return unchanged


# --- Step 2: split into sections at markdown headings -------------------------

def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a document at its headings ('#', '##', ...).

    Returns a list of (heading, section_text) pairs — the document's natural
    sub-topics. This is what makes the chunker "structure-aware": we cut where
    the AUTHOR drew boundaries, not at arbitrary character counts.

    Wrinkle: lines inside ``` code fences can start with '#' (comments in
    terraform/bash) and must NOT be mistaken for headings, so we track whether
    we're inside a fence with a boolean toggle.
    """
    sections: list[tuple[str, str]] = []
    heading = "(intro)"          # text before the first heading goes here
    current_lines: list[str] = []
    in_code_fence = False

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence   # flip on fence open/close

        is_heading = line.startswith("#") and not in_code_fence
        if is_heading:
            # A new sub-topic starts. First, save the section we were building.
            sections.append((heading, "\n".join(current_lines)))
            # '# Foo' -> 'Foo': lstrip("#") removes leading #s, strip() spaces.
            heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    sections.append((heading, "\n".join(current_lines)))  # don't lose the last one

    # Drop sections that are empty or whitespace-only.
    return [(h, body) for h, body in sections if body.strip()]


# --- Step 3: split a section into paragraphs ----------------------------------

def split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines — the standard definition of a paragraph.

    "\\n\\n" is two newlines back-to-back, i.e. an empty line between blocks.
    Oversized paragraphs (e.g. a 50-bullet argument list with no blank lines)
    get broken up line-by-line so no single piece can blow past MAX_TOKENS.
    """
    paragraphs = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if estimate_tokens(para) <= MAX_TOKENS:
            paragraphs.append(para)
        else:
            # Fallback: break the giant paragraph into line-groups of ~TARGET size.
            buf: list[str] = []
            for line in para.split("\n"):
                buf.append(line)
                if estimate_tokens("\n".join(buf)) >= TARGET_TOKENS:
                    paragraphs.append("\n".join(buf))
                    buf = []
            if buf:
                paragraphs.append("\n".join(buf))
    return paragraphs


# --- Step 4: pack paragraphs into chunks, with overlap -------------------------

def pack_paragraphs(paragraphs: list[str]) -> list[str]:
    """Greedily fill chunks up to ~TARGET_TOKENS, then start a new one.

    Overlap: each new chunk is seeded with the LAST paragraph of the previous
    chunk, so content near a cut appears whole in two chunks. That's the
    ~10-15% insurance against slicing an answer in half.
    """
    chunks: list[str] = []
    current: list[str] = []     # paragraphs going into the chunk being built

    for para in paragraphs:
        candidate = "\n\n".join(current + [para])   # what if we added this one?
        if current and estimate_tokens(candidate) > TARGET_TOKENS:
            # Adding it would overshoot -> close the current chunk...
            chunks.append("\n\n".join(current))
            # ...and start the next one with the previous tail (the overlap).
            # current[-1] is the last element of the list.
            overlap = current[-1]
            if estimate_tokens(overlap) < TARGET_TOKENS // 3:
                current = [overlap, para]
            else:
                current = [para]    # tail too big to repeat — skip the overlap
        else:
            current.append(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# --- Step 5: chunk one file, attaching metadata --------------------------------

def chunk_file(path: Path) -> list[dict]:
    """Turn one markdown file into a list of chunk records (dicts)."""
    # read_text() slurps the whole file into one string.
    text = strip_frontmatter(path.read_text(encoding="utf-8"))
    doc_name = path.stem        # 'kinesis_stream.md' -> 'kinesis_stream'

    records = []
    for heading, section_text in split_into_sections(text):
        paragraphs = split_into_paragraphs(section_text)
        for chunk_text in pack_paragraphs(paragraphs):
            # Prepend a breadcrumb so the chunk (and its vector!) knows where
            # it lives. "[kinesis_stream > Argument Reference]" gives both the
            # embedding model and the LLM crucial context that the raw
            # paragraph alone would lack.
            full_text = f"[{doc_name} > {heading}]\n{chunk_text}"

            # Fingerprint the text: same text in -> same hash out, always.
            # In Task 3 this lets us SKIP re-embedding unchanged chunks.
            content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

            records.append(
                {
                    # :03d pads with zeros -> 000, 001, ... keeps sort order.
                    "chunk_id": f"{doc_name}-{len(records):03d}",
                    "source_uri": f"corpus/{path.name}",
                    "position": len(records),
                    "heading": heading,
                    "content_hash": content_hash,
                    "n_tokens_est": estimate_tokens(full_text),
                    "text": full_text,
                }
            )
    return records


# --- Main ----------------------------------------------------------------------

def main() -> None:
    all_records = []
    # sorted(...) makes the run order stable; glob("*.md") finds every markdown file.
    for path in sorted(CORPUS_DIR.glob("*.md")):
        all_records.extend(chunk_file(path))

    # Write JSONL: one JSON object per line. Streamable, appendable, and the
    # de-facto interchange format for data pipelines (S3, Athena, Firehose...).
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # ---- Stats so we can sanity-check the result ----
    sizes = [r["n_tokens_est"] for r in all_records]
    n_docs = len(set(r["source_uri"] for r in all_records))
    print(f"docs processed : {n_docs}")
    print(f"chunks written : {len(all_records)}  ->  {OUT_PATH.name}")
    print(f"chunk size est : min {min(sizes)} / avg {sum(sizes)//len(sizes)} / max {max(sizes)} tokens")

    # Show one full example so 'chunk' stops being an abstraction.
    example = next(r for r in all_records if "kinesis_stream.md" in r["source_uri"])
    print("\n--- example chunk record " + "-" * 40)
    for key, value in example.items():
        if key == "text":
            preview = value[:400] + ("..." if len(value) > 400 else "")
            print(f"text          : {preview}")
        else:
            print(f"{key:14}: {value}")


if __name__ == "__main__":
    main()

# Task 2 Explained — The Chunker (`02_chunk.py`)

> **Goal:** read every markdown doc in `local/corpus/`, split each into
> structure-aware chunks of roughly 500–800 tokens with a little overlap, attach
> metadata (chunk_id, source_uri, position, content hash), and write everything
> to `local/chunks.jsonl` — ready for embedding in Task 3.
>
> Pure standard library — nothing to install.

Run it with:

```bash
uv run python local/02_chunk.py
```

---

# Part I — Concepts

## Why chunk at all? (3 reasons)

**1. One vector can only hold one "gist."** An embedding squashes text into a
single point in space — a *summary of meaning*. `kinesis_stream.md` talks about
encryption, retention, shards, tags, pricing modes… squash all of that into one
vector and you get a mushy average that's *near everything and close to
nothing*. Small pieces = sharp, specific vectors.

**2. Retrieval granularity.** Ask *"what's the max retention for a Kinesis
stream?"* — the answer is **one line** (max 8760 hours). Retrieve the whole
200-line doc and you hand the LLM a haystack; retrieve the paragraph containing
that line and you hand it the needle.

**3. The context window is a budget.** We'll stuff the top-5 retrieved pieces
into the LLM prompt. Five whole documents might not fit; five focused
paragraphs always will.

> **Mental model:** a librarian doesn't answer "which *book* mentions this?" —
> she photocopies the *relevant page*. Chunks are the pages.

## What's a token? Why ~500–800 of them?

Models don't read characters or words — they read **tokens**: word-pieces
(`retention` might be one token; `Firehose` might be `Fire` + `hose`). Rule of
thumb: **1 token ≈ 4 characters** of English. We use that approximation instead
of installing a real tokenizer — sizing chunks doesn't need precision.

The 500–800 range is a trade-off:

| Size | Effect |
|---|---|
| Too small (~50 tokens) | sharp vector, but the retrieved piece lacks context — you get *"The default value is `false`"* with no idea what setting it refers to |
| Too big (~4000 tokens) | back to the mushy-average problem |
| **500–800** | ≈ several paragraphs ≈ one coherent sub-topic — still *means one thing* but stands on its own |

## Why overlap (~10–15%)?

A chunk boundary is a knife cut — sometimes it cuts an answer in half. Sentence
A ends chunk 3, its continuation B starts chunk 4; neither chunk alone answers
the question. **Overlap repeats the tail of each chunk at the head of the
next**, so anything near a boundary appears *whole in at least one chunk*.
~10–15% duplicated storage buys a real bump in retrieval recall. Cheap insurance.

## Why "structure-aware" instead of fixed windows?

The naïve chunker cuts every 700 tokens exactly — mid-sentence, mid-code-block,
wherever the knife lands. But the docs *already have* natural topic boundaries:
`## Example Usage`, `## Argument Reference`, `### stream_mode_details`. Markdown
headings are the author saying "new sub-topic starts here." Cutting **at
headings and paragraph breaks** keeps each chunk semantically whole. That's all
"structure-aware" means: respect the document's own seams.

## The metadata: every chunk carries a passport

| Field | Why it exists |
|---|---|
| `chunk_id` | unique name, e.g. `kinesis_stream-003` |
| `source_uri` | which file it came from → becomes the **citation** in the final answer |
| `position` | order within the doc (chunk 3 of 7) |
| `content_hash` | a **fingerprint** of the text — same text always → same hash. "Have I seen this fingerprint before?" = skip re-embedding unchanged content = the **idempotency** trick that saves money on AWS |
| `n_tokens_est` | size estimate, for stats and prompt budgeting |

---

# Part II — The code, station by station

## The big picture: an assembly line

```
file on disk
  → one big string                 (path.read_text)
  → string without frontmatter     (strip_frontmatter)
  → list of (heading, section)     (split_into_sections)
  → list of paragraphs             (split_into_paragraphs)
  → list of chunk texts            (pack_paragraphs)
  → list of dicts with metadata    (chunk_file)
  → lines in chunks.jsonl          (main)
```

Every function is one station with a clear **shape in → shape out**. Thinking
"what shape is my data right now?" is *the* core habit of data engineering.

## Configuration block

```python
CORPUS_DIR = Path(__file__).parent / "corpus"
```

- `__file__` — magic variable: the path of *the script itself*.
- `Path(...)` — wraps it in a "smart string" that knows it's a file path.
- `.parent` — chops off the filename → the `local/` folder.
- `/` here is **not division** — `Path` redefines it to mean "join paths".

Why not just `"corpus"`? A bare relative path depends on *which folder you run
the command from*. Anchoring to `__file__` means the script finds its corpus no
matter where you launch it.

```python
def estimate_tokens(text): return len(text) // CHARS_PER_TOKEN
```

`//` is **integer division** — divide, throw away the remainder (`7 // 2` → `3`).

## `strip_frontmatter` — removing the YAML header

Corpus files start with a `---`-fenced block (`subcategory: "Kinesis"`, …) —
website configuration, not content. Noise for embeddings.

```python
lines = text.split("\n")
```

`split("\n")` cuts one string into a **list of lines**: `"a\nb"` → `["a", "b"]`.

```python
if lines and lines[0].strip() == "---":
```

- `lines` alone asks "non-empty?" — an empty list counts as `False`.
- `and` evaluates left-to-right and **stops early** if the left is false — so
  `lines[0]` is never touched on an empty list. Order makes the guard safe.
- `.strip()` shaves whitespace so `"--- "` doesn't fool the comparison.

```python
for i in range(1, len(lines)):
    if lines[i].strip() == "---":
        return "\n".join(lines[i + 1:])
```

- `range(1, len(lines))` = every index *after* the first line (line 0 was the
  opening `---`). We hunt for the **closing** fence.
- `lines[i + 1:]` — a **slice**: "sub-list from index i+1 to the end" =
  everything after the closing fence.
- `"\n".join(...)` — the inverse of `split("\n")`: glue lines back into one
  string. Split and join are yin and yang.

```python
return text
```

Safety net: no fence found → return input untouched. A pipeline station should
**pass data through gracefully** when its job doesn't apply — not crash.

## `split_into_sections` — the structure-aware cut (a state machine)

Walks the file line by line; behavior depends on what it has seen before.
Three pieces of memory ("state") persist across the loop:

```python
sections: list[tuple[str, str]] = []   # finished sections pile up here
heading = "(intro)"                    # heading of the section we're INSIDE
current_lines: list[str] = []          # lines collected for that section
in_code_fence = False                  # inside ``` ... ``` right now?
```

- A **tuple** is like a list but fixed-size and unchangeable — perfect for "a
  pair that belongs together": `("Argument Reference", "…section text…")`.
- `heading` starts as `"(intro)"` because text before the first heading needs a
  home too.

### The code-fence toggle

```python
if line.strip().startswith("```"):
    in_code_fence = not in_code_fence
```

`not` flips a boolean → this is a **toggle switch**: fence open → on; fence
close → off. Why care? Terraform code blocks contain comment lines:

```
# This is a comment inside code   ← starts with '#' but is NOT a heading!
```

Without the toggle, comments would be mistaken for headings and code blocks
shredded into nonsense sections. One boolean = the difference between correct
and *subtly* broken — it wouldn't crash, it would quietly produce bad chunks and
bad retrieval. **The worst bugs are the silent ones.**

### Closing out a section

```python
is_heading = line.startswith("#") and not in_code_fence
if is_heading:
    sections.append((heading, "\n".join(current_lines)))   # 1. save old section
    heading = line.lstrip("#").strip()                     # 2. extract new title
    current_lines = []                                     # 3. reset the bucket
```

- Double parentheses in `append((h, x))`: inner pair *creates the tuple*, outer
  pair is the function call.
- `"## Argument Reference".lstrip("#")` strips leading `#`s (left only), then
  `.strip()` removes the leftover space → `"Argument Reference"`.

```python
sections.append((heading, "\n".join(current_lines)))  # after the loop!
```

The *last* section never gets closed inside the loop (closing happens when the
*next* heading appears — and there is no next heading at end-of-file). This is
the **final flush** — forgetting it is one of the most classic bugs in all
accumulate-in-a-loop code.

### The list-comprehension filter

```python
return [(h, body) for h, body in sections if body.strip()]
```

Read: "for each `(h, body)` pair in `sections`, keep it — but only if `body`
isn't just whitespace." Empty string counts as `False`, so this filters out
empty sections. `[x for x in things if condition]` = "filtered copy" — Python's
one-line filter/transform idiom.

## `split_into_paragraphs` — and the oversized-paragraph fallback

```python
for para in text.split("\n\n"):
```

`"\n\n"` = two newlines back-to-back = **a blank line** — the standard
definition of a paragraph boundary.

```python
    if not para:
        continue
```

`continue` = "skip the rest of this iteration, move to the next item."

**The problem the fallback solves:** an Argument Reference with 12+ bullet lines
and *no blank lines between them* is — by the `\n\n` rule — *one paragraph*,
potentially thousands of tokens. `pack_paragraphs` only arranges whole
paragraphs, so a monster paragraph would blow past `MAX_TOKENS`.

```python
    else:
        buf: list[str] = []
        for line in para.split("\n"):
            buf.append(line)
            if estimate_tokens("\n".join(buf)) >= TARGET_TOKENS:
                paragraphs.append("\n".join(buf))
                buf = []
        if buf:
            paragraphs.append("\n".join(buf))
```

Drop one level of granularity — paragraphs → **single lines** — and repackage:
fill a buffer, seal it at target size, start a new one. And `if buf:` at the end
is the **final flush** again (second appearance).

> This "too big? split at the next level down: sections → paragraphs → lines"
> idea is called *recursive splitting* — LangChain's
> `RecursiveCharacterTextSplitter` is exactly this, industrialized.

## `pack_paragraphs` — greedy packing with overlap

**Greedy packing:** stuff paragraphs into a box until the next one won't fit,
seal the box, grab a new one.

```python
candidate = "\n\n".join(current + [para])
```

- `+` on lists = **concatenate**; `[para]` wraps the single item in a list
  because you can only `+` a list with a list.
- `candidate` is a *hypothetical*: "what would the box look like **if** I added
  this?"

```python
if current and estimate_tokens(candidate) > TARGET_TOKENS:
```

The `current` guard matters: without it, a first paragraph already over target
would trigger "seal the box" on an **empty box** — appending an empty chunk.
Edge cases at the *start* of loops deserve as much paranoia as ones at the end.

```python
    chunks.append("\n\n".join(current))
    overlap = current[-1]
    if estimate_tokens(overlap) < TARGET_TOKENS // 3:
        current = [overlap, para]
    else:
        current = [para]
```

- `current[-1]` — **negative indexing**: `-1` = last element, `-2` =
  second-to-last.
- The sealed chunk's last paragraph becomes the next chunk's *first* — text near
  the cut exists **whole in both chunks**. Overlap, implemented in two lines.
- The `if` is a sanity valve: if that tail paragraph is huge (> ⅓ of target),
  repeating it would waste the next chunk's budget on duplication — skip it.

After the loop: `if current: chunks.append(...)` — **final flush, third
appearance.** It's officially a pattern.

## `chunk_file` — where metadata gets attached

```python
text = strip_frontmatter(path.read_text(encoding="utf-8"))
doc_name = path.stem     # 'kinesis_stream.md' -> 'kinesis_stream'
```

- `read_text()` = open + read entire file into one string + close, in one call.
- `encoding="utf-8"` — the same UTF-8 rulebook from Task 1, in the reading
  direction (bytes → characters).

### The breadcrumb trick

```python
full_text = f"[{doc_name} > {heading}]\n{chunk_text}"
```

An **f-string** — the `f` prefix lets you embed variables inside `{}`. Renders
as `[kinesis_stream > Argument Reference]` + the chunk.

Why prepend this *into the text itself*? Because **the vector only knows what's
in the text**. "The default value is `false`" embeds to a nearly meaningless
vector — but "[kinesis_stream > Argument Reference] The default value is
`false`" embeds into *Kinesis-flavored* vector space. We inject the context the
paragraph lost when we cut it out of its document. Cheap trick, measurable
retrieval gains.

### The fingerprint

```python
content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
```

Read inside-out (Python nests like Russian dolls):

1. `.encode("utf-8")` — text → bytes (hashes eat bytes, same rule as networks).
2. `hashlib.sha256(...)` — feed bytes to the fingerprint machine.
3. `.hexdigest()` — result as a 64-char hex string: `7adf2013...`.

Properties that make it our idempotency key: same input → *always* same output;
one character changed → completely different output; collisions practically
impossible. So "is this hash already in my index?" ⇔ "have I already embedded
this exact text?"

### The record

```python
"chunk_id": f"{doc_name}-{len(records):03d}",
```

- `len(records)` as a running counter: before the first append length is 0 →
  id `-000`; after, 1 → `-001`. The list *is* the counter.
- `:03d` — format spec: integer, at least 3 digits, zero-padded → `7` → `007`.
  Why? So alphabetical order equals numerical order — without padding,
  `chunk-10` sorts *before* `chunk-2`. Prevents genuinely annoying bugs in
  anything that lists files or keys (S3 included).

## `main` — orchestration, output, stats

```python
for path in sorted(CORPUS_DIR.glob("*.md")):
    all_records.extend(chunk_file(path))
```

- `.glob("*.md")` — every path matching the pattern (`*` = anything).
- `sorted(...)` — glob's order is *not guaranteed*; sorting makes runs
  **deterministic**.
- `extend` vs `append` — trips everyone once: `append` adds its argument as
  *one item* (you'd get a list of 73 *lists*); `extend` unpacks and adds each
  element (one flat list of 893 records).

```python
with OUT_PATH.open("w", encoding="utf-8") as f:
    for rec in all_records:
        f.write(json.dumps(rec) + "\n")
```

- `with` — borrow-and-return again: file guaranteed closed even on error.
  `"w"` = write mode (creates or **overwrites**).
- One `json.dumps(rec)` + one `\n` per record = the entire **JSONL** format:
  one JSON object per line. Streamable, appendable, processable line-by-line
  without loading everything — the lingua franca of data pipelines (Firehose,
  Athena, Bedrock batch all speak it).

```python
sizes = [r["n_tokens_est"] for r in all_records]        # transform
n_docs = len(set(r["source_uri"] for r in all_records)) # count distinct
```

- First: list comprehension *transforming* (extract one field per record).
- `set(...)` discards duplicates → `len(set(x))` = "how many distinct values" —
  an idiom you'll write weekly for the rest of your career.

```python
example = next(r for r in all_records if "kinesis_stream.md" in r["source_uri"])
```

A comprehension without square brackets is a **generator** — lazy, produces
items one at a time. `next(...)` asks for just the *first* match and stops.
Reads almost like English: "find the first record from kinesis_stream.md."

```python
print(f"{key:14}: {value}")
```

`:14` pads the key to 14 characters — that's what lines up the output columns.

---

# Part III — What the run showed (actual output, 2026-07-02)

```
docs processed : 73
chunks written : 893  ->  chunks.jsonl
chunk size est : min 14 / avg 174 / max 804 tokens
```

Example record:

```
chunk_id      : kinesis_stream-000
source_uri    : corpus/kinesis_stream.md
position      : 0
heading       : Resource: aws_kinesis_stream
content_hash  : 7adf2013568edc53...
n_tokens_est  : 74
text          : [kinesis_stream > Resource: aws_kinesis_stream]
                Provides a Kinesis Stream resource. ...
```

## Reading the stats like a data engineer

Average 174 tokens — well *below* the 500–800 target. Why? We split at **every
heading first**, and these docs have lots of short sections
(`### stream_mode_details` is one line!). Structure-aware splitting respects the
author's seams — this author made many small rooms. Packing to ~600 tokens only
happens *within* a section, so short sections stay short.

**Bug?** No — a **trade-off**, and understanding it is the interview-grade
insight:

- Small chunks → very *precise* retrieval (great for "what's the default
  retention?" questions)
- But a 14-token chunk is nearly context-free — a lone heading with one sentence
- A production refinement: **merge tiny neighboring sections** until they reach
  a minimum size (say 100 tokens). Real pipelines do exactly this.

Decision: keep it and observe the effect in Task 4 retrieval. Data engineering
is iterative — **measure first, tune second**.

---

# Part IV — The four patterns to take with you

1. **Assembly line with explicit shapes** — every function: one transformation,
   shape in → shape out.
2. **Accumulate-and-flush** — collect into a buffer, seal on a trigger, and
   *never forget the final flush* (used three times in this file).
3. **State machine over a stream** — walk items in order with a bit of memory
   (`in_code_fence`, `heading`) deciding how to treat each item. Exactly how
   streaming consumers think (hello, Kinesis).
4. **Pass-through gracefulness + edge-case paranoia** — empty files, missing
   fences, oversized paragraphs, empty first boxes: handled, not crashed.

If you can re-tell the story — "read, strip, cut at headings, cut at blank
lines, pack greedily with overlap, stamp with metadata, write JSONL" — you
understand this file at the level that matters.

---

## AWS mapping (for the port later)

| Local piece | AWS twin |
|---|---|
| this exact chunker code | runs nearly unchanged inside a **Lambda** |
| `chunks.jsonl` on disk | curated chunks in **S3** (JSONL/Parquet, queryable via Athena) |
| `content_hash` | the idempotency key that prevents re-embedding (and re-paying) on every run |

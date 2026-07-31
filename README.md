# EditcoRAG

By [Editco Media](https://editcomedia.com)

Ask questions about huge PDFs. Your documents, the search index and retrieval
all stay on your machine; answers are written by Groq.

Built for documents that are *mixed*: some pages have a real text layer, some
are scans, and some are diagrams and charts whose text only exists as pixels.
All three are read, indexed, and cited.

- **Answers cite pages you can click.** Each citation opens the real page with
  the source region highlighted.
- **It says "not in this document"** instead of inventing an answer.
- **Resumable ingest.** Kill it at page 400 of 800 and it picks up at 400.
- **Consistent everywhere.** One OCR engine (RapidOCR, portable ONNX) on every
  platform, so a document indexes identically on Windows and macOS.
- **macOS and Windows**, as a double-click desktop app or from the CLI.

### What leaves your machine

Only the excerpts needed to answer a question, sent to Groq. Documents are never
uploaded, the index is never uploaded, and embedding and reranking run locally.
If that trade is unacceptable for your documents, this is not the right tool.

---

## Quick start

```bash
# 1. Install (uv manages the Python version for you)
uv sync

# 2. Configure. A free Groq key from https://console.groq.com/keys, and the
#    OCR language setting - which matters more than it looks, see below.
cp .env.example .env && $EDITOR .env

# 3. Check what this machine can do
uv run editcorag doctor --check-structured

# 4. Build the UI
cd web && pnpm install && pnpm build && cd ..

# 5. Go
uv run editcorag serve          # http://127.0.0.1:8000
```

First run downloads ~7 GB of models (embedder 2.3 GB, reranker 2.3 GB, layout
and OCR models). They are cached and reused afterwards; only answering needs
the network after that.

### Desktop app

`desktop/` is an Electron wrapper that starts the engine, downloads the models
with a progress bar on first run, asks for a Groq key once, and opens the UI —
no terminal involved.

```bash
cd desktop && npm install && npm start
```

---

## Before a long ingest, audit

An 800-page ingest takes 30-60 minutes. Spend ten seconds first:

```bash
uv run editcorag audit mydoc.pdf
```

It classifies every page and shows you *why*:

| route | meaning |
|---|---|
| `digital` | trust the text layer; layout + tables, no OCR |
| `scanned` | no usable text layer; render and OCR the whole page |
| `hybrid`  | text layer *and* imagery; OCR the picture regions only |
| `empty`   | nothing renderable; skipped |

Then read a sample of what would actually be indexed:

```bash
uv run editcorag extract mydoc.pdf --pages 1-24 --out /tmp/md
```

**Do this before trusting a full run.** Extraction quality is the ceiling on
answer quality, and it is the one thing no amount of retrieval tuning can fix.

---

## Set your OCR languages

This is the setting most worth getting right:

```bash
LOCALRAG_OCR_LANGUAGES=en          # comma-separated ISO codes: en,de,zh,ja
```

A wrong value does not raise an error - it returns plausible nonsense. Docling's
portable RapidOCR backend bundles only three model sets (`english`, `latin`,
`chinese`) and loads **one per pass**, and its default Chinese model *drops
inter-word spaces from Latin text*.

| your document | what to do |
|---|---|
| English | `en` (the default) |
| Latin-script mix (de/fr/es/it/pt/nl/pl...) | list them; one `latin` pass covers all |
| Chinese | `zh` |
| Chinese + Latin | two ingest passes, one per script |
| ja / ko / ar / ru / hi / th ... | not supported; those pages index empty |

`editcorag doctor` reports the model set it picked and names anything it cannot
read, so an unsupported language is a warning up front rather than a page of
silent nonsense.

There is deliberately no backend to choose. Apple Vision was faster on macOS and
Tesseract read more scripts, but a document that indexes differently depending on
whose machine opened it is worth neither.

---

## Commands

```bash
editcorag doctor      # what this machine can do; run it first
editcorag audit  PDF  # how every page would be read, and why
editcorag extract PDF --pages 1-24   # what would actually be indexed
editcorag ingest PDF  # route, extract, embed, index (resumable, re-runnable)
editcorag query  "..." --trace       # ask from the terminal
editcorag eval --ablate              # measure retrieval quality
editcorag serve       # API + web UI
```

---

## How it works

```
INGEST
  PDF -> per-page router -> digital | scanned | hybrid | empty
      -> docling (layout, reading order, TableFormer) in worker subprocesses
      -> figures cropped, their labels OCR'd, kept as images
      -> structure-aware chunks with heading context + page/bbox provenance
      -> bge-m3 embeddings -> LanceDB (vectors + full-text index)
      -> every page's state committed to SQLite as it completes

QUERY (LangGraph)
  condense -> retrieve (dense + BM25, RRF-fused) -> rerank (bge-reranker-v2-m3)
           -> grade: do these passages answer it?
              yes -> generate with citations -> verify every claim is supported
              no  -> rewrite the query and retry (max 2) -> else abstain
```

**Design notes.** The router scores text-layer *quality*, not just presence, so
pages with CID-encoded or mojibake text are OCR'd instead of indexed as
gibberish. Citations store a normalised bbox and pdf.js renders pages in the
browser, so the server never stores ~1 GB of page images. Chunk ids are content
hashes, making re-ingest idempotent - duplicate rows would distort BM25 term
statistics for every query, not just the affected chunk.

---

## Tuning

Write 25-40 questions against your document in `eval/questions.yaml`, recording
which page genuinely answers each. Include the hard cases: a fact only on a
scanned page, a number only in a table, a label only inside a figure, a
non-English passage, and questions the document *cannot* answer.

```bash
uv run editcorag eval --ablate    # dense vs bm25 vs hybrid vs hybrid+rerank
uv run editcorag eval --answers --detail
```

Knobs that matter most, in order:

| setting | effect |
|---|---|
| `LOCALRAG_OCR_LANGUAGES` | wrong value = silently wrong text |
| `LOCALRAG_ABSTAIN_SCORE_MIN` | too low invents answers; too high refuses good ones |
| `LOCALRAG_CHUNK_MAX_TOKENS` | smaller = sharper retrieval, less context per hit |
| `LOCALRAG_ROUTE_QUALITY_MIN` | raise if garbled pages slip through as `digital` |
| `LOCALRAG_ANSWER_MODE` | `fast` skips grading; `careful` self-corrects |
| `LOCALRAG_BATCH_PAGES` | larger = fewer chunks split at batch boundaries, more memory |

If your model is unreliable at JSON schemas (`doctor --check-structured` will
tell you), set `LOCALRAG_ENABLE_GRADERS=false`. Score-threshold abstention still
works - it costs no LLM call.

---

## Choosing the answering model

```bash
GROQ_API_KEY=gsk_...
LOCALRAG_LLM_MODEL=openai/gpt-oss-120b   # any model Groq serves
```

Groq is the only supported provider. The grader nodes need reliable JSON-schema
output, so check a new model with `editcorag doctor --check-structured` before
trusting `answer_mode=careful` with it.

Embedding and reranking stay local and are the memory cost: budget ~5 GB of RAM
for the two models, or set `LOCALRAG_EMBED_DEVICE=cpu` on a constrained machine.

---

## Development

```bash
uv run pytest              # test suite
uv run ruff check .        # lint
cd web && pnpm dev         # UI with hot reload against a running API
uv run langgraph dev       # step through the query graph in LangGraph Studio
```

Data lives in `data/`: `pdfs/`, `lance/` (index), `editcorag.db` (page state),
`assets/` (figure crops), `checkpoints.db` (chat threads). Delete it to start
clean; keep it to resume. Ingest resumes from `editcorag.db` — it records every
page's outcome as it completes, so a killed run restarts at the first page that
never finished.

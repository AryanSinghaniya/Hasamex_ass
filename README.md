# Expert Interview Analyzer

A Streamlit application for analyzing expert interview transcripts in a
robotic-surgery market research context. The app parses three transcripts
(France, Germany, UK), generates structured Q&A answers using the Groq
API (openai/gpt-oss-120b), verifies every quote programmatically, and provides cross-expert synthesis
and a free-form retrieval-based chat interface.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

**PowerShell (Windows):**
```powershell
$env:GROQ_API_KEY = "your-groq-api-key-here"
```

**Bash / macOS / Linux:**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

Or copy `.env.example` to `.env` and fill in your key. The `python-dotenv`
package will pick it up automatically on startup.

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **Note on Cache Management:**
> Whenever you change the LLM model or prompt configuration, always clear the cache by running:
> ```bash
> rm -rf cache/*      # Linux / macOS / Git Bash
> Remove-Item -Path 'cache\*' -Force -Recurse   # PowerShell (Windows)
> ```
> Cache keys are content-hash based to avoid redundant API calls during development, so clearing `cache/` ensures new answers are freshly generated.

---

## Project Structure

```
Hasamex_ass/
├── app.py           # Main Streamlit entry point — UI rendering
├── parser.py        # Transcript parsing: .txt → structured JSON chunks
├── llm.py           # Groq API wrapper — answers, synthesis, chat
├── verify.py        # Quote verification (anti-hallucination)
├── tests.py         # Full unit test suite (24 tests)
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   ├── Interview_Guide.txt
│   ├── Transcript_1_France.txt
│   ├── Transcript_2_Germany.txt
│   └── Transcript_3_UK.txt
├── scripts/
│   └── check_hallucinations.py  # Hallucination regression test (18 Q&As)
└── cache/           # Auto-created — JSON cache files for parsed transcripts
                     #   and LLM responses
```

---

## Architecture

### Data Flow

```
.txt files
   │
   ▼
parser.py  ──► cache/parsed_<market>.json
   │               (structured chunks)
   ▼
llm.py  ──────► Groq API (openai/gpt-oss-120b)
   │                │
   │                ▼
   │          raw answer JSON
   │          {answer, timestamp, supporting_quote}
   │
   ▼
verify.py  ──► programmatic quote check
   │               (difflib SequenceMatcher)
   │
   ├── pass ──► display verified answer + quote
   │
   └── fail ──► re-prompt once → check again
                    │
                    ├── pass ──► display corrected answer
                    └── fail ──► discard quote, display warning
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `parser.py` | Reads `.txt` files; splits header (expert name, role, market) from timestamped body; returns list of `{expert_name, market, timestamp, speaker, text, chunk_index}` dicts; caches to `cache/parsed_<market>.json` |
| `llm.py` | Calls Groq API (`openai/gpt-oss-120b`) via LPU inference; provides `get_expert_answer()`, `re_prompt_exact_quote()`, `synthesize_question()`, `ask_panel()`, and `retrieve_chunks()`. All answers disk-cached. |
| `verify.py` | `verify_quote()` checks substring presence using exact normalised match then `difflib.SequenceMatcher` (threshold 0.85). `verify_and_repair()` orchestrates the re-prompt flow. |
| `app.py` | Streamlit UI: sidebar file status, per-expert tabs, Themes & Disagreements tab, Ask the Panel chat. |

---

## Model Choice Rationale

**Model:** `openai/gpt-oss-120b` (on Groq LPU Cloud)

Groq was chosen as the single LLM provider for this project for the following reasons:

1. **Blazing Fast LPU Inference**: Groq's custom Language Processing Units deliver 250–500 tokens/second, generating answers and cross-transcript syntheses in fractions of a second with zero UI lag.
2. **Deep Reasoning & Exact Quotes**: The high-capacity 120B parameter model (`openai/gpt-oss-120b`) excels at transcript reading comprehension, accurately identifying verbatim quotes and timestamps without fabrication.
3. **Genuine Free Tier**: Groq Cloud provides a free tier with high rate limits (30 RPM and generous token buckets), supporting seamless evaluation runs without paywalls.
4. **Guaranteed JSON Schema Compliance**: Built-in `response_format: {"type": "json_object"}` reliably formats structured outputs (`answer`, `timestamp`, `supporting_quote`) for automated quote verification.

> **Changelog Note on Model Switch:** The codebase was migrated to Groq as the single high-speed provider to deliver instant response times, reliable JSON output, and 100% free live operation.

---

## How Citations and Hallucination Prevention Work

### The Problem

Language models can fabricate plausible-sounding quotes that don't exist in
the source material — "hallucinations." In a market research context, a
fabricated expert quote presented as real is a serious integrity problem.

### Layer 1 — Prompt Engineering (Necessary but Insufficient)

The system prompt instructs the model to:
- Use **only** the provided transcript text
- Copy quotes **verbatim** (exact characters)
- Return structured JSON with `answer`, `timestamp`, `supporting_quote`
- Say "Not discussed in this transcript" if the topic isn't covered

This sets the right expectation but cannot guarantee accuracy.

### Layer 2 — Programmatic Verification (`verify.py`)

After every API call, `verify_quote()` checks the returned `supporting_quote`
against the source transcript text using two methods:

1. **Exact normalised match**: Collapses whitespace in both strings and checks
   for substring presence. Fast and deterministic.
2. **Fuzzy sliding-window match**: Uses `difflib.SequenceMatcher` with a
   threshold of 0.85 (85% character-level similarity). This tolerates minor
   whitespace or punctuation differences while still catching fabricated text.

### Layer 3 — Re-prompt on Failure

If verification fails, `verify_and_repair()`:
1. Logs a warning
2. Re-prompts the model **once** with an explicit instruction to copy the
   quote character-for-character from the transcript
3. Verifies the new quote

### Layer 4 — Discard and Flag

If verification still fails after re-prompting:
- The `supporting_quote` is **discarded** — never displayed
- The answer card shows an orange warning: "Quote could not be verified"
- The `answer` text (which is harder to fabricate verbatim) is still shown

This means every displayed quote is traceable to a specific location in the
source transcript.

### "Not Discussed" Handling

If the model returns "Not discussed in this transcript", no quote is expected
and verification is skipped. The UI renders a neutral grey tag instead of a
quote block.

---

## Scaling to 30+ Transcripts

The current architecture works well for 3 transcripts but would not scale to
30+ without modifications. Here is the recommended scaling path:

### 1. Replace Keyword Retrieval with Embeddings + a Vector Store

**Current approach**: Keyword frequency scoring across all chunks. Works for 3
transcripts (~150 chunks total) but becomes noisy and slow at scale.

**Scaled approach**:
```
At ingest time:
  For each transcript chunk:
    embedding = embed(chunk["text"])  # e.g. text-embedding-3-small
    vector_store.upsert(
        id=chunk_id,
        vector=embedding,
        metadata={
            "expert_name": ...,
            "market": ...,
            "timestamp": ...,
            "speaker": ...,
        }
    )

At query time:
  query_embedding = embed(user_query)
  results = vector_store.query(
      vector=query_embedding,
      top_k=10,
      filter={"market": {"$in": selected_markets}},  # metadata filter
  )
```

**Recommended vector store**: [Chroma](https://www.trychroma.com/) for local
development; [Pinecone](https://www.pinecone.io/) or
[Weaviate](https://weaviate.io/) for production.

**Why this scales**: Semantic similarity search finds contextually relevant
chunks even when exact keywords don't match. Metadata filtering (by market,
expert role, specialty) prevents irrelevant markets polluting results.

### 2. Metadata Filtering by Expert / Market

Every chunk is stored with rich metadata:
```python
{
    "expert_name": "Dr. Jean Martin",
    "market": "France",
    "role": "Head of Urology, CHU Bordeaux-Pellegrin",
    "specialty_tags": ["urology", "reimbursement"],
    "timestamp": "02:15",
}
```

Filters can be applied at query time:
- "Only search UK and Germany transcripts"
- "Only surgeon experts, not payers"
- "Only chunks after timestamp 05:00"

### 3. Map-Reduce Style Synthesis for Many Experts

**Current approach**: All three experts' answers are stuffed into one synthesis
call. Fine for 3 but infeasible for 30 (context window overflow + poor signal-
to-noise ratio).

**Scaled approach — Map-Reduce**:

```
MAP phase:
  For each expert (parallel):
    expert_summary = summarize(
        question, expert_chunks, max_tokens=300
    )
    # Returns: {key_points: [...], stance: "...", confidence: "..."}

REDUCE phase:
  synthesis = synthesize(
      question,
      expert_summaries,   # 30 × 300-token summaries = ~9k tokens
      max_tokens=800
  )
```

This keeps each API call within a manageable context window regardless of the
number of experts. The MAP phase is fully parallelisable with `asyncio` or a
`ThreadPoolExecutor`.

### 4. Async API Calls

At 30 transcripts × 6 questions = 180 answer calls, sequential execution
would take ~10 minutes. Switch `llm.py` to use asynchronous requests (e.g. `httpx` or
`aiohttp`) and `asyncio.gather()` to parallelise, reducing wall time to ~30 seconds.

### 5. Persistent Cache

Replace the current flat-file JSON cache with a SQLite or Redis cache keyed
by `(transcript_hash, question_hash)`. This survives server restarts and
supports cache invalidation by expert or question.

---

## Running Tests & Diagnostics

### 1. Run the Full Unit Test Suite (24 Tests)
Validates header extraction, exact and whitespace-normalised quote verification, rejection of fabricated quotes, retry logic, cache invalidation, and retrieval grounding.

```bash
python -X utf8 tests.py
```

### 2. Run the Hallucination Regression Check
Scans all 18 expert answers (3 experts x 6 questions), extracts all currency figures, percentages, dates, durations, acronyms, and proper nouns via regex, and verifies them against the source transcripts.

```bash
python -X utf8 scripts/check_hallucinations.py
```

---

## Interview Guide Questions

The 6 questions mapped from `data/Interview_Guide.txt`:
1. **Adoption Level**: Current adoption level, procedure types, and clinical penetration.
2. **Barriers to Adoption**: Primary barriers (capital expenditure, infrastructure retrofitting, clinical evidence, regulatory/cultural).
3. **Budget & ROI Importance**: Hospital budget constraints, DRG/tariff economics, and return-on-investment timelines.
4. **Training & Clinical Outcomes**: Surgeon training pathways, vendor certification vs. national credentialing, and simulator bottlenecks.
5. **3-5 Year Outlook**: Market growth projections, expected installed base evolution, and emerging clinical indications.
6. **Purchase Decision Timeline**: Hospital procurement cycles, stakeholder roles (CFO, clinical leads, tender committees), and leasing vs. capital purchase.

---

## Streamlit Cloud Deployment

When deploying to [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Fork or push this repository to GitHub.
2. Create a new app pointing to `app.py`.
3. Under **App Settings → Secrets**, add your Groq API key in TOML format:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
4. Save the secret. The app will automatically initialize and load Groq.
5. If changing model versions, use **Manage app → ⋮ → Reboot app** to flush the container's ephemeral cache.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (for LLM features) | Groq API key. Get one for free at console.groq.com |

---

## License

Internal use only — Hasamex Research.

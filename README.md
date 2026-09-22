# Expert Interview Analyzer

A Streamlit application for analyzing expert interview transcripts in a
robotic-surgery market research context. The app parses three transcripts
(France, Germany, UK), generates structured Q&A answers using the Anthropic
API, verifies every quote programmatically, and provides cross-expert synthesis
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
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

**Bash / macOS / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Or copy `.env.example` to `.env` and fill in your key. The `python-dotenv`
package will pick it up automatically on startup.

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
Hasamex_ass/
├── app.py           # Main Streamlit entry point — UI rendering
├── parser.py        # Transcript parsing: .txt → structured JSON chunks
├── llm.py           # Anthropic API wrapper — answers, synthesis, chat
├── verify.py        # Quote verification (anti-hallucination)
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   ├── Interview_Guide.txt
│   ├── Transcript_1_France.txt
│   ├── Transcript_2_Germany.txt
│   └── Transcript_3_UK.txt
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
llm.py  ──────► Anthropic API (claude-sonnet-4-5)
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
| `llm.py` | Wraps Anthropic `messages.create()`; provides `get_expert_answer()`, `re_prompt_exact_quote()`, `synthesize_question()`, `ask_panel()`, and `retrieve_chunks()`. All answers disk-cached. |
| `verify.py` | `verify_quote()` checks substring presence using exact normalised match then `difflib.SequenceMatcher` (threshold 0.85). `verify_and_repair()` orchestrates the re-prompt flow. |
| `app.py` | Streamlit UI: sidebar file status, per-expert tabs, Themes & Disagreements tab, Ask the Panel chat. |

---

## Model Choice Rationale

**Model:** `claude-sonnet-4-5`

Claude Sonnet 4.5 was chosen for the following reasons:

1. **Instruction following**: Claude models reliably follow complex, multi-part
   system prompts — critical for enforcing the "no outside knowledge, cite
   timestamps, return JSON" contract.
2. **JSON output reliability**: Claude Sonnet 4.5 consistently returns
   well-formed JSON when instructed to do so, minimising parse failures.
3. **Context window**: The model's large context window comfortably holds a
   full interview transcript (~10 minutes of dialogue) plus a question, leaving
   room for a detailed answer.
4. **Cost/quality balance**: Sonnet sits between Haiku (cheaper, less reliable
   on complex instructions) and Opus (more expensive, not needed here). For a
   production market research tool with 18+ API calls per session, Sonnet
   provides the right tradeoff.

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
    "expert_name": "Dr. Sophie Marchand",
    "market": "France",
    "role": "Head of Minimally Invasive Surgery",
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
would take ~10 minutes. Switch `llm.py` to use `anthropic.AsyncAnthropic` and
`asyncio.gather()` to parallelise, reducing wall time to ~30 seconds.

### 5. Persistent Cache

Replace the current flat-file JSON cache with a SQLite or Redis cache keyed
by `(transcript_hash, question_hash)`. This survives server restarts and
supports cache invalidation by expert or question.

---

## Running Tests

```bash
# Quick sanity check on the parser
python -c "
import parser as p
chunks = p.parse_transcript('France')
print(f'France: {len(chunks)} chunks')
chunks = p.parse_transcript('Germany')
print(f'Germany: {len(chunks)} chunks')
chunks = p.parse_transcript('UK')
print(f'UK: {len(chunks)} chunks')
qs = p.load_interview_questions()
print(f'Questions: {len(qs)}')
"

# Quick sanity check on the verifier
python -c "
from verify import verify_quote
text = 'The primary barrier is capital cost and infrastructure requirements.'
ok, matched = verify_quote('capital cost and infrastructure', text)
print(f'Exact match: {ok}, matched={matched}')
ok2, matched2 = verify_quote('totally fabricated phrase xyz', text)
print(f'Fabricated: {ok2}')
"
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (for LLM features) | Anthropic API key. Get one at console.anthropic.com |

---

## License

Internal use only — Hasamex Research.

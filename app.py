"""
app.py — Expert Interview Analyzer — Main Streamlit Application.

Architecture overview:
  - parser.py   : Reads .txt transcripts → structured JSON chunks
  - llm.py      : Anthropic claude-sonnet-4-5 calls (answers, synthesis, chat)
  - verify.py   : Programmatic quote verification (anti-hallucination)
  - app.py      : Streamlit UI (this file)

Tabs:
  1. Per-expert tabs (France, Germany, UK)
  2. Themes & Disagreements (cross-expert synthesis)
  3. Ask the Panel (free-form retrieval-based chat)
"""

# Load .env file if present — allows setting ANTHROPIC_API_KEY via a .env file
# without manually exporting env vars each session.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # does not override already-set env vars
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

import os
import re
import logging
import functools
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Project modules ──────────────────────────────────────────────────────────
import parser as transcript_parser
import llm
import verify

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expert Interview Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* ── Header banner ── */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0e4d6b 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .app-header h1 {
        color: #e2e8f0;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0;
    }

    /* ── Q&A card ── */
    .qa-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .qa-card .question-label {
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .qa-card .answer-text {
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.65;
        margin-bottom: 0.8rem;
    }

    /* ── Citation badge ── */
    .citation-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #0f3460;
        color: #7dd3fc;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        border: 1px solid #1e4e8c;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }

    /* ── Verified quote block ── */
    .verified-quote {
        background: #0f2a1a;
        border-left: 3px solid #22c55e;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        color: #86efac;
        font-size: 0.88rem;
        font-style: italic;
        margin-top: 0.6rem;
    }
    .quote-label {
        color: #4ade80;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    /* ── Unverified quote warning ── */
    .unverified-warning {
        background: #2a1a0a;
        border-left: 3px solid #f97316;
        padding: 0.6rem 1rem;
        border-radius: 0 6px 6px 0;
        color: #fdba74;
        font-size: 0.84rem;
        margin-top: 0.6rem;
    }

    /* ── Not discussed tag ── */
    .not-discussed {
        display: inline-block;
        background: #27272a;
        color: #71717a;
        font-size: 0.82rem;
        font-style: italic;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #3f3f46;
    }

    /* ── Synthesis card ── */
    .synthesis-section {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .synthesis-section h4 {
        color: #38bdf8;
        font-size: 1rem;
        margin-top: 0;
    }

    /* ── Expert chip ── */
    .expert-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .chip-france { background: #1e3a6e; color: #93c5fd; }
    .chip-germany { background: #2d1f5e; color: #c4b5fd; }
    .chip-uk { background: #1a3a2a; color: #86efac; }

    /* ── Chat message ── */
    .chat-user {
        background: #1e3a5f;
        border-radius: 10px 10px 2px 10px;
        padding: 0.8rem 1.1rem;
        color: #e2e8f0;
        margin-bottom: 0.8rem;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.93rem;
    }
    .chat-assistant {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px 10px 10px 2px;
        padding: 0.8rem 1.1rem;
        color: #e2e8f0;
        margin-bottom: 0.8rem;
        max-width: 90%;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    .status-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .status-ok { color: #4ade80; }
    .status-missing { color: #f87171; }

    /* ── Divider ── */
    hr { border-color: #1e293b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MARKETS = ["France", "Germany", "UK"]
EXPERT_COLORS = {"France": "chip-france", "Germany": "chip-germany", "UK": "chip-uk"}


# Use non-flag emojis — Windows does not render regional-indicator flag emojis
# (e.g. 🇫🇷) in most fonts; they appear as letter codes like 'FR' instead.
MARKET_ICONS   = {"France": "🔵", "Germany": "🟡", "UK": "🔴"}
MARKET_LABELS  = {"France": "France 🔵", "Germany": "Germany 🟡", "UK": "UK 🔴"}

# NOTE: There is intentionally NO hardcoded expert-name dict here.
# Expert names are read directly from the parsed transcript chunks
# (first non-interviewer chunk for each market) via get_expert_name().
# This prevents the class of bug where a hardcoded name diverges from
# the actual transcript file.


def get_expert_name(market: str) -> str:
    """
    Return the expert name for a market by reading it from the parsed
    transcript chunks stored in session state.

    Falls back to reading the file header directly if chunks aren't loaded yet.
    Never returns a hardcoded or invented name.
    """
    # Prefer session-state chunks (most up-to-date after a parse)
    chunks = st.session_state.get("all_chunks", {}).get(market, [])
    for chunk in chunks:
        if chunk.get("speaker", "").lower() not in ("interviewer", ""):
            return chunk["expert_name"]
    # Fallback: read file header directly
    try:
        h = transcript_parser.get_header(market)
        name = h.get("expert_name", "")
        if name:
            return name
    except Exception:
        pass
    return "Unknown Expert"


# ── Session state initialisation ──────────────────────────────────────────────
def _init_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {role, content}
    if "chat_retrieved" not in st.session_state:
        st.session_state.chat_retrieved = []  # retrieved chunks for last query
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = {}
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        # {market: {q_index: verified_answer_dict}}
        st.session_state.answers = {m: {} for m in MARKETS}
    if "syntheses" not in st.session_state:
        st.session_state.syntheses = {}  # {q_index: synthesis_dict}
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    # Re-evaluate api_key_ok from environment (.env)
    st.session_state.api_key_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_all_data(force: bool = False) -> tuple[dict, list[str]]:
    """Load and parse all transcripts and the interview guide. Cached per session."""
    all_chunks = transcript_parser.parse_all_transcripts(force=force)
    questions = transcript_parser.load_interview_questions()
    return all_chunks, questions


def _load_data(force: bool = False):
    """Load data into session state.

    On first load, calls log_all_headers() which prints the parsed expert
    name/role/market to stdout so identity bugs surface immediately in dev.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY not detected in .env file.")

    with st.spinner("📂 Parsing transcripts…"):
        try:
            all_chunks, questions = _load_all_data(force)
            st.session_state.all_chunks = all_chunks
            st.session_state.questions = questions
            st.session_state.data_loaded = True
            # Print parsed identity fields to stdout / server log on every load
            # so wrong names are caught immediately without running any LLM calls.
            transcript_parser.log_all_headers()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.session_state.data_loaded = False


# ── Per-expert answer loading ─────────────────────────────────────────────────

def _get_or_generate_answer(market: str, q_idx: int) -> Optional[dict]:
    """
    Get a verified answer for one market × question pair.
    Uses session state cache first, then disk cache, then API call.
    """
    if q_idx in st.session_state.answers.get(market, {}):
        return st.session_state.answers[market][q_idx]

    question = st.session_state.questions[q_idx]
    chunks = st.session_state.all_chunks.get(market, [])
    expert_name = get_expert_name(market)

    if not chunks:
        return None

    file_hash = transcript_parser.get_file_hash(market)

    # ── Raw LLM answer ──────────────────────────────────────────────────
    raw_answer = llm.get_expert_answer(
        question=question,
        chunks=chunks,
        expert_name=expert_name,
        market=market,
        file_hash=file_hash,
    )

    # ── Quote verification ──────────────────────────────────────────────
    transcript_text = transcript_parser.get_transcript_text(market)

    def _re_prompt(q, ch, en, bad_quote):
        return llm.re_prompt_exact_quote(q, ch, en, market, bad_quote)

    verified = verify.verify_and_repair(
        answer_json=raw_answer,
        transcript_text=transcript_text,
        re_prompt_fn=_re_prompt,
        question=question,
        chunks=chunks,
        expert_name=expert_name,
    )
    verified["expert_name"] = expert_name
    verified["market"] = market

    if market not in st.session_state.answers:
        st.session_state.answers[market] = {}
    st.session_state.answers[market][q_idx] = verified
    return verified


# ── Synthesis loading ─────────────────────────────────────────────────────────

def _get_or_generate_synthesis(q_idx: int) -> Optional[dict]:
    """Generate cross-expert synthesis for one question."""
    if q_idx in st.session_state.syntheses:
        return st.session_state.syntheses[q_idx]

    question = st.session_state.questions[q_idx]
    expert_answers = {}
    for market in MARKETS:
        ans = st.session_state.answers.get(market, {}).get(q_idx)
        if ans:
            expert_answers[market] = ans

    if len(expert_answers) < 2:
        return None

    synthesis = llm.synthesize_question(question, expert_answers)
    st.session_state.syntheses[q_idx] = synthesis
    return synthesis


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _render_answer_card(answer: dict, q_idx: int, q_text: str):
    """Render a single Q&A card with citation, verified quote, and transcript expander."""
    if not answer:
        st.markdown('<div class="not-discussed">⚪ Answer not yet generated</div>', unsafe_allow_html=True)
        return

    is_not_discussed = answer.get("not_discussed", False) or (
        "not discussed" in answer.get("answer", "").lower()
    )

    with st.container():
        # Question label
        q_num = q_idx + 1
        st.markdown(
            f'<div class="qa-card">'
            f'<div class="question-label">Q{q_num} — {_short_q_title(q_text)}</div>'
            f'<div class="answer-text">{answer.get("answer", "")}</div>',
            unsafe_allow_html=True,
        )

        if not is_not_discussed:
            ts = answer.get("timestamp", "")
            if ts:
                st.markdown(
                    f'<span class="citation-badge">🕐 {ts}</span>',
                    unsafe_allow_html=True,
                )

            # Verified quote
            if answer.get("quote_verified") and answer.get("verified_quote"):
                st.markdown(
                    f'<div class="quote-label">✅ Verified Quote</div>'
                    f'<div class="verified-quote">"{answer["verified_quote"]}"</div>',
                    unsafe_allow_html=True,
                )
            elif not answer.get("quote_verified"):
                st.markdown(
                    '<div class="unverified-warning">'
                    "⚠️ Quote could not be verified against the source transcript "
                    "and has been discarded as a precaution."
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span class="not-discussed">⚪ Not discussed in this transcript</span>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Transcript context expander
        if not is_not_discussed and answer.get("timestamp"):
            with st.expander(f"📜 View transcript context around {answer['timestamp']}"):
                _render_transcript_context(answer["market"], answer["timestamp"])


def _render_transcript_context(market: str, timestamp: str):
    """Show 3 turns around the cited timestamp from the parsed chunks."""
    chunks = st.session_state.all_chunks.get(market, [])
    if not chunks:
        st.info("Transcript not loaded.")
        return

    # Find the chunk with this timestamp
    target_idx = next(
        (i for i, c in enumerate(chunks) if c["timestamp"] == timestamp), None
    )
    if target_idx is None:
        st.info(f"Timestamp {timestamp} not found in parsed chunks.")
        return

    start = max(0, target_idx - 1)
    end = min(len(chunks), target_idx + 3)
    window = chunks[start:end]

    for c in window:
        highlight = c["timestamp"] == timestamp
        bg = "#0f2d1a" if highlight else "#1e293b"
        border = "#22c55e" if highlight else "#334155"
        st.markdown(
            f"""<div style="background:{bg};border:1px solid {border};
            border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.5rem;">
            <span style="color:#38bdf8;font-size:0.78rem;font-weight:600;">
            {c['timestamp']}</span>
            <span style="color:#64748b;font-size:0.78rem;margin-left:0.5rem;">
            {c['speaker']}</span><br/>
            <span style="color:#e2e8f0;font-size:0.88rem;">{c['text']}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def _short_q_title(q_text: str) -> str:
    """Extract a concise title from the question heading (first line)."""
    lines = q_text.strip().splitlines()
    if not lines:
        return "Question"
    first_line = lines[0].strip()
    # Strip leading numbering like '1. ', '1) ', 'Q1: '
    title = re.sub(r"^(?:Q\d+[:.]?|\d+[.)])\s*", "", first_line).strip()
    if title:
        # Strip secondary clauses after em-dash or dash
        title = re.split(r"[—–\-:]", title)[0].strip()
        if title.isupper():
            title = title.title()
        return title[:60]
    return lines[0][:60]


def _q_short_label(q_text: str, idx: int) -> str:
    """
    Derive a short label programmatically from parsed question text,
    with an explicit fallback tied 1-to-1 to each question in Interview_Guide.txt.
    """
    # Programmatic derivation: extract title from question text
    derived = _short_q_title(q_text)
    if derived and derived != "Question":
        return derived

    # Fallback list tied explicitly to Question 1–6 in Interview_Guide.txt
    fallback_labels = [
        "Adoption Barriers",               # Q1: Adoption barriers & infrastructure
        "Reimbursement Landscape",         # Q2: Reimbursement & tariff environment
        "Competitive Dynamics",            # Q3: Platform competitive landscape
        "Surgeon Training & Pathway",      # Q4: Surgeon training & credentialing
        "Hospital Procurement Decisions",  # Q5: Hospital procurement & capital purchase
        "Future Outlook",                  # Q6: 3–5 year market outlook
    ]
    return fallback_labels[idx] if idx < len(fallback_labels) else f"Q{idx+1}"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.markdown(
            "### 🔬 Expert Interview Analyzer",
        )
        st.caption("Robotic Surgery Market Research")
        st.markdown("---")

        # Active model status (configured purely via .env)
        model_name = llm.get_active_model_name()
        st.markdown(f"**🤖 Model:** `{model_name}`")
        st.caption("Configured via `.env` file")

        st.markdown("---")
        st.markdown("#### 📁 Data Files")

        status = transcript_parser.file_status()

        # Interview guide
        guide_ok = status["interview_guide"]["exists"]
        icon = "✅" if guide_ok else "❌"
        st.markdown(f"{icon} Interview Guide")

        # Transcripts
        for market in MARKETS:
            m_status = status.get(market, {})
            t_ok = m_status.get("transcript_exists", False)
            c_ok = m_status.get("cache_exists", False)
            t_icon = "✅" if t_ok else "❌"
            c_icon = "💾" if c_ok else "○"
            st.markdown(
                f"{MARKET_ICONS[market]} {t_icon} Transcript · {c_icon} Cache &nbsp; **{market}**",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### ⚙️ Controls")

        if st.button("🔄 Re-parse & Re-analyse", use_container_width=True):
            # Clear caches
            _load_all_data.clear()
            st.session_state.answers = {m: {} for m in MARKETS}
            st.session_state.syntheses = {}
            st.session_state.chat_history = []
            st.session_state.chat_retrieved = []
            _load_data(force=True)
            st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_retrieved = []
            st.rerun()

        st.markdown("---")
        st.markdown("#### 📊 Session Stats")
        total_answers = sum(
            len(v) for v in st.session_state.answers.values()
        )
        st.caption(f"Answers cached: **{total_answers}** / 18")
        st.caption(f"Syntheses cached: **{len(st.session_state.syntheses)}** / 6")
        st.caption(f"Chat turns: **{len(st.session_state.chat_history)}**")

        st.markdown("---")
        st.caption(f"Model: `{llm.MODEL}`")
        st.caption("Anti-hallucination: difflib ≥ 0.85")


# ── Expert tab ─────────────────────────────────────────────────────────────────

def _render_expert_tab(market: str):
    if not st.session_state.data_loaded:
        st.info("Click **Load Data** to begin.")
        return

    questions = st.session_state.questions
    expert_name = get_expert_name(market)
    chunks = st.session_state.all_chunks.get(market, [])

    col1, col2 = st.columns([3, 1])
    with col1:
        icon = MARKET_ICONS[market]
        st.markdown(f"### {icon} {expert_name} &nbsp;·&nbsp; {market}", unsafe_allow_html=True)
    with col2:
        st.caption(f"{len(chunks)} transcript turns parsed")

    if not chunks:
        st.warning(f"No transcript data found for {market}.")
        return

    # Question selector
    q_labels = [f"Q{i+1}: {_q_short_label(q, i)}" for i, q in enumerate(questions)]
    selected_q = st.selectbox(
        "Select question",
        options=range(len(questions)),
        format_func=lambda i: q_labels[i],
        key=f"q_select_{market}",
    )

    st.markdown("---")

    # Show the full question text
    with st.expander("📋 Full question text", expanded=False):
        st.markdown(questions[selected_q])

    # Generate / retrieve answer
    if not st.session_state.api_key_ok:
        st.warning("Please configure your `ANTHROPIC_API_KEY` in `.env` to generate answers.")
        return

    with st.spinner(f"Generating answer for {market}…"):
        answer = _get_or_generate_answer(market, selected_q)

    if answer:
        _render_answer_card(answer, selected_q, questions[selected_q])
    else:
        st.error("Could not generate answer.")

    # "Generate all" button
    st.markdown("---")
    if st.button(f"Generate all 6 answers for {market}", key=f"gen_all_{market}"):
        progress = st.progress(0, text="Generating…")
        for i, q in enumerate(questions):
            _get_or_generate_answer(market, i)
            progress.progress((i + 1) / len(questions), text=f"Q{i+1} done")
        progress.empty()
        st.success(f"All answers generated for {market}!")
        st.rerun()


# ── Themes & Disagreements tab ─────────────────────────────────────────────────

def _render_synthesis_tab():
    st.markdown("### 🧩 Themes & Disagreements — Cross-Expert Synthesis")
    st.caption(
        "Generated by comparing verified answers from all three experts. "
        "Every claim is attributed to a specific expert."
    )

    if not st.session_state.data_loaded:
        st.info("Load data first.")
        return

    if not st.session_state.api_key_ok:
        st.warning("Please configure your `ANTHROPIC_API_KEY` in `.env` to generate synthesis.")
        return

    questions = st.session_state.questions

    # "Generate all syntheses" button
    col1, col2 = st.columns([2, 1])
    with col2:
        if st.button("⚡ Generate all syntheses", use_container_width=True):
            # First ensure all expert answers exist
            for q_idx in range(len(questions)):
                for market in MARKETS:
                    _get_or_generate_answer(market, q_idx)
            progress = st.progress(0, text="Synthesising…")
            for i in range(len(questions)):
                _get_or_generate_synthesis(i)
                progress.progress((i + 1) / len(questions), text=f"Q{i+1} synthesised")
            progress.empty()
            st.success("All syntheses complete!")
            st.rerun()

    st.markdown("---")

    q_labels = [f"Q{i+1}: {_q_short_label(q, i)}" for i, q in enumerate(questions)]
    selected_q = st.selectbox(
        "Select question",
        options=range(len(questions)),
        format_func=lambda i: q_labels[i],
        key="synth_q_select",
    )

    st.markdown("")
    with st.expander("📋 Full question text", expanded=False):
        st.markdown(questions[selected_q])

    # Show per-expert answers side by side
    st.markdown("#### Expert Answers")
    cols = st.columns(3)
    for col, market in zip(cols, MARKETS):
        with col:
            ans = st.session_state.answers.get(market, {}).get(selected_q)
            chip_class = EXPERT_COLORS.get(market, "")
            st.markdown(
                f'<span class="expert-chip {chip_class}">'
                f'{MARKET_ICONS[market]} {market}</span>',
                unsafe_allow_html=True,
            )
            if ans:
                st.markdown(
                    f'<div style="background:#1e293b;border:1px solid #334155;'
                    f'border-radius:8px;padding:0.8rem;font-size:0.85rem;color:#e2e8f0;'
                    f'min-height:100px;">{ans.get("answer", "")}</div>',
                    unsafe_allow_html=True,
                )
                if ans.get("timestamp"):
                    st.caption(f"🕐 {ans['timestamp']}")
            else:
                if st.button(f"Generate {market}", key=f"gen_{market}_{selected_q}"):
                    with st.spinner():
                        _get_or_generate_answer(market, selected_q)
                    st.rerun()

    # Synthesis
    st.markdown("---")
    st.markdown("#### Synthesis")

    # Check if all answers are ready
    answers_ready = all(
        st.session_state.answers.get(m, {}).get(selected_q) for m in MARKETS
    )

    if not answers_ready:
        st.info(
            "Generate answers for all three experts first to enable synthesis. "
            "Use the buttons above or click 'Generate all syntheses'."
        )
        return

    with st.spinner("Synthesising…"):
        synthesis = _get_or_generate_synthesis(selected_q)

    if not synthesis:
        st.warning("Synthesis could not be generated.")
        return

    # Summary line
    summary = synthesis.get("summary_line", "")
    if summary:
        st.markdown(
            f'<div style="background:#1e3a5f;border-radius:8px;padding:0.9rem 1.2rem;'
            f'color:#bae6fd;font-size:0.95rem;font-weight:500;margin-bottom:1rem;">'
            f'📌 {summary}</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🤝 Common Themes")
        st.markdown(
            f'<div class="synthesis-section">'
            f'{synthesis.get("common_themes", "—")}'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("##### ⚡ Disagreements & Differences")
        st.markdown(
            f'<div class="synthesis-section">'
            f'{synthesis.get("disagreements", "—")}'
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Ask the Panel tab ──────────────────────────────────────────────────────────

def _render_chat_tab():
    st.markdown("### 💬 Ask the Panel")
    st.caption(
        "Ask any question about robotic surgery across all three markets. "
        "Answers are grounded in retrieved transcript excerpts only."
    )

    if not st.session_state.data_loaded:
        st.info("Load data first.")
        return

    if not st.session_state.api_key_ok:
        st.warning("Please configure your `ANTHROPIC_API_KEY` in `.env` to use the chat.")
        return

    # Render chat history
    for turn in st.session_state.chat_history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            st.markdown(
                f'<div class="chat-user">👤 {content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">🤖 {content}</div>',
                unsafe_allow_html=True,
            )

    # Show retrieved chunks for the last query
    if st.session_state.chat_retrieved:
        with st.expander(
            f"🔍 Retrieved {len(st.session_state.chat_retrieved)} transcript chunk(s) used as context"
        ):
            for c in st.session_state.chat_retrieved:
                chip = EXPERT_COLORS.get(c["market"], "")
                st.markdown(
                    f'<div style="background:#1e293b;border:1px solid #334155;'
                    f'border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;">'
                    f'<span class="expert-chip {chip}">{MARKET_ICONS.get(c["market"],"")} '
                    f'{c["market"]}</span> '
                    f'<span class="citation-badge">🕐 {c["timestamp"]}</span><br/>'
                    f'<span style="color:#94a3b8;font-size:0.8rem;">{c["speaker"]}</span><br/>'
                    f'<span style="color:#e2e8f0;font-size:0.88rem;">{c["text"]}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Your question",
                placeholder="e.g. How does reimbursement differ between France and Germany?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        query = user_input.strip()

        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Retrieve chunks
        retrieved = llm.get_retrieved_chunks_for_display(
            query, st.session_state.all_chunks
        )
        st.session_state.chat_retrieved = retrieved

        # Build model conversation history (for multi-turn)
        model_history = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.chat_history[:-1]  # exclude current user msg
        ]

        with st.spinner("Searching transcripts…"):
            answer = llm.ask_panel(
                query=query,
                all_chunks=st.session_state.all_chunks,
                chat_history=model_history,
            )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "How does reimbursement differ between France and Germany?",
            "Which market has the most structured surgeon training pathway?",
            "What are the key drivers of platform selection across markets?",
            "How do procurement timelines compare across the three markets?",
            "What role do usage-based pricing models play in each market?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    retrieved = llm.get_retrieved_chunks_for_display(
                        s, st.session_state.all_chunks
                    )
                    st.session_state.chat_retrieved = retrieved
                    with st.spinner("Searching…"):
                        answer = llm.ask_panel(s, st.session_state.all_chunks)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()


# ── Main layout ────────────────────────────────────────────────────────────────

def main():
    _init_session()
    _render_sidebar()

    # Header
    st.markdown(
        """
        <div class="app-header">
          <h1>🔬 Expert Interview Analyzer</h1>
          <p>Robotic Surgery Market Research · France · Germany · United Kingdom</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load data on first run
    if not st.session_state.data_loaded:
        _load_data()

    # API key warning (only shows if .env is missing ANTHROPIC_API_KEY)
    if not st.session_state.api_key_ok:
        st.warning(
            "⚠️ No API key found in `.env`. Please add your `ANTHROPIC_API_KEY` to the `.env` file.",
            icon="🔑",
        )

    # Tabs
    tab_france, tab_germany, tab_uk, tab_themes, tab_chat = st.tabs([
        "France 🔵",
        "Germany 🟡",
        "UK 🔴",
        "🧩 Themes & Disagreements",
        "💬 Ask the Panel",
    ])

    with tab_france:
        _render_expert_tab("France")

    with tab_germany:
        _render_expert_tab("Germany")

    with tab_uk:
        _render_expert_tab("UK")

    with tab_themes:
        _render_synthesis_tab()

    with tab_chat:
        _render_chat_tab()


if __name__ == "__main__":
    main()

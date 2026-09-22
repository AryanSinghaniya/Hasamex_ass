"""
llm.py — Anthropic API wrapper for Expert Interview Analyzer.

Provides three main operations:
  1. get_expert_answer()  — per-expert Q&A with structured JSON output
  2. re_prompt_exact_quote() — retry with explicit "copy exactly" instruction
  3. synthesize_question() — cross-expert synthesis for one question
  4. ask_panel()          — free-form chat with chunk-based retrieval context

All calls use claude-sonnet-4-5.
Disk caching prevents redundant API calls on UI re-runs.
"""

import json
import os
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import requests

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "gemini-3.6-flash"
CACHE_DIR = Path("cache")

# Maximum number of transcript chunks to include in a single API call.
# Keeps context windows manageable.
MAX_CHUNKS_PER_CALL = 40

# Number of top chunks to retrieve for the "Ask the Panel" feature.
RETRIEVAL_TOP_N = 8


def _get_anthropic_key() -> Optional[str]:
    """Retrieve ANTHROPIC_API_KEY from environment or Streamlit secrets."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            import streamlit as st
            if "ANTHROPIC_API_KEY" in st.secrets:
                key = st.secrets["ANTHROPIC_API_KEY"]
                if key:
                    os.environ["ANTHROPIC_API_KEY"] = key
        except Exception:
            pass
    return key


def _get_gemini_key() -> Optional[str]:
    """Retrieve GEMINI_API_KEY from environment or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
                if key:
                    os.environ["GEMINI_API_KEY"] = key
        except Exception:
            pass
    return key


def get_active_model_name() -> str:
    """Return human-readable active model name."""
    if _get_anthropic_key():
        import re
        m = re.match(r"claude-([a-z]+)-(\d+)-(\d+)", MODEL)
        if m:
            tier, major, minor = m.groups()
            return f"Claude {tier.capitalize()} {major}.{minor}"
        m2 = re.match(r"claude-(\d+)-(\d+)-([a-z]+)", MODEL)
        if m2:
            major, minor, tier = m2.groups()
            return f"Claude {major}.{minor} {tier.capitalize()}"
        return MODEL.replace("-", " ").title()
    elif _get_gemini_key():
        return "Gemini 3.6 Flash (Free)"
    return "No API Key Set"


# ── Anthropic client (lazy-initialised) ─────────────────────────────────────

_client: Optional[object] = None


def _get_client():
    global _client
    if _client is None:
        if anthropic is None:
            raise ImportError("anthropic package is not installed.")
        api_key = _get_anthropic_key()
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def reset_client():
    """Reset client instance so it re-initialises on next call."""
    global _client
    _client = None


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(*parts: str) -> str:
    combined = "||".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def _cache_path(prefix: str, key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{prefix}_{key}.json"


def _load_cache(prefix: str, key: str) -> Optional[dict]:
    path = _cache_path(prefix, key)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(prefix: str, key: str, data: dict) -> None:
    path = _cache_path(prefix, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Prompt builders ──────────────────────────────────────────────────────────

def _format_chunks_as_text(chunks: list[dict]) -> str:
    """Convert transcript chunks to a readable text block for the prompt."""
    lines = []
    for c in chunks[:MAX_CHUNKS_PER_CALL]:
        lines.append(f"[{c['timestamp']}] {c['speaker']}: {c['text']}")
    return "\n\n".join(lines)


EXPERT_ANSWER_SYSTEM = """\
You are a rigorous market research analyst. You are given an excerpt from an \
expert interview transcript. Your task is to answer a specific interview guide \
question using ONLY the information present in the provided transcript.

STRICT RULES — you must follow these exactly:
1. Base your answer SOLELY on the transcript text provided. Do NOT use any \
outside knowledge, general industry knowledge, or information not explicitly \
stated in the transcript.
2. After giving your answer, cite the timestamp of the most relevant turn(s) \
in the format MM:SS (e.g. "01:23"). If multiple timestamps are relevant, \
list the most important one as the primary citation.
3. Include a supporting_quote: copy a verbatim sentence or phrase directly \
from the transcript that best supports your answer. Copy it EXACTLY as it \
appears — do not paraphrase, do not add or remove words.
4. If the transcript does not address the question at all, set answer to \
exactly: "Not discussed in this transcript" and leave timestamp and \
supporting_quote empty strings.

Respond ONLY with a valid JSON object matching this schema exactly:
{
  "answer": "<your answer based only on the transcript>",
  "timestamp": "<MM:SS of most relevant turn, or empty string>",
  "supporting_quote": "<exact verbatim quote from the transcript, or empty string>"
}

Do not include any text outside the JSON object."""


EXPERT_ANSWER_RETRY_SYSTEM = """\
You are a rigorous market research analyst. A previous attempt to extract a \
supporting quote from this transcript produced a quote that could not be \
verified as appearing verbatim in the transcript.

Your task: provide a corrected answer with a supporting_quote that is copied \
EXACTLY, CHARACTER FOR CHARACTER, from the transcript text provided. \
Do not alter punctuation, capitalisation, or wording in any way.

STRICT RULES:
1. Answer using ONLY the transcript provided — no outside knowledge.
2. The supporting_quote must be a continuous verbatim substring of the transcript text.
3. Cite the timestamp of the most relevant turn.
4. If the transcript does not address the question, set answer to \
"Not discussed in this transcript".

Respond ONLY with a valid JSON object:
{
  "answer": "<answer based only on transcript>",
  "timestamp": "<MM:SS or empty string>",
  "supporting_quote": "<verbatim substring of transcript, or empty string>"
}"""


SYNTHESIS_SYSTEM = """\
You are a senior market research analyst synthesising findings across multiple \
expert interviews about the robotic surgery market.

You will be given one interview guide question and the verified answers from \
three market experts (France, Germany, UK). Your task is to write a synthesis that:

1. Identifies COMMON THEMES and POINTS OF AGREEMENT across experts. \
Every claim must attribute which expert(s) said it, using their name and market \
in parentheses, e.g. "(Dr. Marchand, France)".
2. Identifies DISAGREEMENTS or DIFFERENCES IN EMPHASIS — where experts' views \
diverge or where emphasis varies by market. Again, attribute every claim.
3. Is structured with clearly labelled sections: \
"## Common Themes" and "## Disagreements & Differences".
4. Does NOT introduce any information or analysis beyond what the experts stated. \
Every sentence must be traceable to a specific expert.

Respond with a JSON object:
{
  "common_themes": "<markdown text — Common Themes section>",
  "disagreements": "<markdown text — Disagreements & Differences section>",
  "summary_line": "<one sentence executive summary of the key finding>"
}"""


PANEL_CHAT_SYSTEM = """\
You are an analyst for a robotic surgery market research project. You have been \
given a set of relevant excerpt chunks from three expert interviews (France, \
Germany, UK). Your task is to answer the user's question using ONLY the provided \
excerpts.

STRICT RULES:
1. Answer ONLY from the provided transcript excerpts — no outside knowledge.
2. For every factual claim you make, cite the expert name, market, and timestamp \
in the format: (Expert Name, Market, MM:SS).
3. If the excerpts do not contain sufficient information to answer the question, \
respond with: "This isn't covered in the transcripts provided."
4. Do not speculate or extrapolate beyond what the experts explicitly said.

Respond in clear, structured prose."""


# ── Core API functions ───────────────────────────────────────────────────────

def _call_claude(system: str, user_message: str) -> str:
    """Make a single Claude API call and return the text response."""
    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def _call_gemini(system: str, user_message: str) -> str:
    """Call Google Gemini API with automatic retry on transient errors."""
    api_key = _get_gemini_key()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            elif resp.status_code in (500, 503, 429) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise e
    return ""


def _call_gemini_chat(system: str, messages: list[dict]) -> str:
    """Call Google Gemini API for multi-turn chat."""
    api_key = _get_gemini_key()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        },
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            elif resp.status_code in (500, 503, 429) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise e
    return ""


def _call_llm(system: str, user_message: str) -> str:
    """Dispatch call to Claude if Anthropic key is set, else Gemini."""
    if _get_anthropic_key():
        return _call_claude(system, user_message)
    elif _get_gemini_key():
        return _call_gemini(system, user_message)
    raise EnvironmentError("No API key set. Provide either ANTHROPIC_API_KEY or GEMINI_API_KEY.")


def _parse_json_response(raw: str) -> dict:
    """
    Safely parse a JSON response from the model.
    Handles markdown code fences the model might accidentally include.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        inner = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(inner)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from model response: %s\nRaw: %s", e, raw[:300])
        return {
            "answer": raw,
            "timestamp": "",
            "supporting_quote": "",
            "parse_error": True,
        }


def _call_llm_json(system: str, user_message: str) -> dict:
    """
    Call the active LLM and parse JSON response. If JSON parsing fails on the first attempt,
    retry once with an explicit instruction appended before falling back to raw-text behavior.
    """
    raw = _call_llm(system, user_message)
    result = _parse_json_response(raw)
    if result.get("parse_error"):
        logger.warning("JSON parse failed on first attempt. Retrying with explicit instruction.")
        retry_message = (
            f"{user_message}\n\n"
            "Your previous response was not valid JSON. Return ONLY the JSON object, no other text."
        )
        retry_raw = _call_llm(system, retry_message)
        retry_result = _parse_json_response(retry_raw)
        return retry_result
    return result


# Keep alias for backwards compatibility
_call_claude_json = _call_llm_json


# ── Public API ───────────────────────────────────────────────────────────────

def get_expert_answer(
    question: str,
    chunks: list[dict],
    expert_name: str,
    market: str,
    file_hash: str = "",
    force: bool = False,
) -> dict:
    """
    Generate a structured answer for one interview question from one expert's
    transcript chunks.

    Uses disk cache keyed by question + expert + transcript hash.
    Returns dict: {answer, timestamp, supporting_quote}
    """
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    key = _cache_key(q_hash, market, file_hash)
    prefix = f"answer_{market}"

    if not force:
        cached = _load_cache(prefix, key)
        if cached:
            logger.debug("Cache hit: %s / %s", prefix, key)
            return cached

    transcript_text = _format_chunks_as_text(chunks)
    user_message = (
        f"TRANSCRIPT EXCERPT (Expert: {expert_name}, Market: {market}):\n\n"
        f"{transcript_text}\n\n"
        f"QUESTION:\n{question}"
    )

    result = _call_claude_json(EXPERT_ANSWER_SYSTEM, user_message)

    _save_cache(prefix, key, result)
    return result


def re_prompt_exact_quote(
    question: str,
    chunks: list[dict],
    expert_name: str,
    market: str,
    bad_quote: str,
) -> dict:
    """
    Re-prompt the model with an explicit instruction to copy the quote exactly.
    Called by verify.py when the first attempt's quote fails verification.

    Returns dict: {answer, timestamp, supporting_quote}
    """
    transcript_text = _format_chunks_as_text(chunks)
    user_message = (
        f"TRANSCRIPT EXCERPT (Expert: {expert_name}, Market: {market}):\n\n"
        f"{transcript_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"PREVIOUS UNVERIFIABLE QUOTE (do not reuse this — it could not be "
        f"found verbatim in the transcript):\n\"{bad_quote}\"\n\n"
        f"Please provide a corrected answer with a supporting_quote that is a "
        f"verbatim, unmodified substring of the transcript text above."
    )
    return _call_claude_json(EXPERT_ANSWER_RETRY_SYSTEM, user_message)


def synthesize_question(
    question: str,
    expert_answers: dict[str, dict],
    force: bool = False,
) -> dict:
    """
    Generate a cross-expert synthesis for one interview question.

    Args:
        question:       The interview guide question text.
        expert_answers: {market: verified_answer_dict} for all three experts.
        force:          If True, bypass cache.

    Returns:
        dict: {common_themes, disagreements, summary_line}
    """
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    # Include a hash of the answers content to invalidate if answers change
    answers_str = json.dumps(expert_answers, sort_keys=True)
    a_hash = hashlib.md5(answers_str.encode()).hexdigest()[:8]
    key = _cache_key(q_hash, a_hash)
    prefix = "synthesis"

    if not force:
        cached = _load_cache(prefix, key)
        if cached:
            return cached

    # Build a readable summary of each expert's answer
    answers_block = []
    for market, ans in expert_answers.items():
        expert = ans.get("expert_name", market)
        answer_text = ans.get("answer", "Not available")
        ts = ans.get("timestamp", "")
        quote = ans.get("verified_quote") or ans.get("supporting_quote") or ""
        block = (
            f"=== {expert} ({market}) ===\n"
            f"Answer: {answer_text}\n"
            f"Timestamp: {ts}\n"
            f"Supporting quote: {quote}"
        )
        answers_block.append(block)

    user_message = (
        f"INTERVIEW GUIDE QUESTION:\n{question}\n\n"
        + "\n\n".join(answers_block)
    )

    result = _call_claude_json(SYNTHESIS_SYSTEM, user_message)

    _save_cache(prefix, key, result)
    return result


# ── Keyword retrieval for "Ask the Panel" ────────────────────────────────────

def _score_chunk(chunk: dict, query_words: list[str]) -> float:
    """
    Simple keyword frequency score for a chunk.
    Returns the fraction of query words that appear in the chunk text.
    """
    text_lower = chunk["text"].lower()
    hits = sum(1 for w in query_words if w in text_lower)
    return hits / max(len(query_words), 1)


def retrieve_chunks(
    query: str,
    all_chunks: dict[str, list[dict]],
    top_n: int = RETRIEVAL_TOP_N,
) -> list[dict]:
    """
    Keyword-based retrieval: score all chunks across all experts against
    the query and return the top_n by score.

    This simulates the retrieval step you'd use with embeddings at scale.
    Only chunks with at least one keyword match are returned.
    """
    # Tokenise query into meaningful words (≥3 chars, not stop words)
    stop_words = {
        "the", "and", "for", "are", "was", "you", "that", "this", "what",
        "how", "why", "did", "does", "can", "will", "with", "from", "have",
        "has", "not", "but", "they", "their", "your", "our", "its", "all",
        "any", "each", "been", "about", "more", "when", "where", "which",
    }
    words = [
        w.strip(".,?!\"'()")
        for w in query.lower().split()
        if len(w) >= 3 and w not in stop_words
    ]

    scored = []
    for market, chunks in all_chunks.items():
        for chunk in chunks:
            score = _score_chunk(chunk, words)
            if score > 0:
                scored.append((score, chunk))

    # Sort by score descending, break ties by chunk_index (prefer earlier)
    scored.sort(key=lambda x: (-x[0], x[1].get("chunk_index", 0)))
    return [c for _, c in scored[:top_n]]


def ask_panel(
    query: str,
    all_chunks: dict[str, list[dict]],
    chat_history: Optional[list[dict]] = None,
) -> str:
    """
    Answer a free-form question using only keyword-retrieved transcript chunks.

    Args:
        query:          The user's question.
        all_chunks:     {market: [chunks]} for all three experts.
        chat_history:   Optional list of {role, content} for conversational context.

    Returns:
        The model's answer string (with inline citations).
    """
    retrieved = retrieve_chunks(query, all_chunks)

    if not retrieved:
        return (
            "This isn't covered in the transcripts provided. "
            "No relevant chunks were found matching your query."
        )

    # Format retrieved chunks
    context_lines = []
    for c in retrieved:
        context_lines.append(
            f"[{c['expert_name']} | {c['market']} | {c['timestamp']}]\n"
            f"{c['speaker']}: {c['text']}"
        )
    context_block = "\n\n---\n\n".join(context_lines)

    user_message = (
        f"RELEVANT TRANSCRIPT EXCERPTS:\n\n"
        f"{context_block}\n\n"
        f"USER QUESTION: {query}"
    )

    # Build messages list (supports multi-turn)
    messages = []
    if chat_history:
        messages.extend(chat_history[-6:])  # keep last 3 turns for context
    messages.append({"role": "user", "content": user_message})

    if not _get_anthropic_key() and _get_gemini_key():
        return _call_gemini_chat(PANEL_CHAT_SYSTEM, messages)

    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=PANEL_CHAT_SYSTEM,
        messages=messages,
    )
    return response.content[0].text.strip()


def get_retrieved_chunks_for_display(
    query: str,
    all_chunks: dict[str, list[dict]],
) -> list[dict]:
    """Return the retrieved chunks for display in the UI."""
    return retrieve_chunks(query, all_chunks)

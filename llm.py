"""
llm.py — Groq API wrapper for Expert Interview Analyzer.

Provides four main operations:
  1. get_expert_answer()     — per-expert Q&A with structured JSON output
  2. re_prompt_exact_quote() — retry with explicit "copy exactly" instruction
  3. synthesize_question()   — cross-expert synthesis for one question
  4. ask_panel()             — free-form chat with chunk-based retrieval context

All calls go via the Groq API (high-speed LPU inference) as the single provider.
The active model is controlled by the GROQ_MODEL constant below.
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
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

import requests

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_DIR = Path("cache")

# Maximum number of transcript chunks to include in a single API call.
MAX_CHUNKS_PER_CALL = 40

# Number of top chunks to retrieve for the "Ask the Panel" feature.
RETRIEVAL_TOP_N = 8


class RateLimitError(Exception):
    """Raised when Groq API rate limits (HTTP 429) are exhausted."""
    pass


def _get_groq_key() -> Optional[str]:
    """Retrieve GROQ_API_KEY from environment or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                key = st.secrets["GROQ_API_KEY"]
                if key:
                    os.environ["GROQ_API_KEY"] = key
        except Exception:
            pass
    return key


def get_active_model_name() -> str:
    """Return human-readable active model name derived directly from GROQ_MODEL."""
    return f"Groq ({GROQ_MODEL})"


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(*parts: str) -> str:
    combined = f"{GROQ_MODEL}||" + "||".join(parts)
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
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(prefix, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Transcript formatting ───────────────────────────────────────────────────

def _format_chunks_as_text(chunks: list[dict], max_chunks: int = MAX_CHUNKS_PER_CALL) -> str:
    """
    Format chunk dicts into a clean dialogue string for the prompt.
    Labels each turn with [Speaker | Timestamp].
    """
    lines = []
    for c in chunks[:max_chunks]:
        speaker = c.get("speaker", "Unknown")
        ts = c.get("timestamp", "00:00")
        text = c.get("text", "").strip()
        lines.append(f"[{speaker} | {ts}]\n{text}")
    return "\n\n".join(lines)


# ── Prompts ──────────────────────────────────────────────────────────────────

EXPERT_ANSWER_SYSTEM = """\
You are an expert market research analyst analysing interview transcripts about \
robotic surgery. Your task is to answer questions about what a specific expert said, \
relying STRICTLY AND ENTIRELY on the provided transcript excerpt.

CRITICAL INSTRUCTIONS — ZERO HALLUCINATION POLICY:
1. Base your answer ONLY on facts, opinions, and figures explicitly stated in the \
provided transcript excerpt. Do NOT infer, extrapolate, or use outside knowledge.
2. If the expert did NOT discuss or mention the topic, you MUST respond with:
   "Not discussed in this transcript" for the answer field, leave timestamp empty, \
and leave supporting_quote empty. Do NOT invent an answer.
3. When the expert DID discuss the topic:
   a. "answer": Write a clear, concise synthesis of what they said (2-4 sentences).
   b. "timestamp": Give the MM:SS timestamp from the transcript turn where they \
said this most directly.
   c. "supporting_quote": Provide a VERBATIM, EXACT quote from the transcript text \
that supports the answer. Do NOT edit, paraphrase, fix grammar, or change even one word. \
It must be a precise substring of the transcript text.
4. Do NOT fabricate numbers, prices, percentages, adoption rates, timelines, or names. If a number is written out in words in the transcript (e.g. "one-point-five"), you MUST write it exactly as words in your answer. Do NOT convert it to digits like "1.5".

Respond ONLY with a valid JSON object matching this schema:
{
  "answer": "<your synthesis, or 'Not discussed in this transcript'>",
  "timestamp": "<MM:SS or empty string>",
  "supporting_quote": "<verbatim substring of transcript, or empty string>"
}"""


EXPERT_ANSWER_RETRY_SYSTEM = """\
You are an expert market research analyst. Your previous response included a quote or factual claims \
that could NOT be found in the transcript.

You must fix this immediately. Both the supporting_quote AND every fact, figure, and claim in the \
answer paragraph must be strictly grounded in the provided transcript text, not just the quote.

CRITICAL INSTRUCTIONS:
1. The "supporting_quote" field MUST be an EXACT, CHARACTER-FOR-CHARACTER substring \
copied directly from the provided transcript text.
2. The "answer" field MUST be strictly grounded: every fact, number, currency amount, \
and proper noun in the answer paragraph must be explicitly stated in the transcript text. \
Do NOT invent, extrapolate, or estimate figures.
3. Do not change punctuation, do not fix spoken grammar, do not omit words in supporting_quote.
4. If a number is written as a word in the transcript (e.g. "one-point-five"), keep it as a word. Do not convert to digits.
5. If you cannot find an exact verbatim quote to support the answer, set \
"answer" to "Not discussed in this transcript", "timestamp" to "", and \
"supporting_quote" to "".
6. Return ONLY valid JSON.

Schema:
{
  "answer": "<synthesis or 'Not discussed in this transcript'>",
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
in parentheses, e.g. "(Dr. Jean Martin, France)".
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


# ── Core Groq API functions ─────────────────────────────────────────────────

def _call_groq(system: str, user_message: str, json_mode: bool = True) -> str:
    """Call Groq API with automatic retry on transient errors or rate limits."""
    api_key = _get_groq_key()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Please set it in .env or Streamlit Cloud Secrets."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(5):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        sleep_s = float(retry_after) + 0.5
                    except ValueError:
                        sleep_s = (attempt + 1) * 3
                else:
                    sleep_s = (attempt + 1) * 3
                if sleep_s > 30:
                    logger.warning("Groq rate limit retry-after too long (%.1fs); raising RateLimitError", sleep_s)
                    raise RateLimitError(f"Groq rate limit exceeded. Retry after {int(sleep_s)}s.")
                logger.warning("Groq rate limit 429 encountered, sleeping %.1fs...", sleep_s)
                time.sleep(sleep_s)
                continue
            else:
                logger.error("Groq API error %s: %s", resp.status_code, resp.text[:200])
                raise RuntimeError(f"Groq API returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            if attempt == 4:
                if "429" in str(e) or (hasattr(e, "response") and getattr(e.response, "status_code", None) == 429):
                    raise RateLimitError("Groq rate limit exceeded.")
                raise e
            time.sleep(1)
    raise RateLimitError("Groq API rate limit exceeded after retries.")


def _call_groq_chat(system: str, messages: list[dict]) -> str:
    """Call Groq API for multi-turn conversational panel chat."""
    api_key = _get_groq_key()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Please set it in .env or Streamlit Cloud Secrets."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    groq_messages = [{"role": "system", "content": system}]
    for m in messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": GROQ_MODEL,
        "messages": groq_messages,
        "temperature": 0.0,
        "max_tokens": 1500,
    }

    for attempt in range(5):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        sleep_s = float(retry_after) + 0.5
                    except ValueError:
                        sleep_s = (attempt + 1) * 3
                else:
                    sleep_s = (attempt + 1) * 3
                if sleep_s > 30:
                    raise RateLimitError(f"Groq rate limit exceeded. Retry after {int(sleep_s)}s.")
                time.sleep(sleep_s)
                continue
            else:
                raise RuntimeError(f"Groq API returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            if attempt == 4:
                if "429" in str(e) or (hasattr(e, "response") and getattr(e.response, "status_code", None) == 429):
                    raise RateLimitError("Groq rate limit exceeded.")
                raise e
            time.sleep(1)
    raise RateLimitError("Groq API rate limit exceeded after retries.")


def _parse_json_response(raw: str) -> dict:
    """
    Safely parse a JSON response from the model.
    Handles markdown code fences if present.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
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


def _call_groq_json(system: str, user_message: str) -> dict:
    """
    Call Groq and parse JSON response. If JSON parsing fails on the first attempt,
    retry ONCE with an explicit instruction appended before falling back to raw-text behavior.
    If 429 rate limit is hit, returns dict with rate_limited=True.
    """
    try:
        raw = _call_groq(system, user_message, json_mode=True)
    except RateLimitError:
        return {
            "answer": "Groq free tier rate limit reached. Please wait a moment and retry.",
            "timestamp": "",
            "supporting_quote": "",
            "rate_limited": True,
        }
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        return {
            "answer": f"Error calling Groq API: {e}",
            "timestamp": "",
            "supporting_quote": "",
            "error": True,
        }

    result = _parse_json_response(raw)
    if result.get("parse_error"):
        logger.warning("JSON parse failed on first attempt. Retrying with explicit instruction.")
        retry_message = (
            f"{user_message}\n\n"
            "Your previous response was not valid JSON. Return ONLY the JSON object matching the schema, "
            "with no markdown formatting, no preamble, no explanation."
        )
        try:
            retry_raw = _call_groq(system, retry_message, json_mode=True)
            retry_result = _parse_json_response(retry_raw)
            return retry_result
        except RateLimitError:
            return {
                "answer": "Groq free tier rate limit reached. Please wait a moment and retry.",
                "timestamp": "",
                "supporting_quote": "",
                "rate_limited": True,
            }
        except Exception as e:
            logger.error("Retry also failed: %s", e)
            return result
    return result


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

    result = _call_groq_json(EXPERT_ANSWER_SYSTEM, user_message)

    if not result.get("rate_limited") and not result.get("error"):
        _save_cache(prefix, key, result)
    return result


def re_prompt_exact_quote(
    question: str,
    chunks: list[dict],
    expert_name: str,
    market: str,
    bad_quote: str = "",
    ungrounded_items: list[str] = None,
) -> dict:
    """
    Re-prompt Groq with explicit instructions if quote or answer grounding fails.
    """
    transcript_text = _format_chunks_as_text(chunks)
    if bad_quote:
        retry_msg = (
            f"Your previous response included this quote which could NOT be found verbatim in the transcript:\n"
            f"\"{bad_quote}\"\n\n"
            f"Please provide a corrected answer with a supporting_quote that is a "
            f"verbatim, unmodified substring of the transcript text above."
        )
    else:
        retry_msg = (
            f"Please provide a corrected answer with a supporting_quote that is a "
            f"verbatim, unmodified substring of the transcript text above."
        )

    user_message = (
        f"TRANSCRIPT EXCERPT (Expert: {expert_name}, Market: {market}):\n\n"
        f"{transcript_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"{retry_msg}"
    )

    if ungrounded_items:
        user_message += (
            f"\n\nADDITIONALLY, your previous answer included these specific facts/figures that could "
            f"NOT be found in the transcript: {', '.join(ungrounded_items)}. Your corrected answer "
            f"must NOT include any of these or similar invented details — describe only what is "
            f"explicitly stated in the transcript, in general terms if no specific figure was given."
        )

    return _call_groq_json(EXPERT_ANSWER_RETRY_SYSTEM, user_message)


def synthesize_question(
    question: str,
    expert_answers: dict[str, dict],
    force: bool = False,
) -> dict:
    """
    Generate a cross-expert synthesis for one interview question using Groq.

    Args:
        question:       The interview guide question text.
        expert_answers: {market: verified_answer_dict} for all three experts.
        force:          If True, bypass cache.

    Returns:
        dict: {common_themes, disagreements, summary_line}
    """
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    answers_str = json.dumps(expert_answers, sort_keys=True)
    a_hash = hashlib.md5(answers_str.encode()).hexdigest()[:8]
    key = _cache_key(q_hash, a_hash)
    prefix = "synthesis"

    if not force:
        cached = _load_cache(prefix, key)
        if cached:
            return cached

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

    result = _call_groq_json(SYNTHESIS_SYSTEM, user_message)

    if not result.get("rate_limited") and not result.get("error"):
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
    """
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

    scored.sort(key=lambda x: (-x[0], x[1].get("chunk_index", 0)))
    return [c for _, c in scored[:top_n]]


def ask_panel(
    query: str,
    all_chunks: dict[str, list[dict]],
    chat_history: Optional[list[dict]] = None,
) -> str:
    """
    Answer a free-form question using only keyword-retrieved transcript chunks.
    Uses Groq as the single provider.
    """
    retrieved = retrieve_chunks(query, all_chunks)

    if not retrieved:
        return (
            "This isn't covered in the transcripts provided. "
            "No relevant chunks were found matching your query."
        )

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

    messages = []
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_message})

    try:
        return _call_groq_chat(PANEL_CHAT_SYSTEM, messages)
    except RateLimitError:
        return "⚠️ Groq free tier rate limit reached. Please wait a moment and retry."
    except Exception as e:
        logger.error("Groq panel chat error: %s", e)
        return f"Error contacting Groq: {e}"


def get_retrieved_chunks_for_display(
    query: str,
    all_chunks: dict[str, list[dict]],
) -> list[dict]:
    """Return the retrieved chunks for display in the UI."""
    return retrieve_chunks(query, all_chunks)

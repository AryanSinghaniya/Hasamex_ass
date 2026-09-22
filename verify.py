"""
verify.py — Quote verification module for Expert Interview Analyzer.

After the LLM returns a supporting_quote, this module checks that the quote
is an exact (or near-exact) substring of the original transcript text.
This is the critical anti-hallucination step — we never display quotes that
cannot be verified against the source transcript.
"""

import difflib
import logging
import unicodedata
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum similarity ratio (0–1) required to accept a fuzzy match.
# 0.85 means 85% character-level similarity after normalisation.
MATCH_THRESHOLD = 0.85

# When searching "near" a timestamp, look at this many characters of
# transcript context on each side of the timestamp mention.
CONTEXT_WINDOW_CHARS = 1500


def _normalize(text: str) -> str:
    """
    Normalise text for comparison:
    - Collapse runs of whitespace (spaces, newlines, tabs) to a single space
    - Strip leading/trailing whitespace
    - Normalise unicode to NFC
    - Lowercase
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _find_context_around_timestamp(transcript_text: str, timestamp: str) -> str:
    """
    Return the portion of the transcript near the given timestamp.
    We locate the timestamp string in the raw text and return
    CONTEXT_WINDOW_CHARS characters either side of it.
    """
    idx = transcript_text.find(timestamp)
    if idx == -1:
        # Timestamp not found verbatim — return full transcript for searching
        return transcript_text
    start = max(0, idx - 200)  # timestamps appear before the text
    end = min(len(transcript_text), idx + CONTEXT_WINDOW_CHARS)
    return transcript_text[start:end]


def verify_quote(
    supporting_quote: str,
    transcript_text: str,
    timestamp: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Check whether supporting_quote is a verifiable substring of transcript_text.

    Strategy:
    1. Exact match (after normalising whitespace) — fastest check.
    2. If timestamp is given, narrow search to context around that timestamp.
    3. Fuzzy match using difflib.SequenceMatcher — accepts if ratio ≥ MATCH_THRESHOLD.

    Returns:
        (is_valid: bool, matched_text: str | None)
        matched_text is the best-matching span found in the transcript,
        or None if no match meets the threshold.
    """
    if not supporting_quote or not supporting_quote.strip():
        return False, None

    norm_quote = _normalize(supporting_quote)
    norm_transcript = _normalize(transcript_text)

    # ── Step 1: Exact (normalised) substring match ───────────────────────
    if norm_quote in norm_transcript:
        # Recover original-casing match span for display
        idx = norm_transcript.find(norm_quote)
        return True, transcript_text[idx: idx + len(supporting_quote) + 20].strip()

    # ── Step 2: Narrow to timestamp context if available ─────────────────
    search_text = transcript_text
    if timestamp:
        context = _find_context_around_timestamp(transcript_text, timestamp)
        norm_context = _normalize(context)
        if norm_quote in norm_context:
            idx = norm_context.find(norm_quote)
            return True, context[idx: idx + len(supporting_quote) + 20].strip()
        # Use the narrowed context for fuzzy matching
        search_text = context

    # ── Step 3: Fuzzy sliding-window match ───────────────────────────────
    norm_search = _normalize(search_text)
    q_len = len(norm_quote)

    if q_len == 0:
        return False, None

    best_ratio = 0.0
    best_start = 0

    # Slide a window of ±30% around the quote length
    min_window = max(10, int(q_len * 0.7))
    max_window = int(q_len * 1.3) + 50

    step = max(1, q_len // 4)  # step size for efficiency

    for start in range(0, max(1, len(norm_search) - min_window), step):
        for window in (q_len, min_window, max_window):
            end = min(start + window, len(norm_search))
            candidate = norm_search[start:end]
            ratio = difflib.SequenceMatcher(
                None, norm_quote, candidate, autojunk=False
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start

    if best_ratio >= MATCH_THRESHOLD:
        # Return original-text snippet for display
        end = min(best_start + q_len + 50, len(search_text))
        matched = search_text[best_start:end].strip()
        logger.debug(
            "verify_quote: fuzzy match ratio=%.2f for quote='%s...'",
            best_ratio,
            supporting_quote[:60],
        )
        return True, matched

    logger.warning(
        "verify_quote: FAILED (best ratio=%.2f < %.2f) for quote='%s...'",
        best_ratio,
        MATCH_THRESHOLD,
        supporting_quote[:80],
    )
    return False, None


def verify_and_repair(
    answer_json: dict,
    transcript_text: str,
    re_prompt_fn,
    question: str,
    chunks: list[dict],
    expert_name: str,
) -> dict:
    """
    Verify the supporting_quote in answer_json.
    If verification fails, call re_prompt_fn once for a corrected answer,
    then verify again.
    If still failing, discard the quote and mark quote_verified=False.

    Args:
        answer_json:    Dict with keys: answer, timestamp, supporting_quote
        transcript_text: Raw full transcript text (body) for substring matching
        re_prompt_fn:   Callable(question, chunks, expert_name, bad_quote) → new answer_json
        question:       The original question text (for re-prompting)
        chunks:         Expert's transcript chunks (for re-prompting)
        expert_name:    Expert identifier string (for re-prompting)

    Returns:
        Updated answer_json with added fields:
            quote_verified (bool)
            verified_quote (str | None)  — the matched text span, or None
    """
    quote = answer_json.get("supporting_quote", "")
    timestamp = answer_json.get("timestamp", "")

    # Handle "not discussed" responses — no verification needed
    if _is_not_discussed(answer_json.get("answer", "")):
        return {
            **answer_json,
            "quote_verified": True,  # vacuously valid — nothing to verify
            "verified_quote": None,
            "not_discussed": True,
        }

    is_valid, matched = verify_quote(quote, transcript_text, timestamp)

    if is_valid:
        return {
            **answer_json,
            "quote_verified": True,
            "verified_quote": matched,
            "not_discussed": False,
        }

    # ── First attempt failed — re-prompt once ────────────────────────────
    logger.warning(
        "[verify] Quote failed for %s. Re-prompting once.", expert_name
    )
    try:
        new_answer = re_prompt_fn(question, chunks, expert_name, bad_quote=quote)
        new_quote = new_answer.get("supporting_quote", "")
        new_timestamp = new_answer.get("timestamp", timestamp)
        is_valid2, matched2 = verify_quote(new_quote, transcript_text, new_timestamp)
    except Exception as e:
        logger.error("[verify] Re-prompt failed: %s", e)
        is_valid2, matched2 = False, None
        new_answer = answer_json

    if is_valid2:
        return {
            **new_answer,
            "quote_verified": True,
            "verified_quote": matched2,
            "not_discussed": False,
        }

    # ── Both attempts failed — discard quote ─────────────────────────────
    logger.warning(
        "[verify] Quote could not be verified after re-prompt for %s. Discarding.",
        expert_name,
    )
    return {
        **answer_json,
        "supporting_quote": None,
        "quote_verified": False,
        "verified_quote": None,
        "not_discussed": False,
    }


def _is_not_discussed(answer_text: str) -> bool:
    """Return True if the answer indicates the topic wasn't covered."""
    if not answer_text:
        return False
    lower = answer_text.lower()
    markers = [
        "not discussed",
        "not addressed",
        "not covered",
        "no mention",
        "not mentioned",
        "transcript does not",
        "transcript doesn't",
    ]
    return any(m in lower for m in markers)

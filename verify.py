"""
verify.py — Quote verification and Answer Grounding module for Expert Interview Analyzer.

Performs dual-layer verification:
1. verify_quote(): Checks that supporting_quote is an exact (or near-exact) substring
   of the original transcript text.
2. check_answer_grounding(): Extracts numbers, metrics, currency, durations, and proper nouns
   from the answer paragraph and ensures every fact is traceable to the source transcript.
3. verify_and_repair(): Orchestrates re-prompting if either quote or answer claims are ungrounded.
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

# Words to ignore from proper noun extraction (sentence starters, generic terms)
STOP_ENTITIES = {
    "The", "This", "That", "These", "Those", "There", "Here", "What", "When",
    "Where", "Which", "While", "However", "Beyond", "First", "Second", "Third",
    "Finally", "Additionally", "Overall", "Specifically", "In", "On", "At", "To",
    "From", "With", "By", "For", "Of", "About", "Although", "Because", "Since",
    "Question", "Answer", "Expert", "Interviewer", "Market", "Role", "Transcript",
    "Yes", "No", "Sure", "Well", "Right", "Many", "Most", "Some", "Several",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Not", "None", "All", "Both", "Key", "Single", "Major", "Public", "Private",
    "Hospital", "Hospitals", "Surgery", "Surgical", "Robotic", "Robots",
    "According", "Furthermore", "Moreover", "Instead", "Consequently",
}

NUMBER_WORDS_MAP = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    12: "twelve", 18: "eighteen", 20: "twenty", 24: "twenty-four",
    40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    75: "seventy-five", 80: "eighty", 100: "hundred",
    180: "one hundred and eighty", 200: "two hundred",
    220: "two hundred and twenty", 300: "three hundred",
    350: "three hundred and fifty", 400: "four hundred",
    450: "four hundred and fifty", 600: "six hundred",
}


def _normalize(text: str) -> str:
    """
    Normalise text for comparison:
    - Collapse runs of whitespace to a single space
    - Strip leading/trailing whitespace
    - Normalise unicode to NFC
    - Lowercase
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _find_context_around_timestamp(transcript_text: str, timestamp: str) -> str:
    """Return the portion of the transcript near the given timestamp."""
    idx = transcript_text.find(timestamp)
    if idx == -1:
        return transcript_text
    start = max(0, idx - 200)
    end = min(len(transcript_text), idx + CONTEXT_WINDOW_CHARS)
    return transcript_text[start:end]


def verify_quote(
    supporting_quote: str,
    transcript_text: str,
    timestamp: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Check whether supporting_quote is a verifiable substring of transcript_text.
    Returns (is_valid: bool, matched_text: str | None).
    """
    if not supporting_quote or not supporting_quote.strip():
        return False, None

    norm_quote = _normalize(supporting_quote)
    norm_transcript = _normalize(transcript_text)

    # 1. Exact normalised substring match
    if norm_quote in norm_transcript:
        idx = norm_transcript.find(norm_quote)
        return True, transcript_text[idx: idx + len(supporting_quote) + 20].strip()

    # 2. Narrow to timestamp context if available
    search_text = transcript_text
    if timestamp:
        context = _find_context_around_timestamp(transcript_text, timestamp)
        norm_context = _normalize(context)
        if norm_quote in norm_context:
            idx = norm_context.find(norm_quote)
            return True, context[idx: idx + len(supporting_quote) + 20].strip()
        search_text = context

    # 3. Fuzzy sliding-window match (only for quotes >= 20 characters)
    norm_search = _normalize(search_text)
    q_len = len(norm_quote)
    if q_len < 20:
        # Require exact substring match for short quotes (< 20 chars) to prevent false-positive matches
        logger.warning(
            "verify_quote: Short quote (%d chars < 20) not found as exact substring: '%s'",
            q_len,
            supporting_quote[:80],
        )
        return False, None

    best_ratio = 0.0
    best_start = 0
    min_window = max(10, int(q_len * 0.7))
    max_window = int(q_len * 1.3) + 50
    step = max(1, q_len // 4)

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
        end = min(best_start + q_len + 50, len(search_text))
        matched = search_text[best_start:end].strip()
        return True, matched

    logger.warning(
        "verify_quote: FAILED (best ratio=%.2f < %.2f) for quote='%s...'",
        best_ratio,
        MATCH_THRESHOLD,
        supporting_quote[:80],
    )
    return False, None


# ── Grounding Verification for the answer field ──────────────────────────────

def extract_numbers_and_measures(text: str) -> list[str]:
    """Extract numbers, currency, percentages, ranges, and durations from text."""
    patterns = [
        # Currency: €1.5M, €1,500, £1.8 million, 200,000 euros, etc.
        r"(?:€|£|\$)\s*\d+(?:[.,]\d+)?(?:\s*(?:[KkMmBb]|million|thousand|billion))?",
        r"\b\d+(?:[.,]\d+)?\s*(?:euros?|pounds?|dollars?)\b",
        # Percentages: 75%, 60 percent, etc.
        r"\b\d+(?:[.,]\d+)?\s*%",
        r"\b\d+(?:[.,]\d+)?\s*percent\b",
        # Ranges and durations: 1.5-2 million, 12-to-18-month, 18 to 24 months, 3-5 years, etc.
        r"\b\d+(?:[.,]\d+)?\s*(?:[-–to]+\s*\d+(?:[.,]\d+)?)?\s*(?:months?|years?|days?|weeks?|hours?|million|billion|thousand)\b",
        # Years / Dates: 2024, 2025, 2026, 2028, 2030
        r"\b(?:19|20)\d{2}\b",
        # Quantities with unit words: 300 systems, 50 cases, 40 centres, etc.
        r"\b\d+(?:[.,]\d+)?\s*(?:systems?|cases?|centres?|centers?|hospitals?|sites?|surgeons?|programmes?|programs?)\b",
        # Standalone significant numbers
        r"\b\d{2,}(?:[.,]\d+)?\b",
    ]
    combined_pattern = "|".join(patterns)
    matches = re.findall(combined_pattern, text, flags=re.IGNORECASE)
    seen = set()
    result = []
    for m in matches:
        cleaned = m.strip()
        if cleaned.lower() not in seen and len(cleaned) > 0:
            seen.add(cleaned.lower())
            result.append(cleaned)
    return result


def extract_proper_nouns_and_phrases(text: str) -> list[str]:
    """Extract acronyms, capitalized multi-word phrases, and proper noun terms."""
    entities = []
    # Acronyms (ARS, HAS, GHS, CHU, DRG, NUB, G-BA, NICE, NHS, ICS, CCS, EMR, CFO)
    acronyms = re.findall(r"\b[A-Z]{2,}(?:-[A-Z]+)?\b", text)
    for acr in acronyms:
        if acr not in STOP_ENTITIES and len(acr) >= 2:
            entities.append(acr)

    # Multi-word proper nouns: e.g. "Intuitive Surgical", "Medtronic Hugo", "da Vinci Xi", "NICE guidance"
    multi_words = re.findall(
        r"\bda\s+Vinci(?:\s+[A-Z0-9][a-z0-9]*)*\b|"
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|"
        r"\b[A-Z]{2,}\s+(?:tariffs?|systems?|guidance|frameworks?|programmes?|programs?)\b",
        text,
    )
    for mw in multi_words:
        cleaned = mw.strip()
        words = cleaned.split()
        if all(w in STOP_ENTITIES for w in words):
            continue
        if len(cleaned) > 2 and cleaned not in STOP_ENTITIES:
            entities.append(cleaned)

    seen = set()
    deduped = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            deduped.append(e)
    return deduped


def check_phrase_grounding(phrase: str, transcript_text: str, norm_transcript: str) -> bool:
    """Check if a specific number or phrase appears in transcript_text."""
    norm_phrase = _normalize(phrase)
    if not norm_phrase:
        return True

    if re.search(rf"\b{re.escape(norm_phrase)}\b", norm_transcript):
        return True

    # Check written-out number equivalence
    phrase_in_words = norm_phrase
    phrase_in_words = phrase_in_words.replace("200,000", "two hundred thousand")
    phrase_in_words = phrase_in_words.replace("200000", "two hundred thousand")
    phrase_in_words = phrase_in_words.replace("1,500", "fifteen hundred")
    phrase_in_words = phrase_in_words.replace("1500", "fifteen hundred")
    phrase_in_words = phrase_in_words.replace("2,000", "two thousand")
    phrase_in_words = phrase_in_words.replace("2000", "two thousand")
    phrase_in_words = phrase_in_words.replace("1,200", "twelve hundred")
    phrase_in_words = phrase_in_words.replace("1200", "twelve hundred")
    phrase_in_words = phrase_in_words.replace("1,800", "eighteen hundred")
    phrase_in_words = phrase_in_words.replace("1800", "eighteen hundred")
    phrase_in_words = phrase_in_words.replace("1.5", "one-point-five")
    phrase_in_words = phrase_in_words.replace("1.8", "one-point-eight")

    for num, word in sorted(NUMBER_WORDS_MAP.items(), key=lambda x: -x[0]):
        phrase_in_words = re.sub(rf"\b{num}\b", word, phrase_in_words)

    if re.search(rf"\b{re.escape(phrase_in_words)}\b", norm_transcript):
        return True

    # Minimum-length gate: Fuzzy matching is ONLY used for phrases with length >= 20 chars.
    # Short phrases (< 20 chars) like acronyms and short names MUST be exact substring matches.
    p_len = len(norm_phrase)
    if p_len < 20:
        return False

    step = max(1, p_len // 4)
    best_ratio = 0.0
    for i in range(0, max(1, len(norm_transcript) - p_len + 1), step):
        window = norm_transcript[i : i + p_len + 4]
        ratio = difflib.SequenceMatcher(None, norm_phrase, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if ratio >= MATCH_THRESHOLD:
            return True

    return False


def check_answer_grounding(answer_text: str, transcript_text: str) -> list[str]:
    """
    Extract numbers, currency amounts, percentages, ranges, durations,
    and capitalized proper-noun phrases from answer_text, and check whether
    each one appears in transcript_text (exact match or fuzzy ratio >= 0.85).

    Returns:
        List of ungrounded phrases/numbers that could NOT be found in the transcript.
    """
    if not answer_text or _is_not_discussed(answer_text):
        return []

    norm_transcript = _normalize(transcript_text)
    numbers = extract_numbers_and_measures(answer_text)
    proper_nouns = extract_proper_nouns_and_phrases(answer_text)

    all_claims = numbers + proper_nouns
    ungrounded = []

    for item in all_claims:
        if not check_phrase_grounding(item, transcript_text, norm_transcript):
            ungrounded.append(item)

    return ungrounded


# ── Dual Verification & Repair ───────────────────────────────────────────────

def verify_and_repair(
    answer_json: dict,
    transcript_text: str,
    re_prompt_fn,
    question: str,
    chunks: list[dict],
    expert_name: str,
) -> dict:
    """
    Verify both the supporting_quote and the facts in the answer paragraph.

    1. Checks supporting_quote with verify_quote().
    2. Checks answer paragraph claims with check_answer_grounding().
    3. If either check fails, re-prompts the model ONCE with explicit feedback.
    4. If the retry still contains ungrounded claims, sets answer_ungrounded=True.
    """
    quote = answer_json.get("supporting_quote", "")
    timestamp = answer_json.get("timestamp", "")
    answer_text = answer_json.get("answer", "")

    # Handle "not discussed" responses
    if _is_not_discussed(answer_text):
        return {
            **answer_json,
            "quote_verified": True,
            "verified_quote": None,
            "not_discussed": True,
            "answer_grounded": True,
            "answer_ungrounded": False,
            "ungrounded_claims": [],
        }

    # Step 1: Check quote
    is_quote_valid, matched_quote = verify_quote(quote, transcript_text, timestamp)

    # Step 2: Check answer grounding
    ungrounded_claims = check_answer_grounding(answer_text, transcript_text)
    is_answer_grounded = len(ungrounded_claims) == 0

    if is_quote_valid and is_answer_grounded:
        return {
            **answer_json,
            "quote_verified": True,
            "verified_quote": matched_quote,
            "not_discussed": False,
            "answer_grounded": True,
            "answer_ungrounded": False,
            "ungrounded_claims": [],
        }

    # Step 3: Failure detected — re-prompt ONCE
    logger.warning(
        "[verify] Verification failed for %s. Quote valid: %s, Ungrounded claims: %s. Re-prompting once.",
        expert_name,
        is_quote_valid,
        ungrounded_claims,
    )

    bad_quote_arg = "" if is_quote_valid else quote
    ungrounded_arg = ungrounded_claims if not is_answer_grounded else []

    try:
        import inspect
        sig = inspect.signature(re_prompt_fn)
        if "ungrounded_items" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            new_answer = re_prompt_fn(
                question,
                chunks,
                expert_name,
                bad_quote=bad_quote_arg,
                ungrounded_items=ungrounded_arg,
            )
        else:
            new_answer = re_prompt_fn(
                question,
                chunks,
                expert_name,
                bad_quote=bad_quote_arg,
            )
        if new_answer.get("rate_limited") or new_answer.get("error"):
            logger.warning("[verify] Re-prompt returned rate limit or error; keeping first attempt.")
            return {
                **answer_json,
                "supporting_quote": matched_quote if is_quote_valid else None,
                "quote_verified": is_quote_valid,
                "verified_quote": matched_quote if is_quote_valid else None,
                "not_discussed": False,
                "answer_grounded": is_answer_grounded,
                "answer_ungrounded": not is_answer_grounded,
                "ungrounded_claims": ungrounded_claims,
            }

        new_quote = new_answer.get("supporting_quote", "")
        new_timestamp = new_answer.get("timestamp", timestamp)
        new_text = new_answer.get("answer", "")

        is_quote_valid2, matched_quote2 = verify_quote(new_quote, transcript_text, new_timestamp)
        ungrounded_claims2 = check_answer_grounding(new_text, transcript_text)
        is_answer_grounded2 = len(ungrounded_claims2) == 0

        final_quote_verified = is_quote_valid2
        final_matched_quote = matched_quote2 if is_quote_valid2 else None
        final_supporting_quote = new_quote if is_quote_valid2 else None

        return {
            **new_answer,
            "supporting_quote": final_supporting_quote,
            "quote_verified": final_quote_verified,
            "verified_quote": final_matched_quote,
            "not_discussed": False,
            "answer_grounded": is_answer_grounded2,
            "answer_ungrounded": not is_answer_grounded2,
            "ungrounded_claims": ungrounded_claims2,
        }
    except Exception as e:
        logger.error("[verify] Re-prompt error: %s", e)
        return {
            **answer_json,
            "supporting_quote": matched_quote if is_quote_valid else None,
            "quote_verified": is_quote_valid,
            "verified_quote": matched_quote if is_quote_valid else None,
            "not_discussed": False,
            "answer_grounded": is_answer_grounded,
            "answer_ungrounded": not is_answer_grounded,
            "ungrounded_claims": ungrounded_claims,
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

"""
parser.py — Transcript parsing module for Expert Interview Analyzer.

Reads transcript .txt files from the data/ directory and splits them into
structured chunks: {expert_name, market, timestamp, speaker, text}.
Results are cached to cache/ as JSON files.
"""

import re
import json
import os
import hashlib
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")

# Map market names to their transcript filenames
TRANSCRIPT_FILES = {
    "France": "Transcript_1_France.txt",
    "Germany": "Transcript_2_Germany.txt",
    "UK": "Transcript_3_UK.txt",
}

INTERVIEW_GUIDE_FILE = DATA_DIR / "Interview_Guide.txt"


# ── Header parsing ──────────────────────────────────────────────────────────

def _parse_header(lines: list[str]) -> dict:
    """
    Extract metadata from the transcript header block.
    Looks for lines like 'Expert Name: ...' and 'Market: ...'.
    Returns a dict with expert_name, role, market.
    """
    header = {}
    for line in lines:
        line = line.strip()
        if line.lower().startswith("expert name:"):
            header["expert_name"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("expert ") and " – " in line:
            header["expert_name"] = line.split(" – ", 1)[1].strip()
        elif line.lower().startswith("role:"):
            header["role"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("market:"):
            header["market"] = line.split(":", 1)[1].strip()
    return header


# ── Body / turn parsing ─────────────────────────────────────────────────────

# Matches timestamp lines like "01:23" or "01:23:45"
TIMESTAMP_RE = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)$", re.MULTILINE)

# Matches speaker lines like "Dr. Martin: some text..." or
# "Anna Keller: text..."  — speaker name ends at first colon.
SPEAKER_RE = re.compile(r"^([^:]{2,60}):\s+(.+)$")


def _parse_body(body_text: str, expert_name: str, market: str) -> list[dict]:
    """
    Parse the timestamped body of a transcript into a list of chunk dicts.

    Each chunk:
        {
            "expert_name": str,
            "market": str,
            "timestamp": str,       # e.g. "01:23"
            "speaker": str,
            "text": str,
            "chunk_index": int,
        }

    Handles multi-line text that belongs to a single turn by accumulating
    lines until the next timestamp is found.
    """
    chunks: list[dict] = []
    current_timestamp: Optional[str] = None
    current_speaker: Optional[str] = None
    current_lines: list[str] = []

    def _flush(ts, spk, lines):
        text = " ".join(lines).strip()
        if ts and spk and text:
            chunks.append({
                "expert_name": expert_name,
                "market": market,
                "timestamp": ts,
                "speaker": spk,
                "text": text,
                "chunk_index": len(chunks),
            })

    for raw_line in body_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        ts_match = TIMESTAMP_RE.match(line)
        if ts_match:
            # Flush previous turn
            _flush(current_timestamp, current_speaker, current_lines)
            current_timestamp = ts_match.group(1)
            current_speaker = None
            current_lines = []
            continue

        spk_match = SPEAKER_RE.match(line)
        if spk_match and current_timestamp:
            # If we were mid-turn with same timestamp, flush
            _flush(current_timestamp, current_speaker, current_lines)
            current_speaker = spk_match.group(1).strip()
            current_lines = [spk_match.group(2).strip()]
            continue

        # Continuation line — append to current turn
        if current_speaker:
            current_lines.append(line)

    # Flush final turn
    _flush(current_timestamp, current_speaker, current_lines)
    return chunks


# ── Public API ──────────────────────────────────────────────────────────────

def parse_transcript(market: str, force: bool = False) -> list[dict]:
    """
    Parse a transcript for the given market and return its chunks.

    Results are cached to cache/parsed_<market>.json.
    The cache is automatically invalidated if the source file has changed
    (detected via MD5 hash stored in cache/<market>_hash.txt).

    Args:
        market: One of 'France', 'Germany', 'UK'
        force:  If True, ignore existing cache and re-parse.

    Returns:
        List of chunk dicts.

    Raises:
        FileNotFoundError if the transcript file does not exist.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"parsed_{market}.json"
    hash_file  = CACHE_DIR / f"{market}_hash.txt"

    # ── Cache-staleness check ─────────────────────────────────────────────
    # Compute current hash of the source file.
    current_hash = get_file_hash(market)
    cached_hash  = hash_file.read_text(encoding="utf-8").strip() if hash_file.exists() else ""

    if not force and cache_file.exists() and current_hash == cached_hash:
        import logging as _log
        _log.getLogger(__name__).debug(
            "[parser] Cache hit for %s (hash=%s)", market, current_hash[:8]
        )
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    if cache_file.exists() and current_hash != cached_hash and not force:
        import logging as _log
        _log.getLogger(__name__).warning(
            "[parser] Transcript for %s changed (old=%s new=%s) — invalidating cache.",
            market, cached_hash[:8], current_hash[:8],
        )

    filename = TRANSCRIPT_FILES.get(market)
    if not filename:
        raise ValueError(f"Unknown market '{market}'. Known: {list(TRANSCRIPT_FILES)}")

    transcript_path = DATA_DIR / filename
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    raw = transcript_path.read_text(encoding="utf-8")

    # Split header from body at the first timestamp line
    body_start_idx = len(raw)
    for match in TIMESTAMP_RE.finditer(raw):
        # find the start of the line containing this match
        line_start = raw.rfind('\n', 0, match.start()) + 1
        if line_start == 0 and match.start() == 0:
            body_start_idx = 0
        else:
            body_start_idx = line_start
        break
    
    header_section = raw[:body_start_idx]
    body_section = raw[body_start_idx:]

    header = _parse_header(header_section.splitlines())
    expert_name = header.get("expert_name", "Unknown Expert")
    market_val  = header.get("market", market)

    chunks = _parse_body(body_section, expert_name, market_val)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Persist current hash so next run can detect changes
    hash_file.write_text(current_hash, encoding="utf-8")

    return chunks


def parse_all_transcripts(force: bool = False) -> dict[str, list[dict]]:
    """
    Parse all three transcripts. Returns {market: [chunks]} dict.
    """
    result = {}
    for market in TRANSCRIPT_FILES:
        try:
            result[market] = parse_transcript(market, force=force)
        except FileNotFoundError as e:
            result[market] = []
            print(f"[parser] Warning: {e}")
    return result


def get_header(market: str) -> dict:
    """
    Read and return only the header metadata from a transcript file
    (expert_name, role, market) WITHOUT doing a full parse.

    Used by app.py to display correct expert names without relying on
    any hardcoded lookup table.

    Returns:
        dict with keys: expert_name, role, market
    """
    filename = TRANSCRIPT_FILES.get(market)
    if not filename:
        raise ValueError(f"Unknown market: {market}")
    path = DATA_DIR / filename
    if not path.exists():
        return {"expert_name": "Unknown", "role": "Unknown", "market": market}
    raw = path.read_text(encoding="utf-8")
    # Find first timestamp
    body_start_idx = len(raw)
    for match in TIMESTAMP_RE.finditer(raw):
        line_start = raw.rfind('\n', 0, match.start()) + 1
        body_start_idx = line_start if line_start > 0 else 0
        break
        
    header_section = raw[:body_start_idx]
    return _parse_header(header_section.splitlines())


def log_all_headers() -> dict[str, dict]:
    """
    Parse and log the expert_name, role, and market from all transcript
    headers.  Call this on app startup to catch identity bugs immediately.

    Logs at INFO level and also prints to stdout for visibility in dev.

    Returns:
        {market: {expert_name, role, market}} for all transcripts.
    """
    import logging as _log
    logger = _log.getLogger(__name__)
    headers = {}
    print("\n" + "=" * 60)
    print("  TRANSCRIPT IDENTITY CHECK (parsed from file headers)")
    print("=" * 60)
    for market in TRANSCRIPT_FILES:
        try:
            h = get_header(market)
            headers[market] = h
            msg = (
                f"  [{market:>7}]  Name : {h.get('expert_name', 'MISSING')}\n"
                f"           Role : {h.get('role', 'MISSING')}\n"
                f"         Market : {h.get('market', 'MISSING')}"
            )
            print(msg)
            logger.info("[parser] Header OK — %s: %s (%s)",
                        market, h.get('expert_name'), h.get('role'))
        except Exception as e:
            print(f"  [{market}] ERROR: {e}")
            logger.error("[parser] Header read failed for %s: %s", market, e)
    print("=" * 60 + "\n")
    return headers


def get_transcript_text(market: str) -> str:
    """
    Return the full raw text of a transcript (body section only).
    Used by verify.py for substring checking.
    """

    filename = TRANSCRIPT_FILES.get(market)
    if not filename:
        raise ValueError(f"Unknown market: {market}")
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    raw = path.read_text(encoding="utf-8")
    # Split at first timestamp
    body_start_idx = len(raw)
    for match in TIMESTAMP_RE.finditer(raw):
        line_start = raw.rfind('\n', 0, match.start()) + 1
        body_start_idx = line_start if line_start > 0 else 0
        break
        
    return raw[body_start_idx:]


def load_interview_questions() -> list[str]:
    """
    Parse Interview_Guide.txt and return a list of the 6 question strings.
    Strips leading numbering (e.g. '1. ADOPTION BARRIERS\n   What...')
    and returns the combined heading + body as a single string per question.
    """
    if not INTERVIEW_GUIDE_FILE.exists():
        raise FileNotFoundError(f"Interview guide not found: {INTERVIEW_GUIDE_FILE}")

    text = INTERVIEW_GUIDE_FILE.read_text(encoding="utf-8")
    # Split on numbered question headers like "1.", "2.", ... "6."
    parts = re.split(r"\n(?=\d+\.\s+[A-Z])", text)

    questions = []
    for part in parts:
        part = part.strip()
        if re.match(r"^\d+\.", part):
            questions.append(part)

    if not questions:
        # Fallback: return whole guide as single block
        questions = [text.strip()]

    # Handle the case where the user pasted the simple questions from 1 to 6 without headers
    if len(questions) == 0 or len(questions) == 1:
        parts = re.split(r"(?m)^\d+\.\s+", text)
        if len(parts) > 1:
            questions = [p.strip() for p in parts[1:]]

    return questions


def file_status() -> dict:
    """
    Return a dict describing the presence of all expected data and cache files.
    Used by the Streamlit sidebar.
    """
    status = {}
    for market, fname in TRANSCRIPT_FILES.items():
        path = DATA_DIR / fname
        cache = CACHE_DIR / f"parsed_{market}.json"
        status[market] = {
            "transcript_exists": path.exists(),
            "transcript_path": str(path),
            "cache_exists": cache.exists(),
            "cache_path": str(cache),
        }
    status["interview_guide"] = {
        "exists": INTERVIEW_GUIDE_FILE.exists(),
        "path": str(INTERVIEW_GUIDE_FILE),
    }
    return status


def get_file_hash(market: str) -> str:
    """Return an MD5 hash of the transcript file content for cache invalidation."""
    filename = TRANSCRIPT_FILES.get(market, "")
    path = DATA_DIR / filename
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()

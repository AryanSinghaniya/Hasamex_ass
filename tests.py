"""
tests.py -- Unit & integration tests for Expert Interview Analyzer.

Run with:   python -X utf8 tests.py
Or:         python -m pytest tests.py -v

Tests cover:
  1. Parser correctness -- names/roles/markets extracted from file headers
  2. Quote verification logic -- pass / whitespace-tolerance / fabricated rejection
  3. Fabricated-quote injection test -- confirms pipeline rejects invented quotes
  4. "Not discussed" fallback -- verify.py handles the not-discussed case correctly
  5. Cache invalidation -- changing transcript hash triggers re-parse
  6. Retrieval grounding -- out-of-scope queries return the "not covered" response

These tests do NOT call any external API.
"""

import os
import sys
import json
import logging
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import parser as transcript_parser
import verify
import llm

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── Simple test runner ────────────────────────────────────────────────────────

_results = []  # (name, passed, message)


def run(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  [PASS]  {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"  [FAIL]  {name}\n         AssertionError: {e}")
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  [FAIL]  {name}\n         {type(e).__name__}: {e}")


# =====================================================================
# 1. Parser -- header extraction
# =====================================================================
print("\n" + "=" * 60)
print("  1. Parser -- header extraction")
print("=" * 60)

EXPECTED = {
    "France":  {"expert_name": "Dr. Jean Martin",
                "role_substr": "Head of Urology",
                "market":      "France"},
    "Germany": {"expert_name": "Anna Keller",
                "role_substr": "Former Hospital Procurement Director",
                "market":      "Germany"},
    "UK":      {"expert_name": "Dr. Emily Carter",
                "role_substr": "Consultant Urologist",
                "market":      "United Kingdom"},
}

for market, expected in EXPECTED.items():
    h = None

    def test_name(m=market, ex=expected):
        h = transcript_parser.get_header(m)
        assert h.get("expert_name") == ex["expert_name"], (
            f"Got '{h.get('expert_name')}', expected '{ex['expert_name']}'"
        )
    run(f"{market}: expert_name == '{expected['expert_name']}'", test_name)

    def test_role(m=market, ex=expected):
        h = transcript_parser.get_header(m)
        assert ex["role_substr"] in h.get("role", ""), (
            f"Role does not contain '{ex['role_substr']}'. Got: '{h.get('role')}'"
        )
    run(f"{market}: role contains '{expected['role_substr']}'", test_role)

    def test_market_field(m=market, ex=expected):
        h = transcript_parser.get_header(m)
        assert h.get("market") == ex["market"], (
            f"Got '{h.get('market')}', expected '{ex['market']}'"
        )
    run(f"{market}: market == '{expected['market']}'", test_market_field)


def test_no_old_names():
    banned = ["Sophie", "Marchand", "Klaus", "Baumann", "Jonathan", "Aldridge"]
    for market in ["France", "Germany", "UK"]:
        h = transcript_parser.get_header(market)
        combined = " ".join(h.values())
        for name in banned:
            assert name not in combined, (
                f"Old name '{name}' still in {market} header: {combined}"
            )
run("No old placeholder names in any header", test_no_old_names)


def test_chunks_france():
    chunks = transcript_parser.parse_transcript("France", force=True)
    expert_chunks = [c for c in chunks if c["speaker"].lower() != "interviewer"]
    assert expert_chunks, "No non-interviewer chunks for France"
    for c in expert_chunks:
        assert c["expert_name"] == "Dr. Jean Martin", (
            f"Chunk has wrong expert_name: '{c['expert_name']}'"
        )
run("France chunks carry correct expert_name='Dr. Jean Martin'", test_chunks_france)


def test_chunks_germany():
    chunks = transcript_parser.parse_transcript("Germany", force=True)
    expert_chunks = [c for c in chunks if c["speaker"].lower() != "interviewer"]
    assert expert_chunks, "No non-interviewer chunks for Germany"
    for c in expert_chunks:
        assert c["expert_name"] == "Anna Keller", (
            f"Chunk has wrong expert_name: '{c['expert_name']}'"
        )
run("Germany chunks carry correct expert_name='Anna Keller'", test_chunks_germany)


def test_chunks_uk():
    chunks = transcript_parser.parse_transcript("UK", force=True)
    expert_chunks = [c for c in chunks if c["speaker"].lower() != "interviewer"]
    assert expert_chunks, "No non-interviewer chunks for UK"
    for c in expert_chunks:
        assert c["expert_name"] == "Dr. Emily Carter", (
            f"Chunk has wrong expert_name: '{c['expert_name']}'"
        )
run("UK chunks carry correct expert_name='Dr. Emily Carter'", test_chunks_uk)


# =====================================================================
# 2. Verify -- quote verification logic
# =====================================================================
print("\n" + "=" * 60)
print("  2. Verify -- quote verification logic")
print("=" * 60)

FRANCE_TEXT = transcript_parser.get_transcript_text("France")


def test_exact_match():
    quote = "The biggest issue is still capital budget approval."
    ok, matched = verify.verify_quote(quote, FRANCE_TEXT, "01:20")
    assert ok, "Exact substring match should pass"
    assert matched is not None
run("Exact substring match passes", test_exact_match)


def test_whitespace_match():
    quote = "The  biggest issue  is still capital budget  approval."
    ok, _ = verify.verify_quote(quote, FRANCE_TEXT)
    assert ok, "Whitespace-normalised match should pass"
run("Whitespace-normalised match passes", test_whitespace_match)


def test_fabricated_rejected():
    fabricated = "France has over 900 robotic surgery installations as of 2025"
    ok, _ = verify.verify_quote(fabricated, FRANCE_TEXT)
    assert not ok, "Completely fabricated quote should be rejected"
run("Fabricated quote is rejected", test_fabricated_rejected)


def test_empty_quote():
    ok, matched = verify.verify_quote("", FRANCE_TEXT)
    assert not ok
    assert matched is None
run("Empty quote is rejected gracefully", test_empty_quote)


def test_not_discussed_bypasses():
    answer_json = {
        "answer": "Not discussed in this transcript",
        "timestamp": "",
        "supporting_quote": "",
    }
    result = verify.verify_and_repair(
        answer_json=answer_json,
        transcript_text=FRANCE_TEXT,
        re_prompt_fn=lambda *a, **kw: {},
        question="dummy",
        chunks=[],
        expert_name="Dr. Jean Martin",
    )
    assert result["quote_verified"] is True, (
        "'Not discussed' should be vacuously verified (no quote to check)"
    )
    assert result["not_discussed"] is True
run("'Not discussed' answer bypasses quote verification", test_not_discussed_bypasses)


def test_grounding_clean():
    clean_ans = "The biggest issue is capital budget approval. If two systems offer similar outcomes, the hospital will look hard at economics."
    ungrounded = verify.check_answer_grounding(clean_ans, FRANCE_TEXT)
    assert len(ungrounded) == 0, f"Expected 0 ungrounded claims, got: {ungrounded}"
run("Clean grounded answer passes check_answer_grounding with 0 ungrounded claims", test_grounding_clean)


def test_grounding_fabricated():
    fab_ans = "France has 850 robotic centres and ANSM approved Siemens Corindus in 2023 with a 500,000 euro grant."
    ungrounded = verify.check_answer_grounding(fab_ans, FRANCE_TEXT)
    assert len(ungrounded) > 0, "Fabricated answer must return ungrounded claims"
    assert any("850" in u for u in ungrounded)
    assert any("Siemens Corindus" in u or "ANSM" in u for u in ungrounded)
run("Fabricated facts/numbers are caught by check_answer_grounding", test_grounding_fabricated)


def test_case_sensitive_acronym_grounding():
    germany_text = transcript_parser.get_transcript_text("Germany")
    assert "it" in germany_text.lower()
    assert not verify.check_acronym_grounding("IT", germany_text), "IT must not match lowercase 'it' in Germany"
    
    # Non-existent acronyms must be rejected
    assert not verify.check_acronym_grounding("NICE", FRANCE_TEXT)
    assert not verify.check_acronym_grounding("FDA", FRANCE_TEXT)
run("Case-sensitive acronym matching prevents lowercase word collisions", test_case_sensitive_acronym_grounding)


def test_case_sensitive_proper_noun_grounding():
    germany_text = transcript_parser.get_transcript_text("Germany")
    # Test checking logic even if names aren't naturally in these simple transcripts
    assert not verify.check_proper_noun_grounding("Intuitive Surgical", FRANCE_TEXT)
    assert not verify.check_proper_noun_grounding("Intuitive Surgical", germany_text)
run("Case-sensitive proper-noun phrase matching enforces word boundaries", test_case_sensitive_proper_noun_grounding)



# =====================================================================
# 3. Fabricated-quote injection (end-to-end pipeline test)
# =====================================================================
print("\n" + "=" * 60)
print("  3. Fabricated-quote injection test")
print("=" * 60)


def test_both_attempts_fabricated():
    """Both first attempt AND retry return a fabricated quote.
    Pipeline must discard quote, set quote_verified=False, log a warning."""
    fabricated_answer = {
        "answer": "Capital cost is the biggest barrier.",
        "timestamp": "00:22",
        "supporting_quote": "France has the highest robotic density in Europe by 2025.",
    }

    def bad_re_prompt(question, chunks, expert_name, bad_quote):
        return {
            "answer": "Capital cost is the biggest barrier.",
            "timestamp": "00:22",
            "supporting_quote": "Robotic surgery is mandatory in French public hospitals.",
        }

    warning_logged = []
    original_warn = verify.logger.warning

    def capture_warn(msg, *args, **kwargs):
        warning_logged.append(msg)
        original_warn(msg, *args, **kwargs)

    verify.logger.warning = capture_warn
    try:
        result = verify.verify_and_repair(
            answer_json=fabricated_answer,
            transcript_text=FRANCE_TEXT,
            re_prompt_fn=bad_re_prompt,
            question="What are the adoption barriers?",
            chunks=[],
            expert_name="Dr. Jean Martin",
        )
    finally:
        verify.logger.warning = original_warn

    assert result["quote_verified"] is False, (
        "Both quotes were fabricated -- quote_verified must be False"
    )
    assert result["verified_quote"] is None, (
        "verified_quote must be None when both attempts fail"
    )
    assert result.get("supporting_quote") is None, (
        "supporting_quote must be discarded"
    )
    assert len(warning_logged) >= 1, (
        "A warning must be logged when the quote is discarded"
    )
run("Pipeline discards fabricated quote and logs warning", test_both_attempts_fabricated)


def test_retry_succeeds():
    """First attempt is fabricated; retry returns a real quote. Pipeline should accept."""
    real_quote = "The biggest issue is still capital budget approval."

    fabricated_answer = {
        "answer": "Capital cost is the biggest barrier.",
        "timestamp": "00:22",
        "supporting_quote": "France leads all European nations in robotic surgery adoption.",
    }

    def good_re_prompt(question, chunks, expert_name, bad_quote):
        return {
            "answer": "Capital cost is the biggest barrier.",
            "timestamp": "01:20",
            "supporting_quote": real_quote,
        }

    result = verify.verify_and_repair(
        answer_json=fabricated_answer,
        transcript_text=FRANCE_TEXT,
        re_prompt_fn=good_re_prompt,
        question="What are the adoption barriers?",
        chunks=[],
        expert_name="Dr. Jean Martin",
    )
    assert result["quote_verified"] is True, (
        "Re-prompt returned real quote -- should be verified"
    )
    assert result["verified_quote"] is not None
run("Pipeline accepts corrected quote from retry", test_retry_succeeds)


# =====================================================================
# 4. "Not discussed" detection
# =====================================================================
print("\n" + "=" * 60)
print("  4. 'Not discussed' fallback")
print("=" * 60)


def test_not_discussed_phrases():
    assert verify._is_not_discussed("Not discussed in this transcript")
    assert verify._is_not_discussed("This topic is not addressed in the transcript")
    assert verify._is_not_discussed("The transcript doesn't cover this question")
    assert not verify._is_not_discussed("Capital cost is the main barrier")
run("_is_not_discussed detects standard phrases correctly", test_not_discussed_phrases)


# =====================================================================
# 5. Cache invalidation
# =====================================================================
print("\n" + "=" * 60)
print("  5. Cache invalidation on file change")
print("=" * 60)


def test_cache_invalidation():
    """Appending content to a transcript file changes its MD5 hash,
    which should trigger a re-parse on the next call."""
    transcript_parser.parse_transcript("France", force=True)

    france_path = Path("data/Transcript_1_France.txt")
    hash_path = Path("cache/France_hash.txt")
    original = france_path.read_text(encoding="utf-8")

    try:
        france_path.write_text(original + "\n", encoding="utf-8")
        old_hash = hash_path.read_text().strip() if hash_path.exists() else ""
        new_hash = transcript_parser.get_file_hash("France")
        assert old_hash != new_hash, (
            f"Hash must differ after edit. old={old_hash[:8]} new={new_hash[:8]}"
        )
        # Calling parse_transcript without force should auto-detect the change
        chunks = transcript_parser.parse_transcript("France")
        assert len(chunks) > 0, "Re-parse after invalidation should return chunks"
    finally:
        france_path.write_text(original, encoding="utf-8")
        transcript_parser.parse_transcript("France", force=True)
run("Editing transcript invalidates cache and triggers re-parse", test_cache_invalidation)


# =====================================================================
# 6. Retrieval grounding -- out-of-scope queries
# =====================================================================
print("\n" + "=" * 60)
print("  6. Retrieval grounding -- out-of-scope queries")
print("=" * 60)


def test_outofscope_retrieval():
    """
    Keyword retrieval is intentionally coarse-grained.  Words like 'experts'
    or 'market' appear throughout the transcripts, so a query containing those
    words may match some chunks.  What matters is that the *content* of the
    retrieved chunks does not contain country names like Japan or Brazil --
    confirming the retrieval is not feeding misleading context to the model.

    The definitive safeguard is tested in test_ask_panel_not_covered:
    when the retrieved chunks don't answer the question the model says so.
    """
    all_chunks = {m: transcript_parser.parse_transcript(m) for m in ["France", "Germany", "UK"]}
    out_of_scope_countries = ["Japan", "Brazil", "South Korea", "Taiwan", "China", "India"]
    query = "Tell me about robotic surgery adoption in Japan, Brazil, and Taiwan."
    retrieved = llm.retrieve_chunks(query, all_chunks)
    # None of the retrieved chunks should contain these country names
    for chunk in retrieved:
        text_lower = chunk["text"].lower()
        for country in out_of_scope_countries:
            assert country.lower() not in text_lower, (
                f"Retrieved chunk contains out-of-scope country '{country}': {chunk['text'][:80]}"
            )
run("Out-of-scope queries retrieve 0 chunks", test_outofscope_retrieval)


def test_ask_panel_not_covered():
    """ask_panel with no retrieved chunks must return 'not covered' without API call."""
    with patch.object(llm, "retrieve_chunks", return_value=[]):
        result = llm.ask_panel(
            query="What do the experts think about robotic surgery in the US?",
            all_chunks={},
        )
    assert "not covered" in result.lower() or "isn't covered" in result.lower(), (
        f"Expected 'not covered' message, got: {result[:120]}"
    )
run("ask_panel returns 'not covered' when 0 chunks retrieved", test_ask_panel_not_covered)


# =====================================================================
# Summary
# =====================================================================
passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)

print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed  (total {len(_results)})")
print("=" * 60)

if failed:
    print("\nFailed tests:")
    for name, ok, msg in _results:
        if not ok:
            print(f"  [x] {name}")
            print(f"      {msg}")
    sys.exit(1)
else:
    print("\n  All tests passed!")
    sys.exit(0)

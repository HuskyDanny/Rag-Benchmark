# sentence_splitter Fast-Path Silently Skips Short Noisy Text

## The Trap
`split_into_atomic_facts` returns early (no LLM call) when `text.isascii() and len(text.split()) < 15`. Many `noisy_fact` test cases are short and ASCII (e.g., `"carol mngz team alpha."` = 4 words). These pass through to `add_episode` unnormalized, so `pipeline_presplit` produces the same result as `pipeline` on those cases — defeating the purpose of pre-splitting.

## The Solution
Remove or widen the ASCII/length fast-path in `sentence_splitter.py`. Either:
- Drop the `text.isascii() and len(text.split()) < 15` guard entirely (always call LLM for non-trivial text)
- Or check `is_likely_compound` OR the test_case has `"noisy"` tag (requires threading tag info through to the splitter)

The simplest fix: only skip the LLM when `not is_likely_compound(text) AND text looks clean` — i.e., remove the length/ASCII guard and rely solely on `is_likely_compound`.

## Context
- **When this applies:** Any change to `src/sentence_splitter.py` or benchmark analysis of `noisy_fact` results
- **Related files:** `src/sentence_splitter.py:split_into_atomic_facts`, `src/test_data/noisy.py`
- **Discovered:** 2026-04-03, architecture review of presplit preprocessor

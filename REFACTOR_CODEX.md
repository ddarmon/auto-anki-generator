# Codex Module Refactor Plan

This document captures the plan to split and simplify `auto_anki/codex.py` in a future session. No code changes are made yet.

## Current Pain Points
- `codex.py` is ~1.1k lines, mixing prompt building, orchestration, backend execution, and JSON parsing.
- `run_codex_pipeline` is ~200 lines, hard to test and reason about.
- Prompt text is embedded as long string literals, making iteration and A/B tests awkward.

## Target Shape
- `auto_anki/prompts.py`: prompt loading/formatting for stage 1 + stage 2 (no I/O beyond reading templates).
- `auto_anki/pipeline.py`: orchestration (chunking, stage 1 filter, stage 2 generation, state updates).
- `auto_anki/codex.py`: thin wrapper over backend execution + response parsing utilities.
- Template files under `auto_anki/prompts/` (Markdown/YAML) with safe substitution (e.g., `string.Template`).

## Migration Steps
1) **Introduce prompt loader**: add template files for stage1/stage2, loader that injects dynamic data (deck list, thresholds) via safe substitution; keep old code paths for one release behind a feature flag/env to ease rollback.
2) **Extract prompt builders**: move `build_conversation_prompt` and `build_conversation_filter_prompt` into `prompts.py`, re-export from `codex.py` temporarily to reduce churn.
3) **Extract orchestration**: move `run_codex_pipeline`, `chunked`, and artifact-writing/state-update logic into `pipeline.py`; adapt `auto_anki_agent.py` to call the new entry point.
4) **Slim codex**: leave only backend execution (`run_codex_exec`) and JSON parsing helpers in `codex.py`; delete deprecated per-turn prompt stubs if still unused.
5) **Wire tests**: add focused tests for prompt loading/substitution (including deck names containing braces), and a small orchestration test with fake backend responses to cover chunking and state updates.

## Guardrails / Risks
- Preserve existing CLI behaviors and artifact filenames during the move; add deprecation warnings if any public imports shift.
- Be careful with dynamic substitutions so deck names containing `{` are handled safely.
- Keep logging and state updates identical to avoid surprising unattended runs.


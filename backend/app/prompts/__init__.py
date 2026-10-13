"""Reviewer prompt templates (Phase 3 plan P3.2, L-6/C-1).

Each `.md` file here is a fixed, whitelisted instruction template — the only text
`InstructionSegment` may carry (see `services/prompt_builder.py`). Nothing in this
package is per-case; per-case content is injected separately as a
`StructuredEvidenceSegment`, never by formatting these files.
"""

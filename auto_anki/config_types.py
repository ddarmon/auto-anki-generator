"""Typed configuration objects for the LLM pipeline and dedup stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LLMPipelineConfig:
    """Structured configuration passed into LLM orchestration helpers."""

    contexts_per_run: int
    two_stage: bool
    dry_run: bool
    verbose: bool
    model_reasoning_effort: Optional[str]
    model_reasoning_effort_stage1: Optional[str]
    model_reasoning_effort_stage2: Optional[str]
    codex_model: Optional[str]
    codex_model_stage1: Optional[str]
    codex_model_stage2: Optional[str]
    llm_backend: str
    llm_model: Optional[str]
    llm_model_stage1: Optional[str]
    llm_model_stage2: Optional[str]
    codex_extra_arg: List[str]
    llm_extra_arg: List[str]
    dedup_method: str
    similarity_threshold: float
    semantic_model: str
    semantic_similarity_threshold: float
    cache_dir: Optional[Path]
    decks: List[str]

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "LLMPipelineConfig":
        """Create a typed config from argparse.Namespace."""
        return cls(
            contexts_per_run=args.contexts_per_run,
            two_stage=args.two_stage,
            dry_run=args.dry_run,
            verbose=args.verbose,
            model_reasoning_effort=getattr(args, "model_reasoning_effort", None),
            model_reasoning_effort_stage1=getattr(args, "model_reasoning_effort_stage1", None),
            model_reasoning_effort_stage2=getattr(args, "model_reasoning_effort_stage2", None),
            codex_model=getattr(args, "codex_model", None),
            codex_model_stage1=getattr(args, "codex_model_stage1", None),
            codex_model_stage2=getattr(args, "codex_model_stage2", None),
            llm_backend=getattr(args, "llm_backend", "codex"),
            llm_model=getattr(args, "llm_model", None),
            llm_model_stage1=getattr(args, "llm_model_stage1", None),
            llm_model_stage2=getattr(args, "llm_model_stage2", None),
            codex_extra_arg=list(getattr(args, "codex_extra_arg", []) or []),
            llm_extra_arg=list(getattr(args, "llm_extra_arg", []) or []),
            dedup_method=getattr(args, "dedup_method", "hybrid"),
            similarity_threshold=getattr(args, "similarity_threshold", 0.82),
            semantic_model=getattr(args, "semantic_model", "all-MiniLM-L6-v2"),
            semantic_similarity_threshold=getattr(args, "semantic_similarity_threshold", 0.85),
            cache_dir=getattr(args, "cache_dir", None),
            decks=list(getattr(args, "decks", []) or []),
        )


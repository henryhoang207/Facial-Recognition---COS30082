"""Shared verification for any model implementing Embedder.embed(face)."""

from .evaluator import PairEvaluator, VerificationConfig, score_metrics

__all__ = ["PairEvaluator", "VerificationConfig", "score_metrics"]

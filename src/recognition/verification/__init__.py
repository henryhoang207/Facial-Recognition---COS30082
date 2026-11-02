"""Embedding training and verification, including the shared Embedder pair API."""

from .config import VerificationConfig
from .evaluator import PairEvaluator, score_metrics

__all__ = ["PairEvaluator", "VerificationConfig", "score_metrics"]

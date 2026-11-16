"""Explicit-only scorer primitive; never enables a model or service path."""

from .config import DecisionHeadConfig
from .head import DecisionHead, DecisionResult

__all__ = ["DecisionHeadConfig", "DecisionHead", "DecisionResult"]

from .bundle import FeatureContract, export_bundle, load_bundle

__all__ += ["FeatureContract", "export_bundle", "load_bundle"]

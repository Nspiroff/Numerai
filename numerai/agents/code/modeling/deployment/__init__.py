"""Portable inference helpers for Numerai model-upload artifacts."""

from .tabm_numpy import build_tabm_numpy_forward, build_tabm_numpy_predictor

__all__ = ["build_tabm_numpy_forward", "build_tabm_numpy_predictor"]

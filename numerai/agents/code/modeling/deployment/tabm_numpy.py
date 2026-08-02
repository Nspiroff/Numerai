"""Self-contained NumPy inference for numerical-feature TabM models.

The builders in this module validate and copy a fitted TabM state into plain
``float32`` NumPy arrays.  They return nested callables so ``cloudpickle`` stores
the inference code and weights by value instead of recording an import from the
local ``agents`` package.

Only the numerical-feature ``arch_type='tabm'`` layout used by the Ender20
experiment is supported.  Dropout is intentionally absent because deployment
inference is the equivalent of ``model.eval()``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_BLOCK_KEYS = frozenset({"weight", "r", "s", "bias"})


def _as_finite_float32(name: str, value: Any, *, ndim: int) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float32 array.") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.shape}.")
    if not array.size:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    array.setflags(write=False)
    return array


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer.") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return integer


def _normalize_tabm_weights(
    blocks: Sequence[Mapping[str, Any]],
    output_weight: Any,
    output_bias: Any,
) -> tuple[
    tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...],
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    try:
        block_specs = tuple(blocks)
    except TypeError as exc:
        raise TypeError("blocks must be a sequence of weight mappings.") from exc
    if not block_specs:
        raise ValueError("At least one TabM block is required.")

    normalized = []
    input_width = None
    ensemble_size = None
    current_width = None
    for index, block in enumerate(block_specs):
        if not isinstance(block, Mapping):
            raise TypeError(f"blocks[{index}] must be a mapping.")
        keys = frozenset(block)
        missing = sorted(_BLOCK_KEYS - keys)
        unexpected = sorted(keys - _BLOCK_KEYS)
        if missing or unexpected:
            raise ValueError(
                f"blocks[{index}] must contain exactly {sorted(_BLOCK_KEYS)}; "
                f"missing={missing}, unexpected={unexpected}."
            )

        weight = _as_finite_float32(
            f"blocks[{index}]['weight']", block["weight"], ndim=2
        )
        r = _as_finite_float32(f"blocks[{index}]['r']", block["r"], ndim=2)
        s = _as_finite_float32(f"blocks[{index}]['s']", block["s"], ndim=2)
        bias = _as_finite_float32(
            f"blocks[{index}]['bias']", block["bias"], ndim=2
        )

        output_width, block_input_width = weight.shape
        if index == 0:
            input_width = block_input_width
            ensemble_size = r.shape[0]
            if ensemble_size <= 0:
                raise ValueError("TabM ensemble size must be positive.")
        elif block_input_width != current_width:
            raise ValueError(
                f"blocks[{index}] expects {block_input_width} inputs, but the "
                f"previous block emits {current_width}."
            )

        expected_r = (ensemble_size, block_input_width)
        expected_output_adapter = (ensemble_size, output_width)
        if r.shape != expected_r:
            raise ValueError(
                f"blocks[{index}]['r'] must have shape {expected_r}; got {r.shape}."
            )
        for name, array in (("s", s), ("bias", bias)):
            if array.shape != expected_output_adapter:
                raise ValueError(
                    f"blocks[{index}]['{name}'] must have shape "
                    f"{expected_output_adapter}; got {array.shape}."
                )
        normalized.append((weight, r, s, bias))
        current_width = output_width

    assert input_width is not None
    assert ensemble_size is not None
    assert current_width is not None
    normalized_output_weight = _as_finite_float32(
        "output_weight", output_weight, ndim=3
    )
    normalized_output_bias = _as_finite_float32(
        "output_bias", output_bias, ndim=2
    )
    expected_output_weight = (ensemble_size, current_width, 1)
    expected_output_bias = (ensemble_size, 1)
    if normalized_output_weight.shape != expected_output_weight:
        raise ValueError(
            "output_weight must have shape "
            f"{expected_output_weight}; got {normalized_output_weight.shape}."
        )
    if normalized_output_bias.shape != expected_output_bias:
        raise ValueError(
            f"output_bias must have shape {expected_output_bias}; "
            f"got {normalized_output_bias.shape}."
        )
    return (
        tuple(normalized),
        normalized_output_weight,
        normalized_output_bias,
        input_width,
        ensemble_size,
    )


def build_tabm_numpy_forward(
    *,
    blocks: Sequence[Mapping[str, Any]],
    output_weight: Any,
    output_bias: Any,
    feature_center: float = 2.0,
    feature_scale: float = 2.0,
    batch_size: int = 32,
    activation: str = "relu",
):
    """Build a cloudpickle-portable raw TabM forward callable.

    Each block mapping corresponds to the four tensors from a fitted TabM state:
    ``backbone.blocks.<i>.0.{weight,r,s,bias}``. ``output_weight`` and
    ``output_bias`` correspond to ``output.{weight,bias}``.
    """

    if str(activation).lower() != "relu":
        raise ValueError("Only the deployed TabM ReLU activation is supported.")
    normalized_batch_size = _positive_int("batch_size", batch_size)
    try:
        center = np.float32(feature_center)
        scale = np.float32(feature_scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("feature_center and feature_scale must be finite numbers.") from exc
    if not np.isfinite(center):
        raise ValueError("feature_center must be finite.")
    if not np.isfinite(scale) or scale == np.float32(0.0):
        raise ValueError("feature_scale must be finite and non-zero.")

    (
        normalized_blocks,
        normalized_output_weight,
        normalized_output_bias,
        input_width,
        ensemble_size,
    ) = _normalize_tabm_weights(blocks, output_weight, output_bias)

    def forward(features):
        import numpy as _np

        try:
            values = _np.asarray(features, dtype=_np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError("TabM features must be numeric.") from exc
        if values.ndim != 2:
            raise ValueError(
                f"TabM features must be two-dimensional; got shape {values.shape}."
            )
        if values.shape[1] != input_width:
            raise ValueError(
                f"TabM expects {input_width} features; got {values.shape[1]}."
            )

        predictions = _np.empty(values.shape[0], dtype=_np.float32)
        with _np.errstate(over="ignore", invalid="ignore"):
            for start in range(0, values.shape[0], normalized_batch_size):
                stop = min(start + normalized_batch_size, values.shape[0])
                hidden = _np.array(
                    values[start:stop], dtype=_np.float32, order="C", copy=True
                )
                _np.subtract(hidden, center, out=hidden)
                _np.divide(hidden, scale, out=hidden)
                _np.nan_to_num(
                    hidden, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                )
                hidden = _np.broadcast_to(
                    hidden[:, _np.newaxis, :],
                    (hidden.shape[0], ensemble_size, input_width),
                )

                for weight, r, s, bias in normalized_blocks:
                    hidden = _np.matmul(
                        hidden * r[_np.newaxis, :, :], weight.T
                    )
                    hidden *= s[_np.newaxis, :, :]
                    hidden += bias[_np.newaxis, :, :]
                    _np.maximum(hidden, _np.float32(0.0), out=hidden)

                ensemble_predictions = _np.einsum(
                    "bkd,kdo->bko",
                    hidden,
                    normalized_output_weight,
                    optimize=False,
                )
                ensemble_predictions += normalized_output_bias[_np.newaxis, :, :]
                chunk = ensemble_predictions[:, :, 0].mean(
                    axis=1, dtype=_np.float32
                )
                if not _np.isfinite(chunk).all():
                    raise FloatingPointError(
                        "TabM produced non-finite raw predictions."
                    )
                predictions[start:stop] = chunk
        return predictions

    forward.feature_count = input_width
    forward.ensemble_size = ensemble_size
    forward.block_count = len(normalized_blocks)
    forward.batch_size = normalized_batch_size
    return forward


def build_tabm_numpy_predictor(
    *,
    feature_names: Sequence[str],
    blocks: Sequence[Mapping[str, Any]],
    output_weight: Any,
    output_bias: Any,
    feature_center: float = 2.0,
    feature_scale: float = 2.0,
    batch_size: int = 32,
    activation: str = "relu",
    era_column: str = "era",
    prediction_column: str = "prediction",
):
    """Build a self-contained Numerai ``predict(features, benchmarks)`` callable."""

    if isinstance(feature_names, (str, bytes)):
        raise TypeError("feature_names must be a sequence of column names.")
    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names must not be empty.")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("Every feature name must be a non-empty string.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    if not isinstance(era_column, str) or not era_column:
        raise TypeError("era_column must be a non-empty string.")
    if not isinstance(prediction_column, str) or not prediction_column:
        raise TypeError("prediction_column must be a non-empty string.")
    if era_column in names:
        raise ValueError("era_column must not also be a model feature.")

    raw_forward = build_tabm_numpy_forward(
        blocks=blocks,
        output_weight=output_weight,
        output_bias=output_bias,
        feature_center=feature_center,
        feature_scale=feature_scale,
        batch_size=batch_size,
        activation=activation,
    )
    if raw_forward.feature_count != len(names):
        raise ValueError(
            f"feature_names contains {len(names)} names, but the first TabM block "
            f"expects {raw_forward.feature_count} features."
        )

    def predict(live_features, live_benchmark_models):
        import numpy as _np
        import pandas as _pd

        if not isinstance(live_features, _pd.DataFrame):
            raise TypeError("live_features must be a pandas DataFrame.")
        if live_benchmark_models is not None and not isinstance(
            live_benchmark_models, _pd.DataFrame
        ):
            raise TypeError(
                "live_benchmark_models must be a pandas DataFrame or None."
            )
        if live_features.empty:
            raise ValueError("live_features must contain at least one row.")
        if not live_features.columns.is_unique:
            raise ValueError("live_features column names must be unique.")
        if live_features.index.nlevels != 1:
            raise ValueError("live_features must use a one-dimensional index.")
        if not live_features.index.is_unique:
            raise ValueError("live_features index must be unique.")
        if live_features.index.hasnans:
            raise ValueError("live_features index must not contain missing values.")
        if era_column not in live_features.columns:
            raise ValueError(
                f"live_features is missing required era column {era_column!r}."
            )
        missing = [name for name in names if name not in live_features.columns]
        if missing:
            preview = missing[:5]
            suffix = "..." if len(missing) > len(preview) else ""
            raise ValueError(
                f"live_features is missing required model features: {preview}{suffix}"
            )

        eras = live_features[era_column]
        if eras.isna().any():
            raise ValueError(f"{era_column!r} must not contain missing values.")
        try:
            feature_values = live_features.loc[:, list(names)].to_numpy(
                dtype=_np.float32, copy=True
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("All required model features must be numeric.") from exc

        raw_predictions = raw_forward(feature_values)
        if raw_predictions.shape != (len(live_features),):
            raise RuntimeError(
                "TabM raw prediction count does not match the live feature rows."
            )
        if not _np.isfinite(raw_predictions).all():
            raise FloatingPointError("TabM raw predictions contain non-finite values.")

        raw = _pd.Series(
            raw_predictions, index=live_features.index, name=prediction_column
        )
        ranked = raw.groupby(eras, sort=False, dropna=False).rank(
            method="average", pct=True
        )
        output = ranked.to_frame(name=prediction_column)
        if not output.index.equals(live_features.index):
            raise RuntimeError("Prediction output index does not match live_features.")
        values = output[prediction_column].to_numpy(copy=False)
        if not _np.isfinite(values).all():
            raise FloatingPointError("Ranked predictions contain non-finite values.")
        if ((values < 0.0) | (values > 1.0)).any():
            raise RuntimeError("Ranked predictions are outside the [0, 1] interval.")
        return output

    predict.__name__ = "predict"
    predict.__qualname__ = "predict"
    predict.feature_names = names
    predict.era_column = era_column
    predict.prediction_column = prediction_column
    predict.uses_live_benchmark_models = False
    predict.tabm_metadata = {
        "feature_count": raw_forward.feature_count,
        "ensemble_size": raw_forward.ensemble_size,
        "block_count": raw_forward.block_count,
        "batch_size": raw_forward.batch_size,
        "activation": "relu",
        "feature_center": float(np.float32(feature_center)),
        "feature_scale": float(np.float32(feature_scale)),
    }
    return predict

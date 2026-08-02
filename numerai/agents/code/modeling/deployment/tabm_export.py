"""Bridge a fitted Torch TabM regressor to portable NumPy inference.

This module is intentionally kept out of the callable returned for Numerai
deployment.  It may import the training wrappers and torch-backed model state;
the resulting predictor from :mod:`tabm_numpy` contains only copied NumPy
weights and its frozen inference code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from agents.code.modeling.deployment.tabm_numpy import (
    build_tabm_numpy_forward,
    build_tabm_numpy_predictor,
)
from agents.code.modeling.models.torch_tabular_regressor import (
    TorchTabularRegressor,
)
from agents.code.modeling.utils.target_transforms import TargetTransformWrapper


_BLOCK_PARAMETER_NAMES = ("weight", "r", "s", "bias")
_OUTPUT_PARAMETER_NAMES = ("output.weight", "output.bias")


def _unwrap_regressor(model: Any) -> TorchTabularRegressor:
    if isinstance(model, TargetTransformWrapper):
        regressor = model.__dict__.get("_model")
        if not isinstance(regressor, TorchTabularRegressor):
            raise TypeError(
                "TargetTransformWrapper must contain a TorchTabularRegressor."
            )
        return regressor
    if isinstance(model, TorchTabularRegressor):
        return model
    raise TypeError(
        "model must be a TorchTabularRegressor or a TargetTransformWrapper "
        "containing one."
    )


def _positive_config_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer.") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return integer


def _identity_if_none(name: str, value: Any, identity: float) -> float:
    if value is None:
        return identity
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite number or None.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if name == "feature_scale" and number == 0.0:
        raise ValueError("feature_scale must be non-zero.")
    return number


def _frozen_feature_names(regressor: TorchTabularRegressor) -> tuple[str, ...]:
    names = regressor.__dict__.get("_input_cols")
    if names is None:
        raise RuntimeError(
            "TorchTabularRegressor has no frozen fitted feature order; fit the "
            "model before export."
        )
    if isinstance(names, (str, bytes)):
        raise TypeError("Fitted feature order must be a sequence of column names.")
    try:
        frozen = tuple(names)
    except TypeError as exc:
        raise TypeError(
            "Fitted feature order must be a sequence of column names."
        ) from exc
    if not frozen:
        raise ValueError("Fitted feature order must not be empty.")
    if any(not isinstance(name, str) or not name for name in frozen):
        raise TypeError("Every fitted feature name must be a non-empty string.")
    if len(set(frozen)) != len(frozen):
        raise ValueError("Fitted feature names must be unique.")
    return frozen


def _state_array(name: str, value: Any) -> np.ndarray:
    if not hasattr(value, "detach"):
        raise TypeError(f"TabM state entry {name!r} is not a torch tensor.")
    try:
        array = value.detach().cpu().numpy()
        return np.array(array, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(
            f"TabM state entry {name!r} cannot be copied to a float32 NumPy array."
        ) from exc


def _extract_state(
    regressor: TorchTabularRegressor,
    *,
    block_count: int,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    fitted_module = regressor.__dict__.get("_model")
    if fitted_module is None:
        raise RuntimeError("TorchTabularRegressor must be fitted before export.")
    state_dict = getattr(fitted_module, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("Fitted TorchTabularRegressor model has no state_dict().")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("Fitted TabM state_dict() must return a mapping.")
    if any(not isinstance(key, str) for key in state):
        raise TypeError("Every fitted TabM state key must be a string.")

    block_keys = {
        f"backbone.blocks.{index}.0.{parameter}"
        for index in range(block_count)
        for parameter in _BLOCK_PARAMETER_NAMES
    }
    expected_keys = block_keys | set(_OUTPUT_PARAMETER_NAMES)
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Fitted TabM state layout does not match the supported numerical "
            f"TabM architecture; missing={missing}, unexpected={unexpected}."
        )

    blocks = []
    for index in range(block_count):
        prefix = f"backbone.blocks.{index}.0"
        blocks.append(
            {
                parameter: _state_array(
                    f"{prefix}.{parameter}", state[f"{prefix}.{parameter}"]
                )
                for parameter in _BLOCK_PARAMETER_NAMES
            }
        )
    return (
        blocks,
        _state_array("output.weight", state["output.weight"]),
        _state_array("output.bias", state["output.bias"]),
    )


def extract_tabm_numpy_predictor_spec(
    model: Any,
    *,
    batch_size: int = 32,
    era_column: str | None = None,
    prediction_column: str = "prediction",
) -> dict[str, Any]:
    """Extract validated NumPy predictor arguments from a fitted TabM wrapper.

    The returned dictionary can be passed directly to
    :func:`build_tabm_numpy_predictor`.  All tensors are detached, copied to the
    CPU, converted to contiguous ``float32`` arrays, and then shape-validated by
    the public NumPy forward builder.
    """

    regressor = _unwrap_regressor(model)
    if str(regressor.architecture).lower() != "tabm":
        raise ValueError("Only architecture='tabm' can be exported by this bridge.")
    if str(regressor.tabm_arch_type).lower() != "tabm":
        raise ValueError("Only tabm_arch_type='tabm' is supported for NumPy export.")
    if str(regressor.activation).lower() != "relu":
        raise ValueError("Only activation='relu' is supported for NumPy export.")

    feature_names = _frozen_feature_names(regressor)
    configured_blocks = _positive_config_int("tabm_blocks", regressor.tabm_blocks)
    configured_k = _positive_config_int("tabm_k", regressor.tabm_k)
    configured_width = _positive_config_int("tabm_width", regressor.tabm_width)
    center = _identity_if_none("feature_center", regressor.feature_center, 0.0)
    scale = _identity_if_none("feature_scale", regressor.feature_scale, 1.0)
    blocks, output_weight, output_bias = _extract_state(
        regressor, block_count=configured_blocks
    )

    forward = build_tabm_numpy_forward(
        blocks=blocks,
        output_weight=output_weight,
        output_bias=output_bias,
        feature_center=center,
        feature_scale=scale,
        batch_size=batch_size,
        activation="relu",
    )
    if forward.feature_count != len(feature_names):
        raise ValueError(
            "Fitted feature order has "
            f"{len(feature_names)} names, but TabM expects {forward.feature_count}."
        )
    if forward.ensemble_size != configured_k:
        raise ValueError(
            f"Fitted TabM has k={forward.ensemble_size}, but its configuration "
            f"declares tabm_k={configured_k}."
        )
    if forward.block_count != configured_blocks:
        raise ValueError(
            f"Fitted TabM has {forward.block_count} blocks, but its configuration "
            f"declares tabm_blocks={configured_blocks}."
        )
    emitted_widths = tuple(block["weight"].shape[0] for block in blocks)
    if any(width != configured_width for width in emitted_widths):
        raise ValueError(
            f"Fitted TabM block widths {emitted_widths} do not match configured "
            f"tabm_width={configured_width}."
        )

    resolved_era_column = regressor.era_col if era_column is None else era_column
    spec = {
        "feature_names": feature_names,
        "blocks": blocks,
        "output_weight": output_weight,
        "output_bias": output_bias,
        "feature_center": center,
        "feature_scale": scale,
        "batch_size": batch_size,
        "activation": "relu",
        "era_column": resolved_era_column,
        "prediction_column": prediction_column,
    }
    # Validate names and output columns now, rather than deferring an invalid
    # export until the artifact is built or invoked.
    build_tabm_numpy_predictor(**spec)
    return spec


def build_tabm_numpy_predictor_from_fitted(
    model: Any,
    *,
    batch_size: int = 32,
    era_column: str | None = None,
    prediction_column: str = "prediction",
):
    """Build the portable Numerai predictor from a fitted training wrapper."""

    spec = extract_tabm_numpy_predictor_spec(
        model,
        batch_size=batch_size,
        era_column=era_column,
        prediction_column=prediction_column,
    )
    return build_tabm_numpy_predictor(**spec)


__all__ = [
    "build_tabm_numpy_predictor_from_fitted",
    "extract_tabm_numpy_predictor_spec",
]

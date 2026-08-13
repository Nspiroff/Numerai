"""Era-based CV helpers and OOF prediction builder."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from agents.code.modeling.utils.model_factory import build_model
from agents.code.modeling.utils.model_data import ModelDataBatch


def _era_sort_key(era):
    try:
        return int(era)
    except (TypeError, ValueError):
        return str(era)


def _sorted_unique_eras(eras: Iterable) -> List:
    return sorted(set(eras), key=_era_sort_key)


def era_cv_splits(
    eras: Sequence,
    n_splits: int = 5,
    embargo: int = 13,
    mode: str = "expanding",
    min_train_size: int = 1,
) -> List[Tuple[List, List]]:
    """Create sequential era splits with optional embargo.

    mode="expanding": train uses eras before validation; embargo removes eras
    immediately before the validation window.

    mode="blocked": train uses all eras except the validation window plus an
    embargo buffer before and after.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    eras_sorted = _sorted_unique_eras(eras)
    if n_splits > len(eras_sorted):
        raise ValueError("n_splits must be <= number of eras")

    fold_size = len(eras_sorted) // n_splits
    remainder = len(eras_sorted) % n_splits
    splits: List[Tuple[List, List]] = []

    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size
        if i == n_splits - 1:
            end += remainder
        val_eras = eras_sorted[start:end]
        if not val_eras:
            continue

        if mode == "expanding":
            train_end = max(0, start - embargo)
            train_eras = eras_sorted[:train_end]
        elif mode == "blocked":
            left = max(0, start - embargo)
            right = min(len(eras_sorted), end + embargo)
            train_eras = eras_sorted[:left] + eras_sorted[right:]
        else:
            raise ValueError("mode must be 'expanding' or 'blocked'")

        if len(train_eras) < min_train_size:
            raise ValueError(
                "train split too small; reduce n_splits/embargo or set min_train_size"
            )

        splits.append((train_eras, val_eras))

    return splits


def build_oof_predictions(
    eras: Sequence,
    data_loader,
    model_type: str,
    model_params: dict,
    model_config: dict,
    cv_config: dict,
    max_train_samples: int | None,
    sample_seed: int,
    id_col: str | None,
    era_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    cv_n_splits = int(cv_config.get("n_splits", 5))
    cv_embargo = int(cv_config.get("embargo", 13))
    cv_mode = cv_config.get("mode", "expanding")
    cv_min_train_size = int(cv_config.get("min_train_size", 0))
    raw_max_train_eras = cv_config.get("max_train_eras")
    if raw_max_train_eras is None:
        max_train_eras = None
    elif (
        isinstance(raw_max_train_eras, (bool, np.bool_))
        or not isinstance(raw_max_train_eras, (int, np.integer))
        or int(raw_max_train_eras) <= 0
    ):
        raise ValueError("cv.max_train_eras must be a positive integer or null")
    else:
        max_train_eras = int(raw_max_train_eras)

    splits = era_cv_splits(
        eras,
        n_splits=cv_n_splits,
        embargo=cv_embargo,
        mode=cv_mode,
        min_train_size=cv_min_train_size,
    )

    predictions = []
    fold_info = []

    for fold_idx, (train_eras, val_eras) in enumerate(splits):
        if not train_eras or not val_eras:
            continue
        available_train_eras = list(train_eras)
        if max_train_eras is not None:
            train_eras = train_eras[-max_train_eras:]
        train_data = _load_data(data_loader, train_eras)
        val_data = _load_data(data_loader, val_eras)

        train_rows = _data_length(train_data)
        val_rows = _data_length(val_data)
        if train_rows == 0 or val_rows == 0:
            continue

        if max_train_samples and train_rows > max_train_samples:
            train_data = _subset_data(train_data, max_train_samples, sample_seed)
            train_rows = max_train_samples
            print(
                f"Downsampled training fold {fold_idx} to {max_train_samples} rows for {model_type}."
            )

        model = build_model(
            model_type,
            model_params,
            model_config,
            feature_cols=feature_cols,
            disk_materialization_max_rows=(
                max_train_samples
                if getattr(train_data.X, "is_disk_feature_view", False)
                else None
            ),
        )
        model.fit(train_data.X, train_data.y)
        preds = model.predict(val_data.X)

        model_diagnostics = {}
        best_epoch = getattr(model, "best_epoch_", None)
        if best_epoch is not None:
            model_diagnostics["best_epoch"] = int(best_epoch)
        n_parameters = getattr(model, "n_parameters_", None)
        if n_parameters is not None:
            model_diagnostics["n_parameters"] = int(n_parameters)
        training_history = getattr(model, "training_history_", None)
        if training_history:
            model_diagnostics["epochs_ran"] = len(training_history)
            model_diagnostics["best_val_loss"] = float(
                min(row["val_loss"] for row in training_history)
            )
            model_diagnostics["final_train_loss"] = float(
                training_history[-1]["train_loss"]
            )
        ema_decay = getattr(model, "ema_decay", None)
        if ema_decay is not None:
            model_diagnostics["ema_decay"] = float(ema_decay)
            model_diagnostics["ema_updates"] = int(model.ema_updates_)
            for attribute, key in (
                ("ema_live_state_sha256_", "ema_live_state_sha256"),
                ("ema_shadow_state_sha256_", "ema_shadow_state_sha256"),
                ("ema_inference_state_sha256_", "ema_inference_state_sha256"),
            ):
                value = getattr(model, attribute, None)
                if value is not None:
                    model_diagnostics[key] = str(value)
        effective_device_type = getattr(model, "effective_device_type_", None)
        if effective_device_type is not None:
            model_diagnostics["effective_device_type"] = str(
                effective_device_type
            )
        gpu_fallback_used = getattr(model, "gpu_fallback_used_", None)
        if gpu_fallback_used is not None:
            model_diagnostics["gpu_fallback_used"] = bool(gpu_fallback_used)
        if getattr(model, "data_mode_", None) == "disk_feature_store":
            model_diagnostics["data_mode"] = "disk_feature_store"
            for attribute, key in (
                ("disk_train_rows_", "disk_train_rows"),
                ("disk_validation_rows_", "disk_validation_rows"),
                (
                    "disk_prediction_batch_size_",
                    "disk_prediction_batch_size",
                ),
                ("disk_prediction_batches_", "disk_prediction_batches"),
            ):
                value = getattr(model, attribute, None)
                if value is not None:
                    model_diagnostics[key] = int(value)
            batches_per_epoch = getattr(model, "disk_batches_per_epoch_", None)
            rows_per_epoch = getattr(model, "disk_rows_per_epoch_", None)
            if batches_per_epoch is not None:
                model_diagnostics["disk_batches_per_epoch"] = [
                    int(value) for value in batches_per_epoch
                ]
            if rows_per_epoch is not None:
                model_diagnostics["disk_rows_per_epoch"] = [
                    int(value) for value in rows_per_epoch
                ]

        fold_predictions = {}
        if id_col and val_data.id is not None:
            fold_predictions[id_col] = _as_array(val_data.id)
        fold_predictions[era_col] = _as_array(val_data.era)
        fold_predictions[target_col] = _as_array(val_data.y)
        fold_predictions["prediction"] = np.asarray(preds).ravel()
        fold_predictions["cv_fold"] = fold_idx
        predictions.append(pd.DataFrame(fold_predictions))
        fold_receipt = {
            "fold": fold_idx,
            "train_eras": len(train_eras),
            "val_eras": len(val_eras),
            "train_rows": int(train_rows),
            "val_rows": int(val_rows),
            "model_diagnostics": model_diagnostics or None,
        }
        if max_train_eras is not None:
            fold_receipt.update(
                {
                    "available_train_eras": len(available_train_eras),
                    "max_train_eras": max_train_eras,
                    "first_train_era": train_eras[0],
                    "last_train_era": train_eras[-1],
                }
            )
        fold_info.append(fold_receipt)

    if not predictions:
        raise ValueError("No CV folds produced predictions; check CV settings.")

    oof = pd.concat(predictions, ignore_index=True)
    if id_col and id_col in oof.columns and oof[id_col].duplicated().any():
        raise ValueError("OOF predictions contain duplicate ids.")

    cv_meta = {
        "n_splits": cv_n_splits,
        "embargo": cv_embargo,
        "mode": cv_mode,
        "min_train_size": cv_min_train_size,
        "folds_used": len(fold_info),
        "folds": fold_info,
    }
    if max_train_eras is not None:
        cv_meta["max_train_eras"] = max_train_eras
    return oof, cv_meta


def _load_data(data_loader, eras: Sequence) -> ModelDataBatch:
    if hasattr(data_loader, "load"):
        return data_loader.load(eras)
    return data_loader(eras)


def _data_length(data: ModelDataBatch) -> int:
    return len(data.X)


def _subset_data(data: ModelDataBatch, max_samples: int, seed: int) -> ModelDataBatch:
    total = _data_length(data)
    if total <= max_samples:
        return data
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=max_samples, replace=False)
    return ModelDataBatch(
        X=_subset_value(data.X, indices),
        y=_subset_value(data.y, indices),
        era=_subset_value(data.era, indices),
        id=_subset_value(data.id, indices) if data.id is not None else None,
    )


def _subset_value(value, indices):
    if value is None:
        return None
    if getattr(value, "is_disk_feature_view", False):
        return value.take(indices)
    if hasattr(value, "iloc"):
        return value.iloc[indices]
    return value[indices]


def _as_array(values):
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)

"""Frozen shared config for Ender27 tempered Gaussian-rank residual research."""

from copy import deepcopy


CONFIG = {
    "data": {
        "data_version": "v5.3",
        "embargo_eras": 13,
        "era_col": "era",
        "feature_set": "all",
        "feature_columns_path": (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/feature_columns_all_v53.json"
        ),
        "target_col": "target_ender_20",
        "id_col": "id",
        "full_data_path": "v5.3/ender21_discovery_full_through_0861.parquet",
        "benchmark_data_path": (
            "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
        ),
        "benchmark_model": "v53_lgbm_ender20",
        "require_benchmark_coverage": True,
        "era_allowlist_path": (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        ),
    },
    "model": {
        "type": "TorchTabularRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "target_transform": {
            "type": "residual_to_benchmark",
            "benchmark_col": "v53_lgbm_ender20",
            "era_col": "era",
            "per_era": True,
            "fit_intercept": True,
            "proportion": 1.0,
        },
        "params": {
            "architecture": "tabm",
            "activation": "relu",
            "tabm_arch_type": "tabm",
            "tabm_k": 64,
            "tabm_width": 512,
            "tabm_blocks": 3,
            "dropout": 0.1,
            "batch_size": 1024,
            "prediction_batch_size": 2048,
            "learning_rate": 0.002,
            "weight_decay": 0.0003,
            "max_epochs": 30,
            "patience": 4,
            "val_fraction": 0.1,
            "val_split": "recent_eras",
            "internal_val_embargo": 13,
            "feature_center": 2.0,
            "feature_scale": 2.0,
            "device": "cuda",
            "amp": True,
            "seed": 1337,
            "loss_mode": "chronological_block_dro",
            "chronological_blocks": 4,
            "dro_temperature": 2.0,
        },
    },
    "output": {
        "output_dir": "experiments/ender27_tempered_gaussian_rank_residual_v53",
        "results_name": "base_r1",
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "training": {
        "max_train_samples": 500_000,
        "sample_seed": 1337,
        "cv": {
            "embargo": 13,
            "enabled": True,
            "max_train_eras": 78,
            "min_train_size": 0,
            "mode": "expanding",
            "n_splits": 5,
        },
    },
}


def variant(
    name: str,
    *,
    model_seed: int,
    benchmark_transform: str | None = None,
    benchmark_transform_strength: float | None = None,
):
    """Return one exact matched raw- or fixed tempered-residual config."""
    if benchmark_transform is None:
        if benchmark_transform_strength is not None:
            raise ValueError(
                "benchmark_transform_strength requires benchmark_transform."
            )
    elif (
        benchmark_transform != "tie_kept_rank_gaussian"
        or benchmark_transform_strength != 0.5
    ):
        raise ValueError(
            "Ender27 challenger is frozen to tie_kept_rank_gaussian at "
            "strength 0.5."
        )

    config = deepcopy(CONFIG)
    config["output"]["results_name"] = name
    config["model"]["params"]["seed"] = model_seed
    if benchmark_transform is not None:
        target_transform = config["model"]["target_transform"]
        target_transform["benchmark_transform"] = benchmark_transform
        target_transform["benchmark_transform_strength"] = (
            benchmark_transform_strength
        )
    return config

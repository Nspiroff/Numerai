"""Frozen component template for the Ender20 auxiliary-target ensemble."""

from copy import deepcopy


BASE_CONFIG = {
    "data": {
        "data_version": "v5.3",
        "feature_set": "medium",
        "target_col": None,
        "era_col": "era",
        "id_col": "id",
        "benchmark_model": "v53_lgbm_ender20",
        "full_data_path": "v5.3/downsampled_full.parquet",
        "benchmark_data_path": "v5.3/downsampled_full_benchmark_models.parquet",
        "require_benchmark_coverage": True,
        "embargo_eras": 13,
    },
    "model": {
        "type": "LGBMRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "n_estimators": 6000,
            "learning_rate": 0.003,
            "max_depth": 8,
            "num_leaves": 255,
            "colsample_bytree": 0.1,
            "min_data_in_leaf": 10000,
            "device_type": "gpu",
            "n_jobs": 12,
            "random_state": 1337,
            "verbosity": -1,
        },
    },
    "training": {
        "max_train_samples": 500000,
        "sample_seed": 1337,
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
    "preprocessing": {
        "nan_missing_all_twos": False,
        "missing_value": 2.0,
    },
    "output": {
        "output_dir": "experiments/ender20_aux_target_rank_ensemble_v53",
        "results_name": None,
    },
}


def make_config(target_col: str, results_name: str) -> dict:
    config = deepcopy(BASE_CONFIG)
    config["data"]["target_col"] = target_col
    config["output"]["results_name"] = results_name
    return config

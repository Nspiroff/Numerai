CONFIG = {
    "data": {
        "data_version": "v5.3",
        "feature_set": "medium",
        "target_col": "target_xerxes_20",
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
            "max_depth": 6,
            "num_leaves": 63,
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
        "output_dir": "experiments/xerxes20_lgbm_challenger_v53",
        "results_name": "r1_base_d6_t6000",
    },
}

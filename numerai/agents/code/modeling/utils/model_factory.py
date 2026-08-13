from __future__ import annotations

def build_model(
    model_type: str,
    model_params: dict,
    model_config: dict | None = None,
    *,
    feature_cols: list[str] | None = None,
    disk_materialization_max_rows: int | None = None,
):
    model_config = model_config or {}
    if model_type == "LGBMRegressor":
        from agents.code.modeling.models.lgbm_regressor import LGBMRegressor
        model = LGBMRegressor(
            feature_cols=feature_cols,
            disk_materialization_max_rows=disk_materialization_max_rows,
            prediction_batch_size=model_config.get(
                "prediction_batch_size", 65_536
            ),
            **model_params,
        )
    elif model_type == "TorchTabularRegressor":
        from agents.code.modeling.models.torch_tabular_regressor import (
            TorchTabularRegressor,
        )

        model = TorchTabularRegressor(feature_cols=feature_cols, **model_params)
    else:
        raise ValueError(
            "Unsupported model type: "
            f"{model_type}. Supported types: LGBMRegressor, "
            "TorchTabularRegressor"
        )

    target_transform = model_config.get("target_transform")
    if target_transform:
        from agents.code.modeling.utils.target_transforms import TargetTransformWrapper

        model = TargetTransformWrapper(model, target_transform)
    return model

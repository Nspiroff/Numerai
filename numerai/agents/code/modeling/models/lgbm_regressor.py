from __future__ import annotations

import numpy as np


_DEFAULT_PREDICTION_BATCH_SIZE = 65_536


class LGBMRegressor:
    """Minimal wrapper exposing fit/predict for the training pipeline."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        disk_materialization_max_rows: int | None = None,
        prediction_batch_size: int = _DEFAULT_PREDICTION_BATCH_SIZE,
        **params,
    ):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LGBMRegressor. Install with `.venv/bin/pip install lightgbm`."
            ) from exc
        self._lgb = lgb
        self._params = dict(params)
        self._model = lgb.LGBMRegressor(**params)
        self._feature_cols = feature_cols
        self._disk_materialization_max_rows = self._validate_disk_row_cap(
            disk_materialization_max_rows
        )
        self.prediction_batch_size = self._validate_prediction_batch_size(
            prediction_batch_size
        )
        self.data_mode_: str | None = None
        self.disk_train_rows_: int | None = None
        self.disk_validation_rows_: int | None = None
        self.disk_prediction_batch_size_: int | None = None
        self.disk_prediction_batches_: int | None = None
        self.disk_batches_per_epoch_: list[int] = []
        self.disk_rows_per_epoch_: list[int] = []
        self.effective_device_type_: str | None = None
        self.gpu_fallback_used_: bool | None = None

    def fit(self, X, y, **kwargs):
        X = self._filter_features(X, self._feature_cols)
        self._reset_fit_diagnostics()
        if self._is_disk_feature_view(X):
            X, batch_count = self._materialize_disk_features(X)
            self.data_mode_ = "disk_feature_store"
            self.disk_train_rows_ = len(X)
            self.disk_batches_per_epoch_ = [batch_count]
            self.disk_rows_per_epoch_ = [len(X)]
        else:
            self.data_mode_ = "eager"
        try:
            self._model.fit(X, y, **kwargs)
        except self._lgb.basic.LightGBMError as exc:
            if self._should_fallback_to_cpu(exc):
                print("LightGBM GPU not available; retrying with device_type='cpu'.")
                self.gpu_fallback_used_ = True
                self._params["device_type"] = "cpu"
                self._model = self._lgb.LGBMRegressor(**self._params)
                self._model.fit(X, y, **kwargs)
                self.effective_device_type_ = "cpu"
            else:
                raise
        else:
            self.effective_device_type_ = self._configured_device_type()
            self.gpu_fallback_used_ = False
        return self

    def predict(self, X):
        X = self._filter_features(X, self._feature_cols)
        if self._is_disk_feature_view(X):
            return self._predict_disk_features(X)
        return self._model.predict(X)

    def _materialize_disk_features(self, X) -> tuple[np.ndarray, int]:
        row_count = len(X)
        if row_count == 0:
            raise ValueError("Disk LightGBM training requires at least one row.")
        cap = self._disk_materialization_max_rows
        if cap is None:
            raise ValueError(
                "Disk LightGBM training requires a positive "
                "training.max_train_samples materialization cap."
            )
        if row_count > cap:
            raise ValueError(
                f"Disk LightGBM training view has {row_count:,} rows, exceeding "
                f"the materialization cap of {cap:,}."
            )

        feature_count = len(X.feature_columns)
        if feature_count == 0:
            raise ValueError("Disk LightGBM training requires at least one feature.")
        values = np.empty((row_count, feature_count), dtype=np.int8, order="C")
        visited = np.zeros(row_count, dtype=bool)
        batch_count = 0
        for batch, positions in X.iter_feature_batches(
            self.prediction_batch_size,
            shuffle_blocks=False,
        ):
            self._validate_disk_batch(
                batch,
                positions,
                row_count=row_count,
                feature_count=feature_count,
                visited=visited,
            )
            values[positions] = batch
            visited[positions] = True
            batch_count += 1
        if not visited.all():
            raise RuntimeError("Disk LightGBM materialization did not visit every row.")
        return values, batch_count

    def _predict_disk_features(self, X) -> np.ndarray:
        row_count = len(X)
        feature_count = len(X.feature_columns)
        if row_count and feature_count == 0:
            raise ValueError("Disk LightGBM prediction requires at least one feature.")
        predictions = np.empty(row_count, dtype=np.float64)
        visited = np.zeros(row_count, dtype=bool)
        batch_count = 0
        for batch, positions in X.iter_feature_batches(
            self.prediction_batch_size,
            shuffle_blocks=False,
        ):
            self._validate_disk_batch(
                batch,
                positions,
                row_count=row_count,
                feature_count=feature_count,
                visited=visited,
            )
            batch_predictions = np.asarray(self._model.predict(batch)).reshape(-1)
            if len(batch_predictions) != len(positions):
                raise RuntimeError(
                    "LightGBM prediction count does not match the disk batch rows."
                )
            predictions[positions] = batch_predictions
            visited[positions] = True
            batch_count += 1
        if not visited.all():
            raise RuntimeError("Disk LightGBM prediction did not visit every row.")
        self.disk_validation_rows_ = row_count
        self.disk_prediction_batch_size_ = self.prediction_batch_size
        self.disk_prediction_batches_ = batch_count
        return predictions

    @staticmethod
    def _validate_disk_batch(
        batch,
        positions,
        *,
        row_count: int,
        feature_count: int,
        visited: np.ndarray,
    ) -> None:
        if not isinstance(batch, np.ndarray) or batch.dtype != np.int8:
            raise TypeError("Disk LightGBM batches must be int8 NumPy arrays.")
        positions = np.asarray(positions)
        if positions.ndim != 1:
            raise ValueError("Disk LightGBM batch positions must be one-dimensional.")
        if not np.issubdtype(positions.dtype, np.integer):
            raise TypeError("Disk LightGBM batch positions must be integers.")
        if batch.shape != (len(positions), feature_count):
            raise ValueError("Disk LightGBM batch shape does not match its positions.")
        if positions.size and (
            positions.min() < 0 or positions.max() >= row_count
        ):
            raise IndexError("Disk LightGBM batch position is out of range.")
        if len(np.unique(positions)) != len(positions) or visited[positions].any():
            raise RuntimeError("Disk LightGBM batches contain duplicate positions.")

    def _reset_fit_diagnostics(self) -> None:
        self.data_mode_ = None
        self.disk_train_rows_ = None
        self.disk_validation_rows_ = None
        self.disk_prediction_batch_size_ = None
        self.disk_prediction_batches_ = None
        self.disk_batches_per_epoch_ = []
        self.disk_rows_per_epoch_ = []
        self.effective_device_type_ = None
        self.gpu_fallback_used_ = None

    @staticmethod
    def _validate_disk_row_cap(value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("disk_materialization_max_rows must be a positive integer.")
        return value

    @staticmethod
    def _validate_prediction_batch_size(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("prediction_batch_size must be a positive integer.")
        return value

    def _configured_device_type(self) -> str:
        return str(self._params.get("device_type", "cpu")).lower()

    @staticmethod
    def _is_disk_feature_view(X) -> bool:
        return bool(getattr(X, "is_disk_feature_view", False))

    def _should_fallback_to_cpu(self, exc: Exception) -> bool:
        message = str(exc)
        device_type = str(self._params.get("device_type", "")).lower()
        return (
            device_type == "gpu"
            and "GPU Tree Learner was not enabled" in message
        )

    @staticmethod
    def _filter_features(X, feature_cols):
        if not feature_cols or not hasattr(X, "columns"):
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for LGBMRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[feature_cols]

    def __getattr__(self, name: str):
        return getattr(self._model, name)

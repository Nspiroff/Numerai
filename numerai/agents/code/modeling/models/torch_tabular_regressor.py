from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np


class _ResidualBlock:
    """Factory namespace kept separate so torch remains an optional dependency."""

    @staticmethod
    def build(torch, width: int, hidden_width: int, dropout: float):
        nn = torch.nn

        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(width)
                self.linear1 = nn.Linear(width, hidden_width)
                self.linear2 = nn.Linear(hidden_width, width)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                residual = x
                x = self.norm(x)
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = self.dropout(x)
                return residual + x

        return ResidualBlock()


class _ResidualNetwork:
    @staticmethod
    def build(
        torch,
        input_dim: int,
        width: int,
        hidden_width: int,
        n_blocks: int,
        dropout: float,
    ):
        nn = torch.nn

        class ResidualNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Linear(input_dim, width)
                self.blocks = nn.ModuleList(
                    _ResidualBlock.build(torch, width, hidden_width, dropout)
                    for _ in range(n_blocks)
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1)
                )

            def forward(self, x):
                x = self.input(x)
                for block in self.blocks:
                    x = block(x)
                return self.head(x)

        return ResidualNetwork()


def _activation(torch, name: str):
    mapping = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "selu": torch.nn.SELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported activation '{name}'. Supported: {sorted(mapping)}"
        )
    return mapping[key]


class TorchTabularRegressor:
    """Batched GPU regressor supporting MLP, residual MLP, and TabM backbones.

    The wrapper deliberately trains only on Numerai feature columns. Era and
    benchmark columns may still be present in ``X`` for leakage-safe internal
    validation and target transforms, but never enter the network.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        architecture: str = "mlp",
        hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
        activation: str = "gelu",
        dropout: float = 0.0,
        resnet_width: int = 512,
        resnet_hidden_width: int = 1024,
        resnet_blocks: int = 4,
        tabm_arch_type: str = "tabm",
        tabm_k: int = 16,
        tabm_width: int = 512,
        tabm_blocks: int = 3,
        batch_size: int = 2048,
        prediction_batch_size: int | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_mode: str = "mse",
        chronological_blocks: int = 8,
        dro_temperature: float = 2.0,
        recency_half_life_eras: float | None = None,
        ema_decay: float | None = None,
        max_epochs: int = 40,
        patience: int = 5,
        val_fraction: float = 0.1,
        val_split: str = "recent_eras",
        era_col: str = "era",
        internal_val_embargo: int = 13,
        feature_center: float | None = 2.0,
        feature_scale: float | None = 2.0,
        max_grad_norm: float | None = 1.0,
        device: str = "cuda",
        amp: bool = True,
        seed: int = 1337,
        num_workers: int = 0,
        disk_block_rows: int = 65_536,
        pin_memory: bool | None = None,
        deterministic: bool = False,
        verbose: bool = True,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for TorchTabularRegressor. Install a "
                "CUDA-enabled build for GPU training."
            ) from exc

        self._torch = torch
        self._feature_cols = list(feature_cols) if feature_cols else None
        self.architecture = str(architecture).lower()
        if self.architecture not in {"mlp", "resnet", "tabm"}:
            raise ValueError("architecture must be one of: mlp, resnet, tabm")
        self.hidden_layer_sizes = tuple(int(x) for x in hidden_layer_sizes)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.resnet_width = int(resnet_width)
        self.resnet_hidden_width = int(resnet_hidden_width)
        self.resnet_blocks = int(resnet_blocks)
        self.tabm_arch_type = str(tabm_arch_type)
        self.tabm_k = int(tabm_k)
        self.tabm_width = int(tabm_width)
        self.tabm_blocks = int(tabm_blocks)
        self.batch_size = int(batch_size)
        self.prediction_batch_size = int(prediction_batch_size or batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss_mode = str(loss_mode).lower()
        if self.loss_mode not in {
            "mse",
            "era_balanced_mse",
            "chronological_block_dro",
        }:
            raise ValueError(
                "loss_mode must be one of: mse, era_balanced_mse, "
                "chronological_block_dro"
            )
        if isinstance(chronological_blocks, (bool, np.bool_)) or not isinstance(
            chronological_blocks, (int, np.integer)
        ):
            raise TypeError("chronological_blocks must be an integer of at least 2")
        self.chronological_blocks = int(chronological_blocks)
        if self.chronological_blocks < 2:
            raise ValueError("chronological_blocks must be at least 2")
        if isinstance(dro_temperature, (bool, np.bool_)):
            raise ValueError("dro_temperature must be finite and positive")
        try:
            self.dro_temperature = float(dro_temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "dro_temperature must be finite and positive"
            ) from exc
        if not np.isfinite(self.dro_temperature) or self.dro_temperature <= 0.0:
            raise ValueError("dro_temperature must be finite and positive")
        if recency_half_life_eras is None:
            self.recency_half_life_eras = None
        else:
            if isinstance(recency_half_life_eras, (bool, np.bool_)):
                raise ValueError(
                    "recency_half_life_eras must be finite and positive"
                )
            try:
                self.recency_half_life_eras = float(recency_half_life_eras)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "recency_half_life_eras must be finite and positive"
                ) from exc
            if (
                not np.isfinite(self.recency_half_life_eras)
                or self.recency_half_life_eras <= 0.0
            ):
                raise ValueError(
                    "recency_half_life_eras must be finite and positive"
                )
            if self.loss_mode != "chronological_block_dro":
                raise ValueError(
                    "recency_half_life_eras is only supported with "
                    "loss_mode='chronological_block_dro'"
                )
        if ema_decay is None:
            self.ema_decay = None
        else:
            if isinstance(ema_decay, (bool, np.bool_)):
                raise ValueError("ema_decay must be finite and in (0, 1)")
            try:
                self.ema_decay = float(ema_decay)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ema_decay must be finite and in (0, 1)"
                ) from exc
            if not np.isfinite(self.ema_decay) or not 0.0 < self.ema_decay < 1.0:
                raise ValueError("ema_decay must be finite and in (0, 1)")
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.val_split = str(val_split).lower()
        if self.val_split not in {"recent_eras", "random_rows", "none"}:
            raise ValueError(
                "val_split must be one of: recent_eras, random_rows, none"
            )
        self.era_col = str(era_col)
        self.internal_val_embargo = int(internal_val_embargo)
        self.feature_center = (
            float(feature_center) if feature_center is not None else None
        )
        self.feature_scale = (
            float(feature_scale) if feature_scale is not None else None
        )
        if self.feature_scale == 0.0:
            raise ValueError("feature_scale must be non-zero")
        self.max_grad_norm = (
            float(max_grad_norm) if max_grad_norm is not None else None
        )
        self.device = str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Install a CUDA-enabled "
                "torch build or set device='cpu'."
            )
        self.amp = bool(amp)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.disk_block_rows = int(disk_block_rows)
        if self.disk_block_rows <= 0:
            raise ValueError("disk_block_rows must be positive")
        self.pin_memory = (
            self.device.startswith("cuda")
            if pin_memory is None
            else bool(pin_memory)
        )
        self.deterministic = bool(deterministic)
        self.verbose = bool(verbose)

        self._device = torch.device(self.device)
        self._model = None
        self._input_cols: list[str] | None = None
        self.training_history_: list[dict[str, float | int]] = []
        self.best_epoch_: int | None = None
        self.epochs_trained_: int = 0
        self.n_parameters_: int | None = None
        self.data_mode_: str | None = None
        self.disk_train_rows_: int | None = None
        self.disk_validation_rows_: int | None = None
        self.disk_rows_per_epoch_: list[int] = []
        self.disk_batches_per_epoch_: list[int] = []
        self.disk_prediction_batches_: int | None = None
        self.ema_updates_: int = 0
        self.ema_live_state_sha256_: str | None = None
        self.ema_shadow_state_sha256_: str | None = None
        self.ema_inference_state_sha256_: str | None = None
        self._ema_state = None

    def fit(self, X, y, **kwargs):
        del kwargs
        torch = self._torch
        self.best_epoch_ = None
        self.epochs_trained_ = 0
        self.ema_updates_ = 0
        self.ema_live_state_sha256_ = None
        self.ema_shadow_state_sha256_ = None
        self.ema_inference_state_sha256_ = None
        self._ema_state = None
        self._seed_everything()

        X_features = self._select_features(X, fit=True)
        y_values = self._as_float_array(y)
        if not y_values.flags.writeable:
            y_values = y_values.copy()
        valid_indices = np.flatnonzero(np.isfinite(y_values))
        if valid_indices.size < 2:
            raise ValueError("TorchTabularRegressor needs at least two finite targets.")

        train_indices, val_indices = self._split_indices(
            X, valid_indices, y_values.shape[0]
        )
        train_loss_aux = self._training_loss_aux(X, train_indices)
        has_validation = bool(val_indices.size)
        is_disk = getattr(X_features, "is_disk_feature_view", False)
        self.data_mode_ = "disk_feature_store" if is_disk else "eager"
        self.disk_rows_per_epoch_ = []
        self.disk_batches_per_epoch_ = []
        self.disk_prediction_batches_ = None
        if is_disk:
            if self.num_workers != 0:
                raise ValueError(
                    "Disk feature-store training requires num_workers=0, "
                    "including on Windows."
                )
            input_dim = len(X_features.feature_columns)
            train_view = X_features.take(train_indices)
            train_targets = np.ascontiguousarray(
                y_values[train_indices], dtype=np.float32
            )
            val_view = X_features.take(val_indices) if val_indices.size else None
            val_targets = (
                np.ascontiguousarray(y_values[val_indices], dtype=np.float32)
                if val_indices.size
                else None
            )
            self.disk_train_rows_ = len(train_view)
            self.disk_validation_rows_ = len(val_view) if val_view is not None else 0
            train_loader = None
            val_loader = None
        else:
            X_values = self._feature_array(X_features)
            if not X_values.flags.writeable:
                X_values = X_values.copy()
            input_dim = X_values.shape[1]
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_values), torch.from_numpy(y_values)
            )
            if train_loss_aux is None:
                train_loader = self._make_loader(
                    dataset, train_indices, batch_size=self.batch_size, shuffle=True
                )
            else:
                loss_aux_values = np.zeros(
                    (y_values.shape[0], *train_loss_aux.shape[1:]),
                    dtype=train_loss_aux.dtype,
                )
                loss_aux_values[train_indices] = train_loss_aux
                train_dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_values),
                    torch.from_numpy(y_values),
                    torch.from_numpy(loss_aux_values),
                )
                train_loader = self._make_loader(
                    train_dataset,
                    train_indices,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
            val_loader = (
                self._make_loader(
                    dataset,
                    val_indices,
                    batch_size=self.prediction_batch_size,
                    shuffle=False,
                )
                if val_indices.size
                else None
            )
            self.disk_train_rows_ = None
            self.disk_validation_rows_ = None

        self._model = self._build_model(input_dim).to(self._device)
        self.n_parameters_ = sum(p.numel() for p in self._model.parameters())
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            self._device.type,
            enabled=self.amp and self._device.type == "cuda",
        )

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.training_history_ = []

        if self.verbose:
            print(
                f"Training {self.architecture}: rows={len(train_indices):,}, "
                f"val_rows={len(val_indices):,}, features={input_dim:,}, "
                f"params={self.n_parameters_:,}, device={self._device}."
            )

        for epoch in range(1, self.max_epochs + 1):
            if is_disk:
                if train_loss_aux is None:
                    train_loss, epoch_rows, epoch_batches = (
                        self._train_epoch_disk(
                            train_view,
                            train_targets,
                            optimizer,
                            scaler,
                            epoch=epoch,
                        )
                    )
                else:
                    train_loss, epoch_rows, epoch_batches = (
                        self._train_epoch_disk(
                            train_view,
                            train_targets,
                            optimizer,
                            scaler,
                            epoch=epoch,
                            loss_aux=train_loss_aux,
                        )
                    )
                self.disk_rows_per_epoch_.append(epoch_rows)
                self.disk_batches_per_epoch_.append(epoch_batches)
            else:
                train_loss = self._train_epoch(train_loader, optimizer, scaler)
            epoch_evaluation_state = None
            if has_validation:
                if self.ema_decay is None:
                    val_loss = (
                        self._validation_loss_disk(val_view, val_targets)
                        if is_disk
                        else self._validation_loss(val_loader)
                    )
                else:
                    with self._ema_evaluation_scope():
                        val_loss = (
                            self._validation_loss_disk(val_view, val_targets)
                            if is_disk
                            else self._validation_loss(val_loader)
                        )
                        epoch_evaluation_state = self._cpu_model_state()
            else:
                val_loss = train_loss
            self.training_history_.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            self.epochs_trained_ = epoch
            if self.verbose:
                print(
                    f"epoch={epoch:03d} train_loss={train_loss:.8f} "
                    f"val_loss={val_loss:.8f}"
                )

            if has_validation:
                if val_loss < best_loss - 1e-8:
                    best_loss = val_loss
                    best_state = (
                        epoch_evaluation_state
                        if epoch_evaluation_state is not None
                        else self._cpu_model_state()
                    )
                    self.best_epoch_ = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        if self.verbose:
                            print(
                                f"Early stopping at epoch {epoch}; "
                                f"best_epoch={self.best_epoch_}."
                            )
                        break

        if self.ema_decay is not None:
            if self._ema_state is None or self.ema_updates_ == 0:
                raise RuntimeError(
                    "EMA training completed without a successful optimizer step."
                )
            self.ema_live_state_sha256_ = self._state_sha256(
                self._model.state_dict()
            )
            self.ema_shadow_state_sha256_ = self._state_sha256(self._ema_state)
        if has_validation and best_state is not None:
            self._model.load_state_dict(best_state)
        elif self.ema_decay is not None:
            self._model.load_state_dict(self._ema_state)
        if self.ema_decay is not None:
            self.ema_inference_state_sha256_ = self._state_sha256(
                self._model.state_dict()
            )
            # Keep the post-fit shadow aligned with the exact checkpoint used
            # by predict/export. The separately recorded shadow hash above is
            # the terminal optimizer-trajectory EMA before best restoration.
            self._ema_state = {
                key: value.detach().clone()
                for key, value in self._model.state_dict().items()
            }
        return self

    def cpu_state_dict(self):
        if self._model is None:
            raise RuntimeError(
                "TorchTabularRegressor must be fitted before exporting state."
            )
        self._model.eval()
        return self._cpu_model_state()

    def _cpu_model_state(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self._model.state_dict().items()
        }

    def _state_sha256(self, state) -> str:
        digest = hashlib.sha256()
        for key in sorted(state):
            value = state[key].detach().cpu().contiguous()
            encoded_key = key.encode("utf-8")
            encoded_dtype = str(value.dtype).encode("ascii")
            encoded_shape = repr(tuple(value.shape)).encode("ascii")
            digest.update(len(encoded_key).to_bytes(8, "little"))
            digest.update(encoded_key)
            digest.update(len(encoded_dtype).to_bytes(8, "little"))
            digest.update(encoded_dtype)
            digest.update(len(encoded_shape).to_bytes(8, "little"))
            digest.update(encoded_shape)
            digest.update(value.view(self._torch.uint8).numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _update_ema_after_step(self) -> None:
        if self.ema_decay is None:
            return
        torch = self._torch
        current_state = self._model.state_dict()
        with torch.no_grad():
            if self._ema_state is None:
                self._ema_state = {
                    key: value.detach().clone()
                    for key, value in current_state.items()
                }
            else:
                for key, value in current_state.items():
                    shadow = self._ema_state[key]
                    if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                        shadow.mul_(self.ema_decay).add_(
                            value.detach(), alpha=1.0 - self.ema_decay
                        )
                    else:
                        shadow.copy_(value.detach())
        self.ema_updates_ += 1

    def _step_optimizer(self, optimizer, scaler) -> None:
        if self.ema_decay is None:
            scaler.step(optimizer)
            scaler.update()
            return
        scale_before = float(scaler.get_scale())
        if not np.isfinite(scale_before) or scale_before <= 0.0:
            raise RuntimeError(
                "EMA training requires a finite positive AMP scale before step."
            )
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if not np.isfinite(scale_after) or scale_after <= 0.0:
            raise RuntimeError(
                "EMA training requires a finite positive AMP scale after update."
            )
        # GradScaler reduces its scale when an overflow skips optimizer.step().
        # Equal or increased valid scale therefore proves that this step completed.
        if scale_after >= scale_before:
            self._update_ema_after_step()

    @contextmanager
    def _ema_evaluation_scope(self):
        if self.ema_decay is None:
            yield
            return
        if self._ema_state is None or self.ema_updates_ == 0:
            raise RuntimeError(
                "EMA validation requires at least one successful optimizer step."
            )
        live_state = self._cpu_model_state()
        try:
            self._model.load_state_dict(self._ema_state)
            yield
        finally:
            self._model.load_state_dict(live_state)

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("TorchTabularRegressor must be fitted before predict().")
        torch = self._torch
        X_features = self._select_features(X, fit=False)
        if getattr(X_features, "is_disk_feature_view", False):
            if self.num_workers != 0:
                raise ValueError(
                    "Disk feature-store prediction requires num_workers=0, "
                    "including on Windows."
                )
            predictions = np.empty(len(X_features), dtype=np.float32)
            prediction_batches = 0
            self._model.eval()
            with torch.inference_mode():
                for batch_values, local_positions in X_features.iter_feature_batches(
                    self.prediction_batch_size,
                    shuffle_blocks=False,
                    block_rows=self.disk_block_rows,
                ):
                    batch_x = self._prepare_batch(
                        self._tensor_from_numpy(batch_values)
                    )
                    with torch.amp.autocast(
                        self._device.type,
                        enabled=self.amp and self._device.type == "cuda",
                    ):
                        batch_predictions = self._model(batch_x)
                        batch_predictions = self._ensemble_mean(batch_predictions)
                    if not torch.isfinite(batch_predictions).all():
                        raise FloatingPointError(
                            "TorchTabularRegressor produced non-finite predictions."
                        )
                    predictions[local_positions] = (
                        batch_predictions.float().cpu().numpy()
                    )
                    prediction_batches += 1
            self.disk_prediction_batches_ = prediction_batches
            return predictions

        X_values = self._feature_array(X_features)
        if not X_values.flags.writeable:
            X_values = X_values.copy()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_values))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.prediction_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        predictions = []
        self._model.eval()
        with torch.inference_mode():
            for (batch_x,) in loader:
                batch_x = self._prepare_batch(batch_x)
                with torch.amp.autocast(
                    self._device.type,
                    enabled=self.amp and self._device.type == "cuda",
                ):
                    batch_predictions = self._model(batch_x)
                    batch_predictions = self._ensemble_mean(batch_predictions)
                if not torch.isfinite(batch_predictions).all():
                    raise FloatingPointError(
                        "TorchTabularRegressor produced non-finite predictions."
                    )
                predictions.append(batch_predictions.float().cpu().numpy())
        if not predictions:
            return np.array([], dtype=np.float32)
        return np.concatenate(predictions).astype(np.float32, copy=False)

    def _build_model(self, input_dim: int):
        torch = self._torch
        if self.architecture == "mlp":
            activation_cls = _activation(torch, self.activation)
            layers = []
            in_dim = input_dim
            for hidden_dim in self.hidden_layer_sizes:
                layers.extend((torch.nn.Linear(in_dim, hidden_dim), activation_cls()))
                if self.dropout > 0.0:
                    layers.append(torch.nn.Dropout(self.dropout))
                in_dim = hidden_dim
            layers.append(torch.nn.Linear(in_dim, 1))
            return torch.nn.Sequential(*layers)

        if self.architecture == "resnet":
            return _ResidualNetwork.build(
                torch,
                input_dim=input_dim,
                width=self.resnet_width,
                hidden_width=self.resnet_hidden_width,
                n_blocks=self.resnet_blocks,
                dropout=self.dropout,
            )

        try:
            from tabm import TabM
        except ImportError as exc:
            raise ImportError(
                "tabm is required for architecture='tabm'. Install with "
                "`.venv/Scripts/pip install tabm`."
            ) from exc
        tabm_activation = {
            "relu": "ReLU",
            "gelu": "GELU",
            "silu": "SiLU",
        }.get(self.activation.lower(), self.activation)
        return TabM.make(
            n_num_features=input_dim,
            d_out=1,
            arch_type=self.tabm_arch_type,
            k=self.tabm_k,
            d_block=self.tabm_width,
            n_blocks=self.tabm_blocks,
            dropout=self.dropout,
            activation=tabm_activation,
        )

    def _train_epoch(self, loader, optimizer, scaler) -> float:
        torch = self._torch
        self._model.train()
        total_loss = torch.zeros((), device=self._device)
        total_rows = 0
        for batch in loader:
            batch_x, batch_y = batch[:2]
            batch_loss_aux = batch[2] if len(batch) == 3 else None
            batch_x = self._prepare_batch(batch_x)
            batch_y = batch_y.to(self._device, non_blocking=True)
            if batch_loss_aux is not None:
                batch_loss_aux = batch_loss_aux.to(
                    self._device, non_blocking=True
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                self._device.type,
                enabled=self.amp and self._device.type == "cuda",
            ):
                predictions = self._model(batch_x)
                loss = self._training_loss(
                    predictions, batch_y, batch_loss_aux
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "TorchTabularRegressor encountered a non-finite training loss."
                )
            scaler.scale(loss).backward()
            if self.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.max_grad_norm
                )
            self._step_optimizer(optimizer, scaler)
            rows = batch_y.shape[0]
            total_loss += loss.detach() * rows
            total_rows += rows
        return float(total_loss) / max(total_rows, 1)

    def _validation_loss(self, loader) -> float:
        torch = self._torch
        self._model.eval()
        total_loss = torch.zeros((), device=self._device)
        total_rows = 0
        with torch.inference_mode():
            for batch_x, batch_y in loader:
                batch_x = self._prepare_batch(batch_x)
                batch_y = batch_y.to(self._device, non_blocking=True)
                with torch.amp.autocast(
                    self._device.type,
                    enabled=self.amp and self._device.type == "cuda",
                ):
                    predictions = self._ensemble_mean(self._model(batch_x))
                    loss = torch.nn.functional.mse_loss(predictions, batch_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "TorchTabularRegressor encountered a non-finite "
                        "validation loss."
                    )
                rows = batch_y.shape[0]
                total_loss += loss * rows
                total_rows += rows
        return float(total_loss) / max(total_rows, 1)

    def _train_epoch_disk(
        self,
        view,
        targets,
        optimizer,
        scaler,
        *,
        epoch: int,
        loss_aux=None,
    ):
        torch = self._torch
        self._model.train()
        total_loss = torch.zeros((), device=self._device)
        total_rows = 0
        total_batches = 0
        for batch_values, local_positions in view.iter_feature_batches(
            self.batch_size,
            shuffle_blocks=True,
            seed=self.seed + epoch,
            block_rows=self.disk_block_rows,
        ):
            batch_x = self._prepare_batch(self._tensor_from_numpy(batch_values))
            batch_targets = np.ascontiguousarray(
                targets[local_positions], dtype=np.float32
            )
            batch_y = self._tensor_from_numpy(batch_targets).to(
                self._device, non_blocking=True
            )
            batch_loss_aux = None
            if loss_aux is not None:
                batch_loss_aux_values = np.ascontiguousarray(
                    loss_aux[local_positions]
                )
                batch_loss_aux = self._tensor_from_numpy(
                    batch_loss_aux_values
                ).to(self._device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                self._device.type,
                enabled=self.amp and self._device.type == "cuda",
            ):
                batch_predictions = self._model(batch_x)
                loss = self._training_loss(
                    batch_predictions, batch_y, batch_loss_aux
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "TorchTabularRegressor encountered a non-finite training loss."
                )
            scaler.scale(loss).backward()
            if self.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.max_grad_norm
                )
            self._step_optimizer(optimizer, scaler)
            rows = batch_y.shape[0]
            total_loss += loss.detach() * rows
            total_rows += rows
            total_batches += 1
        return float(total_loss) / max(total_rows, 1), total_rows, total_batches

    def _validation_loss_disk(self, view, targets) -> float:
        torch = self._torch
        self._model.eval()
        total_loss = torch.zeros((), device=self._device)
        total_rows = 0
        with torch.inference_mode():
            for batch_values, local_positions in view.iter_feature_batches(
                self.prediction_batch_size,
                shuffle_blocks=False,
                block_rows=self.disk_block_rows,
            ):
                batch_x = self._prepare_batch(
                    self._tensor_from_numpy(batch_values)
                )
                batch_targets = np.ascontiguousarray(
                    targets[local_positions], dtype=np.float32
                )
                batch_y = self._tensor_from_numpy(batch_targets).to(
                    self._device, non_blocking=True
                )
                with torch.amp.autocast(
                    self._device.type,
                    enabled=self.amp and self._device.type == "cuda",
                ):
                    batch_predictions = self._ensemble_mean(self._model(batch_x))
                    loss = torch.nn.functional.mse_loss(batch_predictions, batch_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "TorchTabularRegressor encountered a non-finite "
                        "validation loss."
                    )
                rows = batch_y.shape[0]
                total_loss += loss * rows
                total_rows += rows
        return float(total_loss) / max(total_rows, 1)

    def _independent_mse(self, predictions, targets):
        if self.architecture == "tabm":
            return ((predictions.squeeze(-1) - targets[:, None]) ** 2).mean()
        return self._torch.nn.functional.mse_loss(predictions.squeeze(-1), targets)

    def _training_loss(self, predictions, targets, loss_aux=None):
        if self.loss_mode == "mse":
            # Keep the established default path byte-for-byte equivalent.
            return self._independent_mse(predictions, targets)
        if loss_aux is None:
            raise RuntimeError(
                f"{self.loss_mode} requires era-derived training metadata"
            )

        row_losses = self._row_squared_error(predictions, targets)
        if self.loss_mode == "era_balanced_mse":
            weights = loss_aux.to(dtype=row_losses.dtype)
            return (row_losses * weights).mean()

        if loss_aux.ndim == 1:
            # Preserve the established chronological-block DRO path when
            # recency weighting is disabled.
            group_ids = loss_aux.to(dtype=self._torch.long)
            block_losses = self._torch.stack(
                [
                    row_losses[group_ids == group_id].mean()
                    for group_id in group_ids.unique()
                ]
            )
        else:
            if loss_aux.ndim != 2 or loss_aux.shape[1] != 2:
                raise RuntimeError(
                    "recency-weighted chronological_block_dro requires "
                    "N x 2 loss metadata"
                )
            group_ids = loss_aux[:, 0].to(dtype=self._torch.long)
            recency_weights = loss_aux[:, 1].to(dtype=row_losses.dtype)
            unique_group_ids = group_ids.unique()
            block_losses = self._torch.stack(
                [
                    (
                        row_losses[group_ids == group_id]
                        * recency_weights[group_ids == group_id]
                    ).sum()
                    / recency_weights[group_ids == group_id].sum()
                    for group_id in unique_group_ids
                ]
            )
            block_recency_mass = self._torch.stack(
                [
                    recency_weights[group_ids == group_id].sum()
                    for group_id in unique_group_ids
                ]
            )
            block_recency_prior = (
                block_recency_mass / block_recency_mass.sum()
            ).detach()
        mean_block_loss = block_losses.mean().clamp_min(
            self._torch.finfo(block_losses.dtype).eps
        )
        logits = self.dro_temperature * (block_losses / mean_block_loss - 1.0)
        block_weights = self._torch.softmax(logits, dim=0).detach()
        if loss_aux.ndim == 2:
            block_weights = block_weights * block_recency_prior
            block_weights = block_weights / block_weights.sum()
        return (block_weights * block_losses).sum()

    def _row_squared_error(self, predictions, targets):
        if self.architecture == "tabm":
            member_errors = predictions.squeeze(-1) - targets[:, None]
            return member_errors.square().mean(dim=1)
        return (predictions.squeeze(-1) - targets).square()

    def _ensemble_mean(self, predictions):
        if self.architecture == "tabm":
            return predictions.squeeze(-1).mean(dim=1)
        return predictions.squeeze(-1)

    def _prepare_batch(self, batch_x):
        batch_x = batch_x.to(self._device, non_blocking=True, dtype=self._torch.float32)
        if self.feature_center is not None:
            batch_x = batch_x - self.feature_center
        if self.feature_scale is not None:
            batch_x = batch_x / self.feature_scale
        return self._torch.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0)

    def _tensor_from_numpy(self, values):
        tensor = self._torch.from_numpy(values)
        if self.pin_memory and self._device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor

    def _make_loader(self, dataset, indices, *, batch_size: int, shuffle: bool):
        subset = self._torch.utils.data.Subset(dataset, indices)
        return self._torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _split_indices(self, X, valid_indices, total_rows: int):
        if self.val_split == "none" or self.val_fraction <= 0.0:
            return valid_indices, np.array([], dtype=np.int64)

        if self.val_split == "random_rows":
            rng = np.random.default_rng(self.seed)
            shuffled = rng.permutation(valid_indices)
            val_size = max(1, int(len(shuffled) * self.val_fraction))
            return shuffled[val_size:], shuffled[:val_size]

        if not hasattr(X, "columns") or self.era_col not in X.columns:
            raise ValueError(
                f"recent_eras validation requires era column '{self.era_col}'."
            )
        eras = np.asarray(X[self.era_col])
        valid_eras = eras[valid_indices]
        unique_eras = sorted(set(valid_eras), key=self._era_sort_key)
        if len(unique_eras) < 2:
            return valid_indices, np.array([], dtype=np.int64)
        val_era_count = max(1, int(len(unique_eras) * self.val_fraction))
        split = len(unique_eras) - val_era_count
        val_eras = set(unique_eras[split:])
        train_end = max(0, split - self.internal_val_embargo)
        train_eras = set(unique_eras[:train_end])
        train_mask = np.fromiter(
            (era in train_eras for era in valid_eras), dtype=bool, count=len(valid_eras)
        )
        val_mask = np.fromiter(
            (era in val_eras for era in valid_eras), dtype=bool, count=len(valid_eras)
        )
        train_indices = valid_indices[train_mask]
        val_indices = valid_indices[val_mask]
        if train_indices.size == 0 or val_indices.size == 0:
            raise ValueError(
                "Internal era validation produced an empty split; reduce "
                "val_fraction/internal_val_embargo."
            )
        return train_indices, val_indices

    def _training_loss_aux(self, X, train_indices):
        if self.loss_mode == "mse":
            return None
        if not hasattr(X, "columns") or self.era_col not in X.columns:
            raise ValueError(
                f"{self.loss_mode} requires era column '{self.era_col}'."
            )

        eras = np.asarray(X[self.era_col])
        train_eras = eras[train_indices]
        if any(self._is_missing_era(era) for era in train_eras):
            raise ValueError(
                f"{self.loss_mode} requires non-missing training era labels."
            )
        unique_eras = sorted(set(train_eras), key=self._era_sort_key)
        if len(unique_eras) < 2:
            raise ValueError(
                f"{self.loss_mode} requires at least two training era groups."
            )

        if self.loss_mode == "era_balanced_mse":
            era_counts = {
                era: int(np.count_nonzero(train_eras == era))
                for era in unique_eras
            }
            weights = np.fromiter(
                (
                    len(train_eras) / (len(unique_eras) * era_counts[era])
                    for era in train_eras
                ),
                dtype=np.float32,
                count=len(train_eras),
            )
            # The formula above is exact in real arithmetic; normalize once in
            # float32 so its stored mean is exactly the training convention.
            weights /= weights.mean(dtype=np.float64)
            return np.ascontiguousarray(weights, dtype=np.float32)

        if len(unique_eras) < self.chronological_blocks:
            raise ValueError(
                "chronological_block_dro requires at least "
                f"{self.chronological_blocks} distinct training eras; got "
                f"{len(unique_eras)}."
            )
        era_to_block = {
            era: block_id
            for block_id, block_eras in enumerate(
                np.array_split(
                    np.asarray(unique_eras, dtype=object),
                    self.chronological_blocks,
                )
            )
            for era in block_eras
        }
        block_ids = np.fromiter(
            (era_to_block[era] for era in train_eras),
            dtype=np.int64,
            count=len(train_eras),
        )
        if self.recency_half_life_eras is None:
            return block_ids

        newest_index = len(unique_eras) - 1
        era_to_age = {
            era: newest_index - era_index
            for era_index, era in enumerate(unique_eras)
        }
        recency_weights = np.fromiter(
            (
                2.0
                ** (-era_to_age[era] / self.recency_half_life_eras)
                for era in train_eras
            ),
            dtype=np.float32,
            count=len(train_eras),
        )
        return np.ascontiguousarray(
            np.column_stack((block_ids, recency_weights)), dtype=np.float32
        )

    def _select_features(self, X, *, fit: bool):
        if not hasattr(X, "columns"):
            raise TypeError(
                "TorchTabularRegressor requires a pandas DataFrame or "
                "DiskFeatureView."
            )
        cols = self._feature_cols if fit or self._input_cols is None else self._input_cols
        if not cols:
            raise ValueError("feature_cols must be supplied by the model factory.")
        missing = [col for col in cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for TorchTabularRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        if fit:
            self._input_cols = list(cols)
        if getattr(X, "is_disk_feature_view", False):
            return X.select_features(cols)
        return X[list(cols)]

    def _seed_everything(self):
        torch = self._torch
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def _as_float_array(values):
        if hasattr(values, "to_numpy"):
            return values.to_numpy(dtype=np.float32, copy=False)
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _feature_array(frame):
        """Keep compact integer Numerai features compact until each GPU batch."""
        values = frame.to_numpy(copy=False)
        if values.dtype.kind not in {"b", "i", "u", "f"}:
            values = values.astype(np.float32)
        elif values.dtype == np.float64:
            values = values.astype(np.float32)
        return values

    @staticmethod
    def _era_sort_key(era):
        try:
            return int(era)
        except (TypeError, ValueError):
            return str(era)

    @staticmethod
    def _is_missing_era(era):
        if era is None:
            return True
        try:
            return bool(np.isnan(era))
        except (TypeError, ValueError):
            return False

    def __getattr__(self, name: str):
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

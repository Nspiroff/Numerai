"""Custody and pure validation for the Ender25 Ender24 evaluation recovery.

This module imports only the standard library.  It deliberately keeps raw-byte
identity separate from newline-canonical JSON identity.  Runtime/scoring
dependencies and the frozen Ender24 scoring module are injected only after the
bootstrap has leased the complete immutable envelope.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path


FAMILY = "ender25_ender24_evaluation_recovery_v53"
STAGE = "ender25-ender24-round1-evaluation-recovery"
POSITIVE_STATE = "ENDER25_ROUND2_SOURCE_GATE_AUTHORIZED"
NEGATIVE_STATE = "ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN"
DECISION_RELATIVE = (
    "numerai/agents/experiments/ender25_ender24_evaluation_recovery_v53/"
    "receipts/ender24_round1_recovery_decision.json"
)
ENDER24_PREFIX = "numerai/agents/experiments/ender24_ema_seed_stability_v53"
ROUND1_NAMES = (
    "r1_control_seed1337",
    "r1_ema995_seed1337",
    "r1_control_seed2027",
    "r1_ema995_seed2027",
)
PAIR_NAMES = {
    "1337": ("r1_control_seed1337", "r1_ema995_seed1337"),
    "2027": ("r1_control_seed2027", "r1_ema995_seed2027"),
}

ERA_CANONICAL_RECEIPT = {
    "size_bytes": 1_763,
    "sha256": "be0c212a8e910f56dbdae4e1e134fa36ce7e5e1a95e43faa1ccc9e6330f544ca",
}
FEATURE_CANONICAL_RECEIPT = {
    "size_bytes": 148_179,
    "sha256": "663184191e17d2fa4fac6dae017890f0e762368e638d46cfaa489297b9b2049b",
}
TEXT_AUTHORITY = {
    "era_allowlist": {"canonical_lf": ERA_CANONICAL_RECEIPT},
    "feature_columns": {"canonical_lf": FEATURE_CANONICAL_RECEIPT},
}
ENDER24_CONFIG_SHA256 = {
    "r1_control_seed1337": "4aa28eb6cccd3ceb376b3e2a0d439b715c087beacd6c96a865a7069aadefccc4",
    "r1_ema995_seed1337": "ae064a2cbdf8aad663d30cfa4b4d200bdaec5513383213420d7e26884b91ed30",
    "r1_control_seed2027": "fbf91f7817807368bbadb4c24de0d33501107fbda1bdd222c8f132d51b56453a",
    "r1_ema995_seed2027": "c42c6f8e00ab2b67cecc2683e16245dc581de77af638a210285561c93c7cd16d",
}
DISCOVERY_EXTERNAL_AUTHORITY = {
    "full": {
        "path": "numerai/v5.3/ender21_discovery_full_through_0861.parquet",
        "size_bytes": 1_302_848_771,
        "sha256": "476d561ba8515a0066e892c5489a5ae1db6443587e9d5d06a9a6280400a701b9",
        "last_era": "0861",
    },
    "benchmark": {
        "path": (
            "numerai/v5.3/"
            "ender21_discovery_benchmark_models_through_0861.parquet"
        ),
        "size_bytes": 23_325_224,
        "sha256": "c2db9a77811390e9b9c47926b62fbfb7a6c7af24bb9c4db63137798e61b955b6",
        "last_era": "0861",
    },
}


def verify_bytes_receipt(raw: bytes, expected: dict, label: str) -> dict:
    """Verify a byte string against an exact two-field size/hash receipt."""

    if not isinstance(raw, bytes):
        raise TypeError(f"{label} must be bytes.")
    if not isinstance(expected, dict) or set(expected) != {
        "size_bytes",
        "sha256",
    }:
        raise ValueError(f"{label} receipt schema differs.")
    observed = {
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if observed != expected:
        raise ValueError(f"{label} byte receipt differs.")
    return observed


def canonical_json_bytes(raw: bytes, label: str) -> bytes:
    """Return strict UTF-8 JSON bytes with only CRLF-to-LF normalization.

    All-LF and all-CRLF files are portable representations of the same frozen
    text.  BOM, NUL, bare CR, and mixed LF/CRLF encodings are rejected.
    """

    if not isinstance(raw, bytes):
        raise TypeError(f"{label} must be bytes.")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not contain a UTF-8 BOM.")
    if b"\x00" in raw:
        raise ValueError(f"{label} must not contain NUL bytes.")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not strict UTF-8.") from error
    without_crlf = raw.replace(b"\r\n", b"")
    if b"\r" in without_crlf:
        raise ValueError(f"{label} contains a bare carriage return.")
    has_crlf = b"\r\n" in raw
    has_bare_lf = b"\n" in raw.replace(b"\r\n", b"")
    if has_crlf and has_bare_lf:
        raise ValueError(f"{label} mixes LF and CRLF line endings.")
    return raw.replace(b"\r\n", b"\n")


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json(raw: bytes, label: str) -> object:
    canonical = canonical_json_bytes(raw, label)
    try:
        return json.loads(
            canonical.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON.") from error


def load_canonical_authorities(
    era_bytes: bytes,
    feature_bytes: bytes,
    authority: dict,
) -> tuple[dict, list[str], list[str]]:
    """Validate both canonical text identities and their exact semantics."""

    era_canonical = canonical_json_bytes(era_bytes, "era allowlist")
    feature_canonical = canonical_json_bytes(feature_bytes, "feature columns")
    verify_bytes_receipt(
        era_canonical, ERA_CANONICAL_RECEIPT, "canonical era allowlist"
    )
    verify_bytes_receipt(
        feature_canonical, FEATURE_CANONICAL_RECEIPT, "canonical feature columns"
    )
    if not isinstance(authority, dict):
        raise ValueError("Ender24 discovery authority must be an object.")
    if set(authority) == set(TEXT_AUTHORITY):
        if authority != TEXT_AUTHORITY:
            raise ValueError("Recovery text authority differs from the freeze.")
    else:
        if (
            authority.get("schema_version") != 1
            or authority.get("authority") != "ender24-discovery-only"
            or authority.get("forbidden_historical_confirmation")
            != {
                "first_era": "0865",
                "last_era": "1021",
                "rule": "never read, score, select, tune, or report in Ender24",
            }
            or authority.get("prospective_confirmation")
            != {
                "first_era": "1231",
                "last_era": "1282",
                "era_count": 52,
                "rule": "future resolved eras only; no local historical substitute",
            }
        ):
            raise ValueError("Ender24 discovery authority envelope differs.")
        for label, expected in (
            ("era_allowlist", ERA_CANONICAL_RECEIPT),
            ("feature_columns", FEATURE_CANONICAL_RECEIPT),
            ("full", DISCOVERY_EXTERNAL_AUTHORITY["full"]),
            ("benchmark", DISCOVERY_EXTERNAL_AUTHORITY["benchmark"]),
        ):
            section = authority.get(label)
            if (
                not isinstance(section, dict)
                or section.get("size_bytes") != expected["size_bytes"]
                or section.get("sha256") != expected["sha256"]
            ):
                raise ValueError(f"Recorded Ender24 authority differs for {label}.")
    eras = strict_json(era_canonical, "canonical era allowlist")
    expected_eras = [f"{era:04d}" for era in range(161, 862, 4)]
    if eras != expected_eras or len(set(eras)) != 176:
        raise ValueError("Discovery era allowlist semantics differ.")
    features = strict_json(feature_canonical, "canonical feature columns")
    if (
        not isinstance(features, list)
        or len(features) != 3_555
        or any(type(value) is not str or not value for value in features)
        or len(set(features)) != 3_555
    ):
        raise ValueError("Feature authority differs from the ordered freeze.")
    return authority, eras, features


def validate_completion_envelope(
    name: str,
    payload: object,
    manifest_receipt: dict,
    artifact_receipts: dict,
) -> dict:
    """Validate one completion without opening a result or prediction payload."""

    if name not in ROUND1_NAMES:
        raise ValueError(f"Unknown Ender24 completion component: {name}")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "stage",
        "state",
        "component",
        "manifest",
        "config",
        "outputs",
    }:
        raise ValueError(f"{name} completion schema differs.")
    if (
        payload["schema_version"] != 1
        or payload["stage"] != "ender24-round1-training-completion"
        or payload["state"] != "OUTPUTS_FINALIZED"
        or payload["component"] != name
        or payload["manifest"] != manifest_receipt
    ):
        raise ValueError(f"{name} completion envelope differs.")
    config_path = f"{ENDER24_PREFIX}/configs/{name}.py"
    config = payload["config"]
    if (
        not isinstance(config, dict)
        or set(config) != {"path", "sha256"}
        or config.get("path") != config_path
        or config.get("sha256") != ENDER24_CONFIG_SHA256[name]
    ):
        raise ValueError(f"{name} completion config binding differs.")
    if not isinstance(artifact_receipts, dict) or set(artifact_receipts) != {
        "predictions",
        "result",
    }:
        raise ValueError(f"{name} artifact receipt set differs.")
    if payload["outputs"] != artifact_receipts:
        raise ValueError(f"{name} completion artifact identity differs.")
    return payload


def preflight_all_completions(
    completions: dict[str, object],
    manifest_receipt: dict,
    artifact_receipts: dict[str, dict],
) -> dict[str, dict]:
    """Parse/validate all four opaque completion envelopes as one barrier."""

    if not isinstance(completions, dict) or tuple(completions) != ROUND1_NAMES:
        raise ValueError("Ender24 completion cohort or order differs.")
    if not isinstance(artifact_receipts, dict) or set(artifact_receipts) != set(
        ROUND1_NAMES
    ):
        raise ValueError("Ender24 completion artifact cohort differs.")
    return {
        name: validate_completion_envelope(
            name, completions[name], manifest_receipt, artifact_receipts[name]
        )
        for name in ROUND1_NAMES
    }


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def exact_era_cv_splits(
    eras,
    n_splits: int = 5,
    embargo: int = 13,
    mode: str = "expanding",
    min_train_size: int = 1,
):
    """Frozen copy of the small Ender24 CV geometry helper."""

    if n_splits < 1 or embargo < 0:
        raise ValueError("Invalid CV split parameters.")
    eras_sorted = sorted(set(eras), key=lambda value: int(value))
    if n_splits > len(eras_sorted):
        raise ValueError("n_splits exceeds era count.")
    fold_size = len(eras_sorted) // n_splits
    splits = []
    for index in range(n_splits):
        start = index * fold_size
        end = (index + 1) * fold_size
        if index == n_splits - 1:
            end += len(eras_sorted) % n_splits
        validation = eras_sorted[start:end]
        if mode == "expanding":
            train = eras_sorted[: max(0, start - embargo)]
        elif mode == "blocked":
            left = max(0, start - embargo)
            right = min(len(eras_sorted), end + embargo)
            train = eras_sorted[:left] + eras_sorted[right:]
        else:
            raise ValueError("Unknown CV split mode.")
        if len(train) < min_train_size:
            raise ValueError("CV train split is too small.")
        splits.append((train, validation))
    return splits


class RecoveryCustody:
    """Adapter over held leases; no governed path may be opened outside it."""

    def __init__(
        self,
        repo_dir: Path,
        experiment: Path,
        numerai_dir: Path,
        manifest: dict,
        leases: dict[Path, object],
    ) -> None:
        self.repo_dir = Path(os.path.abspath(repo_dir))
        self.experiment = Path(os.path.abspath(experiment))
        self.numerai_dir = Path(os.path.abspath(numerai_dir))
        self.manifest = manifest
        self.leases = {Path(os.path.abspath(path)): lease for path, lease in leases.items()}

    def lease(self, path: Path):
        canonical = Path(os.path.abspath(path))
        try:
            return self.leases[canonical]
        except KeyError as error:
            raise ValueError(f"Governed path has no held lease: {canonical}") from error

    def read_bytes(self, path: Path) -> bytes:
        return self.lease(path).read_bytes()

    def read_json(self, path: Path) -> object:
        return strict_json(self.read_bytes(path), f"governed JSON {path}")

    def receipt(self, path: Path) -> dict:
        lease = self.lease(path)
        try:
            relative = Path(os.path.abspath(path)).relative_to(self.repo_dir).as_posix()
        except ValueError:
            relative = str(Path(os.path.abspath(path)))
        return {
            "path": relative,
            "size_bytes": lease.size_bytes(),
            "sha256": lease.sha256(),
        }

    def _identity_receipt(self, path: Path) -> dict:
        lease = self.lease(path)
        inspected = lease.stat()
        return {
            "path": str(Path(os.path.abspath(path))),
            "device": int(inspected.st_dev),
            "inode": int(inspected.st_ino),
            "size_bytes": int(inspected.st_size),
            "sha256": lease.sha256(),
        }

    def load_config(self, path: Path) -> dict:
        source = self.read_bytes(path)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            raise ValueError(f"Frozen config syntax is invalid: {path}") from error
        assignment = tree.body[-1] if tree.body else None
        if (
            len(tree.body) != 4
            or not isinstance(assignment, ast.Assign)
            or len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
            or assignment.targets[0].id != "CONFIG"
            or not isinstance(assignment.value, ast.Call)
            or not isinstance(assignment.value.func, ast.Name)
            or assignment.value.func.id != "variant"
            or any(keyword.arg is None for keyword in assignment.value.keywords)
        ):
            raise ValueError(f"Frozen config envelope differs: {path}")
        expected_prefix = [
            "from pathlib import Path",
            "import runpy",
            "variant = runpy.run_path(str(Path(__file__).with_name('base_r1.py')))['variant']",
        ]
        if [ast.unparse(node) for node in tree.body[:3]] != expected_prefix:
            raise ValueError(f"Frozen config loader differs: {path}")
        try:
            arguments = [ast.literal_eval(value) for value in assignment.value.args]
            keywords = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in assignment.value.keywords
            }
        except (ValueError, TypeError) as error:
            raise ValueError(f"Frozen config arguments are not literals: {path}") from error
        base_path = path.with_name("base_r1.py")
        namespace = {"__file__": str(base_path), "__name__": "__ender25_frozen_base__"}
        exec(compile(self.read_bytes(base_path), str(base_path), "exec"), namespace)
        variant = namespace.get("variant")
        if not callable(variant):
            raise ValueError("Frozen Ender24 base config has no variant.")
        return variant(*arguments, **keywords)

    def preflight_completions(self, _frozen_common=None) -> dict[str, dict]:
        old_manifest_path = self.repo_dir / f"{ENDER24_PREFIX}/source_manifest_round1.json"
        old_manifest = self.read_json(old_manifest_path)
        manifest_receipt = {
            "path": f"{ENDER24_PREFIX}/source_manifest_round1.json",
            "sha256": self.lease(old_manifest_path).sha256(),
            "git_head": old_manifest["git_head"],
        }
        completions = {}
        artifacts = {}
        for name in ROUND1_NAMES:
            completion_path = self.experiment.parent / (
                f"ender24_ema_seed_stability_v53/receipts/{name}.completion.json"
            )
            completions[name] = self.read_json(completion_path)
            artifacts[name] = {
                "predictions": self._identity_receipt(
                    self.experiment.parent
                    / f"ender24_ema_seed_stability_v53/predictions/{name}.parquet"
                ),
                "result": self._identity_receipt(
                    self.experiment.parent
                    / f"ender24_ema_seed_stability_v53/results/{name}.json"
                ),
            }
        return preflight_all_completions(completions, manifest_receipt, artifacts)

    def load_authority(self, _frozen_common=None):
        old_experiment = self.experiment.parent / "ender24_ema_seed_stability_v53"
        authority = self.read_json(old_experiment / "protocol/discovery_data_authority.json")
        era_path = self.repo_dir / (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        )
        feature_path = self.repo_dir / (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/feature_columns_all_v53.json"
        )
        _, eras, features = load_canonical_authorities(
            self.read_bytes(era_path), self.read_bytes(feature_path), authority
        )
        return (
            {
                "era_allowlist": {
                    "raw": self.receipt(era_path),
                    "canonical_lf": ERA_CANONICAL_RECEIPT,
                },
                "feature_columns": {
                    "raw": self.receipt(feature_path),
                    "canonical_lf": FEATURE_CANONICAL_RECEIPT,
                },
            },
            eras,
            features,
        )

    def load_truth(self, allowed: list[str], frozen_common):
        return frozen_common.load_truth(self.numerai_dir, allowed, self)

    def score_candidate(
        self,
        name: str,
        allowed: list[str],
        truth,
        completion: dict,
        frozen_common,
    ):
        return frozen_common.score_candidate(
            self.experiment.parent / "ender24_ema_seed_stability_v53",
            name,
            allowed,
            truth,
            completion,
            self,
        )

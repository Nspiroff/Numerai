# Research Source CI scope

This file documents what `.github/workflows/research-source-ci.yml` checks,
why that selection is safe on a GitHub-hosted runner, and what it deliberately
does not check. It is CI documentation only: it records no scientific metric,
decision, or terminal state, and it grants no authority of any kind.

## What a passing run means — and does not mean

A green Research Source CI run means only that the governed `numerai/agents`
Python tree compiles and that the selected synthetic source-contract test
modules pass on a fresh CPU-only runner.

A passing run is **not** training, scoring, confirmation, prospective
validation, final fitting, deployment, upload, assignment, staking,
submission, model creation, or any Numerai account authority. The governed
local GPU/data gates — which require the local Numerai `v5.3` datasets,
ignored local artifacts, the sealed CUDA Torch runtime, and separately
authorized one-shot runners and evaluators — remain entirely separate and are
never executed here.

## Selected checks

1. **Compile gate** — every `*.py` file under `numerai/agents` is compiled
   in-memory (no bytecode written). 199 files at selection time.
2. **Source-contract test modules**, run with
   `python -m unittest` from the repository root with `PYTHONPATH=numerai`:

| Module | Tests (local, 2026-08-16) | Why it is synthetic and safe |
| --- | --- | --- |
| `agents.tests.test_target_transforms` | 14 | Pure-function tests of `agents.code.modeling.utils.target_transforms` on small in-memory numpy/pandas fixtures; cross-checked against `numerai_tools.scoring`; no file, dataset, or network access. |
| `agents.tests.test_ender22_evaluators` | 4 | Contract tests of the archived Ender22 evaluator helpers on synthetic in-memory inputs. |
| `agents.tests.test_ender23_protocol` | 6 | Read-only assertions over tracked Ender23 protocol documents; verifies frozen wording/contract, writes nothing. |
| `agents.tests.test_ender24_evaluator` | 14 | Mock-based contract tests of the archived Ender24 evaluator modules; all writes go to `tempfile` directories. |
| `agents.tests.test_ender26_evaluator` | 19 | Mock-based contract tests of the archived Ender26 evaluator (custody, decision-reservation, gate-law envelope) on synthetic metrics; parquet reads are mocked. |
| `agents.tests.test_ender27_evaluator` | 19 | Mock-based contract tests of the terminal Ender27 evaluator (preflight/custody ordering, decision-reservation evidence rules, frozen 22-condition gate law, authority-denial envelope) on synthetic metrics; `load_truth`/`score_candidate` are mocked and all writes go to `tempfile` directories. |

**Total: 76 tests.** Local clean-environment validation ran the suite twice
with identical counts and results (76/76 OK), left the worktree with no
tracked change and no untracked file, and an `sys.addaudithook` run recorded
zero opens of any `v5.3` path, zero reads of the artifact-bearing original
checkout, and zero socket/DNS/urllib events.

## Runtime and dependencies

- GitHub-hosted `ubuntu-24.04` runner, fresh environment, CPU only.
- Python pinned to **3.13.14** — the sealed Ender27 runtime Python, verified
  present in the `actions/python-versions` manifest for `linux-24.04`.
- Exact pins in `.github/ci/research-source-requirements.txt`:
  `numpy==2.5.1`, `pandas==3.0.5`, `scipy==1.18.0`, `numerai-tools==0.6.0`
  (direct imports of the selected tests and modules under test, matching the
  sealed Ender27 runtime versions), `scikit-learn==1.9.0` (required by
  `numerai-tools`, pinned to the sealed runtime version), plus the six
  transitive packages frozen exactly as resolved during clean-environment
  validation. **Torch is not required and not installed.** Neither are
  `pyarrow`, `numerapi`, `lightgbm`, `cloudpickle`, or any dataset,
  credential, or GPU package.

## Deliberately excluded test categories

| Excluded | Examples | Reason |
| --- | --- | --- |
| GPU/Torch model tests | `test_torch_tabular_regressor`, `test_ender22_evaluator_contracts`, `test_ender22_temporal_retention`, `test_ender24_ema_seed_stability`, `test_tabm_export_bridge`, `test_tabm_final_fit_export` | Require Torch (sealed runtime pins CUDA Torch); GPU-adjacent; out of source-contract scope. |
| Deployment/export tests | `test_tabm_numpy_deployment`, `test_tabm_rank_ensemble` | Deployment-bundle scope (`cloudpickle`); deployment remains separately gated. |
| Pipeline/data-loader tests | `test_integration_pipeline`, `test_ender21_round2`, `test_ender23_memory_safe_eager`, `test_scoring_and_prediction_semantics`, all `*_pipeline_custody` modules, `test_ender26_gaussian_rank_residual`, `test_ender27_tempered_gaussian_rank_residual` | Import `agents.code.modeling.utils.pipeline`, which imports `pyarrow` and `numerapi` at module level; keeping the API-client library and parquet stack out of CI keeps the dependency surface minimal and credential-free. |
| Dataset-builder tests | `test_build_downsampled_datasets`, `test_build_feature_store`, `test_disk_feature_store_modeling`, `test_lgbm_disk_feature_store` | Data-build scope; some require `lightgbm`; local-data adjacent. |
| Analysis/report tests | `test_show_experiment`, `test_ender20_*`, `test_ender21_confirmation`, `test_ender21_round1_evaluator`, `test_xerxes20_lgbm_challenger` | Analysis/plotting scope (`pyarrow`/`matplotlib` adjacency); not part of the minimal source contract. |
| Subprocess-executing tests | `test_ender22_training_bootstrap` | Executes a child process; excluded to keep the CI job free of process execution beyond the test runner itself. |
| Lifecycle-phase tests | `test_ender25_ema_evaluation_recovery` | One test asserts the Ender25 recovery decision receipt does not yet exist; on terminal `main` that receipt exists by design, so the module encodes a pre-decision lifecycle phase and permanently fails post-archive. The scientific source is unchanged and untouched; the module is simply not CI-compatible. |
| Notebook execution | the six tutorial notebooks | Manual-only concern of `build-models.yml`; never part of source CI. |
| Canonical one-shot runners/evaluators against canonical paths | `run_round1.py`, `evaluate_round1.py` entry points | One-shot identities are governed evidence; CI must never invoke them. |

Changing this selection (adding or removing modules) is a reviewed change to
this file and the workflow, not a silent edit.

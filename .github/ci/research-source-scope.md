# Research Source CI scope

This file documents what `.github/workflows/research-source-ci.yml` checks,
why that selection is safe on GitHub-hosted runners, how the checks are split
across platforms, and what the workflow deliberately does not check. It is CI
documentation only: it records no scientific metric, decision, or terminal
state, and it grants no authority of any kind.

## What a passing run means — and does not mean

A green Research Source CI run means exactly two things:

1. the platform-portable subset of the governed source-contract suite passes
   on a fresh Ubuntu runner (after the whole `numerai/agents` tree compiles),
   and
2. the complete, unmodified Ender26 and Ender27 archived evaluator custody
   suites pass on their governed Windows platform.

A green run does **not** prove that the archived Ender26/27 evaluator source
is platform-neutral — a known, unresolved POSIX exception-envelope divergence
is documented below.

A passing run is **not** training, scoring, confirmation, prospective
validation, final fitting, deployment, upload, assignment, staking,
submission, model creation, or any Numerai account authority. The governed
local GPU/data gates — which require the local Numerai `v5.3` datasets,
ignored local artifacts, the sealed CUDA Torch runtime, and separately
authorized one-shot runners and evaluators — remain entirely separate and are
never executed here.

## Platform split

The workflow contains **two jobs**. Across both jobs the selected suite is
**76 tests**; **no individual test method is skipped** anywhere, and no
`continue-on-error` is used.

### Job 1 — `portable-source-contract` (ubuntu-24.04)

1. **Compile gate** — every `*.py` file under `numerai/agents` is compiled
   in-memory (no bytecode written). 199 files at selection time.
2. **Portable source-contract modules** (38 tests), run with
   `python -m unittest` from the repository root with `PYTHONPATH=numerai`:

| Module | Tests | Why it is synthetic, safe, and portable |
| --- | --- | --- |
| `agents.tests.test_target_transforms` | 14 | Pure-function tests of `agents.code.modeling.utils.target_transforms` on small in-memory numpy/pandas fixtures; cross-checked against `numerai_tools.scoring`; no file, dataset, or network access. |
| `agents.tests.test_ender22_evaluators` | 4 | Contract tests of the archived Ender22 evaluator helpers on synthetic in-memory inputs. |
| `agents.tests.test_ender23_protocol` | 6 | Read-only assertions over tracked Ender23 protocol documents; verifies frozen wording/contract, writes nothing. |
| `agents.tests.test_ender24_evaluator` | 14 | Mock-based contract tests of the archived Ender24 evaluator modules; all writes go to `tempfile` directories. |

### Job 2 — `windows-terminal-custody-contract` (windows-2022)

The **complete, unmodified** archived evaluator custody suites (38 tests):

| Module | Tests | Why it runs on Windows |
| --- | --- | --- |
| `agents.tests.test_ender26_evaluator` | 19 | Mock-based contract tests of the terminal Ender26 evaluator (custody, decision-reservation, gate-law envelope) on synthetic metrics; parquet reads are mocked; all writes go to `tempfile` directories. |
| `agents.tests.test_ender27_evaluator` | 19 | Mock-based contract tests of the terminal Ender27 evaluator (preflight/custody ordering, decision-reservation evidence rules, frozen 22-condition gate law, authority-denial envelope) on synthetic metrics; `load_truth`/`score_candidate` are mocked and all writes go to `tempfile` directories. |

The Ender26 and Ender27 research families were **governed and executed on
Windows**: the canonical one-shot evaluators ran there, and the frozen tests
in these two modules encode the Windows reservation-refusal error envelope —
`training_bootstrap._DecisionReservation` translates a `CreateFileW`
(`CREATE_NEW`) failure into
`ValueError("Cannot reserve … evaluation decision …")`. Because the terminal
packets (source, tests, manifests, decisions, postmortems) are immutable
archived evidence, their custody contract runs on the platform whose behavior
they froze.

## Known, unresolved portability finding (documented, not repaired)

The first Ubuntu CI run (run `31966777919`, 2026-08-16, on the original
single-job workflow) faithfully exposed that the archived bootstrap's POSIX
branch **leaks `FileExistsError`** from
`os.open(path, O_RDWR | O_CREAT | O_EXCL, 0o600)` instead of the frozen
`ValueError` envelope, causing the matching
`test_bootstrap_failed_decision_is_terminal_zero_byte_evidence` methods in
both archived modules to error on Linux (74 of 76 tests passed there; the
other four modules were fully green).

- The **POSIX branch still preserves fail-closed `CREATE_NEW` custody**:
  creation is create-new-only and existing terminal evidence is never opened
  or truncated on either platform. The divergence is the *exception
  envelope*, not the custody property.
- The **divergence remains known and unresolved**. This CI change does not
  repair, patch, wrap, monkeypatch, or otherwise alter the terminal packets,
  and a green workflow must not be read as a portability fix.
- Any portable reservation helper, or error-envelope normalization for a
  **future** research family, requires its own separate bounded maintenance
  gate and independent review. Terminal Ender26/27 bytes remain the
  canonical record either way.

## Runtime and dependencies

- GitHub-hosted `ubuntu-24.04` and `windows-2022` runners, fresh
  environments, CPU only, read-only checkouts (`persist-credentials: false`).
- Python pinned to **3.13.14** on both jobs — the sealed Ender27 runtime
  Python, verified present in the `actions/python-versions` manifest for
  both `linux-24.04` and `win32-x64`.
- Exact pins in `.github/ci/research-source-requirements.txt`:
  `numpy==2.5.1`, `pandas==3.0.5`, `scipy==1.18.0`, `numerai-tools==0.6.0`
  (direct imports of the selected tests and modules under test, matching the
  sealed Ender27 runtime versions), `scikit-learn==1.9.0` (required by
  `numerai-tools`, pinned to the sealed runtime version), plus the six
  transitive packages frozen exactly as resolved during clean-environment
  validation. **Torch is not required and not installed.** Neither are
  `pyarrow`, `numerapi`, `lightgbm`, `cloudpickle`, or any dataset,
  credential, or GPU package.

## Local validation record (platform-specific)

- **Windows clean environment** (fresh venv, Python 3.13.14, outside every
  repository/worktree, only the pinned requirements): the complete
  six-module suite passed **76/76** repeatedly, including both archived
  modules at **38/38**; an `sys.addaudithook` run recorded zero opens of any
  `v5.3` path, zero reads of the artifact-bearing original checkout, and
  zero socket/DNS/urllib events; the worktree showed no tracked change and
  no untracked file afterward.
- **Ubuntu evidence**: CI run `31966777919` established that the four
  portable modules and the compile gate passed on `ubuntu-24.04`, and that
  the only errors were the two archived exception-envelope custody tests
  described above.

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

Changing this selection or the platform allocation (adding, removing, or
moving modules) is a reviewed change to this file and the workflow, not a
silent edit.

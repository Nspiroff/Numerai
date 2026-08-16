# Numerai governed research status

- **Last verified:** 2026-08-16
- **Ender27 terminal-evidence checkpoint:** `99df04ff07ca7cdc9b02bc0cd275d3ecb105a520` (archived by tag `numerai-ender27-terminal`)
- **Lifecycle issue:** [#22 — Ender27 terminal](https://github.com/Nspiroff/Numerai/issues/22) (closed as completed)

This document is a human-readable control-plane summary for the repository. It
does not replace any experiment's `experiment.md`, `gate.md`, manifest,
completion envelope, decision receipt, or terminal postmortem.

## Executive state

The repository contains a governed Numerai Classic research program centered
on benchmark-unique contribution against `v53_lgbm_ender20`.

No model is approved for deployment. No pickle upload, model assignment,
staking, submission, model creation, or Numerai account mutation is currently
authorized.

**Ender20 through Ender27 are all terminal research families. There is
currently no active scientific gate.** Any successor modeling requires a newly
named, separately frozen hypothesis and fresh explicit user authorization.

## Ender27 terminal state

Lifecycle state:

`ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN`

The separately authorized one-shot evaluator was invoked exactly once, exited
0, and durably committed the canonical decision. It was not retried. The
canonical decision and terminal postmortem are merged and archived on `main`:

| Item | Identity |
| --- | --- |
| Decision path | `numerai/agents/experiments/ender27_tempered_gaussian_rank_residual_v53/receipts/round1_tempered_alignment.json` |
| Decision size | 21,670 bytes |
| Decision SHA-256 | `261ed2661dbc282388498ecd2fd8fec668a14ef78bf256b7549bbbdda73fc007` |
| Decision Git blob | `b6613f93a8bea18c06c60b5f7e26c2e09e2f7748` |
| Terminal postmortem | `numerai/agents/experiments/ender27_tempered_gaussian_rank_residual_v53/terminal_postmortem.md` |
| Evidence PR | [#27](https://github.com/Nspiroff/Numerai/pull/27) (merged) |
| Evidence merge commit | `99df04ff07ca7cdc9b02bc0cd275d3ecb105a520` |
| Archival tag | `numerai-ender27-terminal` |

Exact result under the frozen 22-condition decision law:

- **20 of 22 checks passed; exactly two failed**, both in the recent-40
  window: the aggregate recent-40 uplift gate
  (`mean_recent40_bmc_at_least_control_plus_0_00030`) and seed 1337's
  matched recent-40 gate (`matched_recent40_bmc_at_least_control`).
- Mean full BMC improved **+5.855608759689358%**
  (0.005811426364338923 → 0.00615172075559205); the aggregate full-BMC gate
  passed.
- Mean recent-40 BMC declined **-4.209843297506477%**
  (0.007174454375737901 → 0.006872421089068239) instead of clearing the
  required +0.00030 uplift.
- Seed 1337's matched recent-40 check failed
  (0.006780431226718131 < 0.008490224382548142).
- Seed 1337's drawdown gate **passed**, so tempering repaired one of
  Ender26's two seed-1337 failures — but not the decisive recent-window
  failure.
- The recent-window two-seed gap contracted from
  0.0026315400136204814 to 0.00018397972470021468.

The scientifically correct interpretation is **promising but non-robust
negative evidence**, not a no-signal result. Under the frozen all-conditions
law the family is terminal.

The four training runs each completed exactly once, serially, exit 0; the
twelve generated training artifacts remain local, nonempty, and intentionally
Git-ignored, with their identities bound by the completion envelopes and the
canonical decision's input receipts.

## Ender27 authority chain

| Stage | Identity |
| --- | --- |
| Reviewed source commit | `a23ed023f3d592b05cb5368ca68b8a84a83658d8` |
| Merged source checkpoint A | `f578fef884b1c38a0c7141cd65f7fe3f221b6c59` |
| Manifest-only commit B | `2d95e361dbc7f724a14835c3b4c491112e80f2bb` |
| Merged seal checkpoint M | `c63e0465426c580d6144bcc092e199bc2f1dbbe4` |
| Manifest Git blob | `4942af7b5576d465678e07ad004e329555c7cf0e` |
| Source PR | #23 |
| Manifest PR | #24 |
| Evidence PR | #27 |
| Evidence merge (terminal-evidence checkpoint) | `99df04ff07ca7cdc9b02bc0cd275d3ecb105a520` |
| Archival tag | `numerai-ender27-terminal` |
| Lifecycle issue | #22 (closed as completed) |

The sealed inventory contains exactly 31 governed source files, two external
Parquet identities, Python 3.13.14, and ten exact package versions.

## Frozen Ender27 question (as tested)

Ender27 tested a single preregistered midpoint between two completed target
residuals:

`r_0.5 = 0.5 * r_identity + 0.5 * r_grank`

The challenger changed only:

- `model.target_transform.benchmark_transform = tie_kept_rank_gaussian`
- `model.target_transform.benchmark_transform_strength = 0.5`

There was no strength sweep, alternate seed selection, adaptive rule, rescue
blend, architecture change, or threshold change. The historical scoring cohort
was exactly 768,362 OOF rows across 141 eras `0301`-`0861`, with recent-40
`0705`-`0861`. Those eras are consumed discovery evidence; even a pass would
have been a historical source gate, not fresh confirmation.

## Exact next action

There is no pending scientific action. Ender27 is terminal.

1. Repository maintenance is tracked separately in
   [Issue #26](https://github.com/Nspiroff/Numerai/issues/26) and requires its
   own bounded authorization; it grants no research authority.
2. Any successor modeling requires a newly named, separately frozen,
   preregistered hypothesis and fresh explicit user authorization. Consumed
   eras `0301`-`1021` must not be reused for selection; eras `1022`-`1230` are
   not a substitute confirmation cohort.
3. Deployment remains separately gated behind a successful, separately frozen
   prospective validation and explicit account-action authorization. No such
   authority exists.

## Research lineage

| Family | Outcome | Durable lesson |
| --- | --- | --- |
| Ender20 | No eligible deployment candidate | K64/500k residual TabM was the strongest tested architecture direction, but stability and deployment gates rejected every candidate. |
| Ender21 | Negative confirmation | The signal remained positive and stable but retained only 40.745% of discovery BMC versus the required 60%. |
| Ender22 | Operationally invalid | A pre-fit 3.17 GiB allocation failure prevented a valid cohort decision. |
| Ender23 | `NEGATIVE_SEED_INSTABILITY` | Window-78 improved recent BMC, but the model-seed replication and fixed ensemble failed stability guards. |
| Ender24 | Evaluator precondition failure | Training completed, but CRLF/LF authority fingerprints prevented a scientific decision. |
| Ender25 | `ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN` | EMA compressed seed variance but weakened mean signal and worsened seed-1337 risk. |
| Ender26 | `ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN` | Gaussian-rank target residualization improved aggregate BMC, but seed 1337 failed recent-window and drawdown guards. |
| Ender27 | `ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN` | Half-strength tempering kept the full-period BMC gain and repaired seed-1337 drawdown, and it compressed recent-window seed dispersion — but seed-1337 recent-40 BMC and the aggregate recent-40 uplift still failed, so the attenuation was insufficient where it mattered. |

The canonical navigation ledger is
[`numerai/agents/experiments/README.md`](numerai/agents/experiments/README.md).

## Repository map

- `AGENTS.md` — root instructions for coding agents and Numerai workflows
- `numerai/agents/AGENTS.md` — active research-framework instructions
- `numerai/agents/code/` — shared data, metrics, analysis, modeling, and export code
- `numerai/agents/baselines/` — baseline configurations
- `numerai/agents/experiments/` — one governed directory per research family
- `numerai/agents/tests/` — synthetic, custody, evaluator, and regression tests
- `numerai/agents/skills/` — experiment-design, implementation, reporting, and upload workflows
- `numerai/*.ipynb`, `signals/`, `crypto/`, `cached-pickles/` — inherited tutorial/example assets rather than active Ender authority

## Non-negotiable operating rules

1. Never silently retry a one-shot training or evaluator identity.
2. Never delete zero-byte or partial reservations; they are terminal evidence.
3. Never overwrite, rename, or reuse a terminal prediction, result, completion,
   or decision path.
4. Never modify a manifest-governed source packet after sealing without a new
   source-review and manifest lifecycle.
5. Never squash, rebase, or rewrite commits that manifests or receipts bind.
6. Never treat a historical source-gate pass as prospective confirmation.
7. Never deploy, upload, assign, stake, submit, create a model, or mutate the
   Numerai account without a separate explicit authorization.
8. Keep model implementation and experiment execution separate from account
   actions.

## Current GitHub hygiene posture

- The Ender27 terminal evidence is archived at merge
  `99df04ff07ca7cdc9b02bc0cd275d3ecb105a520`, tagged
  `numerai-ender27-terminal`.
- Issue #22 is closed as completed and records the full terminal evidence
  chain. It grants no successor-research authority.
- Issue #26 is the active repository-maintenance tracker (CI trigger
  modernization, branch review, labels, protection). Maintenance remains
  separately gated from research authority.
- The control-plane documents (this file, the root README status note, and
  the experiment ledger) are current through the Ender27 terminal archive;
  PR #25 is their publication vehicle and carries only ungoverned status
  documentation.
- Historical source/seal/evidence branches remain available as redundant
  review anchors. Branch cleanup should occur only after merge ancestry and
  archival tags are independently confirmed, under Issue #26.
- The inherited GitHub Actions workflow still targets `master` rather than
  `main`; CI modernization is repository maintenance under Issue #26.
- Branch protection is currently disabled on `main`. A future maintenance gate
  should add safeguards compatible with required merge-commit topology.

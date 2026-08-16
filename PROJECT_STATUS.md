# Numerai governed research status

**Last verified:** 2026-08-16  
**Live default-branch checkpoint:** `c63e0465426c580d6144bcc092e199bc2f1dbbe4`  
**Active lifecycle issue:** [#22 — Ender27](https://github.com/Nspiroff/Numerai/issues/22)

This document is a human-readable control-plane summary for the repository. It
does not replace any experiment's `experiment.md`, `gate.md`, manifest,
completion envelope, decision receipt, or terminal postmortem.

## Executive state

The repository contains a governed Numerai Classic research program centered
on benchmark-unique contribution against `v53_lgbm_ender20`.

No model is approved for deployment. No pickle upload, model assignment,
staking, submission, model creation, or Numerai account mutation is currently
authorized.

Ender20 through Ender26 are terminal research families. Ender27 is the only
active scientific gate.

## Current Ender27 state

Lifecycle state:

`ENDER27_ROUND1_FOUR_RUN_COHORT_COMPLETE_AWAITING_SEPARATE_EVALUATOR_AUTHORITY`

The exact four-run Round-1 cohort completed locally, once per component, in the
frozen order and under the sealed source/runtime/data custody packet:

| Run | Exit | Prediction bytes | Result bytes | Completion bytes |
| --- | ---: | ---: | ---: | ---: |
| `r1_control_rawresid_seed1337` | 0 | 15,779,944 | 6,170 | 1,455 |
| `r1_tempered_grank_resid_seed1337` | 0 | 15,772,440 | 6,374 | 1,471 |
| `r1_control_rawresid_seed2027` | 0 | 15,786,216 | 6,166 | 1,455 |
| `r1_tempered_grank_resid_seed2027` | 0 | 15,788,359 | 6,379 | 1,471 |

The twelve generated training artifacts are intentionally Git-ignored. No run
was retried. No output was deleted, repaired, redirected, or manually scored.

The canonical decision path remains absent:

`numerai/agents/experiments/ender27_tempered_gaussian_rank_residual_v53/receipts/round1_tempered_alignment.json`

Therefore Ender27 has no scientific verdict yet.

## Ender27 authority chain

| Stage | Identity |
| --- | --- |
| Reviewed source commit | `a23ed023f3d592b05cb5368ca68b8a84a83658d8` |
| Merged source checkpoint A | `f578fef884b1c38a0c7141cd65f7fe3f221b6c59` |
| Manifest-only commit B | `2d95e361dbc7f724a14835c3b4c491112e80f2bb` |
| Merged seal checkpoint M / `main` | `c63e0465426c580d6144bcc092e199bc2f1dbbe4` |
| Manifest Git blob | `4942af7b5576d465678e07ad004e329555c7cf0e` |
| Source PR | #23 |
| Manifest PR | #24 |
| Lifecycle issue | #22 |

The sealed inventory contains exactly 31 governed source files, two external
Parquet identities, Python 3.13.14, and ten exact package versions.

## Frozen Ender27 question

Ender27 tests a single preregistered midpoint between two completed target
residuals:

`r_0.5 = 0.5 * r_identity + 0.5 * r_grank`

The challenger changes only:

- `model.target_transform.benchmark_transform = tie_kept_rank_gaussian`
- `model.target_transform.benchmark_transform_strength = 0.5`

There is no strength sweep, alternate seed selection, adaptive rule, rescue
blend, architecture change, or threshold change.

The historical scoring cohort is exactly 768,362 OOF rows across 141 eras
`0301`-`0861`, with recent-40 `0705`-`0861`. Those eras are consumed discovery
evidence; even a pass would be a historical source gate, not fresh
confirmation.

## Exact next action

Do not launch the evaluator while another machine job may start Unreal Engine,
Blender, another trainer, or another substantial workload.

After the competing job finishes:

1. keep local HEAD and `origin/main` exactly at
   `c63e0465426c580d6144bcc092e199bc2f1dbbe4`;
2. perform a fresh read-only evaluator preflight;
3. verify the sealed source/runtime envelope, both governed Parquets, all twelve
   training artifact identities, resource posture, and decision-path absence;
4. obtain separate explicit user authorization for exactly one canonical
   evaluator invocation;
5. preserve the first truthful evaluator result or failure without retry;
6. review the scientific decision before publishing any terminal evidence or
   authoring any Round-2 proposal.

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
| Ender27 | Evaluator pending | Tests whether fixed half-strength attenuation preserves Ender26's gains while repairing its matched-seed failures. |

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

- `main` is intentionally pinned for the pending Ender27 evaluator.
- Issue #22 is the live active-gate record and is updated through the completed
  four-run cohort.
- There are no open pull requests on `main` at the pinned checkpoint.
- Historical source/seal/evidence branches remain available as redundant review
  anchors. Branch cleanup should occur only after merge ancestry and archival
  tags are independently confirmed.
- The inherited GitHub Actions workflow still targets `master` rather than
  `main`; CI modernization is repository maintenance, not part of the sealed
  Ender27 evaluator gate.
- Branch protection is currently disabled on `main`. A future maintenance gate
  should add safeguards compatible with required merge-commit topology.

## Maintenance work that must remain separate from Ender27

The following work is useful but must not change `main` before the evaluator:

- correct the inherited Actions trigger from `master` to `main`;
- add a lightweight source/test workflow that does not require local Numerai
  datasets or GPUs;
- add a repository description and topics;
- review historical branches for safe deletion after verifying merge/tag
  custody;
- add governance-oriented issue labels and milestones;
- consider branch protection that permits merge commits and blocks force-pushes;
- publish the final Ender27 decision, postmortem, ledger update, and archival
  tag only after the evaluator result is reviewed.

These items should be tracked and merged independently from scientific source,
manifest, execution, evaluator, and account-action gates.

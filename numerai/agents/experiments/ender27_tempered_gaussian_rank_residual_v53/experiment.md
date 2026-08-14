# Ender27 tempered Gaussian-rank benchmark-residual experiment

Status: **frozen source-only Round-1 candidate; no manifest, data access,
training, scoring, or evaluation is authorized by this document**.

## Research question

Can one fixed half-strength target-space blend preserve Ender26's aggregate
benchmark-unique BMC gain while avoiding its seed-1337 recent-window and
drawdown failures, without changing the proven window-78 Block-DRO TabM
procedure?

Ender27 changes only the target residual supplied to model fitting. The
control retains the legacy per-era linear residual to the raw
`v53_lgbm_ender20` benchmark. The challenger computes both that legacy
residual and Ender26's full tie-kept Gaussian-rank benchmark residual, then
uses their exact arithmetic midpoint. The fixed strength is `lambda=0.5`.

This is a new experiment family, not an Ender26 retry. It does not reuse an
Ender26 prediction, result, completion, or decision as an experiment input.
It does not select a transform strength, add seeds, relax a gate, or reopen an
earlier Round 2.

## Frozen target transform

For one era, define:

- `r_identity` as the established intercept-fitted linear residual of the
  target to the raw benchmark; and
- `r_grank` as the Ender26 target-side projection residual. If `y_c` is the
  centered target and `m = gaussian(tie_kept_rank(benchmark))`, then
  `r_grank = y_c - m * (m.T @ y_c) / (m.T @ m)`.

The challenger's complete scientific change is:

```text
r_0.5 = (1 - 0.5) * r_identity + 0.5 * r_grank
```

This is a blend of the two completed **target residuals**. It is not a blend
of the raw and Gaussian benchmark representations, and it is not the existing
`proportion` blend between an original target and one residual. The existing
`model.target_transform.proportion` remains `1.0`.

The challenger config expresses this as:

```text
model.target_transform.benchmark_transform = tie_kept_rank_gaussian
model.target_transform.benchmark_transform_strength = 0.5
```

The control omits both keys, preserving legacy identity behavior. The
challenger strength is fixed before any Ender27 data access. No other strength,
sweep, adaptive rule, per-era strength, per-seed strength, or post-hoc choice
is permitted.

Each Gaussian branch must retain Ender26's normative transform: average ranks
keep ties, each finite value maps to `(rank - 0.5) / n` independently within
its era, and the standard-normal inverse CDF is applied. The target is
centered, while the Gaussian benchmark is not recentered before projection.
The legacy branch retains its established intercept-fitted raw-benchmark
semantics. Both branches must restore original row order and fail closed
before fitting on malformed, missing, non-finite, constant, or otherwise
undefined Gaussian groups. Persisted predictions remain direct raw
`model.predict` output; no prediction transform or blend is permitted.

## Fixed procedure and discovery data

- Data version: v5.3.
- Inputs: the exact Ender21 discovery-only full and benchmark Parquets through
  era `0861`, identified in `protocol/discovery_data_authority.json`.
- Discovery allowlist: 176 every-fourth eras `0161`-`0861`.
- Features: the exact ordered 3,555-feature Ender21 authority list.
- Target: `target_ender_20`.
- Benchmark: `v53_lgbm_ender20`.
- Architecture: K64 TabM, three 512-wide ReLU blocks, dropout `0.1`.
- Loss: existing four-group chronological Block-DRO over row MSE.
- Optimizer: AdamW, learning rate `0.002`, weight decay `0.0003`.
- Training: at most 30 epochs, patience 4, batch size 1,024, AMP enabled.
- Outer CV: five expanding splits with a 13-retained-era embargo; the first
  empty-training fold is skipped.
- Training window: most recent 78 retained eras per usable outer fold.
- Inner validation: latest 10% of training eras with its own 13-era embargo;
  validation and early stopping retain ordinary MSE.
- Configured row cap: 500,000. It remains mechanically inactive under the
  frozen window-78 folds, so sample seed 1337 is fixed and is not replication
  evidence.
- EMA: disabled and absent from all configs.

The historical OOF evaluation geometry remains exactly 768,362 matched rows
across 141 eras `0301`-`0861`. Its recent window is the final 40 retained eras
`0705`-`0861`, with fixed ten-era blocks `0705`-`0741`, `0745`-`0781`,
`0785`-`0821`, and `0825`-`0861`.

Those eras are already consumed discovery evidence. An eventual Ender27
historical result could therefore be only a source-gate result, not fresh
confirmation. Eras `0865`-`1021` are consumed Ender21 confirmation evidence
and are forbidden for Ender27 selection, tuning, or reporting. Eras
`1022`-`1230` are not a substitute confirmation cohort. A historical pass
could support only a separately frozen prospective protocol over newly
resolved eras `1231`-`1282`.

## Round 1: exact matched two-seed cohort

| Run | Procedure | Model seed | Sample seed |
| --- | --- | ---: | ---: |
| `r1_control_rawresid_seed1337` | raw-benchmark residual control | 1337 | 1337 |
| `r1_tempered_grank_resid_seed1337` | fixed `lambda=0.5` tempered residual | 1337 | 1337 |
| `r1_control_rawresid_seed2027` | raw-benchmark residual control | 2027 | 1337 |
| `r1_tempered_grank_resid_seed2027` | fixed `lambda=0.5` tempered residual | 2027 | 1337 |

Within each matched pair, the challenger's only scientific config additions
are the two frozen transform keys above. `output.results_name` changes only to
give every future run a new canonical Ender27 identity. Model seeds 1337 and
2027 are fixed matched replications, not candidates from which a favorable
seed may be selected.

The four runs are one indivisible future cohort. No output identity has been
created or reserved by this scaffold. The source candidate implements the
fixed launch order, create-new destinations, opaque completion validation,
and one-shot evaluator law. Those controls must still be independently
reviewed and sealed in a separately authorized manifest before invocation.

## Frozen Round-1 decision law

The tempered procedure is source-gate eligible only if every condition below
holds:

1. its two-seed mean full BMC is at least the control mean plus `0.00020`;
2. its two-seed mean recent-40 BMC is at least the control mean plus `0.00030`;
3. at each matched seed, challenger recent-40 BMC is at least control;
4. at each matched seed, challenger full BMC retains at least 95% of control;
5. every used outer fold has strictly positive challenger BMC;
6. each challenger has at least three of four positive recent blocks and its
   worst recent block is strictly greater than `-0.001`;
7. each challenger BMC Sharpe is at least its matched control minus `0.05`;
8. each challenger max drawdown is no greater than matched control plus
   `0.01`;
9. each challenger Corr is in `[0.008, 0.04)`; and
10. each challenger's average correlation with the benchmark is below `0.25`.

A completed future scientific evaluation has exactly one of two states:

- `ENDER27_ROUND2_SOURCE_GATE_ELIGIBLE` when every check passes; or
- `ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN` otherwise.

The eligible state permits only a proposal for independently reviewed Round-2
source. It does not authorize Round-2 source creation, a Round-2 scaffold,
manifest creation, data access, output reservation, training, prediction
reading, target reading, scoring, evaluation, prospective validation,
deployment, upload, model creation, staking, submission, or account/API
action.

## Current negative authority

This directory may contain source-only runner, bootstrap, evaluator, and
custody candidates alongside the specifications, configs, and documentary
discovery authority. Their presence is not execution authority. It contains no
manifest, prediction, result, completion, decision, output directory, receipt
directory, or Round-2 source.

The generic target-transform implementation, synthetic tests, and all
Ender27-only orchestration and custody source must pass independent source
review. A later separate manifest-only seal and explicit user launch
authorization would still be required. This source candidate itself grants no
execution or account authority.

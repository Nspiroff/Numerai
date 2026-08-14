# Ender26 terminal postmortem

## Verdict

`ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN`

The sole authorized Ender26 Round-1 evaluator invocation completed
successfully with exit code `0` and published the canonical decision receipt.
The target-side Gaussian-rank benchmark residual improved mean full and
recent-40 BMC and materially reduced benchmark correlation, but the gain was
not robust at model seed 1337: recent-40 BMC fell and drawdown exceeded its
matched-control allowance. The frozen gate therefore rejects the procedure and
closes this family without Round 2.

## Frozen identity

- Family: `ender26_gaussian_rank_residual_v53`
- Stage: `ender26-round1-bmc-alignment`
- Source checkpoint A: `18ea8bbae1976fadf2574cce8c6c231c7fd2dc5d`
- Manifest-only child B: `dee1b1b21638edfb104f6516948d1d1aa315b79d`
- Merged launch base: `2797c9e5d2e8a78e9090fcef478679ea236c1d20`
- Live manifest: 5,504 bytes, SHA-256
  `942afcbdb23f246dba337cad16086c9a4c938b07abdb7bde6932c2b03a4c6909`
- Decision: [round1_bmc_alignment.json](receipts/round1_bmc_alignment.json)
- Decision size: 20,856 bytes
- Decision SHA-256:
  `29579cce359378853c0fa5d8bc3e155029bc1d8f1ac31a42bef1a8376d854feb`
- Decision Git blob: `9189f38c9b84d868a185727a1c8bc6d66523930d`

The [experiment](experiment.md), [gate](gate.md), manifest, configs, runner,
evaluator, and all other governed sources remain byte-identical to the exact
31-file source manifest. This terminal packet does not revise the scientific
protocol.

## Execution record

The user explicitly authorized the exact four-run Round-1 cohort and later the
single evaluator invocation under GitHub Issue `#18`. Each component used the
sealed Python 3.13.14 environment and a fresh external bytecode cache. All four
original training processes exited `0`; all completion and opaque output
identity checks passed before evaluation.

| Component | Completion receipt SHA-256 |
| --- | --- |
| `r1_control_rawresid_seed1337` | `c14b899e88332b7d1e6c87afc1f01b313ce14d71afed294eb84204c9431f2778` |
| `r1_grank_resid_seed1337` | `3e9850084b67730377037e8d32b5fffb8f2524d03b3cfba7d50f43e16169c316` |
| `r1_control_rawresid_seed2027` | `3159fe19966e83712592639f92dd46cff634ee93ca0ce31d40ff1d4a9081fceb` |
| `r1_grank_resid_seed2027` | `31e02a76e7ade198a34f9ddd8b09c4ac8474099d70456a7cce6cd1fd1539d77f` |

- Decision reservation: `2026-08-13T19:17:54.8649899-07:00`
- Decision finalization: `2026-08-13T19:19:36.0104957-07:00`
- Evaluator invocations: `1`
- Evaluator retry: `false`
- Process exit code: `0`
- Terminal output: `state=ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN`

The evaluator reserved the decision CREATE_NEW before governed reads, held all
source, input, completion, result, and prediction leases through decision
`fsync`, and validated all four completion envelopes before parsing any
scientific artifact.

## Exact run metrics

| Run | Full BMC | Recent-40 BMC | BMC Sharpe | Max drawdown | Corr | Benchmark corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control, seed 1337 | 0.006664912 | 0.008490224 | 0.629060 | 0.043614 | 0.010707 | 0.094266 |
| Gaussian-rank residual, seed 1337 | 0.007319795 | 0.007952763 | 0.675619 | 0.061982 | 0.010679 | 0.080104 |
| Control, seed 2027 | 0.004957941 | 0.005858684 | 0.440663 | 0.058255 | 0.013883 | 0.214082 |
| Gaussian-rank residual, seed 2027 | 0.006073291 | 0.007627567 | 0.538288 | 0.057594 | 0.010859 | 0.122511 |

Every run used the exact same 768,362 rows across 141 eras (`0301`-`0861`),
the same 2,866,944-parameter procedure, and the predeclared matched-pair config
delta only.

## Aggregate result

| Procedure | Mean full BMC | Worst-seed full BMC | Full seed gap | Mean recent-40 BMC | Worst-seed recent-40 BMC | Recent seed gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 0.005811426 | 0.004957941 | 0.001706971 | 0.007174454 | 0.005858684 | 0.002631540 |
| Gaussian-rank residual | 0.006696543 | 0.006073291 | 0.001246504 | 0.007790165 | 0.007627567 | 0.000325196 |

The aggregate full BMC gain was `+0.000885116` (`+15.23%`), and the aggregate
recent-40 gain was `+0.000615711` (`+8.58%`). Both aggregate improvement gates
passed. Average benchmark correlation fell from `0.154174` to `0.101307`, and
the two-seed gaps contracted by 26.98% full-period and 87.64% recent-40.

Seed 2027 passed every individual guard. Seed 1337 failed exactly two required
guards:

- recent-40 BMC was `0.007952762779041245`, below its matched control
  `0.008490224382548142` by `0.0005374616035068967`; and
- drawdown was `0.06198157371018209`, above the matched-control-plus-0.01
  ceiling `0.05361362545889927` by `0.008367948251282821`.

The scientific conclusion is therefore narrower than “no signal.” Full BMC
improved at both seeds, but the challenger did not preserve the predeclared
recent-performance and path-risk robustness at seed 1337. Aggregate gains
cannot override a required matched-seed failure.

## Authority and stopping rule

The decision records all of these as `false`:

- `round2_source_gate_eligible`
- `continuation_authorized`
- `round2_source_gate_authorized`
- `round2_authorized`
- `training_authorized`
- `scoring_authorized`
- `deployment_authorized`
- `submission_authorized`
- `account_actions_authorized`

Ender26 is terminal. Do not retry the evaluator, replace cohort members, add or
select seeds, loosen thresholds, change the transform/window/architecture, or
launch Round 2.

The smallest defensible next research hypothesis is a new Ender27 source-only
family with one preregistered fixed-strength partial Gaussianization treatment:
`lambda=0.5` between the identity and tie-kept rank-Gaussian benchmark
transforms. Architecture, features, training procedure, matched seeds, sample
seed, cohort, and gates should remain fixed. This mechanism is provisional—the
receipt does not prove that transform tails caused the drawdown—and no lambda
sweep or post-hoc tuning is admissible on the consumed discovery eras.

Any Ender27 work requires a new identity, protocol, reviewed source, manifest,
and explicit execution authorization. Eras `0301`-`0861` are consumed for
discovery selection, eras `0865`-`1021` remain consumed historical
confirmation, and eras `1022`-`1230` are not a substitute confirmation cohort.
Deployment remains a separate prospective-validation and account-action gate.

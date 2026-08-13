# Ender25 terminal postmortem

## Verdict

`ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN`

The sole authorized Ender25 evaluator invocation completed successfully with
exit code `0` and published the canonical decision receipt. EMA 0.995 reduced
the variation between the two seeds, but it did not preserve enough average
BMC and materially worsened the seed-1337 Sharpe and drawdown. The frozen gate
therefore rejects EMA and closes this branch without Round 2.

## Frozen identity

- Family: `ender25_ender24_evaluation_recovery_v53`
- Stage: `ender25-ender24-round1-evaluation-recovery`
- Source checkpoint A2: `cf6873623be4cc5826000ae4c214b553960091e4`
- Manifest-only child B2: `d6b4422e644d4602d29efafc923f659727e5bf67`
- Merged launch base: `95dd8b9ef37484ca531b9c4e7d3c7d3e1979e662`
- Recovery manifest: 11,279 bytes, SHA-256
  `d15583d454d0befbc7fa561fa8ba5360bbfb664502742c14acfed551b3ca9a7f`
- Decision: [ender24_round1_recovery_decision.json](receipts/ender24_round1_recovery_decision.json)
- Decision size: 21,104 bytes
- Decision SHA-256:
  `5e9317548d2bd83e79485b375ef49542ac07ae57db479b82d46f4fbc919238ea`
- Decision Git blob: `b60aaaa12f48ccc0abe8c708b634331930d43871`

The [experiment](experiment.md), [gate](gate.md), authority JSON, and evaluator
sources remain byte-identical to the 19-file manifest. This terminal packet
does not revise the scientific protocol.

## Execution record

The user explicitly authorized one recovery invocation under GitHub Issue
`#12`. Independent prelaunch checks passed the source, topology, runtime,
decision-absence, path, and size contracts. The evaluator ran once using the
sealed Python 3.13.14 environment and a fresh external bytecode cache.

- Decision creation: `2026-08-13T16:11:06Z`
- Decision finalization: `2026-08-13T16:13:23Z`
- Process exit code: `0`
- Terminal output:
  `state=ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN`
- `scientific_decision`: `true`
- `reused_training`: `true`
- `evaluator_retry`: `false`

An earlier PTY allocation request was rejected by the terminal service before
PowerShell, Python, the cache directory, or the decision reservation existed.
Read-only footprint checks proved that no evaluator process had started. The
subsequent non-PTY process was the sole evaluator invocation.

## Exact run metrics

| Run | Full BMC | Recent-40 BMC | BMC Sharpe | Max drawdown | Corr | Benchmark corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control, seed 1337 | 0.006664912 | 0.008490224 | 0.629060 | 0.043614 | 0.010707 | 0.094266 |
| EMA 0.995, seed 1337 | 0.005293690 | 0.006640162 | 0.441032 | 0.083817 | 0.013858 | 0.206289 |
| Control, seed 2027 | 0.004957941 | 0.005858684 | 0.440663 | 0.058255 | 0.013883 | 0.214082 |
| EMA 0.995, seed 2027 | 0.004938359 | 0.006559310 | 0.430940 | 0.061977 | 0.014220 | 0.230700 |

Both matched pairs used the exact same 768,362 rows across 141 eras
(`0301`-`0861`) with cohort SHA-256
`46b98f2474af29f13b0f39f9320782f4371580a0a82da1d480bf1743253665df`.

## Aggregate result

| Procedure | Mean full BMC | Worst-seed full BMC | Full seed gap | Mean recent-40 BMC | Worst-seed recent-40 BMC | Recent seed gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 0.005811426 | 0.004957941 | 0.001706971 | 0.007174454 | 0.005858684 | 0.002631540 |
| EMA 0.995 | 0.005116024 | 0.004938359 | 0.000355331 | 0.006599736 | 0.006559310 | 0.000080853 |

EMA compressed the full-period two-seed gap by 79.18% and the recent-40 gap
by 96.93%. That variance reduction was not sufficient:

- mean full BMC retention was 88.03%, below the required 95%;
- mean recent-40 BMC retention was 91.99%, below the control requirement;
- seed-1337 EMA Sharpe was 0.441032, below its matched-control floor of
  0.579060; and
- seed-1337 EMA drawdown was 0.083817, above its matched-control ceiling of
  0.053614.

All other aggregate and per-run checks passed. The scientific conclusion is
that EMA smoothing made the procedure more seed-consistent but weaker on the
mean signal and materially worse on one seed's risk profile. Variance
compression alone is not a promotion-worthy gain.

## Authority and stopping rule

The decision records all of these as `false`:

- `round2_source_gate_authorized`
- `round2_authorized`
- `training_authorized`
- `scoring_authorized`
- `deployment_authorized`
- `account_actions_authorized`

Ender25 and the EMA-0.995 branch are terminal. Do not rerun the evaluator,
retrain or replace cohort members, add seeds, change EMA decay or thresholds,
populate the superseded Ender24 decision, or launch Round 2.

The next admissible research is a new Ender26 family with a new hypothesis,
identity, protocol, source seal, and explicit execution authorization. It must
target stable benchmark-unique BMC rather than treating lower seed variance as
an objective by itself. Eras `0865`-`1021` remain consumed, and eras
`1022`-`1230` are not a substitute confirmation cohort. Deployment remains a
separate prospective-validation and account-authorization gate.

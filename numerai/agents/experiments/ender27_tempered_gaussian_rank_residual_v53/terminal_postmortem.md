# Ender27 terminal postmortem

## Verdict

`ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN`

The sealed one-shot evaluator was invoked exactly once under separate explicit
authorization, exited 0, and durably committed the valid canonical decision at
its CREATE_NEW-reserved path. It was not retried. The frozen Round-1 decision
law rejects the tempered procedure, and the Ender27 family is scientifically
terminal under this cohort.

## Frozen identity

| Item | Value |
| --- | --- |
| Family | `ender27_tempered_gaussian_rank_residual_v53` |
| Evaluator stage | `ender27-round1-tempered-alignment` |
| Reviewed source checkpoint A | `f578fef884b1c38a0c7141cd65f7fe3f221b6c59` |
| Manifest-only commit B | `2d95e361dbc7f724a14835c3b4c491112e80f2bb` |
| Merged seal checkpoint M | `c63e0465426c580d6144bcc092e199bc2f1dbbe4` |
| Manifest Git blob | `4942af7b5576d465678e07ad004e329555c7cf0e` |
| Decision path | `numerai/agents/experiments/ender27_tempered_gaussian_rank_residual_v53/receipts/round1_tempered_alignment.json` |
| Decision size | 21,670 bytes |
| Decision SHA-256 | `261ed2661dbc282388498ecd2fd8fec668a14ef78bf256b7549bbbdda73fc007` |
| Decision Git blob after commit | `b6613f93a8bea18c06c60b5f7e26c2e09e2f7748` |
| Evaluator invocation count | 1 |
| Evaluator exit code | 0 |

Schema version 1; `passed=false`; `round2_source_gate_eligible=false`; every
authority field (`continuation_authorized`, `round2_source_gate_authorized`,
`round2_authorized`, `training_authorized`, `scoring_authorized`,
`deployment_authorized`, `submission_authorized`,
`account_actions_authorized`) is false.

## Exact run metrics

All values below are full-precision canonical decision values. Every run used
the identical 2,866,944-parameter procedure over the exact frozen OOF cohort
of 768,362 rows across 141 eras `0301`-`0861` (recent-40 `0705`-`0861`).

| Canonical metric | `r1_control_rawresid_seed1337` | `r1_tempered_grank_resid_seed1337` | `r1_control_rawresid_seed2027` | `r1_tempered_grank_resid_seed2027` |
| --- | --- | --- | --- | --- |
| Full BMC mean | 0.006664911685985717 | 0.006972731243817692 | 0.004957941042692131 | 0.005330710267366409 |
| Full BMC std | 0.010595037010667135 | 0.01047182404535322 | 0.011251105987274234 | 0.011238405982085581 |
| BMC Sharpe | 0.6290597832999971 | 0.6658564175275443 | 0.4406625489351802 | 0.4743297471068184 |
| BMC max drawdown | 0.04361362545889927 | 0.04096116868058008 | 0.05825455222025591 | 0.053661116464899294 |
| Recent-40 BMC mean | 0.008490224382548142 | 0.006780431226718131 | 0.00585868436892766 | 0.006964410951418346 |
| Corr mean | 0.010706598203043503 | 0.012773026809066057 | 0.013883286525787852 | 0.012550920025478444 |
| Avg corr with benchmark | 0.09426613361228738 | 0.1247648464202894 | 0.2140816674471981 | 0.17226494371871448 |
| Block `0705-0741` BMC | 0.009938452293358671 | 0.007999829080576266 | 0.007807250744614878 | 0.009028261899060593 |
| Block `0745-0781` BMC | 0.011171348584415475 | 0.009524912018795252 | 0.007862119323628996 | 0.00875213151976173 |
| Block `0785-0821` BMC | 0.008998627719899554 | 0.00897299511444996 | 0.006613825144808717 | 0.008338133550552112 |
| Block `0825-0861` BMC | 0.0038524689325188655 | 0.0006239886930510439 | 0.0011515422626580515 | 0.001739116836298946 |
| Fold 1 BMC mean | 0.0025533472216259 | 0.004658868008810997 | 0.00027958917398765773 | 0.0005775469129503596 |
| Fold 2 BMC mean | 0.007086776409377295 | 0.006705889556553676 | 0.004430976311528405 | 0.005553672228045372 |
| Fold 3 BMC mean | 0.008564303749308676 | 0.009871547434434275 | 0.009647265664754285 | 0.00829624877801822 |
| Fold 4 BMC mean | 0.008405488594807518 | 0.006663456399592536 | 0.0054595999100035646 | 0.006851910292588203 |

## Procedure aggregates

Canonical two-seed aggregates from the decision:

| Aggregate | Control | Tempered Gaussian-rank residual |
| --- | --- | --- |
| Mean full BMC | 0.005811426364338923 | 0.00615172075559205 |
| Worst-seed full BMC | 0.004957941042692131 | 0.005330710267366409 |
| Full two-seed gap | 0.0017069706432935863 | 0.0016420209764512828 |
| Mean recent-40 BMC | 0.007174454375737901 | 0.006872421089068239 |
| Worst-seed recent-40 BMC | 0.00585868436892766 | 0.006780431226718131 |
| Recent two-seed gap | 0.0026315400136204814 | 0.00018397972470021468 |

The following values are arithmetic derived from the canonical decision values
above for reporting; they are not additional scoring:

| Derived quantity | Value |
| --- | --- |
| Mean full-BMC gain (tempered − control) | 0.0003402943912531268 (≈ +5.86%; exact 5.855608759689358%) |
| Mean recent-40 difference (tempered − control) | -0.00030203328666966213 (≈ −4.21%; exact -4.209843297506477%) |
| Full-period seed-gap change | -6.494966684230351e-05 |
| Recent-window seed-gap change | -0.0024475602889202667 |

## Frozen gate result

20 of 22 frozen checks passed. Exactly two failed:

1. aggregate `mean_recent40_bmc_at_least_control_plus_0_00030`
2. seed 1337 `matched_recent40_bmc_at_least_control`

Complete 22-check matrix (canonical booleans):

| Check | Outcome |
| --- | --- |
| aggregate `mean_full_bmc_at_least_control_plus_0_00020` | PASS |
| aggregate `mean_recent40_bmc_at_least_control_plus_0_00030` | FAIL |
| seed1337 `matched_recent40_bmc_at_least_control` | FAIL |
| seed1337 `full_bmc_retains_95pct_matched_control` | PASS |
| seed1337 `all_used_folds_bmc_positive` | PASS |
| seed1337 `three_of_four_recent_blocks_positive` | PASS |
| seed1337 `worst_recent_block_above_minus_0_001` | PASS |
| seed1337 `corr_at_least_0_008` | PASS |
| seed1337 `corr_below_0_04` | PASS |
| seed1337 `benchmark_corr_below_0_25` | PASS |
| seed1337 `sharpe_not_below_matched_control_minus_0_05` | PASS |
| seed1337 `drawdown_no_greater_than_matched_control_plus_0_01` | PASS |
| seed2027 `matched_recent40_bmc_at_least_control` | PASS |
| seed2027 `full_bmc_retains_95pct_matched_control` | PASS |
| seed2027 `all_used_folds_bmc_positive` | PASS |
| seed2027 `three_of_four_recent_blocks_positive` | PASS |
| seed2027 `worst_recent_block_above_minus_0_001` | PASS |
| seed2027 `corr_at_least_0_008` | PASS |
| seed2027 `corr_below_0_04` | PASS |
| seed2027 `benchmark_corr_below_0_25` | PASS |
| seed2027 `sharpe_not_below_matched_control_minus_0_05` | PASS |
| seed2027 `drawdown_no_greater_than_matched_control_plus_0_01` | PASS |

`passed=false` is exactly the conjunction of these 22 booleans, and the
terminal state follows mechanically from the frozen law.

## Interpretation

- Tempering retained a full-period BMC improvement: the mean full-BMC gate
  passed with a +0.0003402943912531268 gain over control.
- It repaired Ender26 seed 1337's matched drawdown failure: seed 1337's
  drawdown check passed (0.04096116868058008 against an allowance of matched
  control 0.04361362545889927 plus 0.01).
- It substantially compressed recent-window two-seed dispersion, from
  0.0026315400136204814 to 0.00018397972470021468.
- It did not repair seed 1337's recent-40 degradation: 0.006780431226718131
  remained below its matched control 0.008490224382548142.
- Aggregate recent-40 BMC fell instead of clearing the required uplift:
  0.006872421089068239 versus a floor of control 0.007174454375737901 plus
  0.00030.
- The final `0825-0861` block was the largest source of seed-1337 recent
  weakness (0.0006239886930510439 versus control 0.0038524689325188655).
- Because the frozen law requires every condition, the two failed
  recent-window checks are terminal.
- This is promising but non-robust negative evidence, not a no-signal result.

No claim is made that adjusting the blend strength would necessarily fix the
result, and no lambda sweep over the consumed discovery eras is recommended or
permitted; eras `0301`-`0861` are consumed evidence and any strength selection
on them would be post-hoc.

## Artifact and evaluator custody

- One evaluator invocation (detached task `bjovzs63w`, launcher PID 12968,
  bootstrap PID 35372); exit code 0; decision fsync
  2026-08-16 09:39:01.118 -07:00. No retry occurred.
- All twelve training artifacts retained their exact path, device, inode,
  byte size, and modification time through evaluation (0/12 changed).
- No independent raw-prediction rescore occurred; all scientific reads
  happened inside the sealed evaluator under held-handle custody.
- The decision was not altered after evaluator fsync; its identity and
  SHA-256 were re-verified before publication.
- The decision file was copied byte-exactly for publication (identical
  21,670-byte content, SHA-256
  `261ed2661dbc282388498ecd2fd8fec668a14ef78bf256b7549bbbdda73fc007`,
  binary comparison with zero differences) and is pinned `-text` in
  `.gitattributes` so Windows `core.autocrlf=true` checkouts preserve the
  evaluator-produced physical bytes.
- Generated predictions, results, and completion files remain outside Git;
  their identities stay bound by the committed-style completion envelopes and
  the canonical decision's input receipts.

## Authority and stopping rule

Every continuation and account authority in the canonical decision remains
false. Explicitly:

- Ender27 is terminal under the frozen Round-1 decision law.
- No evaluator retry is permitted.
- No alternate strength, additional seed, threshold relaxation, rescue blend,
  or Round 2 is authorized.
- No prospective validation, final fitting, pickle, deployment, upload,
  assignment, staking, submission, model creation, or account mutation is
  authorized.
- Any further modeling requires a newly named, separately frozen research
  hypothesis and fresh user authorization.
- Eras `0301`-`1021` remain consumed.
- Eras `1022`-`1230` are not a substitute confirmation cohort.

This postmortem records the terminal state only; it does not design or
authorize any successor family.

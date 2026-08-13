# Ender22 execution gate

## Negative authority

- This scaffold does not authorize training, target reads, prediction reads,
  scoring, confirmation, upload, model creation, staking, or API/account change.
- Eras 0865-1021 are consumed Ender21 confirmation evidence and are forbidden
  for every Ender22 action. Eras 1022-1230 are not an alternate confirmation.
- The only possible Ender22 confirmation is prospective resolved data for
  consecutive eras 1231-1282, under a separately frozen runner and manifest.
- All experiment decisions are create-new receipts. Existing decisions are
  never overwritten or recomputed under the same path.

## Before Round 1 training

1. Review and commit this protocol, the three Round-1 configs, both possible
   Round-2 pairs, and both evaluators without reading target or prediction data.
2. Create and commit `source_manifest_round1.json`. It must bind the exact Git
   source commit, runtime, all protocol/config/evaluator files, the reused
   Ender21 allowlist and feature list, and both physical discovery Parquets from
   `protocol/discovery_data_authority.json`.
3. Launch each exact config only through `run_round1.py`, under isolated
   Python `-I -B`
   and a fresh external `-X pycache_prefix`. The stdlib bootstrap verifies and
   leases the committed manifest, every governed source, and both physical
   inputs in one fresh isolated bootstrap process, then invokes the pipeline
   in-process while those leases remain held. Invoking the generic CLI directly
   is not authorized.
4. Run A, B, and C exactly once. Each run must produce its normal prediction
   and result plus a create-new completion receipt binding the source manifest,
   config hash, and finalized output identities and hashes.
5. Run `evaluate_round1.py` exactly once to create
   `receipts/round1_discovery.json`. Review the terminal state before any Round
   2 activity.

## Before Round 2 training

1. Round 1 must name exactly B or C with state `SCOUT_WINNER`.
2. Create and commit `source_manifest_round2.json`, binding the immutable Round
   1 receipt and only the exact model-seed-2027 and sample-seed-2027 configs for
   the selected family. The other predeclared pair remains unauthorized.
3. Use `run_round2.py` with the same fail-closed isolated launch, input, and
   completion-receipt controls as Round 1.
4. Run the two selected-family replications once, then run
   `evaluate_round2.py` once to create
   `receipts/round2_seed_replication.json`.

## Stop conditions

Stop immediately on a missing/changed artifact; non-clean or non-committed
manifest source; mismatched runtime/config/result/provenance; output overwrite;
non-finite values; duplicate or mismatched IDs; an era outside 0301-0861 in OOF
predictions; any attempt to read 0865 or later historical data; unexpected CV
geometry; an unselected Round-2 config; or a terminal negative state.

The exact metric rules and ranking are normative in `experiment.md` and are
implemented independently in the evaluators. A historical pass still requires
the prospective 1231-1282 validation and carries no production authority.

# Ender22 Round 1 execution postmortem

## Outcome

Ender22 Round 1 produced no valid experiment decision. Round 2 is not
authorized.

The frozen control and window-78 procedures completed and finalized their
artifacts. The half-life-52 procedure consumed its one-shot output reservations
and then failed before its first model fit with a NumPy allocation error. Its
three zero-byte files are preserved as terminal failure evidence. The failed
procedure was not retried, renamed, deleted, or substituted.

The Round-1 evaluator was deliberately not executed. Its contract requires the
exact three-candidate cohort, and the half-life completion is invalid. Running
it would also load discovery truth before reaching that invalid completion,
creating partial scored knowledge from an inadmissible cohort. Consequently:

- `receipts/round1_discovery.json` remains absent;
- no candidate was selected;
- no stored candidate metrics were used to select, tune, or report a model;
- no Round-2 manifest, training, or evaluation is authorized;
- no Numerai account, upload, model assignment, or staking action occurred.

## Frozen authority

- Launch-wrapper repair commit: `c08fc77dc4f28ae61d402eea4aaf9e01c4191512`
- Round-1 manifest-only seal: `211e70ab716b95f319e1d7df4eed864d6ec4768c`
- `source_manifest_round1.json` SHA-256:
  `c90428ead39286251b66f6c94287197a774d8da68882021c14d2e23f6ac59845`
- Remote branch matched the local seal before execution.
- Selection inputs were limited to the frozen discovery extracts through era
  `0861`; consumed Ender21 confirmation eras `0865`-`1021` and protected later
  cohorts were not used.

Before the sealed execution, a Windows `os.execv` wrapper returned success
without launching a child. It created no artifacts and opened no governed data.
The wrappers were changed to blocking `subprocess.run`, tested with a real
invalid invocation, committed, and resealed before any actual Round-1 fit.

## One-shot executions

### A — `r1_control_block_dro`

Completed and finalized. Completion receipt SHA-256:
`d98cf6ad51e1d9b1ff1a6377fd5db2bc1230b1619c386b69a9b9e5a567c26cf8`.
The completion exact-binds a nonempty 15,779,944-byte prediction artifact and a
nonempty 5,556-byte result artifact. Their live size, device/inode, and SHA-256
were revalidated after the process exited.

### B — `r1_recent_half_life52`

Failed before its first model fit while applying the required benchmark-coverage
filter to the eager 3,555-feature discovery frame:

```text
numpy._core._exceptions._ArrayMemoryError:
Unable to allocate 3.17 GiB for an array with shape (3555, 957366)
and data type int8
```

The canonical prediction, result, and completion paths are each zero-byte,
regular, non-link files with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
They remain in place as the no-rerun evidence required by the frozen gate.
The captured failure log is `logs/r1_recent_half_life52.failure.log`, SHA-256
`23c937f1935f5ab26d26afdbd3d45882cc59a07762ef7d37c64e149eaba92e5e`.

### C — `r1_recent_window78`

Completed and finalized. Completion receipt SHA-256:
`9fcaa4efdb1b5fa2013bab8dab2a5a22f38eade13927ebef6f9199a7e0cdb37e`.
The completion exact-binds a nonempty 15,779,944-byte prediction artifact and a
nonempty 6,147-byte result artifact. Their live size, device/inode, and SHA-256
were revalidated after the process exited.

## Classification and next admissible work

This is an operationally invalid Round-1 cohort, not
`NEGATIVE_NO_TEMPORAL_RETENTION_GAIN` and not `SCOUT_WINNER`; neither frozen
decision state can be computed without all three finalized candidates.

Any future attempt must be a newly named, separately frozen experiment. It may
repair eager-memory materialization before execution, but it must not reuse
these one-shot paths, treat A or C as a winner, inspect this invalid cohort to
choose a replacement procedure, or use eras beyond the discovery boundary for
selection.

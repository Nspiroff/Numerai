# Ender25 source gate

## Verdict

`SOURCE_ONLY_NOT_SEALED_NOT_AUTHORIZED`

This gate freezes a reviewable recovery design. It does not create the future
source manifest and does not authorize execution.

## Fixed scope

The only future scientific operation in scope is one evaluation of the exact
four-run Ender24 Round-1 cohort under the unchanged Ender24 matched-pair scoring
and decision law. Training is complete and must not be repeated. The new
Ender25 evaluator repairs only the authority layer that previously confused
physical CRLF bytes with canonical LF bytes.

The frozen machine-readable authority is
`protocol/ender24_input_authority.json`. Any conflict between implementation
defaults and that authority must fail closed.

## Conditions for a later manifest-only seal

Before `source_manifest_evaluation_recovery.json` may be created, an independent
review must establish all of the following:

- source commit A contains the complete implementation, synthetic regression
  tests, documents, authority JSON, and `receipts/.gitkeep`;
- every runtime-loaded source is included in the finalized manifest source set;
- the bootstrap begins in isolated mode, reserves the new decision path with
  `CREATE_NEW`, and acquires source custody before importing governed code;
- the old Ender24 decision path is proved absent and is never created, replaced,
  truncated, or renamed;
- all four completion envelopes are leased and preflighted before any
  prediction, result, or truth payload is parsed;
- raw-byte manifest identity, strict text normalization, canonical identity,
  and exact JSON semantics are separate mandatory checks;
- preserved input and source leases remain held through decision-file
  publication and file `fsync` on the original create-new handle;
- failure cleanup cannot delete or alter pre-existing evidence;
- synthetic tests cover LF and CRLF equivalence plus fail-closed mutation,
  malformed-newline, BOM, wrong-raw-receipt, custody, ordering, and output-path
  cases; and
- no code path can train, launch Round 2, submit, deploy, or access a Numerai
  account.

The later seal must have exactly this topology:

- commit A: reviewed source implementation;
- commit B: direct child of A;
- commit B diff: only
  `source_manifest_evaluation_recovery.json`; and
- manifest `git_head`: exactly commit A.

Creating B and launching from B are separate authorities. Neither is granted by
this document.

## Conditions for a later one-shot launch

A launch requires all of the following after the manifest-only child is merged
or otherwise frozen:

1. independent acceptance of commit A, commit B, their parent relationship, and
   the manifest-only diff;
2. verification that the new decision path and old Ender24 decision path are
   both absent;
3. explicit user authorization naming this Ender25 recovery launch;
4. exactly one isolated evaluator invocation; and
5. no retry after success or failure.

An infrastructure, custody, authority, parsing, or publication failure is not a
scientific negative. It stops the gate without promoting either scientific
state. The first truthful failure must be preserved.

## Scientific terminal states

After—and only after—a complete scoring pass, the evaluator may atomically
publish one of:

- `ENDER25_ROUND2_SOURCE_GATE_AUTHORIZED`
- `ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN`

`ENDER25_ROUND2_SOURCE_GATE_AUTHORIZED` grants permission only to author and
review a separately sealed Round 2 source gate. Its authority flags for source
manifest creation, data access, training, Round 2 execution, submission,
deployment, and Numerai account actions remain false. A later explicit launch
authorization is mandatory.

`ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN` terminates this EMA branch for the
frozen cohort. It does not authorize threshold changes, new seeds, replacement
runs, or another evaluator attempt.

## Current prohibited actions

Until the later sequence is separately authorized, do not:

- create `source_manifest_evaluation_recovery.json`;
- open, hash, parse, copy, or score Ender24 predictions, results, completions,
  postmortem evidence, external Parquets, or other Numerai data;
- run the recovery evaluator, an old Ender24 evaluator, or any training command;
- create either the old or new decision receipt;
- prepare or launch Round 2;
- submit predictions, deploy a model, or perform Numerai account actions; or
- infer a scientific decision from the Ender24 infrastructure failure.

Synthetic tests over temporary, invented fixtures are allowed during source
review. Passing those tests proves only source behavior; it is not evidence that
the real recovery evaluation ran or that either scientific terminal state was
reached.

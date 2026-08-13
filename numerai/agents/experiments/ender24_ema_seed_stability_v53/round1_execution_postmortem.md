# Ender24 Round 1 execution postmortem

## Outcome

Ender24 Round 1 produced no scientific model decision. All four frozen
training components completed once and finalized their canonical prediction,
result, and completion artifacts. The sole frozen evaluator invocation then
stopped before truth or result parsing with:

```text
ValueError: Physical authority differs for era_allowlist
```

The durable documentary classification is
`STOPPED_AT_ENDER24_ROUND1_EVALUATOR_PRECONDITION_FAILURE`. This is not a third
evaluator state: the decision state is absent, the metric outcome is
unevaluated, and Round 2 is not authorized. In particular, neither
`ROUND2_AUTHORIZED` nor `NEGATIVE_NO_EMA_STABILITY_GAIN` was produced.

The create-new decision reservation was rolled back after the exception, so
`receipts/round1_ema_stability.json` remains absent. Its absence does not grant
retry authority. The evaluator was not retried, no original output was
deleted or replaced, and no metric was manually opened or interpreted.

## Frozen authority

- Launch issue: [#9](https://github.com/Nspiroff/Numerai/issues/9).
- Source merge: `aebc577249d202ab9f32e4dac2bc939f496a6ddc`.
- Mechanical receipt-only commit:
  `a2bfe0fce7ac1a6a6b075a65b8538aa32165e3c6`.
- Manifest-only commit: `5a1a75d1b00f639fb04a522dda6c390d5535732f`.
- Seal merge: `789a91f`.
- Launch head: `7adc6724bd41689e34e8d21effa088b0ff606022`.
- `source_manifest_round1.json` SHA-256:
  `bd55280e4a99a1b45be87cc5af73aea2615da14a4de0e0662d1ac6c008ab1b35`.
- The manifest binds the exact 31 governed files, the frozen runtime, and the
  two discovery Parquets through era `0861`.

## One-shot training evidence

The four original launchers ran serially in the frozen order and each exited
zero. All 12 outputs remain nonempty, single-link regular files at their exact
canonical paths. The completion envelopes bind their prediction and result
paths, sizes, file identities, and hashes.

| Component | Prediction bytes | Result bytes | Completion bytes | Completion SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `r1_control_seed1337` | 15,779,944 | 6,148 | 1,370 | `5a38b9f7211155b7ce9ea71db8c6815a72940f8c999c53e7b2f0ef6d4bd65b4e` |
| `r1_ema995_seed1337` | 15,786,841 | 7,678 | 1,365 | `0b6d500c571abe376ae388ace4abc56430ac8df9c5bce71be7a41657a149431f` |
| `r1_control_seed2027` | 15,786,216 | 6,144 | 1,369 | `f33f40f537413fdd8fc80cb7d656192fbf02d906b6885723894e9fc1776653e4` |
| `r1_ema995_seed2027` | 15,787,673 | 7,673 | 1,363 | `c63793b84ef0e3fa03a9dc1a8c1aa167a1fb6a29863313e82259301f12f512ed` |

The successful prediction Parquets remain outside Git. Their identities and
SHA-256 values are retained by the committed completion envelopes.

## Evaluator failure and root cause

The evaluator first validated the four completion envelopes and their opaque
artifact bindings. It then failed in `load_authority`, before loading discovery
truth, parsing result metrics, or scoring predictions.

The failure was a deterministic text-byte portability defect. The evaluator
and discovery authority recorded canonical LF identities for two reused JSON
lists, while the sealed Windows checkout used CRLF physical bytes under
`core.autocrlf=true`. The source manifest correctly bound those physical CRLF
bytes, but the evaluator compared them against the incompatible LF literals.

| Authority file | Canonical LF identity | Sealed Windows identity |
| --- | --- | --- |
| discovery era allowlist | 1,763 bytes, `be0c212a8e910f56dbdae4e1e134fa36ce7e5e1a95e43faa1ccc9e6330f544ca` | 1,941 bytes, `4ffd0ef68092d935c121b45c83a89ef67afe832b48fc259e425d3fe3f51deae7` |
| feature-column list | 148,179 bytes, `663184191e17d2fa4fac6dae017890f0e762368e638d46cfaa489297b9b2049b` | 151,736 bytes, `e4df25383aff5ddf9446df275f55a8a93ca64f926a842f4cf84a68280adf769d` |

Normalizing the physical text to LF reproduces the canonical hashes exactly.
This establishes a source-portability defect rather than model, data, or
artifact drift. The evaluator stopped on the allowlist first; the feature list
contained the same latent mismatch.

## Classification and next admissible work

Ender24 is closed without a scientific decision. Do not retry its evaluator,
rerun any component, populate its old decision path, infer a winner from the
stored results, or launch Round 2.

Any recovery must use a separately named experiment, new evaluator stage,
new create-only decision path, independently reviewed source, and a new
manifest-only seal. That gate may explicitly consume the immutable Ender24
manifest, four completion envelopes, and their bound artifacts, so training
does not need to be repeated. It must compare textual authority canonically
across LF and CRLF while continuing to bind raw physical bytes through the
manifest, and it requires fresh user authorization before execution.

No upload, model assignment, submission, staking, deployment, or Numerai
account mutation occurred or is authorized by this record.

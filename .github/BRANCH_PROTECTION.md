# Main branch protection policy

> **Classic branch protection is enabled on `main`.** The exact live
> configuration and enforcement record are documented below (PE30,
> 2026-08-17). Protection was applied through the GitHub REST API under
> separate explicit authority *before* the branch carrying this revision
> was created; the pull request that publishes this revision only records
> that state — it did not apply protection itself. Modifying or removing
> protection still requires separate explicit user authorization under the
> emergency process below.

## Historical readiness baseline (BP26, 2026-08-16 — pre-enforcement)

Everything in this section is a **historical snapshot** preserved unchanged
from the BP26 readiness gate. It describes the repository **before** PE30
applied protection on 2026-08-17 and is deliberately not updated. As
verified against the live repository on 2026-08-16:

- Current `main` checkpoint: `95721aec64e810d16b7ca7e4c896e08ad9ae91ea`
  (the merge commit of CI maintenance PR #28; parents
  `06a4effe65594d1aabfb0d1f0303e360d37e4382` and
  `6c08d401569d27e7181f41911829210441fc0b9f`).
- Phase 1 CI modernization is merged and green: post-merge push run
  `31971144661` on `main` succeeded (Ubuntu compiled 199 governed files and
  passed 38/38 portable tests; Windows passed 38/38 archived custody tests).
- Branch protection on `main`: **disabled**
  (`GET /repos/Nspiroff/Numerai/branches/main/protection` returns
  404 "Branch not protected").
- Repository rulesets: **none** (`GET /repos/Nspiroff/Numerai/rulesets`
  returns an empty list).
- Required status checks: **none configured**.
- Direct collaborators: exactly one — `Nspiroff`, with admin access.
- The exact two check contexts emitted on `main`'s head, both produced by
  the **GitHub Actions app (app ID `15368`, slug `github-actions`)**:
  1. `Portable compile and source-contract tests` (ubuntu-24.04)
  2. `Windows terminal evaluator custody contracts` (windows-2022)

## Why readiness was required first (historical rationale)

1. **Skipped required workflows remain pending.** GitHub documents that a
   workflow skipped by path filtering, branch filtering, or a commit message
   leaves its associated checks in a "Pending" state, and "a pull request
   that requires those checks to be successful will be blocked from
   merging." Before BP26, `.github/workflows/research-source-ci.yml` used
   `paths` filters on its `pull_request` trigger, so documentation-only,
   one-file manifest, and terminal-evidence PRs would never emit the two
   contexts. Making those contexts required in that state would deadlock
   every out-of-scope PR. The pull-request path filters therefore had to be
   removed (BP26) before the checks can ever be required. Sources:
   - <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/troubleshooting-required-status-checks>
     ("Handling skipped but required checks")
   - <https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax>
     (`on.<event>.paths` semantics)
2. **The sole collaborator cannot approve their own PR.** GitHub documents
   that "Pull request authors cannot approve their own pull requests"
   (<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/approving-a-pull-request-with-required-reviews>).
   With exactly one collaborator, requiring even one approving review now
   would deadlock every normal merge. The required approving review count
   must therefore remain **0** until a genuinely distinct collaborator with
   suitable access exists.

## Classic branch-protection configuration (reviewed BP26, applied by PE30)

Target: classic branch protection (not a ruleset) on branch `main`.

Endpoint (originally validated by BP26 against the 2022-11-28 REST
documentation,
<https://docs.github.com/en/rest/branches/branch-protection?apiVersion=2022-11-28#update-branch-protection>,
and revalidated immediately before enforcement against the then-current
supported API versions — `2026-03-10` (latest) and `2022-11-28` — with
every field below accepted, unchanged, in both):

```
PUT /repos/Nspiroff/Numerai/branches/main/protection
Accept: application/vnd.github+json
X-GitHub-Api-Version: 2026-03-10
```

Payload (exact JSON; this is the payload PE30 applied verbatim on
2026-08-17 — see the live enforcement record below; every field is an
accepted parameter of the endpoint above and no unsupported fields are
included):

```json
{
  "required_status_checks": {
    "strict": false,
    "checks": [
      {
        "context": "Portable compile and source-contract tests",
        "app_id": 15368
      },
      {
        "context": "Windows terminal evaluator custody contracts",
        "app_id": 15368
      }
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": false
}
```

Field-by-field intent:

- `required_status_checks.checks` — require **both exact app-bound
  contexts**, each bound to GitHub Actions app ID `15368` via `app_id`. The
  documented `app_id` semantics: omitting it auto-selects the app that
  recently provided the check, and `-1` allows any app. Neither loose form
  is used: the contexts are explicitly bound so a same-named status from any
  other app cannot satisfy the requirement. The deprecated `contexts` array
  is deliberately not used; `checks` is its documented replacement.
- `required_status_checks.strict: false` — do **not** require branches to be
  up to date with `main` before merging (see "Merge-topology law").
- `enforce_admins: true` — protection applies to administrators too; the
  sole admin collaborator gains no silent bypass.
- `required_pull_request_reviews` present with
  `required_approving_review_count: 0` — pull requests are required before
  merging, but zero approving reviews are required (the REST documentation
  explicitly allows "0 to not require reviewers"). No code-owner review, no
  last-push approval, no stale-review dismissal.
- `restrictions: null` — no push restrictions (user/team/app allowlists).
- `required_linear_history: false` — merge commits remain permitted; GitHub
  documents that enforcing linear history "prevents collaborators from
  pushing merge commits to the branch", which would break the custody
  topology.
- `allow_force_pushes: false` and `allow_deletions: false` — force pushes
  and deletion of `main` are blocked.
- `block_creations: false` — irrelevant to the literal branch `main`
  (already existing); left at the default.
- `required_conversation_resolution: true` — all PR conversations must be
  resolved before merging.
- `lock_branch: false` — `main` is not locked and stays writable through
  pull requests.
- `allow_fork_syncing: false` — default; only meaningful for locked
  branches.
- Not required signed commits, no deployment-environment requirement, no
  merge queue (both are unavailable in classic protection payloads and are
  not wanted), and no bypass allowances of any kind.

## Documentation-only canary record (DC29, 2026-08-17)

- Canary PR: **#30** — `test(ci): documentation-only required-check canary`
  — final state **closed, draft, unmerged** (`merged_at: null`).
- Head: `cbf20e44a481d341a0be5cf85681f0d326af7a3f`; base: `main` at
  `7eae9e4850782bbc5362834f65b28fb1cd6c2818`.
- Only changed path: `BRANCH_PROTECTION_CANARY.md` (root level) — outside
  every former pull-request path filter (`numerai/agents/**`,
  `.github/workflows/research-source-ci.yml`, `.github/ci/**`), so the
  pre-BP26 workflow would have skipped it entirely.
- Workflow run `31981489705` (`pull_request`, attempt 1) → **success**:
  - `Portable compile and source-contract tests` — job `95248932726`,
    ubuntu-24.04: compiled 199 files, 38/38 tests OK;
  - `Windows terminal evaluator custody contracts` — job `95248932710`,
    windows-2022: 38/38 tests OK.
- Both checks were emitted by GitHub Actions app ID `15368` with
  byte-identical names on the exact canary head.
- The temporary canary branch and worktree were deleted after success; the
  closed PR and its Actions run remain preserved as evidence.
- Conclusion: `DOCS_ONLY_PR_EMITS_BOTH_REQUIRED_CHECK_CANDIDATES`.

## Live enforcement record (PE30, 2026-08-17)

- Gate: **PE30**, under separate explicit user authorization obtained after
  the DC29 canary.
- Enforcement time: PUT submitted **2026-08-17T01:19:58Z** (UTC); HTTP 200
  received by 2026-08-17T01:19:59Z; local machine time
  2026-08-16 18:19:59 PDT.
- `main` at enforcement: `7eae9e4850782bbc5362834f65b28fb1cd6c2818`
  (enforcement moved no ref).
- Endpoint: `PUT /repos/Nspiroff/Numerai/branches/main/protection` with
  `Accept: application/vnd.github+json` and
  `X-GitHub-Api-Version: 2026-03-10` — the latest supported version at
  enforcement time (live supported set: `2026-03-10`, `2022-11-28`), and
  the version the server confirmed via
  `X-Github-Api-Version-Selected: 2026-03-10`.
- Request payload: the exact JSON above — 767 bytes, SHA-256
  `1b827801123342d2fe46616c00789c071808d0960cffd0dabd06f7c49c159c31`.
- Response: **HTTP/2.0 200 OK**; the captured raw response (status line,
  headers, and body) is 2,823 bytes, SHA-256
  `7f851791a0dddda7a680fe7d386351af8ddc1ca9d49504af9fb65b196e7936b5`.
  Exactly one PUT was submitted — nothing was retried, configured
  piecemeal, or applied through the web UI, and no ruleset was created.
- Immediate read-back verification — **29/29 live fields match** the
  payload:
  - required status checks: `strict: false`; exactly two checks —
    `Portable compile and source-contract tests` and
    `Windows terminal evaluator custody contracts` — both `app_id: 15368`,
    no third check;
  - `enforce_admins: true`; required pull-request flow enabled with
    `required_approving_review_count: 0`; stale-review dismissal off;
    code-owner reviews off; last-push approval off; no dismissal
    restrictions; no bypass allowances;
  - no push restrictions (`restrictions` null); linear history not
    required; force pushes blocked; deletions blocked; branch-creation
    blocking off; conversation resolution required; branch not locked;
    fork syncing off;
  - `main.protected: true`; **no repository ruleset**; merge methods
    unchanged (merge, squash, and rebase all still enabled; automatic
    branch deletion still disabled); exactly one direct collaborator
    (`Nspiroff`, admin); `main` still at
    `7eae9e4850782bbc5362834f65b28fb1cd6c2818`.
- **No direct-push probe was performed.** Direct-push blocking is verified
  through the live configuration (required pull-request flow with
  `enforce_admins: true` and no push restrictions), not by an empirical
  push attempt — a genuine negative probe could mutate `main` if protection
  were misconfigured. Any mutating direct-push test remains separately
  gated.
- Authority boundaries: this protection is repository governance, not
  scientific authority. Passing required checks grants no training,
  scoring, confirmation, deployment, or Numerai account authority. Changing
  required checks, approval counts, app IDs, strictness, admin enforcement,
  or bypass posture requires separate explicit authority; protection
  removal requires the recorded emergency process above, including
  before/after configuration capture and restoration verification.

## Merge-topology law

- Source, manifest, and terminal-evidence packets in this repository use
  **merge commits**; manifests, receipts, and evidence records bind exact
  merge-parent identities.
- Squash and rebase merges remain **prohibited by task authority** for
  governed packets, regardless of what repository settings technically
  allow.
- Protection must **not** force branch updating: `strict: false` and
  `required_linear_history: false` are deliberate and load-bearing.
- The orchestrator preflight pins exact base and head identities before any
  governed merge. If `main` moves unexpectedly, the gate **stops** rather
  than rebasing, updating, or fast-forwarding the branch.
- `strict: false` does **not** waive that identity preflight; it only keeps
  GitHub from demanding a mechanical branch update that would rewrite
  governed topology.

## Review law for a one-person repository

- Independent Claude/ChatGPT review remains **required by orchestration**
  for every governed merge. It is recorded in PR reviews, PR comments, and
  written reports.
- That AI review is orchestration evidence, **not** a second GitHub approval
  identity: reviews posted through the owner's own account cannot satisfy an
  approving-review requirement, because pull request authors cannot approve
  their own pull requests.
- The GitHub approving-review count therefore remains **0** until a
  genuinely distinct authorized collaborator with suitable access exists.
- Adding a collaborator, or raising the approving-review count, is a
  separate decision requiring its own explicit authority; this document does
  not authorize either.

## Required-check exception process

Required checks must never be bypassed silently. For every source,
manifest, terminal-evidence, documentation, and maintenance PR:

- both contexts must appear on the PR head;
- both must complete successfully;
- a **missing** check is a blocker;
- a **renamed** check is a blocker;
- a check emitted by the **wrong app** (any app other than GitHub Actions
  app ID `15368`) is a blocker;
- a **failed** check is preserved and investigated — never deleted,
  re-run-to-green without diagnosis, or papered over;
- **no admin bypass is authorized by this document**, and `enforce_admins:
  true` is live precisely so none exists mechanically.

Emergency modification or removal of branch protection requires all of:

1. explicit user authorization;
2. a recorded before/after configuration (the live protection JSON captured
   both before and after the change);
3. a recorded reason;
4. restoration verification (the protection re-queried and compared
   field-by-field once the emergency is over).

## Canary and enforcement sequence

Status as of PE30 (2026-08-17):

**Completed:**

1. The BP26 readiness PR (#29) was independently reviewed (review
   4947425481 at head `cabf45a6273470ebc15a1bdf7bbf679c5b8f5e6f`) and
   merged with a merge commit (`7eae9e4850782bbc5362834f65b28fb1cd6c2818`)
   — RM29.
2. A temporary documentation-only canary PR (#30) was created against
   `main` — DC29.
3. Both exact contexts appeared and passed on the canary (run
   `31981489705`) despite no governed-source change.
4. The canary was closed without merging and only its temporary branch and
   worktree were deleted, under explicit cleanup authority.
5. The payload above was independently reviewed again and the current
   GitHub API schema was revalidated immediately before enforcement
   (supported versions `2026-03-10` and `2022-11-28`; every authorized
   field unchanged).
6. Explicit user authority to apply protection was obtained (PE30).
7. The protection was applied to `main` (a single PUT, HTTP 200).
8. The live protection was queried and every field compared to the approved
   payload — 29/29 match (see the live enforcement record above).

**Current:**

9. The first protected-flow PR — the pull request carrying this revision —
   was created after protection went live and changes only this file.
10. Both required checks must appear and pass on it.
11. It remains draft and awaits independent review before merge; only after
    that merge and a post-merge re-verification are the two checks treated
    as operationally trusted requirements.

**Not performed (and not authorized by PE30):**

- Any destructive direct-push probe against `main`. Direct-push blocking is
  asserted from the live API configuration, not from an empirical push
  attempt (see the live enforcement record above).
- The protected-flow merge itself.
- Repository-wide squash/rebase disabling (see the merge-method caveat
  below).

## Merge-method setting caveat

Repository settings currently allow **merge commits, squash merges, and
rebase merges** (`allow_merge_commit`, `allow_squash_merge`, and
`allow_rebase_merge` are all `true`; automatic branch deletion after merge
is disabled). **Neither the BP26 readiness gate nor the PE30 enforcement
gate changed those settings.**

Disabling squash and rebase repository-wide is a separate, explicitly
authorized decision. Until it is made, governed prompts must continue to
explicitly require merge commits and reject squash/rebase for source,
manifest, and terminal-evidence packets — branch protection as applied
above permits merge commits but does not by itself forbid the other two
methods.

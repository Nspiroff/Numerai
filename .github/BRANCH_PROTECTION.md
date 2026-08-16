# Main branch protection policy

> **Branch protection is not enabled by this PR.** This file is a policy and
> proposed-configuration document only (Issue #26, Phase 2 readiness — BP26).
> It grants no enforcement authority. Applying, modifying, or removing any
> branch protection on `main` requires separate explicit user authorization
> after independent review and the docs-only canary described below.

## Current state

Verified against the live repository on 2026-08-16:

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

## Why readiness is required first

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

## Proposed classic branch-protection configuration

Target: classic branch protection (not a ruleset) on branch `main`.

Endpoint (validated against the current GitHub REST documentation,
<https://docs.github.com/en/rest/branches/branch-protection?apiVersion=2022-11-28#update-branch-protection>):

```
PUT /repos/Nspiroff/Numerai/branches/main/protection
Accept: application/vnd.github+json
X-GitHub-Api-Version: 2022-11-28
```

Proposed payload (exact JSON; every field below is an accepted parameter of
the endpoint above — no unsupported fields are included):

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
  true` is proposed precisely so none exists mechanically.

Emergency modification or removal of branch protection requires all of:

1. explicit user authorization;
2. a recorded before/after configuration (the live protection JSON captured
   both before and after the change);
3. a recorded reason;
4. restoration verification (the protection re-queried and compared
   field-by-field once the emergency is over).

## Canary and enforcement sequence

The exact future sequence (none of it performed by BP26):

1. Independently review and merge the BP26 readiness PR.
2. Create a temporary documentation-only canary PR against `main`.
3. Verify both exact contexts appear and pass on the canary despite no
   governed-source change.
4. Close the canary without merging and delete only its temporary branch,
   under explicit cleanup authority.
5. Independently audit the proposed protection payload above against the
   then-current GitHub API documentation again.
6. Obtain explicit user authority to apply protection.
7. Apply the protection to `main`.
8. Query the live protection
   (`GET /repos/Nspiroff/Numerai/branches/main/protection`) and compare
   every field to the approved payload.
9. Create a protected-branch canary PR.
10. Verify direct pushes to `main` are blocked and the merge-commit PR flow
    remains usable end to end.
11. Only then consider the two checks operationally trusted requirements.

## Merge-method setting caveat

Repository settings currently allow **merge commits, squash merges, and
rebase merges** (`allow_merge_commit`, `allow_squash_merge`, and
`allow_rebase_merge` are all `true`; automatic branch deletion after merge
is disabled). **This BP26 gate does not change those settings.**

Disabling squash and rebase repository-wide is a separate, explicitly
authorized decision. Until it is made, governed prompts must continue to
explicitly require merge commits and reject squash/rebase for source,
manifest, and terminal-evidence packets — branch protection as proposed
above permits merge commits but does not by itself forbid the other two
methods.

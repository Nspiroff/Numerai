# Documentation-only required-check canary

This temporary file exists only to verify that every pull request targeting
`main` emits the complete Research Source CI contract even when the change is
outside the workflow's former path filters.

- Canary: DC29
- Base: `7eae9e4850782bbc5362834f65b28fb1cd6c2818`
- Expected checks:
  - `Portable compile and source-contract tests`
  - `Windows terminal evaluator custody contracts`
- Required provider: GitHub Actions app ID `15368`
- Disposition: close without merging, then delete the temporary branch

This file must never be merged into `main`.

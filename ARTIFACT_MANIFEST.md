# Artifact Manifest

## Package Summary

- 15 experiment/analysis scripts under `scripts/stage_a/` and
  `scripts/stage_b/`.
- One lightweight bundled-result verifier:
  `scripts/verify_key_results.py`.
- Seven Stage A result families.
- Eight Stage B result families.
- Paper-facing figures in PNG and PDF formats.
- Paper-facing Markdown summary tables.
- Public-data acquisition and full/partial reproduction instructions.
- Explicit GAIA reconstruction and 900-discovered/890-complete case-set notes.

## Included

- Derived aggregate and case-level outputs needed to verify reported claims.
- GAIA integrated aggregates required by Stage B follow-up analyses.
- Original supporting experiments still used by the revised study.
- New dense-budget, selector-expansion, weak-anchor, and learned-scorer
  experiments.

## Not Included

- Raw RCAEval and GAIA benchmark datasets.
- Manuscript source or drafts.
- Personal notes.
- Python caches, local logs, or machine-specific paths.

## Size

The package is approximately 198 MB. No individual file exceeds GitHub's
100 MB file limit.

## Fast Integrity Check

From the repository root:

```bash
python scripts/verify_key_results.py
```

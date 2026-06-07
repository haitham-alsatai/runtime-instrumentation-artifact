# Publishing This Artifact as a New GitHub Repository

This repository was created separately from the previous artifact and must
remain connected only to its own remote. Publishing or updating it does not
modify the previous `artifact-anonymous` repository.

## Recommended Repository Name

For an anonymous review submission, use a neutral name such as:

```text
runtime-instrumentation-artifact
```

Avoid placing author names, institutions, or paper drafts inside the
repository. Repository ownership on a personal GitHub account is not anonymous;
use the venue's recommended anonymous-artifact process if anonymity is
required.

## Create and Push

When publishing a fresh copy to another account or archival location, run from
that fresh local copy:

```bash
git remote add origin https://github.com/<account>/<new-repository>.git
git branch -M main
git push -u origin main
```

Do not point this repository at the previous artifact remote.

## Before Publishing

1. Run `python scripts/verify_key_results.py`.
2. Confirm `git status` is clean.
3. Confirm no raw datasets or manuscript drafts are staged.
4. Confirm no file exceeds GitHub's 100 MB individual-file limit.
5. Inspect the rendered GitHub README and Markdown tables.
6. Create a release or archival DOI only after the artifact is frozen.

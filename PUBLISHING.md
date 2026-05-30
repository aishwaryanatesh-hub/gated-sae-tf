# Publishing `gated-sae-tf`

These steps require **your** GitHub and PyPI credentials, so run them yourself.
Everything up to this point (package, tests, CI + publish workflows) is ready.

## 0. Sanity-check the build locally

```bash
pip install build twine
python -m build              # -> dist/*.whl and dist/*.tar.gz
twine check dist/*           # validates metadata + README rendering
```

## 1. Create the public GitHub repo and push

This folder is already its own git repo (`git init` + an initial commit), so:

```bash
gh repo create gated-sae-tf --public --source=. --remote=origin --push
```

(or create the repo in the GitHub UI and `git remote add origin … && git push -u origin main`).

## 2. Configure PyPI Trusted Publishing (no API token needed)

On <https://pypi.org>:

1. Create the project `gated-sae-tf` (it's confirmed available) — or add a
   *pending publisher* before the first upload.
2. Add a **Trusted Publisher** pointing at this repo's `publish.yml`:
   - Owner: `aishwaryanatesh-hub`
   - Repository: `gated-sae-tf`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. Create a GitHub Environment named `pypi` in the repo settings.

## 3. Dry-run on TestPyPI (recommended)

Configure an analogous trusted publisher on <https://test.pypi.org>, then either
trigger a pre-release or upload a local build:

```bash
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ gated-sae-tf
```

## 4. Release → automatic PyPI publish

```bash
git tag v0.1.0
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes "First public release."
```

Cutting the GitHub release fires `.github/workflows/publish.yml`, which builds
the package and uploads it to PyPI via OIDC Trusted Publishing. After it
succeeds:

```bash
pip install gated-sae-tf   # works from any clean environment
```

# Release Checklist

This document describes the minimal release flow for OpenClawBrain.

## 1) Create and verify the release commit

```bash
git checkout chore/release-v12.2.6
python3 -m pytest -q
PYTHONPATH=. python3 -m openclawbrain --version
```

Expected version output for this release: `openclawbrain 12.2.6`.

## 2) Tag the release

```bash
git tag -a v12.2.6 -m "OpenClawBrain v12.2.6"
git push origin chore/release-v12.2.6
git push origin v12.2.6
```

## 3) Build distribution artifacts

```bash
python3 -m pip install --upgrade build
python3 -m build
ls -lh dist/
```

## 4) Publish to PyPI

```bash
python3 -m pip install --upgrade twine
python3 -m twine upload dist/*
```

## 5) Post-publish sanity checks

```bash
python3 -m pip install --upgrade openclawbrain==12.2.6
openclawbrain --version
python3 -m openclawbrain --version
```

Optional smoke check:

```bash
openclawbrain doctor --state /tmp/brain/state.json
```

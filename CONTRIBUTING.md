# Contributing to CrabPath

Welcome! This guide helps you get changes into CrabPath quickly and safely.

## Developer setup

Install editable with extras:

```bash
pip install -e '.[dev]'
```

## Running tests

Run the full suite locally:

```bash
pytest
```

For quick feedback on output, use the requested command:

```bash
python3 -m pytest --tb=short -q
```

## Code style

- Use `ruff` for linting and formatting alignment where applicable.
- Keep type hints on public functions and dataclasses.
- Prefer explicit imports and small utility functions.

## PR process

1. Open an issue or discuss the change before large refactors.
2. Add or update tests for behavior changes.
3. Run tests and include results in the PR description.
4. Attach a short rationale and list potential risks.
5. Request review from a maintainer familiar with the touched modules.

## Architecture overview

- `crabpath/graph.py`: graph persistence, nodes, edges, and consolidation helpers.
- `crabpath/controller.py`: query orchestration and result shaping.
- `crabpath/learning.py`: learning, rewards, and policy-step updates.
- `crabpath/memory`: (via `controller`, `mitosis`, `feedback`) runtime adaptation.
- `crabpath/embeddings.py`: embedding index and provider adapters.
- `crabpath/cli.py`: CLI interface and persistence commands.
- `crabpath/mitosis.py`: graph growth, splits, merges, and workspace bootstrap workflow.\n- `crabpath/synaptogenesis.py`: proto-linking, co-firing analysis, and edge evolution.
- `crabpath/legacy/activation.py`: backward-compatible activation helpers.

## Release expectations

Keep API compatibility in mind: avoid breaking existing imports from `crabpath.__init__`,
and document schema or persistence changes in `CHANGELOG.md`.

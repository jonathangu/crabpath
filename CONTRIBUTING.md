# Contributing

## Development setup

1. Clone the repository and enter it:
   - `git clone <repo-url>`
   - `cd <repo>`
2. Create or activate your environment, then install editable:
   - `pip install -e .`
3. Run the test suite:
   - `pytest`

## Code style

- No linter is currently required by the project.
- Public functions must have docstrings.
- Keep interfaces focused and backward-compatible where possible.

## Test requirements

- All tests must pass before opening a PR.
- Run the full suite with:
  - `python3 -m pytest tests/ -x -q`
- Add tests for new features and bug fixes when behavior changes or new API
  shapes are introduced.

## Release process

1. Bump version in both:
   - `pyproject.toml`
   - `crabpath/__init__.py`
2. Create a release commit/tag:
   - `git tag vX.Y.Z`
3. Push commits and tags:
   - `git push`
   - `git push --tags`

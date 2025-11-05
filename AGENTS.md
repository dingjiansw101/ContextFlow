# Repository Guidelines

## Project Structure & Module Organization
The core Python packages live under `src/openpi`, covering `models`, `policies`, `shared`, `serving`, and `training`. Training and inference entry points stay in `scripts/` (for example `scripts/train.py`, `scripts/serve_policy.py`), while configuration assets and walkthroughs sit in `examples/` and `docs/`. Client-facing utilities and msgpack helpers are maintained in the sibling workspace at `packages/openpi-client`. Tests are split between the top-level `tests/` suite for integration and benchmarking and co-located module unit tests such as `src/openpi/transforms_test.py`.

## Build, Test, and Development Commands
- `GIT_LFS_SKIP_SMUDGE=1 uv sync` installs dependencies, honoring the workspace defined in `pyproject.toml`.
- `PYTHONPATH=src uv run pytest -m "not manual"` runs the fast test suite; use `-m manual` only when GPU checkpoints are available.
- `uv run scripts/train.py <config_name> --exp-name=<run>` starts training; tune memory via `XLA_PYTHON_CLIENT_MEM_FRACTION`.
- `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<path>` launches a policy server for local inference while iterating.

## Coding Style & Naming Conventions
We target Python 3.11 with `ruff` enforcing lint and `ruff format` handling formatting (line length 120). Prefer type hints, module constants in `UPPER_SNAKE_CASE`, functions and modules in `snake_case`, and user-facing classes in `PascalCase`. Before pushing, run `uv run ruff check --fix` or `pre-commit run --all-files` to keep imports sorted and code formatted.

## Testing Guidelines
Pytest drives validation; shared fixtures live in `src/openpi/conftest.py`. Mark GPU- or data-heavy checks with `@pytest.mark.manual` and leave them skipped in CI-style runs. When editing tests, mirror the directory of the code under test and name files with the `_test.py` suffix.

## Commit & Pull Request Guidelines
Commits trend toward short, present-tense descriptions (`add cache for actions`, `fix custom dataset inference`). Follow that style and keep scopes focused. For pull requests, include a concise summary of the change, links to related issues or experiments, reproduction steps (commands, config names, checkpoints), and screenshots or logs when touching serving or UI flows. Keep checkpoints and other large artifacts out of Git—reference their storage location instead.

## Environment & Configuration Tips
Set `PYTHONPATH=src` in development shells or `.env` files so scripts resolve workspace imports. Use `uv run` to execute any Python entry point to ensure consistent virtualenv resolution. When working on GPU-hosted tasks, note the minimum hardware in `README.md` (RTX 4090 for inference, A100 for full fine-tuning) and mention your hardware in PR descriptions if it affects reproducibility.

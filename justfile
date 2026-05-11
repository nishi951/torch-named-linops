# Justfile for torch-named-linops
# Install just: uv tool install just

# Build tutorials from marimo notebooks
tutorials:
    uv run python scripts/build_tutorials.py

# Build documentation (runs tutorials first)
docs: tutorials
    uv run zensical build

# Serve documentation locally
serve:
    uv run zensical serve

# Build and serve documentation
dev: docs serve

# Run all tests
test:
    uv run pytest

# Run linting
lint:
    uv run ruff check .

# Run formatting
fmt:
    uv run ruff format .

# Run all pre-commit hooks
check:
    uv run pre-commit run --all-files

# Install dependencies
install:
    uv sync

# Install all dependencies (including dev and sigpy)
install-all:
    uv sync --all
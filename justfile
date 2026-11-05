# Justfile for torch-named-linops
# Install just: uv tool install just

# Build tutorials from marimo notebooks
tutorials:
    uv run python scripts/build_tutorials.py

# Build documentation (runs tutorials and benchmark report first)
# Pass --clean to clear cache: just docs --clean
docs *args: tutorials bench-report
    uv run zensical build {{args}}

# Serve documentation locally
serve:
    uv run zensical serve

# Build and serve documentation
dev: docs serve

# Run all tests (excludes benchmarks by default)
test:
    uv run pytest

# Run CPU benchmarks (eager + reduced compiled)
bench-cpu:
    uv run pytest benchmarks/ -m "benchmark and not gpu"

# Run all benchmarks (eager + reduced compiled, CPU + GPU)
bench-gpu:
    uv run pytest benchmarks/ -m benchmark

# Run all benchmarks (alias for bench-gpu)
bench: bench-gpu

# Run the full compiled matrix (slow)
bench-compiled:
    uv run pytest benchmarks/test_compiled_benchmarks.py -m benchmark --compiled-level full

# Run only the reduced compiled matrix
bench-compiled-reduced:
    uv run pytest benchmarks/test_compiled_benchmarks.py -m benchmark --compiled-level reduced

# Generate benchmark report for docs
bench-report:
    uv run python benchmarks/_report.py

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
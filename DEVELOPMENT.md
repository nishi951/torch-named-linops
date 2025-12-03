# Development Readme

## Installation

This project uses uv for dependency management and development. If you don't have uv installed, install it first.

To install the repository and its dependencies, run the following command in your terminal:

```sh
# With development dependencies
uv sync --group dev
# With everything
uv sync --group all
```

## Development Workflow

### Making Changes

1. **Create a feature branch**: Always work on a separate branch for new features or bug fixes:
   ```sh
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**: Edit the source code in the `src/torchlinops/` directory.

3. **Run tests**: Before committing, ensure all tests pass:
   ```sh
   uv run pytest
   ```

4. **Code formatting**: Format your code to maintain consistency:
   ```sh
   uv run ruff format --check src/
   uv run isort src/
   ```

5. **Type checking**: (TODO) Run type checks to ensure type safety:
   ```sh
   uv run mypy src/
   ```

6. **Security checking**: (TODO) Run security checks:
    ```sh
    uv run bandit -c pyproject.toml -r src
    ```

### Project Structure

The project is organized as follows:

- `src/torchlinops/`: Main source code
  - `linops/`: Linear operator implementations
  - `functional/`: Functional utilities and operations
  - `utils/`: Utility functions and helpers
- `dev/`: Development utilities and CUDA stream handling
- `docs/`: Documentation source files

## Documentation

To build the documentation using sphinx-autobuild, which provides a live-reloading server, run the following command:

Note: Moving doctrees outside of `build/html` is suspected to help with re-rendering issues.

```sh
uv run sphinx-autobuild docs/source docs/build/html --watch src/torchlinops --doctree-dir docs/build/doctrees
```

To build documentation without the live server:

```sh
uv run sphinx-build docs/source docs/build/html
```

If you run into issues, completely deleting and regenerating the `docs/build` directory is a good place to start.

## Testing

To run the unit tests, use the following command:

```sh
uv run pytest
```

To run tests with coverage:

```sh
uv run pytest --cov=src/torchlinops --cov-report=html
```

To run specific test files:

```sh
uv run pytest src/torchlinops/tests/test_diagonal.py
```

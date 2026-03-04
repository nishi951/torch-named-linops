# Development Readme

## Installation

This project uses uv for dependency management and development. If you don't have uv installed, install it first.

To install the repository and its dependencies, run the following command in your terminal:

```sh
# With development dependencies
uv sync
# With all optional dependencies (including sigpy)
uv sync --all
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
   uv run ruff format src/
   uv run ruff check src/
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

To build the documentation using `mkdocs serve`, which provides a live-reloading server, run the following command:

```sh
uv run mkdocs serve
```

### Documentation strategy
#### Linops
Use the following conventions:

- In the class-level docstring:
  - Document the important attributes
  - Put an example (with math, ideally) of the linop in use.
- In the `__init__` docstring:
  - Document the input arguments.
- Document any other useful methods (e.g. alternative `@classmethod` constructors or `@staticmethod` helper functions)

#### Functions
Use the following conventions:

- In the function-level docstring:
  - Describe what the function does.
  - Document all parameters with types.
  - Document the return value.
  - Add examples where helpful.
- Use numpy-style docstrings throughout.

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

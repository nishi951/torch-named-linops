# Development Readme

## Installation

This project uses uv for dependency management and development. If you don't have uv installed, install it first.

To install the repository and its dependencies, run the following command in your terminal:

```sh
uv sync
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
   uv run black src/
   uv run isort src/
   ```

5. **Type checking**: Run type checks to ensure type safety:
   ```sh
   uv run mypy src/
   ```

### Project Structure

The project is organized as follows:

- `src/torchlinops/`: Main source code
  - `linops/`: Linear operator implementations
  - `functional/`: Functional utilities and operations
  - `utils/`: Utility functions and helpers
- `dev/`: Development utilities and CUDA stream handling
- `docs/`: Documentation source files

### Building from Source

To build the project in development mode:

```sh
uv run pip install -e .
```

This will install the package in editable mode, allowing changes to be immediately reflected without reinstallation.

## Documentation

To build the documentation using sphinx-autobuild, which provides a live-reloading server, run the following command:

```sh
uv run sphinx-autobuild docs/source docs/build/html --watch src/torchlinops
```

To build documentation without the live server:

```sh
uv run sphinx-build docs/source docs/build/html
```

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

## Code Quality

### Linting

Run linting checks:

```sh
uv run ruff check src/
uv run ruff format --check src/
```

### Type Checking TODO

Run static type analysis:

```sh
uv run mypy src/
```

### Security Scanning

Run security checks:

```sh
 uv run bandit -c pyproject.toml -r src
```


## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the package is installed in development mode:
   ```sh
   uv run pip install -e .
   ```

2. **CUDA errors**: Verify GPU drivers and CUDA compatibility:
   ```sh
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Test failures**: Check the test output for specific error messages and ensure all dependencies are installed.

### Getting Help

- Check existing issues in the repository
- Review the documentation in `docs/source/`
- Run tests with verbose output for debugging:
  ```sh
  uv run pytest -v -s
  ```

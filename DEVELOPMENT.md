# Development Readme

## Installation

This project uses uv for dependency management and development. If you don't have uv installed, install it first.

To install the repository and its dependencies, run the following command in your terminal:

```sh
uv sync
```

## Documentation

To build the documentation using sphinx-autobuild, which provides a live-reloading server, run the following command:

```sh
uv run sphinx-autobuild docs/source docs/build/html --watch src/torchlinops
```
## Testing

To run the unit tests, use the following command:

```sh
uv run pytest
```

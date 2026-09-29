# Contributing

Thank you for your interest in contributing to `vib-sqd`. This project is under active
development (see the disclaimer in [README.md](README.md)), and contributions, issues, and
feedback are welcome.

## Setup

Install the package in editable mode along with the development extras:

```bash
pip install -e ".[dev]"
```

For work touching the real-hardware runner, also install the `hardware` extra (see
[README.md](README.md#installation)):

```bash
pip install -e ".[dev,hardware]"
```

## Running checks

### Tests

```bash
pytest test/ -q
```

Slow / end-to-end simulation tests are marked `slow` and can be deselected:

```bash
pytest test/ -q -m "not slow"
```

Some tests require a local [MidasCpp](https://gitlab.com/midascpp/midascpp) installation and are
skipped automatically when one isn't found — see
[docs/midascpp_installation.md](docs/midascpp_installation.md).

### Lint

```bash
ruff check .
```

### Format

```bash
black .
```

### Type check

```bash
mypy vib_sqd
```

## Pull requests

- Keep pull requests focused on a single change where possible.
- Add or update tests for any behavior change.
- Make sure `pytest test/`, `ruff check .`, and `black --check .` pass before requesting review.

## Code of Conduct

This project adheres to the [Qiskit Code of Conduct](CODE_OF_CONDUCT.md). By participating, you
are expected to uphold this code.

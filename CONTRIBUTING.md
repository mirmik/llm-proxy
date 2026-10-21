# Contributing

Issues and pull requests are welcome. The project is evolving; discuss large
API or architecture changes in an issue before implementing them.

## Local setup

Use Python 3.10 or newer on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
```

The tests use local HTTP listeners and mocked model engines. They do not need
a GPU, external model servers, or downloaded weights. Live inference checks
are separate and should name the model, runtime version, and hardware used.

## Changes

Keep pull requests focused. Describe the problem, resulting behavior, and how
you verified it. Add regression coverage for behavior changes, and update the
example config or documentation when changing configuration or protocols.

The proxy's core dependencies should stay small. Put model-specific imports
and GPU dependencies in runners that can use their own Python environments.

## Bug reports

Include the reproduction steps, expected and actual results, Python and backend
versions, and relevant logs. Use a minimal configuration with placeholder host
names and paths. Remove API tokens, passwords, private keys, and user content
before sharing configuration or logs.

## Repository layout

- `llm_proxy/`: API, control panel, routing, process supervision, and state.
- `llm_proxy/drivers/`: runtime validation and launch configuration.
- `llm_proxy/runners/`: standalone speech servers.
- `tests/`: unit tests and local HTTP integration tests.
- `docs/`: configuration, protocols, and deployment guides.
- `deploy/`: service definitions.
- `scripts/`: optional live smoke checks.

Contributions are distributed under the repository's [MIT license](LICENSE).

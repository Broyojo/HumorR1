# humor-rlhf

## Setup (PACE)

Run the setup script once after cloning:

```bash
bash scripts/setup.sh
source ~/.bashrc
```

This installs [uv](https://docs.astral.sh/uv/), creates scratch cache
directories, adds the required environment variables to your `~/.bashrc`,
and runs `uv sync` to install Python and project dependencies. It's safe
to re-run.

## Usage

Run scripts:
```bash
uv run python scripts/<script_name>.py
```

Add dependencies:
```bash
uv add <package_name>
```

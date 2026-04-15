# humor-rlhf

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package and project management.

1. Install uv (if you don't have it):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync the project (installs Python and dependencies):
   ```bash
   uv sync
   ```

3. Run scripts:
   ```bash
   uv run python scripts/<script_name>.py
   ```

4. Add dependencies:
   ```bash
   uv add <package_name>
   ```

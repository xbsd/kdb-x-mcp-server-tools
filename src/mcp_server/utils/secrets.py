"""Read API keys from the environment, falling back to ~/.env.

Secrets must never be hardcoded in source. Resolution order for a name:
  1. the process environment (e.g. an MCP client's "env" block, or the shell);
  2. a KEY=value line in ~/.env (or the file named by $SECRETS_ENV_FILE).
Values from the file never override variables already set in the environment.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _env_file_values() -> dict[str, str]:
    path = Path(os.environ.get("SECRETS_ENV_FILE", "~/.env")).expanduser()
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = (s.strip() for s in line.split("=", 1))
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        else:
            val = val.split(" #", 1)[0].strip()
        if key:
            values[key] = val
    return values


def get_secret(name: str, required: bool = True) -> str | None:
    """Return the secret called `name` from the environment or ~/.env.

    Raises RuntimeError (without revealing any value) if it is required and missing.
    """
    value = os.environ.get(name) or _env_file_values().get(name)
    if not value and required:
        raise RuntimeError(
            f"{name} is not set. Export it, or add '{name}=...' to ~/.env "
            f"(or the file named by SECRETS_ENV_FILE)."
        )
    return value

"""Apply recipe environment defaults as data, never as executable shell code."""

from __future__ import annotations

import os


def apply_defaults(defaults, environment=None):
    environment = os.environ if environment is None else environment
    for key, value in defaults.items():
        if value and not environment.get(key):
            environment[key] = str(value)
    return environment


def main():
    import sys
    from recipe.verl.research.config import cfg

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m recipe.verl.launch.environment COMMAND [ARG ...]")
    apply_defaults(cfg.to_env_dict())
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()

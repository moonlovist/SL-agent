"""Backward-compatible entry point; use ``python -m expert_agent`` for the CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

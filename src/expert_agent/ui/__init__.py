"""Desktop review application for LensAgent."""


def main(argv: list[str] | None = None) -> int:
    from .app import main as run

    return run(argv)


__all__ = ["main"]

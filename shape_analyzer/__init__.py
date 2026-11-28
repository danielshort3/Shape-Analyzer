"""Shape Analyzer package."""

__all__ = ["cfg"]


def __getattr__(name):
    """Keep the public training configuration available without importing it on a health check."""
    if name == "cfg":
        from .config import cfg
        return cfg
    raise AttributeError(name)

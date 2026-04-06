"""ChurnOps baseline training package."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("churnops")
except PackageNotFoundError:
    __version__ = "0.1.0"

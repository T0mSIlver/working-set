"""workingset — serving-capacity model, CLI and measurement harness.

`workingset.model` is the source of truth for every number the study and the
explorer publish; the explorer's JS mirror is verified against it.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    # from the installed package's metadata, so it follows pyproject.toml
    __version__ = version("workingset")
except PackageNotFoundError:             # a source tree that was never installed
    __version__ = "0+unknown"

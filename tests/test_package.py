"""Sanity check that the package is importable and exposes its version."""

import ncomp


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(ncomp.__version__, str)
    assert ncomp.__version__

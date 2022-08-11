"""Miscellaneous for testing."""

import numpy as np

from math import isclose


def eps_float_equality(x, y):
    """Compare two floats for equality with eps as tolerance."""
    return isclose(x, y, abs_tol=np.finfo(float).eps)

"""Miscellaneous functionality for trajectories module."""

from typing import List


def mod_range(arr: List[int], n_range: int):
    """Modulus a list of integer into a range starting from 0."""
    return [num % n_range for num in arr]

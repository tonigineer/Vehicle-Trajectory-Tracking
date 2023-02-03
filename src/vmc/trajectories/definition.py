"""Collection of definitions for several entities."""

from __future__ import annotations

import json

import numpy
import numpy as np

from dataclasses import dataclass

from vmc.trajectories.misc import mod_range


NUM_ATTRIBUTES = 7  # Number of attributes per node


def load_trajectory(filepath) -> Trajectory:
    """Load trajectory data from `.json`."""
    with open(filepath, 'r') as j:
        data = json.loads(j.read())

    return Trajectory(
        *[np.array([data['nodes'][i]]).flatten() for i in range(NUM_ATTRIBUTES)]
    )


def validate_dataclass_types(self) -> None:
    """Validate dataclass types."""
    for field_name, field_def in self.__dataclass_fields__.items():
        var = getattr(self, field_name)
        if not isinstance(var, eval(field_def.type)):  # eval is super slow
            raise TypeError(
                f'{field_name} is a {type(var)} '
                f'instead instance of {eval(field_def.type)}'
            )


@dataclass
class Node:
    """Class to store a trajectory node information."""

    x: float
    y: float
    s: float
    psi: float
    kappa: float
    v: float
    a: float

    def __post_init__(self) -> None:
        """Validate types."""
        validate_dataclass_types(self)


@dataclass
class Position:
    """Class to store only positional data of a node."""

    x: float
    y: float

    def __post_init__(self) -> None:
        """Check data types."""
        validate_dataclass_types(self)

    def as_array(self) -> np.ndarray:
        """Return Position as array."""
        return np.array([self.x, self.y])


@dataclass
class Trajectory:
    """Class to store trajectory information."""

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def __post_init__(self) -> None:
        """Validate types."""
        validate_dataclass_types(self)
        self.n_nodes = len(self.s)

    def get_nodes(self, start_idx: int, n_nodes: int = 1,
                  lead_node: bool = True):
        """Return `n_nodes` according to `start_idx`.

        Arguments
        ---------
        `start_idx` : int
            Starting index in trajectory nodes
        `n_nodes` : int
            Number of nodes to return
        `lead` : bool
            If `True`, there will be always at least one node before
            closest node to position. Needed as query point for possible
            interpolation.

        Returns
        -------
        `nodes` : Node or Trajectory
            Base on the number of nodes return, the return class is
            either a node (n_nodes==1) or a trajectory.
        """
        if lead_node:
            start_idx -= 1

        idc = mod_range(
            range(start_idx, start_idx + n_nodes), self.n_nodes
        )

        # First and last node are identical, remove first when both present
        if 0 in idc and idc.index(0) != 0:
            idc.remove(0)
            idc.insert(0, idc[0]-1)

        if n_nodes == 1:
            return Node(
                self.x[idc][0], self.y[idc][0], self.s[idc][0],
                self.psi[idc][0], self.kappa[idc][0], self.v[idc][0],
                self.a[idc][0]
            )

        return Trajectory(
            self.x[idc], self.y[idc], self.s[idc], self.psi[idc],
            self.kappa[idc], self.v[idc], self.a[idc]
        )

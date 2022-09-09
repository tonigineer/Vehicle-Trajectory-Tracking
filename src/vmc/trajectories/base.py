"""Collection of scenarios for the simulator class."""

import json
import os

import numpy as np

from dataclasses import dataclass


def mod_range(arr: list, n_range: int):
    """Modulus a list of numbers into a range starting from 0."""
    return [_ % n_range for _ in arr]


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


@dataclass
class Position:
    """Class to store only positional data of a node."""

    x: float
    y: float

    def __post_init__(self):
        """Check data types."""
        if not isinstance(self.x, float):
            raise ValueError('value not an float')

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

    n_nodes_traj: bool = None

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
        """
        if self.n_nodes_traj is None:
            self.n_nodes_traj = len(self.s)

        if lead_node:
            start_idx -= 1

        idc = mod_range(
            range(start_idx, start_idx + n_nodes), self.n_nodes_traj
        )

        # First and last node are identical, remove first when both present
        if 0 in idc and idc.index(0) != 0:
            idc.remove(0)
            idc.insert(0, idc[0]-1)

        return_class = Node if n_nodes == 1 else Trajectory

        return return_class(
            self.x[idc], self.y[idc], self.s[idc], self.psi[idc],
            self.kappa[idc], self.v[idc], self.a[idc]
        )


def load_trajectory(filepath) -> Trajectory:
    """Load trajectory data from `.json`."""
    with open(filepath, 'r') as j:
        data = json.loads(j.read())

    return Trajectory(
        np.array([data['nodes'][0]]).flatten(),
        np.array([data['nodes'][1]]).flatten(),
        np.array([data['nodes'][2]]).flatten(),
        np.array([data['nodes'][3]]).flatten(),
        np.array([data['nodes'][4]]).flatten(),
        np.array([data['nodes'][5]]).flatten(),
        np.array([data['nodes'][6]]).flatten()
    )


class OfflineReference():
    """Provide a trajectory from offline generated closed loop circuits.

    The reference is provided according to the vehicle's current position
    along the trajectory and the horizon `N_hor`.
    """

    lead_node = True
    starting_node = 5  # node of x0 (beginning of simulation)

    def __init__(self, track_filepath: str, n_nodes: int = 20):
        """Initialize by selecting a track and a time horizon."""
        self.track_filepath = track_filepath
        self.track_name = os.path.basename(track_filepath).split('.')[0]
        self.trajectory = load_trajectory(track_filepath)
        self.t = self.trajectory

        # Trajectory information
        self.n_nodes = n_nodes
        self.total_length = self.t.s[-1]
        self.close_loop = self.t.x[0] == self.t.x[-1] and \
            self.t.y[0] == self.t.y[-1]
        self.ay_max = max(self.t.v**2 * self.t.kappa)
        self.vx_max = max(self.t.v)

        # Scenario information
        N = self.t.get_nodes(self.starting_node, 1)
        self.x0 = np.array([[N.x[0], N.y[0], N.psi[0], 0, N.v[0], 0]]).T

    def eval(self, position: Position):
        """Return a segment of reference according to current position."""
        localization_idx = self.__localize_on_trajectory(
            pos=position
        )

        return self.t.get_nodes(
            start_idx=localization_idx, n_nodes=self.n_nodes,
            lead_node=self.lead_node
        )

    def __localize_on_trajectory(self, pos: Position) -> int:
        """Determine closest node to position."""
        # TODO: could be to expensive > keeping track of index and increasing
        #       might be cheaper
        nodes = np.asarray(np.array([self.t.x, self.t.y]).T)
        dist_2 = np.sum((nodes - pos.as_array())**2, axis=1)
        return np.argmin(dist_2)

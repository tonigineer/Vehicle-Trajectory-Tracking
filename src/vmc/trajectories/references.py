"""Collection of references for scenarios."""

import os

import numpy as np

from vmc.trajectories.definition import Position, load_trajectory


class ReferenceTube():
    pass


class ReferencePath():
    """Provides a trajectory from offline generated closed loop circuits.

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
        self.x0 = np.array([[N.x, N.y, N.psi, 0, N.v, 0]]).T

    def __localize_on_trajectory(self, pos: Position) -> int:
        """Determine closest node to position."""
        # TODO: could be to expensive > keeping track of index and increasing
        #       might be cheaper
        nodes = np.asarray(np.array([self.t.x, self.t.y]).T)
        dist_2 = np.sum((nodes - pos.as_array())**2, axis=1)
        return np.argmin(dist_2)

    def eval(self, position: Position):
        """Return a segment of reference according to current position."""
        localization_idx = self.__localize_on_trajectory(
            pos=position
        )

        return self.t.get_nodes(
            start_idx=localization_idx, n_nodes=self.n_nodes,
            lead_node=self.lead_node
        )

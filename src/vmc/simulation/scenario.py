"""Contains Scenario class for Simulator."""

import numpy as np

from vmc.trajectories import Position


class Scenario:
    """Class to handle reference and controller as one entity.

    Aims to have a fixed interface for the Simulator to only hand over
    time step and current state vector. If a controller is open loop or
    based on a reference is handled here.
    """

    def __init__(self, controller, reference=None):
        """Get information for Simulator from either control of reference."""
        self.controller = controller
        self.reference = reference

        self.dt = self.controller.dt

        # if no reference given, open loop controller applied
        self.x0 = self.reference.x0 if reference else self.controller.x0
        self.t_end = self.reference.t_end if reference else self.controller.t_end

    def eval(self, t: float, state_vector: np.ndarray) -> np.ndarray:
        """Interface for Simulator to get uk for next time step.

        Arguments
        ---------
        `t` : float
            Current time step in seconds.
        `state_vector` : ndarray
            Current state vector of model.

        Returns
        -------
        `uk` : ndarray
            Control output for system.
        `ref` : Trajectory (opt)
            Current reference for controller.
        """
        ref = None
        if self.reference:
            X, Y = state_vector[0:2][0:2, 0]  # bare values needed
            ref = self.reference.eval(Position(X, Y))

        ctrl_out = self.controller.eval(
            t=t,
            state_vector=state_vector,
            ref=ref
        )
        return ctrl_out, ref

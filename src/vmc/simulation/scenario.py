"""Contains Scenario class for Simulator."""

import numpy as np

from vmc.models import FSVehSingleTrack
from vmc.controller import SteerRamp
from vmc.simulation import Simulator
from vmc.trajectories import OfflineReference


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

        Return
        ------
        `uk` : ndarray
            Control output for system.
        `ref` : Trajectory (opt)
            Current reference for controller.
        """
        ref = None
        if self.reference:
            pass

        return self.controller.eval(t=t, state_vector=state_vector, ref=ref)


# if __name__ == "__main__":
#     TRACK_FILEPATH = './examples/tracks/Algarve_International_Circuit_02g_02g_128.json'
#     N_NODES = 25

#     scenario = Scenario(
#         SteerRamp(derivative=True),
#         OfflineReference(TRACK_FILEPATH, N_NODES)
#     )

#     fs_veh_model = FSVehSingleTrack()

#     Sim = Simulator(model=fs_veh_model, scenario=scenario)
#     Sim.enable_animation = False
#     Sim.run()

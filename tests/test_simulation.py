"""Tests for simulation module of vmc package."""

import numpy as np

from vmc.controller import SteerRamp
from vmc.models import FSVehSingleTrack
from vmc.simulation import Simulator, Scenario


def test_simulator_for_open_loop():
    """Run open loop and check if simulation data is plausible."""
    scenario_ol_ctrl = Scenario(SteerRamp(derivative=True))
    fs_veh_model = FSVehSingleTrack()

    Sim = Simulator(model=fs_veh_model, scenario=scenario_ol_ctrl)
    Sim.run()

    msg = f'Number of simulation steps does not correspond to \
        t_end={scenario_ol_ctrl.t_end} and dt={scenario_ol_ctrl.dt}'
    assert Sim.steps == int(scenario_ol_ctrl.t_end/scenario_ol_ctrl.dt) + 1

    msg = 'Simulation is not strictly monotonic'
    assert all(np.diff(Sim.sim['t'].flatten()) > 0), msg

    # TODO: currently only called for code coverage. Not really tested.
    Sim.show_states_and_input()

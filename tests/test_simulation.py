"""Tests for simulation module of vmc package."""

import numpy as np

from vmc.controller import SteerRamp, TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import Simulator, Scenario
from vmc.trajectories import OfflineReference


def test_simulator_for_open_loop():
    """Run open loop and check if simulation data is plausible."""
    scenario_ol_ctrl = Scenario(SteerRamp(derivative=False))
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


def test_simulator_for_reference_circuit():
    """Run open loop and check if simulation data is plausible."""
    TRACK_FILEPATH = './examples/tracks/Algarve_International_Circuit_03g_06g_130.json'
    N_NODES = 25

    scenario = Scenario(
        TrajTrackPID(),
        OfflineReference(TRACK_FILEPATH, N_NODES)
    )
    scenario.t_end = 200
    fs_veh_model = FSVehSingleTrack()

    Sim = Simulator(model=fs_veh_model, scenario=scenario)
    Sim.laps_max = 1
    Sim.run()
    Sim.show_states_and_input()
    Sim.show_tracking()

    msg = f'Number of simulation steps does not correspond to \
        t_end={scenario.t_end} and dt={scenario.dt}'
    assert Sim.steps == int(scenario.t_end/scenario.dt) + 1

    msg = 'Simulation is not strictly monotonic'
    assert all(np.diff(Sim.sim['t'].flatten()) > 0), msg

    msg = 'Controller performance is not plausible.'
    assert all(np.abs(Sim.sim['e_y'].flatten()) < 2), msg


if __name__ == "__main__":
    test_simulator_for_open_loop()
    test_simulator_for_reference_circuit()

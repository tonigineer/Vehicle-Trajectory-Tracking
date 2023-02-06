"""Tests for simulation module of vmc package."""

import numpy as np

from vmc.controller import SteerRamp, TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import BasicSimulator, Scenario
from vmc.trajectories import ReferencePath


def test_simulator_for_open_loop():
    """Run open loop and check if simulation data is plausible."""
    scenario_ol_ctrl = Scenario(SteerRamp(derivative=False))
    fs_veh_model = FSVehSingleTrack()

    t_max = scenario_ol_ctrl.controller.t_end

    Sim = BasicSimulator(model=fs_veh_model, scenario=scenario_ol_ctrl)
    Sim.t_max = t_max
    Sim.run()

    msg = f'Number of simulation steps does not correspond to \
        t_end={t_max} and dt={scenario_ol_ctrl.dt}'
    assert Sim.steps == int(t_max/scenario_ol_ctrl.dt) + 1

    msg = 'Simulation is not strictly monotonic'
    assert all(np.diff(Sim.sim['t'].flatten()) > 0), msg

    # TODO: currently only called for code coverage. Not really tested.
    Sim.show_states_and_input()


def test_simulator_for_reference_circuit():
    """Run open loop and check if simulation data is plausible."""
    TRACK_FILEPATH = './examples/basic/tracks/Algarve_International_Circuit_03g_06g_130.json'
    N_NODES = 25

    scenario = Scenario(
        TrajTrackPID(),
        ReferencePath(TRACK_FILEPATH, N_NODES)
    )
    fs_veh_model = FSVehSingleTrack()

    Sim = BasicSimulator(model=fs_veh_model, scenario=scenario)
    Sim.t_max = 200
    Sim.laps_max = 1
    Sim.ani.fps_max = 60
    Sim.run()
    Sim.show_states_and_input()
    Sim.show_tracking()

    msg = f'Number of simulation steps does not correspond to \
        t_end={Sim.t_max} and dt={scenario.dt}'
    assert Sim.steps == int(Sim.t_max/scenario.dt) + 1

    msg = 'Simulation is not strictly monotonic'
    assert all(np.diff(Sim.sim['t'].flatten()) > 0), msg

    msg = 'Controller performance is not plausible.'
    assert all(np.abs(Sim.sim['e_y'].flatten()) < 2), msg


# if __name__ == "__main__":
    # test_simulator_for_open_loop()
    # test_simulator_for_reference_circuit()

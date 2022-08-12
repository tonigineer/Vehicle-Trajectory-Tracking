"""Run open loop maneuver."""

from vmc.models import FSVehSingleTrack
from vmc.simulation import Simulator, Scenario
from vmc.controller import SteerRamp


if __name__ == "__main__":
    scenario_open_loop_ctrl = Scenario(SteerRamp(derivative=True))
    fs_veh_model = FSVehSingleTrack()

    Sim = Simulator(model=fs_veh_model, scenario=scenario_open_loop_ctrl)
    Sim.run()
    Sim.show_states_and_input()

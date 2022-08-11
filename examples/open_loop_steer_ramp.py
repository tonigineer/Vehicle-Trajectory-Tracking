"""Run open loop maneuver."""

from vmc.models import FSVehSingleTrack
from vmc.simulation import Simulator
from vmc.controller import SteerRamp


if __name__ == "__main__":
    steer_ramp = SteerRamp(derivative=True)
    fs_veh_model = FSVehSingleTrack()

    Sim = Simulator(model=fs_veh_model, scenario=steer_ramp)
    Sim.run()
    Sim.show_states_and_input()

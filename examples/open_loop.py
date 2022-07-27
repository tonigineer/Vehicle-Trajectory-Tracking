"""Run open loop maneuver."""

from vmc import FSVehSingleTrack
from vmc import Simulator, SteerRamp


if __name__ == "__main__":
    steer_ramp = SteerRamp()
    fs_veh_model = FSVehSingleTrack()

    Sim = Simulator(model=fs_veh_model, scenario=steer_ramp)
    Sim.run()
    Sim.show_states_and_input()

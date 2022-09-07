"""Run trajectory tracking with PID controller."""

from vmc.controller import TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import Simulator, Scenario
from vmc.trajectories import OfflineReference


def main():
    """Exemplary simulation with visualized results."""
    TRACK_FILEPATH = './examples/tracks/Hockenheimring_04g_06g_117.json'
    N_NODES = 15

    scenario = Scenario(
        TrajTrackPID(),
        OfflineReference(TRACK_FILEPATH, N_NODES)
    )
    scenario.t_end = 600

    fs_veh_model = FSVehSingleTrack()

    Sim = Simulator(model=fs_veh_model, scenario=scenario)
    Sim.enable_animation = True
    Sim.laps_max = 1
    Sim.run()

    Sim.show_states_and_input()
    Sim.show_tracking()


if __name__ == "__main__":
    main()

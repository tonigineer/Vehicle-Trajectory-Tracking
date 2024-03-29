"""Run trajectory tracking with PID controller."""

from vmc.controller import TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import BasicSimulator, Scenario
from vmc.trajectories import ReferencePath


def main():
    """Exemplary simulation with visualized results."""
    TRACK_FILEPATH = './examples/basic/tracks/Hockenheimring_04g_06g_117.json'
    # TRACK_FILEPATH = './examples/carla/tracks/Along_The_Ring_03g_06g_75.json'
    N_NODES = 12

    scenario = Scenario(
        TrajTrackPID(),
        ReferencePath(TRACK_FILEPATH, N_NODES)
    )

    fs_veh_model = FSVehSingleTrack()

    Sim = BasicSimulator(model=fs_veh_model, scenario=scenario)
    Sim.run()

    Sim.show_states_and_input()


if __name__ == "__main__":
    main()

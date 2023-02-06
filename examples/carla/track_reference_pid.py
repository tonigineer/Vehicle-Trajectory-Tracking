"""Run trajectory tracking with PID controller."""

import carla

from vmc.controller import TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import CarlaSimulator, Scenario
from vmc.trajectories import ReferencePath


def main():
    """Exemplary simulation with visualized results."""
    TRACK_FILEPATH = './examples/carla/tracks/ssass_01g_01g_30.json'
    N_NODES = 12

    scenario = Scenario(
        TrajTrackPID(),
        ReferencePath(TRACK_FILEPATH, N_NODES)
    )

    scenario.controller.kp_e_psi = 0
    scenario.controller.kp_e_y = 0.1
    scenario.controller.ki_e_y = 0.0
    scenario.controller.kd_e_y = 0.0

    fs_veh_model = FSVehSingleTrack()

    Sim = CarlaSimulator(model=fs_veh_model, scenario=scenario)

    # Carla
    spawn_point = carla.Transform(carla.Location(-7.269806349,-68.79465062,0.6),carla.Rotation(0,180,0))
    


    Sim.enable_animation = False
    try:
        Sim.run()
    finally:
        Sim.show_states_and_input()


if __name__ == "__main__":
    main()

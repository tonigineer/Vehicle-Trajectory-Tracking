"""Run trajectory tracking with PID controller."""

import carla

import numpy as np

from time import sleep
from vmc import CarlaApi

from vmc.controller import TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import CarlaSimulator, Scenario
from vmc.trajectories import ReferencePath


def prepare_carla():
    """Start Carla Server, spawn vehicle and launch monitor."""
    if not CarlaApi.server_is_running():
        screen_w, screen_h = CarlaApi.get_full_resolution()

        CarlaApi.start_dedicated_server()
        sleep(5)
        CarlaApi.arrange_window(
            window_name='CarlaUE4',
            position=(1, 1, screen_w // 4, screen_h // 2)
        )

        CarlaApi.start_dedicated_vehicle_monitor()
        sleep(2)
        CarlaApi.arrange_window(
            window_name='pygame window',
            position=(1, screen_h // 2 + 1, screen_w // 4, screen_h // 2)
        )

        sleep(3)


def main():
    """Exemplary simulation of a PID with Carla as Simulator."""
    TRACK_FILEPATH = './examples/carla/tracks/Along_The_Ring_03g_04g_68.json'
    TRACK_FILEPATH = './examples/carla/tracks/Along_The_Ring_03g_06g_75.json'
    TRACK_FILEPATH = './examples/carla/tracks/Along_The_Ring_02g_08g_81.json'
    # TRACK_FILEPATH = './examples/carla/tracks/Along_The_Ring_02g_10g_87.json'
    N_NODES = 20

    scenario = Scenario(
        TrajTrackPID(),
        ReferencePath(TRACK_FILEPATH, N_NODES)
    )

    # Spawn vehicle at fifth node of full trajectory.
    # NOTE: heading angle offset of 90deg between Carla and trajectory
    starting_node = scenario.reference.trajectory.get_nodes(10)
    spawn_point = carla.Transform(
        carla.Location(x=starting_node.x, y=starting_node.y, z=.6),
        carla.Rotation(roll=0, pitch=0, yaw=np.rad2deg(starting_node.psi) + 90)
    )
    CarlaApi.respawn_ego_vehicle(spawn_point)

    # Set sample time to 20 Hz. 100 Hz are not possible in real time.
    fs_veh_model = FSVehSingleTrack()
    fs_veh_model.dt = 0.05
    CarlaApi.set_fixed_delta_seconds(fs_veh_model.dt)

    Sim = CarlaSimulator(model=fs_veh_model, scenario=scenario)

    # Controller parameter
    Sim.scenario.controller.kp_e_y = 0.4
    Sim.scenario.controller.ki_e_y = 0.1
    Sim.scenario.controller.kd_e_y = 0.05
    Sim.scenario.controller.kp_e_psi = 0.05
    Sim.scenario.controller.kp_vx = 0.5

    try:
        Sim.run()
    finally:
        Sim.show_states_and_input()
        CarlaApi.respawn_ego_vehicle(spawn_point)


if __name__ == "__main__":
    # prepare_carla()
    main()

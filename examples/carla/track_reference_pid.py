"""Run trajectory tracking with PID controller."""

import carla

import numpy as np

from time import sleep
from vmc import CarlaApi

from vmc.controller import TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import CarlaSimulator, Scenario
from vmc.trajectories import ReferencePath


def main():
    """Exemplary simulation of a PID with Carla as Simulator."""
    TRACK_FILEPATH = './examples/carla/tracks/Along_The_Ring_01g_01g_30.json'
    N_NODES = 12

    scenario = Scenario(
        TrajTrackPID(),
        ReferencePath(TRACK_FILEPATH, N_NODES)
    )

    # Prepare Carla Simulator
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

    # Spawn vehicle at fifth node of full trajectory.
    # NOTE: heading angle offset of 90deg between Carla and trajectory
    starting_node = scenario.reference.trajectory.get_nodes(5)
    spawn_point = carla.Transform(
        carla.Location(x=starting_node.x, y=starting_node.y, z=.6),
        carla.Rotation(roll=0, pitch=0, yaw=np.rad2deg(starting_node.psi) + 90)
    )
    CarlaApi.respawn_ego_vehicle(spawn_point)

    scenario.controller.kp_e_psi = 0
    scenario.controller.kp_e_y = 0.01
    scenario.controller.ki_e_y = 0.0
    scenario.controller.kd_e_y = 0.0

    fs_veh_model = FSVehSingleTrack()

    Sim = CarlaSimulator(model=fs_veh_model, scenario=scenario)

    #             carla.Transform(
    #             carla.Location(x=1, y=1, z=1),
    #             carla.Rotation(roll=0, pitch=0, yaw=-90)
    #         )# Carla


    # spawn_point = carla.Transform(carla.Location(-7.269806349,-68.79465062,0.6),carla.Rotation(0,180,0))
    # CarlaApi.start_server()
    # sleep(5)
    #     # Arrange windows
    #     display_manager.render()  # render empty to show window for arrangement
    #     cls._arrange_windows([
    #         ('pygame window', 1, 1081, 1920, 1080),
    #         ('CarlaUE4', 1, 1, 1920, 1080*0.875)
    #     ])
    try:
        Sim.run()
    finally:
        Sim.emergency_stop()
        Sim.show_states_and_input()
        # CarlaApi.respawn_ego_vehicle(spawn_point)


if __name__ == "__main__":
    main()

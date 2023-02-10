"""Run trajectory tracking with PID controller."""

import os
import carla

from time import sleep
from vmc import CarlaApi

from vmc.controller import TrajTrackPID
from vmc.models import FSVehSingleTrack
from vmc.simulation import CarlaSimulator, Scenario
from vmc.trajectories import ReferencePath


def main():
    """Exemplary simulation with visualized results."""

    os.system('python -c "from vmc import CarlaApi; CarlaApi.start_server()" &')
    os.system('python -c "from vmc import CarlaApi; CarlaApi.prepare_vehicle()" &')

    TRACK_FILEPATH = './examples/carla/tracks/ssass_01g_01g_30.json'
    N_NODES = 12

    scenario = Scenario(
        TrajTrackPID(),
        ReferencePath(TRACK_FILEPATH, N_NODES)
    )
    print('srs')
    return
    scenario.controller.kp_e_psi = 0
    scenario.controller.kp_e_y = 0.1
    scenario.controller.ki_e_y = 0.0
    scenario.controller.kd_e_y = 0.0

    fs_veh_model = FSVehSingleTrack()

    Sim = CarlaSimulator(model=fs_veh_model, scenario=scenario)

    # Carla
    spawn_point = carla.Transform(carla.Location(-7.269806349,-68.79465062,0.6),carla.Rotation(0,180,0))
    CarlaApi.start_server()
    sleep(5)
        # Arrange windows
        display_manager.render()  # render empty to show window for arrangement
        cls._arrange_windows([
            ('pygame window', 1, 1081, 1920, 1080),
            ('CarlaUE4', 1, 1, 1920, 1080*0.875)
        ])
    try:
        Sim.run()
    finally:
        CarlaApi.remove_population()
        Sim.show_states_and_input()


if __name__ == "__main__":


    main()

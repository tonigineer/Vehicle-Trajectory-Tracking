import os
import carla
import random
import platform

import pygame as pg
import numpy as np

from time import perf_counter, sleep
from typing import Tuple, List, Dict


DEFAULT_PATH = '~/carla/0_9_13/'
DEFAULT_PATH_WIN = 'Carla/releases/0_9_13'

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 2000


def move_resize_window(window_name: str, x: int, y: int, w: int, h: int):
    os.system(f'wmctrl -r {window_name} -e 0,{x},{y},{w},{h} &')


class DisplayManager:
    """Pygame framework to display sensor data.

    Taken from Carla example for `visualization`.
    Small adjustments made though.
    """

    EXIT_KEYS = [pg.K_ESCAPE, pg.K_q]
    MAX_FPS = 144

    sensor_list = []

    def __init__(self, grid_size: Tuple[int, int],
                 window_size: Tuple[int, int]):
        pg.init()
        pg.font.init()

        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Arial", 18)

        self.display = pg.display.set_mode(
            window_size, pg.HWSURFACE | pg.DOUBLEBUF
        )

        self.grid_size = grid_size
        self.window_size = window_size

    def update_fps(self):
        fps = str(int(self.clock.get_fps()))
        fps_text = self.font.render(fps, 1, pg.Color("coral"))
        return fps_text

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [
            int(self.window_size[0]/self.grid_size[1]),
            int(self.window_size[1]/self.grid_size[0])
        ]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return
        for s in self.sensor_list:
            s.render()

        self.display.blit(self.update_fps(), (10, 0))

        pg.display.flip()
        self.clock.tick(self.MAX_FPS)

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display

    def check_events(self) -> bool:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            elif event.type == pg.KEYDOWN:
                pg.K_ESCAPE
                if event.key in self.EXIT_KEYS:
                    return True
        return False


class SensorManager:
    """Handler for sensors placed as actors in Carla.

    Taken from Carla example for `visualization`.
    Small adjustments made though.
    """

    def __init__(self, world, display_man, sensor_type, transform,
                 attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(
            sensor_type, transform, attached, sensor_options
        )
        self.sensor_options = sensor_options

        self.frame_rate = 0.0
        self.frame = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(
                camera_bp, transform, attach_to=attached
            )
            camera.listen(self.save_rgb_image)

            return camera
        else:
            return None

    def save_rgb_image(self, image):
        t_start = perf_counter()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pg.surfarray.make_surface(array.swapaxes(0, 1))

        self.frame += 1
        self.frame_rate = 1 / (perf_counter() - t_start)

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


class CarlaApi:

    # Hardcoded for 4k
    # TODO based on current resolution
    VEH_MONITOR_WIDTH = 1920
    VEH_MONITOR_HEIGHT = 1080

    @classmethod
    def start_server(cls) -> None:
        """Start Carla from default path and resize window."""
        if cls._server_is_running():
            return

        if platform.system() == 'Windows':
            os.system(
                f'start ../{DEFAULT_PATH_WIN}/WindowsNoEditor/CarlaUE4.exe'
            )
        elif platform.system() == 'Linux':
            # NOTE: `sudo apt install wmctrl` on Ubuntu 20.04 needed
            os.system(
                f'{DEFAULT_PATH}/CarlaUE4.sh -windowed -ResX=480 -ResY=320 &'
            )
        else:
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )

    @classmethod
    def kill_server(cls) -> None:
        """Terminate Carla via command line."""
        if platform.system() == 'Windows':
            os.system('taskkill /IM CarlaUE4.exe /F')
        elif platform.system() == 'Linux':
            os.system('kill -9 $(pgrep -f carla)')
        else:
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )

    @classmethod
    def _server_is_running(cls) -> bool:
        if platform.system() == 'Windows':
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )
        elif platform.system() == 'Linux':
            pid = os.system('pgrep -f carla')
        else:
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )

        return int(pid) != 0

    @staticmethod
    def _connect_to_server() -> carla.Client:
        """Connect to Carla as client and return object."""
        client = carla.Client(DEFAULT_HOST, DEFAULT_PORT)
        client.set_timeout(1.0)
        return client

    @classmethod
    def remove_population(cls):
        """Remove all vehicle and pedestrian from world."""
        client = cls._connect_to_server()
        world = client.get_world()

        for actor in world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.destroy()
            if 'walker' in actor.type_id:
                actor.destroy()

    @classmethod
    def prepare_vehicle(cls, *,
                        spawn_point=None, sensory=None) -> None:
        """Open Pygame window to show camera mounted on ego vehicle."""
        client = cls._connect_to_server()
        world = client.get_world()

        # Handle defaults
        if not sensory:
            sensory = [
                {"type": "RGBCamera", "x": -4, "y": 0, "z": 2.4,
                 "roll": 0, "pitch": -8, "yaw": 0},
                {"type": "RGBCamera", "x": 0, "y": 0, "z": 20,
                 "roll": 0, "pitch": -90, "yaw": 0}
            ]

        if not spawn_point:
            spawn_point = random.choice(world.get_map().get_spawn_points())

        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("mercedes*")[0]

        if not spawn_point:
            spawn_point = cls.random_spawn_point(world)

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        # Set up sensory and monitoring system in Pygame
        display_manager = DisplayManager(
            grid_size=[1, len(sensory)],
            window_size=[cls.VEH_MONITOR_WIDTH, cls.VEH_MONITOR_HEIGHT]
        )

        for idx, sensor in enumerate(sensory):
            name, x, y, z, roll, pitch, yaw = sensor.values()
            SensorManager(
                world, display_manager, name,
                carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
                ), vehicle, {}, display_pos=[0, idx]
            )

        # Arrange windows
        display_manager.render()  # render empty to show window for arrangement
        cls._arrange_windows([
            ('pygame window', 1, 1081, 1920, 1080),
            ('CarlaUE4', 1, 1, 1920, 1080*0.875)
        ])

        try:
            while True:
                # Carla Tick
                # if args.sync:
                # world.tick()
                # else:
                # world.wait_for_tick()

                # Render received data
                display_manager.render()
                if display_manager.check_events():
                    break
        finally:
            display_manager.destroy()


    def _arrange_windows(settings):
        """Arrange windows according to settings."""
        for item in settings:
            move_resize_window(*item)


if __name__ == '__main__':
    """Example including all functionality."""
    try:
        CarlaApi.start_server()
        sleep(5)
        CarlaApi.remove_population()
        CarlaApi.prepare_vehicle()
    except Exception as E:
        print(E)
    finally:
        CarlaApi.kill_server()

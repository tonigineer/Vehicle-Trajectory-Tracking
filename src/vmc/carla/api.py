import os
import re
import carla
import random
import platform
import subprocess

import numpy as np

from time import perf_counter, sleep
from typing import Tuple, List

from dataclasses import dataclass

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame as pg

DEFAULT_PATH = '~/carla/0_9_13/'
DEFAULT_PATH_WIN = 'Carla/releases/0_9_13'

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 2000


@dataclass
class SensorSetup:

    sensor_type: str
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def __init__(self, sensor_type: str, xyz: Tuple[float], rpy: Tuple[float]):
        self.sensor_type = sensor_type
        self.x, self.y, self.z = xyz
        self.roll, self.pitch, self.yaw = rpy


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

    EGO_VEHICLE = 'model3'

    @classmethod
    def start_dedicated_server(cls) -> bool:
        """Start a Carla in a `dedicated` process.

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> CarlaApi.start_dedicated_server()
        True
        # ... continue with your code ...

        Returns
        -------
        status : bool
            `True` when server was started

        """
        # py_code = \
        #     f'from vmc import CarlaApi; CarlaApi.__start_server()'
        # os.system(f'python -c "{py_code}" &')
        cls.__start_server()

        WAIT_SECONDS = 0.5
        MAX_TIMEOUTS = 10

        timeouts = 0
        while timeouts < MAX_TIMEOUTS:
            if cls.server_is_running():
                return True
            timeouts += 1
            sleep(0.5)

        print(f'Server was not ready after {MAX_TIMEOUTS*WAIT_SECONDS} seconds.')
        return False

    @classmethod
    def __start_server(cls) -> None:
        """Start Carla from default path as `dedicated` process.

        Distinguish between operating systems.
        """
        if cls.server_is_running():
            return

        if platform.system() == 'Windows':
            os.system(
                f'start ../{DEFAULT_PATH_WIN}/WindowsNoEditor/CarlaUE4.exe'
            )
        elif platform.system() == 'Linux':
            # NOTE: `sudo apt install wmctrl` on Ubuntu 20.04 needed
            os.system(
                f'{DEFAULT_PATH}/CarlaUE4.sh &'
            )
        else:
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )

    @staticmethod
    def server_is_running() -> bool:
        """Check if server is running by looking for process name.

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> CarlaApi.server_is_running()
        True

        Returns
        -------
        status : bool
            `True` if server is running (process called `CarlaUE4` active)

        """
        if platform.system() == 'Windows':
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )
        elif platform.system() == 'Linux':
            stdout = subprocess.check_output('pgrep -f CarlaUE4', shell=True)
            return len(stdout.splitlines()) > 1
        else:
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )

    @classmethod
    def kill_server(cls) -> None:
        """Terminate Carla via command line.

        Function distinguish between operating systems and
        uses appropriate command.

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> CarlaApi.kill_server()

        """
        if not cls.server_is_running():
            return

        if platform.system() == 'Windows':
            os.system('taskkill /IM CarlaUE4.exe /F')
        elif platform.system() == 'Linux':
            os.system('kill -9 $(pgrep -f carla)')
        else:
            raise NotImplementedError(
                f'No method implemented for operating system: {platform.system()}'
            )

    @staticmethod
    def connect_to_server(timeout: float = 2.0) -> carla.Client:
        """Connect to Carla as client and return object.

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> client = CarlaApi.connect_to_server(3.0)
        # ... continue with your code ...

        Arguments
        ---------
        timeout : float (default 2.0)
            `Time` before client connection is timed out

        Returns
        -------
        client : carla.Client
            Client `object` to interact with Carla server

        """
        client = carla.Client(DEFAULT_HOST, DEFAULT_PORT)
        client.set_timeout(timeout)
        return client

    @classmethod
    def remove_all_actors(cls, types: List[str]) -> None:
        """Remove all vehicle and pedestrian from world.

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> CarlaApi.remove_actors(['vehicle', 'walker'])

        Arguments
        ---------
        types : list of str
            `Actor` types to be removed (checked if `type-string`
            in `actor.type_id`)

        """
        client = cls.connect_to_server()
        world = client.get_world()

        for actor in world.get_actors():
            for _type in types:
                if _type in actor.type_id:
                    actor.destroy()

    @staticmethod
    def start_dedicated_vehicle_monitor() -> None:
        """Run `start_vehicle_monitor` in a dedicated process with defaults.

        Note: Refer to`CarlaApi.start_vehicle_monitor()` for a more detailed
        description. Default settings contain two RGB cameras and a random
        spawn point. Use `CarlaApi.respawn_ego_vehicle()` to move vehicle
        to desired starting position.

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> CarlaApi.start_dedicated_vehicle_monitor()

        """
        os.system(
            'python -c "from vmc import CarlaApi; CarlaApi.start_vehicle_monitor()" & '
        )

    @classmethod
    def start_vehicle_monitor(cls, *,
                              spawn_point=None,
                              sensory=None,
                              window_size=None
                              ) -> None:
        """Prepare Ego Vehicle and open monitor app in Pygame.

        Note: Vehicle is spawned at `spawn_point` with a mounted set of
        `sensory`. If keyword arguments are not given, a hardcoded default
        sensory with a random spawn point is used.

        Arguments
        ---------
        spawn_point : carla.Transform
            `Location` and `orientation`, where Ego Vehicle is spawned.

        sensory : List with sensor setups (SensorSetup class)
            Sensor setup with different sensors mounted within the vehicle's
            coordinate system.

        window_size : Tuple of int
            Width and height of monitor window.

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> CarlaApi.prepare_ego_vehicle(
                spawn_point=carla.Transform(
                    carla.Location(-76.0, -70.0, 0.6),
                    carla.Rotation(0,180,0)
                ),
                sensory=[
                    SensorSetup("RGBCamera", (-4, 0, 2.4), (0, -8, 0))
                ]
            )
        # Or use default setup
        >>> CarlaApi.prepare_ego_vehicle()

        """
        client = cls.connect_to_server()
        world = client.get_world()

        # Handle defaults
        if not sensory:
            sensory = [
                SensorSetup("RGBCamera", (-4, 0, 2.4), (0, -8, 0)),
                SensorSetup("RGBCamera", (0, 0, 20), (0, -90, 0))
            ]

        if not spawn_point:
            spawn_point = random.choice(world.get_map().get_spawn_points())

        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter(cls.EGO_VEHICLE)[0]

        if not spawn_point:
            spawn_point = cls.random_spawn_point(world)

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        # Set up sensory and monitoring system in Pygame
        if window_size:
            width, height = window_size
        else:
            width, height = cls.get_full_resolution()

        display_manager = DisplayManager(
            grid_size=[1, len(sensory)],
            window_size=[width // 4, height // 2]
        )

        for idx, sensor in enumerate(sensory):
            SensorManager(
                world, display_manager, sensor.sensor_type,
                carla.Transform(
                    carla.Location(
                        x=sensor.x, y=sensor.y, z=sensor.z
                    ),
                    carla.Rotation(
                        roll=sensor.roll, pitch=sensor.pitch, yaw=sensor.yaw
                    )
                ), vehicle, {}, display_pos=[0, idx]
            )

        try:
            while True:
                # TODO: Sync needed?
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

    @classmethod
    def get_ego_vehicle(cls) -> carla.Actor:
        client = cls.connect_to_server()
        world = client.get_world()

        for actor in world.get_actors():
            if cls.EGO_VEHICLE in actor.type_id:
                return actor

        raise ValueError(f'Could not get ego_vehicle `{cls.EGO_VEHICLE}`')

    @classmethod
    def respawn_ego_vehicle(cls, spawn_point: carla.Transform) -> None:
        """Move ego vehicle to desired position with orientation."""
        cls.get_ego_vehicle().set_transform(spawn_point)

    @staticmethod
    def arrange_window(*, window_name: str, position: Tuple[int]) -> None:
        """Arrange window on desktop according to position definition.

        Arguments
        ---------
        window_name : str
            `Name` of window to move and resize
        position : Tuple with (x, y, w, h)
            `x` and `y` position of the top left  corner of window,
            width `w` and height `h` of window

        Usage
        -----
        >>> from vmc import CarlaApi
        >>> cls.arrange_windows(
                window_name='CarlaUE4',
                position=(1, 1, 1920, 1080))
            )

        """
        x, y, w, h = position
        os.system(f'wmctrl -r {window_name} -e 0,{x},{y},{w},{h} &')

    @staticmethod
    def get_full_resolution() -> Tuple[int, int]:
        """Get `width` and `height` of full display."""
        stdout = subprocess.check_output(
            "xdpyinfo | awk '/dimensions/{print $2}'", shell=True
        )
        return tuple(map(int, re.findall('[0-9]+', str(stdout))))

    @classmethod
    def get_map_name(cls) -> str:
        client = cls.connect_to_server()
        world = client.get_world()
        return world.get_map().name.split('/')[-1]

    @classmethod
    def set_fixed_delta_seconds(cls, ts: float) -> None:
        """Set server sample time."""
        client = cls.connect_to_server()
        world = client.get_world()
        settings = world.get_settings()
        # settings.synchronous_mode = True
        settings.fixed_delta_seconds = ts
        world.apply_settings(settings)

        settings = world.get_settings()
        print(f'[SERVER] fixed_delta_seconds = {settings.fixed_delta_seconds}')


def main():
    """Call functionality for development and testing."""
    CarlaApi.start_dedicated_server()
    sleep(5)
    CarlaApi.arrange_window(
        window_name='CarlaUE4',
        position=(1, 1, 1920, 1080*0.8)
    )

    CarlaApi.start_dedicated_vehicle_monitor()
    sleep(2)
    CarlaApi.arrange_window(
        window_name='pygame window',
        position=(1, 1081, 1920, 1080)
    )

    sleep(3)
    # spawn_point = carla.Transform(
    #     carla.Location(x=1, y=1, z=1),
    #     carla.Rotation(roll=0, pitch=0, yaw=-90)
    # )
    # CarlaApi.respawn_ego_vehicle(spawn_point)


if __name__ == '__main__':
    main()

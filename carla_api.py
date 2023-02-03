import os
import carla
import random


import pygame as pg
import numpy as np

from time import perf_counter, sleep
from typing import Tuple


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


class CarlaApi():

    DEFAULT_PATH = '~/carla/0_9_13/'

    DEFAULT_HOST = '127.0.0.1'
    DEFAULT_PORT = 2000

    MONITOR_WIDTH = 1920
    MONITOR_HEIGHT = 1080

    def __init__(self):
        pass

    @classmethod
    def start_server(cls) -> None:
        # Start CarlaUE4 from default path and resize window
        #   NOTE: `sudo apt install wmctrl` on Ubuntu 20.04 needed
        #
        #   TODO: differentiate between OS
        os.system(f'{cls.DEFAULT_PATH}/CarlaUE4.sh -windowed -ResX={cls.WIDTH} -ResY={cls.HEIGHT} &')
        sleep(5)
        os.system(f'wmctrl -r CarlaUE4 -e 0,1,1,{cls.WIDTH},{cls.HEIGHT} &')

    def kill_server(cls) -> None:
        # Terminate server via command line.
        #   TODO: differentiate between OS
        os.system('kill -9 $(pgrep -f carla)')

    @staticmethod
    def populate_world(client=None, num_vehicle=30, num_pedestrians=100):
        if not client:
            client = CarlaApi.connect()
        world = client.get_world()

        # bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        # Add random vehicle
        vehicle_bp = world.get_blueprint_library().filter('vehicle*')
        for _ in range(num_vehicle):
            spawn_point = random.choice(spawn_points)
            spawn_points.pop(spawn_points.index(spawn_point))
            world.try_spawn_actor(
                random.choice(vehicle_bp), spawn_point
            )

        # Add random pedestrians
        # pedestrian_bp = world.get_blueprint_library().filter('walker*')
        # for _ in range(num_pedestrians):
        #     spawn_point = random.choice(spawn_points)
        #     spawn_points.pop(spawn_points.index(spawn_point))
        #     world.try_spawn_actor(
        #         random.choice(pedestrian_bp), spawn_point
        #     )

        print('done :)')

    @classmethod
    def connect_client(cls) -> carla.Client:

        client = carla.Client(cls.DEFAULT_HOST, cls.DEFAULT_PORT)
        client.set_timeout(5.0)

        return client

    @classmethod
    def remove_population(cls):
        client = cls.connect()
        world = client.get_world()

        for actor in world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.destroy()
            if 'walker' in actor.type_id:
                actor.destroy()

    @classmethod
    def set_autopilot_for_population(cls, autopilot: bool):
        client = cls.connect()
        world = client.get_world()

        for actor in world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.set_autopilot(autopilot)

    @classmethod
    def spawn_ego_vehicle(cls):
        client = cls.connect_client()
        world = client.get_world()

        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("mercedes*")[0]

        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        # print(vehicle)
        # print(dir(vehicle))
        vehicle.set_autopilot(True)

        return vehicle

        # # Move spectator behind spawned vehicle
        # spectator = world.get_spectator()
        # transform = carla.Transform(
        #     vehicle.get_transform().transform(
        #         carla.Location(x=-8, z=2.5)
        #     ), vehicle.get_transform().rotation
        # )
        # spectator.set_transform(transform)

        # # Place camera to observe vehicle
        # # Create a transform to place the camera on top of the vehicle
        # camera_init_trans = carla.Transform(carla.Location(z=1.5))

        # # We create the camera through a blueprint that defines its properties
        # camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

        # # We spawn the camera and attach it to our ego vehicle
        # camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        # camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

        # return (vehicle.id, camera.id)

    @classmethod
    def prepare_vehicle(cls) -> None:
        """Open """

        client = cls.connect_client()
        world = client.get_world()

        display_manager = DisplayManager(
            grid_size=[1, 3],
            window_size=[cls.MONITOR_WIDTH, cls.MONITOR_HEIGHT]
        )

        vehicle = cls.spawn_ego_vehicle()

        # Spawn sensory
        # - left
        SensorManager(
            world, display_manager, 'RGBCamera',
            carla.Transform(
                carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)
            ), vehicle, {}, display_pos=[0, 0]
        )
        # - front
        SensorManager(
            world, display_manager, 'RGBCamera',
            carla.Transform(
                carla.Location(x=0, z=2.4), carla.Rotation(yaw=0)
            ), vehicle, {}, display_pos=[0, 1]
        )
        # right
        SensorManager(
            world, display_manager, 'RGBCamera',
            carla.Transform(
                carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)
            ), vehicle, {}, display_pos=[0, 2]
        )

        try:
            while True:
                # Carla Tick
                # if args.sync:
                # world.tick()
                # else:
                #     world.wait_for_tick()

                # Render received data
                display_manager.render()
                if display_manager.check_events():
                    break
        finally:
            display_manager.destroy()


if __name__ == '__main__':
    # CarlaApi.start_server()
    # sleep(6)
    CarlaApi.open_vehicle_monitor()

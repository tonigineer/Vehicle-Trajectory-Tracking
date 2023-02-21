"""Simulation framework."""

import carla

import numpy as np

from time import sleep

from vmc.evaluation.evaluation import Evaluation, ScatterEntry, SubplotLayout
from vmc.evaluation.animation import AnimateVehicle, AnimationData
from vmc.carla.api import CarlaApi


class CarlaSimulator():
    """Framework to run simulation of models with a desired scenario."""

    CARLA_HOST = '127.0.0.1'
    CARLA_PORT = 2000

    laps_max = 10
    lap_steps = []

    t_max = 2000  # needed for memory allocation

    def __init__(self, *, model, scenario) -> None:
        """Initiate class with model."""
        self.model = model
        self.scenario = scenario

    def __prepare_run(self) -> None:
        self.exec_time_sim = 0
        self.exec_time_control = 0

        self.lap = 1
        self.ref_s_prev = 0
        self.step = 0
        self.steps = int(self.t_max/self.model.dt) + 1

        self.sim = None
        self.__collect_sim_data()

        self.vehicle = CarlaApi.get_ego_vehicle()
        self.carla_parse_vehicle_config()

        self.client = CarlaApi.connect_to_server()
        self.world = self.client.get_world()

    def __count_laps(self, ctrl) -> bool:
        """Count driven laps and increment counter for termination.

        New lap is determined by travel distance along full trajectory.
        """
        if self.ref_s_prev > ctrl.ref_s:
            self.lap += 1
            self.lap_steps.append(self.step)

            if self.lap > self.laps_max:
                print(f'Max. number of laps {self.laps_max} reached, simulation is terminated.')
                return True

        self.ref_s_prev = ctrl.ref_s
        return False

    def __collect_sim_data(self, **kwargs) -> None:
        """Take care of collecting simulation data from simulation."""
        if self.sim is None:
            self.sim = {
                'xk': np.zeros([self.steps, self.model.Nx, 1]),
                'uk': np.zeros([self.steps-1, self.model.Nu, 1]),
                't': np.zeros([self.steps, 1]),
                'e_y': np.zeros([self.steps, 1]),
                'e_psi': np.zeros([self.steps, 1]),
                'e_vx': np.zeros([self.steps, 1]),
                'ref_s': np.zeros([self.steps, 1])
            }
            return

        k = self.step

        try:
            self.sim['t'][k+1, :] = kwargs['t']
            self.sim['xk'][k+1, :, :] = kwargs['xk']
        except KeyError:
            raise KeyError('t and xk must be collected during simulation.')

        if 'ctrl' in kwargs.keys():
            ctrl = kwargs['ctrl']
            self.sim['uk'][k, :, :] = ctrl.uk
            self.sim['e_y'][k+1, :] = ctrl.e_y
            self.sim['e_psi'][k+1, :] = ctrl.e_psi
            self.sim['e_vx'][k+1, :] = ctrl.e_vx
            self.sim['ref_s'][k+1, :] = ctrl.ref_s

    def __trim_sim_data(self):
        """Remove obsolete allocated sim data."""
        idx = self.lap_steps[-1] if len(self.lap_steps) > 0 else self.step
        for key, value in self.sim.items():
            self.sim[key] = value[1:idx, :, :] if len(value.shape) == 3 else value[1:idx, :]

    def __show_progress(self, t):
        info_string = f'Sim running ...  {t:4.2f}s /  {self.t_max}s [ﯩ: {self.lap} / {self.laps_max}]'
        print(info_string, end='\r')

    def carla_parse_vehicle_config(self) -> None:
        """Parse configuration parameter of vehicle."""
        # Max steering angle is needed to convert steering command
        # to `carla.VehicleControl(steer=)` which is between -1 and 1.
        physics_control = self.vehicle.get_physics_control()
        self.max_steering_angle_deg = physics_control.wheels[0].max_steer_angle

        assert physics_control.wheels[0].max_steer_angle == \
            physics_control.wheels[1].max_steer_angle, \
            f'Front tires have different max steering angle!'

    def carla_apply_emergency_stop(self):
        """Perform emergency stop until vehicle reaches standstill."""
        print('')

        while True:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0, brake=1)
            )
            sleep(0.1)

            v = self.vehicle.get_velocity()
            vehicle_vx = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2) / 2

            print(f'Emergency braking: v={vehicle_vx:0.2f}\r', end='')
            if vehicle_vx < 0.01:
                break

    def carla_set_vehicle_control(self, ctrl) -> None:
        """Convert `ControlOutput` to `carla.VehicleControl`.

        https://carla.readthedocs.io/en/latest/python_api/#methods_55

        NOTE: Normally, we need a lookup table for `throttle` and `brake
        command`, depending on the current speed and the desired ax, the
        pedal position is determined. We would need to run some measurements
        to obtain such a table. For the PID controller this should work
        anyway. For the application of the MPC, we must obtain those tables
        of implement PID controller here to control the ax of the vehicle.

        TODO: PID controller for `throttle` and `brake command`.

        The `steering command` must be calculate according to the maximum
        steering angle.
        """
        throttle = ctrl.ax/1.5 if ctrl.ax > 0 else 0
        brake = np.abs(ctrl.ax)/1.5 if ctrl.ax < 0 else 0
        steer = np.rad2deg(ctrl.delta_v) / self.max_steering_angle_deg

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer
            )
        )

    def carla_get_vehicle_information(self) -> np.ndarray:
        v = self.vehicle.get_velocity()
        a = self.vehicle.get_acceleration()
        return np.array([[
            self.vehicle.get_transform().location.x,
            self.vehicle.get_transform().location.y,
            self.vehicle.get_transform().rotation.yaw/180*np.pi-np.pi/2,
            self.vehicle.get_angular_velocity().z/180*np.pi,  # supposed to be rad/s, but value does make any sense without deg2rad
            np.sqrt(v.x**2 + v.y**2 + v.z**2),
            0
        ]]).T

    def carla_draw_reference(self, ref):
        """Draw reference trajectory on the road.

        Arguments
        ---------
        ref : class Trajectory
            Trajectory class with information for all nodes

        """
        for i in range(ref.x.shape[0] - 1):
            self.world.debug.draw_arrow(
                carla.Location(x=ref.x[i], y=ref.y[i], z=0),
                carla.Location(x=ref.x[i+1], y=ref.y[i+1], z=0),
                thickness=0.05,
                life_time=0.25,
                color=carla.Color(1, 0, 0, 0)
            )

    def run(self):
        """Run simulation for `t_end` of scenario."""
        try:
            self.__prepare_run()

            xk = self.scenario.x0
            t = 0
            self.sim['xk'][0, :, :] = xk

            while self.step < self.steps-1:
                # Calculate inputs and advance a time step
                xk = self.carla_get_vehicle_information()
                ctrl, ref = self.scenario.eval(t, xk)

                self.carla_draw_reference(ref)

                self.carla_set_vehicle_control(ctrl)

                if self.__count_laps(ctrl):
                    break

                xk = np.array(xk)  # convert from CasADi data type
                t = (self.step+1) * self.model.dt
                self.__collect_sim_data(t=t, xk=xk, ctrl=ctrl)

                self.step += 1
                self.__show_progress(t)

                self.world.wait_for_tick()
        finally:
            self.__trim_sim_data()
            self.carla_apply_emergency_stop()

    def show_states_and_input(self):
        """Visualize results of simulation.

        PublishResults class needs a certain organization of data, this
        is done here.

        Layout settings are also set here.
        """
        t, xk, uk = self.sim['t'], self.sim['xk'], self.sim['uk']
        data = list()

        # Layout settings
        sl = SubplotLayout('State and input vector over simulation time')

        # Organize data
        for i, signal in enumerate(self.model.state_names):
            _ = ScatterEntry(
                x=t.flatten(),
                y=xk[:, i, :].flatten(),
                name=signal,
                x_label='[s]',
                y_label=f'[{self.model.state_units[i]}]'
            )
            data.append(_)

        for i, signal in enumerate(self.model.input_names):
            _ = ScatterEntry(
                x=t.flatten(),
                y=uk[:, i, :].flatten(),
                name=signal,
                x_label='[s]',
                y_label=f'[{self.model.input_units[i]}]'
            )
            data.append(_)

        custom_data = [
            {'var': "e_y", 'name': "Lateral deviation", 'label_y': "[m]"},
            {'var': "e_psi", 'name': "Heading error", 'label_y': "[rad]"},
            {'var': "e_vx", 'name': "Velocity error", 'label_y': "[m/s]"},
            {'var': "ref_s", 'name': "Reference traveled distance", 'label_y': "[m]"},
        ]

        for item in custom_data:
            _ = ScatterEntry(
                x=t.flatten(),
                y=self.sim[item['var']].flatten(),
                name=item['name'],
                x_label='[s]',
                y_label=item['label_y']
            )
            data.append(_)

        # TODO: self.lap_steps to separate laps in plots
        Evaluation.subplots(data, sl, columns=3)

    def show_tracking(self):
        """Plot full reference trajectory and driven' paths of vehicle."""
        # Layout settings
        sl = SubplotLayout('Driven paths of vehicle')

        self.sim  # full sim data with
        Evaluation.tracking_plot(
            self.sim,
            self.scenario.reference.trajectory,
            sl
        )

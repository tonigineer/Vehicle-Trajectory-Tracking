"""Simulation framework."""

import numpy as np

from vmc.evaluation.evaluation import Evaluation, ScatterEntry, SubplotLayout
from vmc.evaluation.animation import AnimateVehicle, AnimationData


class BasicSimulator():
    """Framework to run simulation of models with a desired scenario."""

    enable_animation = True
    laps_max = 1
    lap_steps = []

    t_max = 2000  # needed for memory allocation
                  # TODO: dynamic memory allocation during sim

    def __init__(self, *, model, scenario) -> None:
        """Initiate class with model."""
        self.model = model
        self.scenario = scenario
        self.ani = AnimateVehicle(
            dt=self.model.dt, draw_rate=0.1
        )

    def __prepare_run(self) -> None:
        self.exec_time_sim = 0
        self.exec_time_control = 0

        if self.enable_animation:
            self.ani_data = AnimationData()
            self.ani_data.laps_max = self.laps_max
            if self.scenario.reference:
                self.ani_data.full_ref = self.scenario.reference.trajectory
                self.ani_data.full_ref_length = self.ani_data.full_ref.s[-1]

        self.lap = 1
        self.ref_s_prev = 0
        self.step = 0
        self.steps = int(self.t_max/self.model.dt) + 1

        self.sim = None
        self.__collect_sim_data()

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
        """Remove obsolete allocated sim data.

        Happens, if simulation is terminated by maximum number of laps.
        """
        idx = self.lap_steps[-1]
        for key, value in self.sim.items():
            self.sim[key] = value[1:idx, :, :] if len(value.shape) == 3 else value[1:idx, :]

    def __show_progress(self, t):
        info_string = f'Sim running ...  {t:4.2f}s /  {self.t_max}s [ﯩ: {self.lap} / {self.laps_max}]'
        print(info_string, end='\r')

    def run(self):
        """Run simulation for `t_end` of scenario."""
        self.__prepare_run()

        xk = self.scenario.x0
        t = 0
        self.sim['xk'][0, :, :] = xk

        while self.step < self.steps-1:
            # Calculate inputs and advance a time step
            ctrl, ref = self.scenario.eval(t, xk)
            xk = self.model.dxdt_nominal(xk, ctrl.uk)

            if self.__count_laps(ctrl):
                self.__trim_sim_data()
                break

            xk = np.array(xk)  # convert from CasADi data type
            t = (self.step+1)*self.model.dt
            self.__collect_sim_data(t=t, xk=xk, ctrl=ctrl)

            self.step += 1
            if self.enable_animation:
                self.ani_data.assign_sim_data(
                        t, self.step, ref, xk, ctrl, self.lap
                    )
                self.ani.draw_next_frame(self.ani_data)
            else:
                self.__show_progress(t)

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

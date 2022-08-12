"""Simulation framework."""

import numpy as np

from vmc.evaluation.evaluation import Evaluation, ScatterEntry, SubplotLayout
from vmc.evaluation.animation import AnimateVehicle


class Simulator():
    """Framework to run simulation of models with a desired scenario."""

    enable_animation = True

    def __init__(self, *, model, scenario):
        """Initiate class with model."""
        self.model = model
        self.scenario = scenario

    def __prepare_run(self):
        self.exec_time_sim = 0
        self.exec_time_control = 0

        self.ani = AnimateVehicle(dt=self.model.dt, draw_rate=0.1)

        self.step = 0
        self.steps = int(self.scenario.t_end/self.model.dt) + 1

        self.sim = {
            'x': np.zeros([self.steps, self.model.Nx, 1]),
            'u': np.empty([self.steps-1, self.model.Nu, 1]),
            't': np.empty([self.steps, 1]),
        }

    def run(self):
        """Run simulation for `t_end` of scenario."""
        self.__prepare_run()

        xk = self.scenario.x0
        t = 0
        self.sim['x'][0, :, :] = xk
        self.sim['t'][0, :] = t

        for k in range(self.steps-1):
            # Calculate inputs and advance a time step
            uk, ref = self.scenario.eval(t, xk)
            xk = self.model.dxdt_nominal(xk, uk)

            # avoid floating point arithmetic errors in t
            self.step += 1
            t = self.step*self.model.dt

            # Collect simulation data
            self.sim['x'][k+1, :, :] = xk
            self.sim['u'][k, :, :] = uk
            self.sim['t'][k+1, :] = t

            if self.enable_animation:
                self.ani.draw_next_frame(self.step, xk, uk)

    def show_states_and_input(self):
        """Visualize results of simulation.

        PublishResults class needs a certain organization of data, this
        is done here.

        Layout settings are also set here.
        """
        t, x, u = self.sim['t'], self.sim['x'], self.sim['u']
        data = list()

        # Layout settings
        sl = SubplotLayout('State and input vector over simulation time')

        # Organize data
        for i, signal in enumerate(self.model.state_names):
            _ = ScatterEntry(
                x=t.flatten(),
                y=x[:, i, :].flatten(),
                name=signal,
                x_label='[s]',
                y_label=f'[{self.model.state_units[i]}]'
            )
            data.append(_)

        for i, signal in enumerate(self.model.input_names):
            _ = ScatterEntry(
                x=t.flatten(),
                y=u[:, i, :].flatten(),
                name=signal,
                x_label='[s]',
                y_label=f'[{self.model.input_units[i]}]'
            )
            data.append(_)

        Evaluation.subplots(data, sl, columns=3)

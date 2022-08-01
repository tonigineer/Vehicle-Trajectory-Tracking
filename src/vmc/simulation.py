"""Simulation framework."""

import numpy as np

from vmc.evaluation import Evaluation, ScatterEntry, SubplotLayout


class SteerRamp():
    """Steering ramp at constant velocity as open loop scenario.

    Eval function returns input vector `u = [delta_v, ax]^T`
    for given time step `t`.

    Optional parameter `derivative==True` returns the derivative of
    the steering ramp, `steering velocity`.
    """

    name = "Open loop steering ramp."

    def __init__(self, *, dt=0.01, t_start=0, t_end=10, delta_v_max=np.deg2rad(4),
                 delta_vp=np.deg2rad(2), vx=100/3.6, derivative=False):
        """Initialize scenario for an open loop steering ramp."""
        self.dt = dt
        self.t_start = t_start
        self.t_end = t_end
        self.delta_v_max = delta_v_max
        self.delta_vp = delta_vp
        self.vx = vx
        self.derivative = derivative

        self.x0 = np.array([[0, 0, 0, 0, self.vx, 0, 0]]).T

    def eval(self, t):
        """Return input vector `u` according to time step `t`."""
        delta_v = min(
            max(t-self.t_start, 0) * self.delta_vp, self.delta_v_max
        )
        delta_v_prev = min(
            max(t-self.t_start-self.dt, 0) * self.delta_vp, self.delta_v_max
        )
        delta_vp = (delta_v-delta_v_prev) / self.dt

        steering_input = delta_vp if self.derivative else delta_v
        return np.array([[steering_input, 0]]).T


class Simulator():
    """Framework to run simulation of models with a desired scenario."""

    def __init__(self, *, model, scenario):
        """Initiate class with model."""
        self.model = model
        self.scenario = scenario

    def __prepare_run(self):
        self.exec_time_sim = 0
        self.exec_time_control = 0

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
            uk = self.scenario.eval(t)
            xk = self.model.dxdt_nominal(xk, uk)

            # Collect simulation data
            t += self.model.dt
            self.sim['x'][k+1, :, :] = xk
            self.sim['u'][k, :, :] = uk
            self.sim['t'][k+1, :] = t

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

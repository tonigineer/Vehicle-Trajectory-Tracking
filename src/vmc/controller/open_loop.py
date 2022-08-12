"""Collection of open loop controller.

These `controller` are not real controller, more scenarios with
a dynamic sequence of controller outputs.
"""

import numpy as np


class SteerRamp():
    """Steering ramp at constant velocity as open loop scenario.

    Eval function returns input vector `u = [delta_v, ax]^T`
    for given time step `t`.

    Optional parameter `derivative==True` returns the derivative of
    the steering ramp, `steering velocity`.
    """

    name = "Open loop steering ramp."

    def __init__(self, *, dt=0.01, t_start=0, t_end=10,
                 delta_v_max=np.deg2rad(4), delta_vp=np.deg2rad(2),
                 vx=100/3.6, derivative=False):
        """Initialize scenario for an open loop steering ramp."""
        self.dt = dt
        self.t_start = t_start
        self.t_end = t_end
        self.delta_v_max = delta_v_max
        self.delta_vp = delta_vp
        self.vx = vx
        self.derivative = derivative

        self.x0 = np.array([[0, 0, 0, 0, self.vx, 0, 0]]).T

    def eval(self, **kargs) -> np.ndarray:
        """Return input vector `u` according to time step `t`."""
        t = kargs['t']

        delta_v = min(
            max(t-self.t_start, 0) * self.delta_vp, self.delta_v_max
        )
        delta_v_prev = min(
            max(t-self.t_start-self.dt, 0) * self.delta_vp, self.delta_v_max
        )
        delta_vp = (delta_v-delta_v_prev) / self.dt

        steering_input = delta_vp if self.derivative else delta_v
        return np.array([[steering_input, 0]]).T

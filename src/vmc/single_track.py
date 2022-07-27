"""Single track model for vehicle dynamics."""

import casadi as cs
import casadi.tools as ct
import numpy as np

from vmc.common import BaseModel


class FSVehSingleTrack(BaseModel):
    """`Full scale vehicle` modeled via `Bicycle model` and `disturbances`.

    System:
        `xk+1 = fd(xk, uk)`

    State vector:
        `x = [x, y, psi, psip, vx, vy, delta_v]^T`

    Input vector:
        `u = [delta_vp, ax]^T`
    """

    # System
    Nx = 7
    Nu = 2

    dt = 0.01

    # Symbolics for CasADI
    x = ct.struct_symMX(['X', 'Y', 'psi', 'psip', 'vx', 'vy', 'delta_v'])
    u = ct.struct_symMX(['delta_vp', 'ax'])

    state_names = ['Position in x', 'Position in y', 'Yaw angel', 'Yaw rate',
                   'Velocity in x (vehicle)', 'Velocity in y (vehicle)',
                   'Steering angle']
    state_units = ['m', 'm', 'rad', 'rad/s', 'm/s', 'm/s', 'rad']

    input_names = ['Steering change', 'Acceleration']
    input_units = ['rad/s', 'm/ss']

    # Parameter
    m = 2900            # mass [kg]
    J_z = 2860          # yaw inertia [kg m^2]
    l_f = 1.47          # distance CoG to front axle [m]
    l_r = 1.50          # distance CoG to rear axle [m]

    # linear tire model
    calpha_f = 120000   # cornering stiffness front axle
    calpha_r = 180000   # cornering stiffness rear axle

    # nonlinear tire model
    # TODO: implement simple Pacejka Tire Model

    delta_v_max = np.deg2rad(30)    # maximum steering angle
    ax_max = 3                      # maximum acceleration
    ax_min = -8                     # maximum deceleration
    # TODO: implement powertrain with pedal position 1 to -1 and forces

    def __init__(self):
        """Initialize bicycle model for vehicle."""
        self.f_nominal = self.define_nominal_model(self.x, self.u)
        self.dxdt_nominal = self.integration(self.dt, self.f_nominal, self.x, self.u)

    def define_nominal_model(self, x, u):
        """Define model for CasADi as symbolic function.

        Arguments
        ---------
        `x : array`
            Symbolic struct of state vector.
        `u : array`
            Symbolic struct of input vector.

        Return
        ------
        `f : function`
            f(xk+1) = A*xk + B*uk

        NOTE: Consider to use only CasADi function. And return
        state vector as cs.vectcat
        """
        _, _, psi, psip, vx, vy, delta_v = 0, 0, 0, 0, 0, 0, x['delta_v']
        delta_vp, ax = u['delta_vp'], 0

        # Inputs
        # delta_v = np.clip(delta_v+delta_vp, -self.delta_v_max, self.delta_v_max)
        # ax = np.clip(ax, self.ax_min, self.ax_max)

        # delta_v = delta_v+delta_vp

        # Tire forces
        alpha_f = cs.atan2(vy-self.l_r*psip, vx)
        alpha_r = cs.atan2(vy+self.l_f*psip, vx) - delta_v

        Fy_f = alpha_f * self.calpha_f
        Fy_r = alpha_r * self.calpha_r

        # Derivatives
        xp = vx*np.cos(psi) - vy*np.sin(psi)
        yp = vx*np.sin(psi) + vy*np.cos(psi)
        psipp = (Fy_f*self.l_f - Fy_r*self.l_r) / self.J_z
        # vxp = ax                                # TODO: not suitable for high dynamics, because tire might be saturated
        vyp = (Fy_f+Fy_r) / self.m - vx*psip    # Calculation differs quite a lot from Lucas' implementation

        rhs = cs.vertcat(xp, yp, psip, psipp, ax, vyp, delta_vp)
        return cs.Function('f', [x, u], [rhs], ['x', 'u'], ['dx/dt'])


# if __name__ == '__main__':
#     model = FSVehSingleTrack()

#     x0 = np.array([[0, 0, 0, 0, 20, 0, 0]]).T
#     u = np.array([[0.05, 0]]).T
#     print(model.dxdt_nominal(x0, u))

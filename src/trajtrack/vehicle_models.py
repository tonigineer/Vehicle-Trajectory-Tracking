
"""Collection of vehicle models, vehicle parameter and integration methods."""

import casadi as cs
import casadi.tools as ct
import numpy as np

RHO = 1.225     # Air density
GRAVITY = 9.81  # Gravitational constant


class BaseModel:
    """Class with basic functionality for a vehicle model."""

    @staticmethod
    def integration(dt, f, x, u):
        """Define `Runge-Kutta-4` integrator.

        Normally, CasADi uses SUNIDALS integrator, which comes
        with an additional overhead. So, using this manual integration
        saves computational effort.
        """
        RK4 = cs.integrator(
            'RK4', 'rk',
            {'x': x, 'p': u, 'ode': f(x, u)},
            {'number_of_finite_elements': 1}
        )

        # Discretized (sampling time dt) system dynamics as a CasADi Function
        F_RK4 = cs.Function('F_RK4', [x, u], [RK4(x0=x, p=u)["xf"]], ['x[k]', 'u[k]'], ['x[k+1]'])

        # RK4
        k1 = f(x, u)
        k2 = f(x + dt / 2.0 * k1, u)
        k3 = f(x + dt / 2.0 * k2, u)
        k4 = f(x + dt * k3, u)
        xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Single step time propagation
        F_RK4 = cs.Function("F_RK4", [x, u], [xf], ['x[k]', 'u[k]'], ['x[k+1]'])
        return F_RK4


class BicycleModel(BaseModel):
    """`Full scale vehicle` modeled via `bicycle model` and `resistances`.

    System:
        `xk+1 = fd(xk, uk)`
    State vector:
        `x = [x, y, psi, psip, vx, vy]^T`
    Input vector:
        `u = [delta_v, ax]^T`
    """

    dt = 0.01  # seconds
    tire_model = 'nonlinear'

    # Symbolics for CasADI
    x = ct.struct_symMX(['X', 'Y', 'psi', 'psip', 'vx', 'vy'])
    u = ct.struct_symMX(['delta_v', 'ax'])

    state_names = ['Position in x', 'Position in y', 'Yaw angle', 'Yaw rate',
                   'Velocity in x (vehicle)', 'Velocity in y (vehicle)']
    state_units = ['m', 'm', 'rad', 'rad/s', 'm/s', 'm/s', 'rad']

    input_names = ['Steering angle', 'Acceleration']
    input_units = ['rad', 'm/ss']

    # Vehicle parameter
    m = 2900            # mass [kg]
    J_z = 2860          # yaw inertia [kg m^2]
    l_f = 1.47          # distance CoG to front axle [m]
    l_r = 1.50          # distance CoG to rear axle [m]
    cog_h = 0.6         # hight CoG from ground [m]

    # Linear tire model
    calpha_f = 120000   # cornering stiffness front axle
    calpha_r = 180000   # cornering stiffness rear axle

    # Nonlinear tire model
    C_f, C_r = 1.0, 1.25
    B_f, B_r = 18.5, 16.5

    # Drag and roll resistance
    A = 1.80            # cross sectional area
    cw = 0.33           # drag coefficient
    f_roll = 0.0125     # Roll resistance coefficient

    # Actuator constraints
    delta_v_max = np.deg2rad(45)    # maximum steering angle [deg]
    ax_max = 3                      # maximum acceleration [m/ss]
    ax_min = -8                     # maximum deceleration [m/ss]

    def __init__(self):
        """Initialize bicycle model for vehicle."""
        self.Nx = self.x.shape[0]
        self.Nu = self.u.shape[0]

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
        psi, psip, vx, vy = x['psi'], x['psip'], x['vx'], x['vy']
        delta_v, ax = u['delta_v'], u['ax']

        vx = cs.fmax(vx, 1e-5)  # catch division by zero during standstill

        # Apply actuator constraints
        delta_v = cs.fmax(cs.fmin(delta_v, self.delta_v_max), -self.delta_v_max)
        ax = cs.fmax(cs.fmin(ax, self.ax_max), self.ax_min)

        # Tire slip angle
        alpha_f = delta_v - cs.atan(self.l_f*psip+vy / vx)
        alpha_r = - cs.atan(vy-self.l_r*psip / vx)

        # Lateral tire forces
        if self.tire_model == 'linear':
            Fy_f = alpha_f * self.calpha_f
            Fy_r = alpha_r * self.calpha_r
        elif self.tire_model == 'nonlinear':
            # Simple model for vertical tire force with pitch
            Fz_f = self.m * (-ax * self.cog_h + GRAVITY * self.l_r) / \
                (self.l_f + self.l_r)
            Fz_r = self.m * (ax * self.cog_h + GRAVITY * self.l_f) / \
                (self.l_f + self.l_r)

            # Simple Pacejka tire model
            Fy_f = Fz_f * np.sin(self.C_f * np.arctan(self.B_f * alpha_f))
            Fy_r = Fz_r * np.sin(self.C_r * np.arctan(self.B_r * alpha_r))
        else:
            raise NotImplementedError(
                f'Tire model {self.tire_model} not implemented yet.'
            )

        # Dynamics
        xp = vx*np.cos(psi) - vy*np.sin(psi)
        yp = vx*np.sin(psi) + vy*np.cos(psi)

        psipp = (Fy_f*self.l_f - Fy_r*self.l_r) / self.J_z
        vyp = (Fy_f+Fy_r) / self.m - vx*psip

        # Longitudinal modeling
        # Only applying drag and roll resistance to accel command.
        ax_retard = +RHO/2 * self.A * self.cw * vx**2 + \
            self.f_roll * self.m * GRAVITY
        ax -= ax_retard/self.m
        # TODO: vxp = ax is not suitable for high dynamics, because
        # tires may be saturated!

        rhs = cs.vertcat(xp, yp, psip, psipp, ax, vyp)
        return cs.Function('f', [x, u], [rhs], ['x', 'u'], ['dx/dt'])

    def plot_tire_models(self):
        """Plot tire models for current parameters."""
        import matplotlib.pyplot as plt

        plt.figure()

        # Linear tire model
        ALPHA_MAX = 7.5
        alpha = np.linspace(np.deg2rad(-ALPHA_MAX), np.deg2rad(ALPHA_MAX), 100)
        Fy_f, Fy_r = alpha * self.calpha_f, alpha * self.calpha_r
        plt.plot(alpha, Fy_f, alpha, Fy_r)

        # Nonlinear tire model with Fz as half of vehicle mass
        Fy_f = self.m/2*GRAVITY * np.sin(self.C_f * np.arctan(self.B_f * alpha))
        Fy_r = self.m/2*GRAVITY * np.sin(self.C_r * np.arctan(self.B_r * alpha))
        plt.plot(alpha, Fy_f, alpha, Fy_r)

        plt.legend(
            ['Fy_f linear', 'Fy_r linear', 'Fy_f nonlinear', 'Fy_r nonlinear']
        )

        plt.show()


class Vehicle:
    """Wrapper class to have a standardize vehicle model interface."""

    def __init__(self, vehicle_model):
        self.model = vehicle_model()

    def apply(
        self, current_state: np.ndarray, control_input: np.ndarray
    ) -> np.ndarray:
        """Call model's integration function to advance `dt` to next state."""
        return np.array(
            self.model.dxdt_nominal(current_state, control_input)
        )


def development():
    """Run simple test for module."""
    # Exemplary state and input vector
    x0 = np.array([[0, 0, 0, 0, 20, 0]]).T
    u = np.array([[0.05, 0]]).T

    # Create vehicle with model
    vehicle = Vehicle(BicycleModel)

    # Simulate one time step
    print(f'Integrate model for one time step dt: {vehicle.model.dt} s')
    print(f'x_t0:  {x0.flatten()} with input: {u.flatten()}')
    print(f'x_t+1: {vehicle.apply(x0, u)}')

    vehicle.model.plot_tire_models()


if __name__ == '__main__':
    development()

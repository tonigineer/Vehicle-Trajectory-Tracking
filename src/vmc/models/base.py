"""Collection of common functionality for vehicle models."""

import numpy as np
import casadi as cs


class BaseModel:
    """Class with basic functionality for a vehicle model."""

    @staticmethod
    def integration(dt, f, x, u):
        """Define `Runge-Kutta-4` integrator.

        Normally, CasADi uses SUNIDALS integrator, which comes
        with an additional overhead. So, using this manual integration
        saves computational effort.
        """
        RK4 = cs.integrator('RK4', 'rk', {'x': x, 'p': u, 'ode': f(x, u)}, {'number_of_finite_elements': 1, 'tf': dt})

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

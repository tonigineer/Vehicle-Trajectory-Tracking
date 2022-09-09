"""Units tests for common module of vmc package."""

import numpy as np
import casadi as cs
import casadi.tools as ct

from misc import eps_float_equality

from vmc.models import BaseModel
from vmc.models import FSVehSingleTrack

base_model = BaseModel()


def test_rk4():
    """Test RK4-integration with simple system."""
    dt = 1

    # System of uniform acceleration
    x = ct.struct_symMX(['s', 'v'])
    u = ct.struct_symMX(['a'])
    rhs = cs.vertcat(x['v'], u['a'])
    f = cs.Function('f', [x, u], [rhs], ['x', 'u'], ['dx/dt'])

    dxdt = base_model.integration(dt, f, x, u)

    x0 = np.array([[0, 0]])
    u = np.array([[1]])
    xk = dxdt(x0, u)

    assert eps_float_equality(xk[0], 0.5)
    assert eps_float_equality(xk[1], 1.0)


def test_FSVehSingleTrack():
    """Test vehicle model with one time step for left turn."""
    model = FSVehSingleTrack()

    x0 = np.array([[0, 0, 0, 0, 20, 0]]).T  # heading north!
    u = np.array([[0.05, 2]]).T

    X, Y, psi, psip, vx, vy = np.array(model.dxdt_nominal(x0, u))

    assert X < x0[0]
    assert Y > x0[1]
    assert psi > x0[2]
    assert psip > x0[3]
    assert (vx < x0[4]+u[1]*model.dt and vx > x0[4])
    assert vy > x0[5]

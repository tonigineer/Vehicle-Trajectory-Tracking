"""Define tests for open loop controller of vmc package."""

import numpy as np

from misc import eps_float_equality
from vmc.controller import SteerRamp


def test_SteerRamp():
    """Test functionality of open loop SteerRamp scenario."""
    t_start = 1
    t_end = 3
    dt = 0.1
    dv = np.deg2rad(1)
    dvp = np.deg2rad(1)

    steer_ramp = SteerRamp(dt=dt, t_start=t_start, t_end=t_end,
                           delta_v_max=dv, delta_vp=dvp, derivative=False)

    assert eps_float_equality(steer_ramp.eval(t_start)[0], 0.0)
    assert eps_float_equality(steer_ramp.eval(t_start)[1], 0.0)

    assert steer_ramp.eval(t_start+dt)[0] > 0.0
    assert eps_float_equality(steer_ramp.eval(t_start+dt)[1], 0.0)

    assert eps_float_equality(steer_ramp.eval(t_start+dv/dvp)[0], dv)
    assert eps_float_equality(steer_ramp.eval(t_start+dv/dvp)[1], 0.0)

    assert eps_float_equality(steer_ramp.eval(t_end)[0], dv)
    assert eps_float_equality(steer_ramp.eval(t_end)[1], 0.0)


def test_SteerRamp_derivative():
    """Test functionality of open loop SteerRamp scenario."""
    t_start = 1
    t_end = 3
    dt = 0.1
    dv = np.deg2rad(1)
    dvp = np.deg2rad(1)

    steer_ramp = SteerRamp(dt=dt, t_start=t_start, t_end=t_end,
                           delta_v_max=dv, delta_vp=dvp, derivative=True)

    assert eps_float_equality(steer_ramp.eval(t_start)[0], 0.0)
    assert eps_float_equality(steer_ramp.eval(t_start)[1], 0.0)

    assert steer_ramp.eval(t_start+dt)[0] > 0.0
    assert eps_float_equality(steer_ramp.eval(t_start+dt)[1], 0.0)

    assert eps_float_equality(steer_ramp.eval(t_start+dv/dvp)[0], dvp)
    assert eps_float_equality(steer_ramp.eval(t_start+dv/dvp)[1], 0.0)

    assert eps_float_equality(steer_ramp.eval(t_end)[0], 0.0)
    assert eps_float_equality(steer_ramp.eval(t_end)[1], 0.0)

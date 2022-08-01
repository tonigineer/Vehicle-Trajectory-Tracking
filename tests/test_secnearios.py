"""Define tests for simulation of vmc package."""

from vmc.simulation import SteerRamp
import numpy as np


def test_SteerRamp():
    """Test functionality of open loop SteerRamp scenario."""
    t_start = 1
    t_end = 3
    dt = 1
    dv = np.deg2rad(1)
    dvp = np.deg2rad(1)

    steer_ramp = SteerRamp(dt=dt, t_start=t_start, t_end=t_end,
                           delta_v_max=dv, delta_vp=dvp)

    assert steer_ramp.eval(t_start, False)[0] == 0.0
    assert steer_ramp.eval(t_start, False)[1] == 0.0

    assert steer_ramp.eval(t_start+dt, False)[0] == dv
    assert steer_ramp.eval(t_start+dt, False)[1] == 0.0

    assert steer_ramp.eval(t_end+dt, False)[0] == dv
    assert steer_ramp.eval(t_end+dt, False)[1] == 0.0


def test_SteerRamp_derivative():
    """Test functionality of open loop SteerRamp scenario."""
    t_start = 1
    t_end = 3
    dt = 1
    dv = np.deg2rad(1)
    dvp = np.deg2rad(1)

    steer_ramp = SteerRamp(dt=dt, t_start=t_start, t_end=t_end,
                           delta_v_max=dv, delta_vp=dvp, derivative=True)

    assert steer_ramp.eval(t_start)[0] == 0.0
    assert steer_ramp.eval(t_start)[1] == 0.0

    assert steer_ramp.eval(t_start+dt)[0] > 0.0
    assert steer_ramp.eval(t_start+dt)[1] == 0.0

    assert steer_ramp.eval(t_end-dt)[0] > 0
    assert steer_ramp.eval(t_end-dt)[1] == 0

    assert steer_ramp.eval(t_end)[0] == 0.0
    assert steer_ramp.eval(t_end)[1] == 0.0

"""Test basic functions for controller."""

import numpy as np

from misc import eps_float_equality
from vmc.controller import TrajTrackPID
from vmc.trajectories import Trajectory, Position

pid = TrajTrackPID()


def test_heading_angle_error():
    """Check if heading angle calculation is correct."""
    ANGLE_DIFF = np.deg2rad(5)
    psi_traj = np.arange(-np.pi/2, np.pi*5/2, 0.1)

    # Create wrap around at 0/2pi (really interesting test cases)
    psi_traj = psi_traj % (2*np.pi)

    psi_vehicle = psi_traj + ANGLE_DIFF

    for a1, a2 in zip(psi_traj, psi_vehicle):
        assert eps_float_equality(pid.heading_error(a1, a2), ANGLE_DIFF)
        assert eps_float_equality(pid.heading_error(a2, a1), -ANGLE_DIFF)


def test_localize_on_trajectory_exactly():
    """Check if lateral deviation calculation is correct.

    Test case uses a quarter arc as trajectory and the same points
    as vehicle position.

    `s_localized` must be the same as `s` of the trajectory.
    """
    N_NODES = 20
    RADIUS = 25

    psi = np.linspace(0, np.pi/2, N_NODES)
    x = RADIUS * np.cos(psi)
    y = RADIUS * np.sin(psi)
    s = np.linspace(0, np.pi/2 * RADIUS, N_NODES)

    vec0 = np.zeros([N_NODES, 1])  # other attributes not needed now

    T = Trajectory(x, y, s, vec0, vec0, vec0, vec0)

    for k in range(len(x)):
        P = Position(x[k], y[k])
        s_localized = pid.localize_on_trajectory(T, P, N_NODES-1)

        # NOTE: Normally, the error should be even smaller, here another
        # look is needed, why the error for identical ego position is not
        # just an EPS.
        msg = 'Travel distances does not match.'
        assert np.diff([s[k], s_localized]) < 0.001, msg


def test_localize_on_trajectory_inside():
    """Check if lateral deviation calculation is correct.

    Test case uses a quarter arc as trajectory and `center`
    as vehicle position.

    `s_localized` must always be between two nodes.
    """
    N_NODES = 20
    RADIUS = 25
    RADIUS_EGO = 1  # > 0

    psi = np.linspace(0, np.pi/2, N_NODES)
    x = RADIUS * np.cos(psi)
    y = RADIUS * np.sin(psi)
    s = np.linspace(0, np.pi/2 * RADIUS, N_NODES)

    psi = np.linspace(
        (psi[1]-psi[0])/2,
        psi[-2] + (psi[-1]-psi[-2])/2,
        N_NODES-1
    )  # in between other nodes
    x_ego = RADIUS_EGO * np.cos(psi)
    y_ego = RADIUS_EGO * np.sin(psi)

    vec0 = np.zeros([N_NODES, 1])  # other attributes not needed now
    T = Trajectory(x, y, s, vec0, vec0, vec0, vec0)

    for k in range(len(x_ego)):
        P = Position(x_ego[k], y_ego[k])
        s_localized = pid.localize_on_trajectory(T, P, N_NODES-1)

        msg = 'Travel distances not between nodes.'
        assert s[k+1] > s_localized > s[k], msg


def test_localize_on_trajectory_outside():
    """Check if lateral deviation calculation is correct.

    Test case uses a quarter arc as trajectory and a quarter arc
    in a higher resolution as ego position.

    `s_localized` must be monotonic while checking for the whole
    arc of the ego position.
    """
    N_NODES = 20
    RADIUS = 25

    psi = np.linspace(0, np.pi/2, N_NODES)
    x = RADIUS * np.cos(psi)
    y = RADIUS * np.sin(psi)
    s = np.linspace(0, np.pi/2 * RADIUS, N_NODES)

    psi = np.linspace(0, np.pi/2, N_NODES*100)
    x_ego = RADIUS * np.cos(psi)
    y_ego = RADIUS * np.sin(psi)

    vec0 = np.zeros([N_NODES, 1])  # other attributes not needed now

    T = Trajectory(x, y, s, vec0, vec0, vec0, vec0)

    s_current = 0
    for k in range(len(x_ego)-1):
        P = Position(x_ego[k], y_ego[k])
        s_localized = pid.localize_on_trajectory(T, P, N_NODES-1)
        assert s_localized >= s_current
        s_current = s_localized


def test_interpolate_node():
    """Check interpolate node.

    All attributes are the same, so when interpolating
    according to `s`, all values must be `s_current`.
    """
    dummy_vec = np.linspace(0, 19, 20)

    T = Trajectory(
        dummy_vec, dummy_vec, dummy_vec, dummy_vec,
        dummy_vec, dummy_vec, dummy_vec
    )

    for s_current in np.linspace(0, 19, 1000):
        node = pid.interpolate_node(T, s_current)
        assert eps_float_equality(node.x, s_current), 'X not correctly interpolated'
        assert eps_float_equality(node.y, s_current), 'Y not correctly interpolated'
        assert eps_float_equality(node.s, s_current), 's not correctly interpolated'
        assert eps_float_equality(node.psi, s_current), 'psi not correctly interpolated'
        assert eps_float_equality(node.kappa, s_current), 'kappa not correctly interpolated'
        assert eps_float_equality(node.v, s_current), 'v not correctly interpolated'
        assert eps_float_equality(node.a, s_current), 'a not correctly interpolated'


def test_make_s_strictly_monotonic():
    """Check with hardcoded edge cases."""
    N_NODES = 3

    zeros_vec = np.zeros([N_NODES, 1])
    s_sample = [np.array([3, 1, 2]), np.array([1, 2, 3]), np.array([2, 3, 1])]

    for s in s_sample:
        T = Trajectory(
            zeros_vec, zeros_vec, s, zeros_vec, zeros_vec,
            zeros_vec, zeros_vec
        )

        T = pid.make_s_strictly_monotonic(T)
        assert np.all(np.diff(T.s) > 0), 's not strictly monotonic'

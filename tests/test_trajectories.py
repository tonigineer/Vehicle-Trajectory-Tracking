"""Tests for simulation module of vmc package."""

import numpy as np

from vmc.trajectories import ReferencePath, Position


def test_ReferencePath():
    """Return trajectory from offline reference."""
    TRACK_FILEPATH = './examples/basic/tracks/Algarve_International_Circuit_02g_02g_128.json'
    N_NODES = 25

    Ref = ReferencePath(TRACK_FILEPATH, N_NODES)

    for x, y in zip(Ref.t.x, Ref.t.y):
        pos = Position(x, y)
        traj = Ref.eval(pos)

        msg = 'Trajectory contains different number of nodes'
        assert len(traj.x) == N_NODES, msg

        msg = 'Trajectory contains identical x position'
        assert len(set(traj.x)) == len(traj.x), msg
        msg = 'Trajectory contains identical y position'
        assert len(set(traj.y)) == len(traj.y), msg

        msg = 'Trajectory contains first (s=0) node in between'
        assert not np.isin(0, traj.s) or traj.s[0] == 0, msg
        msg = 'Trajectory nodes are not strictly monotonic (ones allowed, \
            lap wrap)'
        assert len(np.diff(traj.s) > 0) >= N_NODES-1, msg

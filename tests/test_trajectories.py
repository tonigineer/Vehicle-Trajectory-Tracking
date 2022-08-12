"""Tests for simulation module of vmc package."""

import numpy as np

from vmc.trajectories import OfflineReference, Position


def test_OfflineReference():
    """Return trajectory from offline reference."""
    TRACK_FILEPATH = './examples/tracks/Algarve_International_Circuit_02g_02g_128.json'
    N_NODES = 25

    Ref = OfflineReference(TRACK_FILEPATH, N_NODES)

    for x, y in zip(Ref.t.X, Ref.t.Y):
        pos = Position(x, y)
        traj = Ref.eval(pos)

        msg = 'Trajectory contains different number of nodes'
        assert len(traj.X) == N_NODES, msg

        msg = 'Trajectory contains identical x position'
        assert len(set(traj.X)) == len(traj.X), msg
        msg = 'Trajectory contains identical y position'
        assert len(set(traj.Y)) == len(traj.Y), msg

        msg = 'Trajectory contains first (s=0) node in between'
        assert not np.isin(0, traj.s) or traj.s[0] == 0, msg
        msg = 'Trajectory nodes are not strictly monotonic (ones allowed, \
            lap wrap)'
        assert len(np.diff(traj.s) > 0) >= N_NODES-1, msg

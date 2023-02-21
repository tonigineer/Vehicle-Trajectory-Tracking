"""Collection of controller for trajectory tracking."""

from dataclasses import dataclass
import numpy as np

from vmc.trajectories import Trajectory, Position, Node


@dataclass
class ControlOutput:
    """Standardized controller output for simulator."""

    uk: np.ndarray      # uk array for state space representation
    delta_v: float     # steering velocity
    ax: float           # vehicle acceleration in x
    e_y: float          # lateral deviation
    e_psi: float        # heading error
    e_vx: float         # velocity error
    ref_s: float        # reference traveled distance

    def __post_init__(self):
        """Check data types."""
        assert type(self.uk) == np.ndarray
        assert self.uk.shape == (2, 1)
        assert type(self.delta_v) == np.float64
        assert type(self.ax) == np.float64
        assert type(self.e_y) == np.float64
        assert type(self.e_psi) == np.float64
        assert type(self.e_vx) == np.float64
        assert type(self.ref_s) == np.float64


def localize_on_trajectory(t: Trajectory, p: Position, nrn: int = 5) -> float:
    """Determine traveled distance on trajectory according to vehicle position.

    The algorithm determines the shortest point from `p` to the trajectory,
    that is perpendicular to a tangent along this trajectory. If the shortest
    point lays outside two points of the trajectory, the closest point used.

    Returned is the traveled distance to this point. This way, all other
    attributes can be interpolated.

    Arguments
    ---------
    t : Trajectory
        Trajectory with `n` nodes, each node containing all seven attributes.
    p : Position
        Ego position of vehicle.
    nrn : int
        Number of relevant nodes, that are taken into consideration for calc.
        If start of trajectory is close to ego position, less `nrn` are needed.
        Default implementation ensures that.

    Return
    ------
    ref_s : float
        Traveled distance along trajectory, can be used to interpolate current
        reference point on trajectory.

    """
    pp = np.array([p.x, p.y])
    # plt.plot(pp[0], pp[1], marker="x")
    assert len(t.x) > nrn, f'Number of relevant nodes ({nrn}) must be \
        smaller than the number nodes ({len(t.x)}) of the trajectory'

    # Allocate memory
    pcs = np.zeros([nrn, 2])
    pc_diffs = np.zeros([nrn, 1])
    distances = np.zeros([nrn, 1])
    in_between = np.zeros([nrn, 1])

    for i in range(nrn):
        # Points of line
        pa = np.array([t.x[i], t.y[i]])
        pb = np.array([t.x[i+1], t.y[i+1]])

        # Vectors
        AB = pb-pa
        AP = pp-pa

        # Length from pa to projection pc on AB, where vector through pp is
        # perpendicular on AB. May be outside of AB!
        pc_diff = np.dot(AB, AP) / np.dot(AB, AB) * AB
        pc = pa + pc_diff

        AC = pc-pa

        # Source: https://lucidar.me/en/mathematics/check-if-a-point-belongs-on-a-line-segment/
        on_vector = not(np.dot(AB, AC) < 0 or np.dot(AB, AC) > np.dot(AB, AB))

        pcs[i, :] = pc
        pc_diffs[i, :] = np.linalg.norm(pc-pa)
        distances[i, :] = np.linalg.norm(pc-pp)
        in_between[i, :] = on_vector

    #     plt.plot(pc[0], pc[1], marker="o")
    #     plt.plot([pc[0], pa[0]], [pc[1], pa[1]])
    # plt.plot(t.x, t.y)
    # plt.show()

    if np.any(in_between):
        distances[in_between == 0] = np.inf
        k = np.argmin(distances)
    else:
        k = np.argmin(pc_diffs)

    return t.s[k] + pc_diffs[k, 0]


def interpolate_node(t: Trajectory, ref_s: float) -> Node:
    """Interpolate node in between `trajectory` according to `ref_s`."""
    x_ref = np.interp(ref_s, t.s, t.x)
    y_ref = np.interp(ref_s, t.s, t.y)
    psi_ref = np.interp(ref_s, t.s, t.psi)
    kappa_ref = np.interp(ref_s, t.s, t.kappa)
    v_ref = np.interp(ref_s, t.s, t.v)
    a_ref = np.interp(ref_s, t.s, t.a)
    return Node(x_ref, y_ref, ref_s, psi_ref, kappa_ref, v_ref, a_ref)


def heading_error(psi_ref: float, psi_vehicle: float) -> float:
    """Calculate error between two heading angle.

    Function handles warp around at 0/2pi.

    Arguments:
    ---------
    psi_ref : float
        Heading of trajectory/reference
    psi_vehicle : float
        Heading of vehicle

    Return:
    ------
    heading_error : float
        Angle difference between both inputs in rad.

    """
    cp = np.cos(psi_ref)*np.sin(psi_vehicle) - \
        np.sin(psi_ref)*np.cos(psi_vehicle)
    dp = np.cos(psi_ref)*np.cos(psi_vehicle) + \
        np.sin(psi_ref)*np.sin(psi_vehicle)
    return np.arctan2(cp, dp)


def make_s_strictly_monotonic(t: Trajectory) -> Trajectory:
    """Make `s` strictly monotonic."""
    if not all(np.diff(t.s) > 0):
        idc = t.s < t.s[0]
        t.s[idc] += np.max(t.s)
    return t


def make_psi_continuous(t: Trajectory) -> Trajectory:
    """Make `psi` continuous for interpolation."""
    t.psi = np.unwrap(2 * t.psi) / 2

    msg = f'heading angle appears not to be continuous {t.psi}'
    assert np.any(np.abs(np.diff(t.psi)) < np.deg2rad(45)), msg

    return t


class TrajTrackPID:
    """PID controller for trajectory tracking."""

    dt = 0.01

    # Controller parameter
    kp_e_y = 0.5
    ki_e_y = 0.1
    kd_e_y = 0.1

    kp_e_psi = 1

    kp_vx = 2

    __e_y_last = 0
    __e_y_integral = 0

    def __pid_ffw_control(self, ref_node, e_y, e_psi, e_vx):
        """Apply PID to for delta_v and ax."""
        delta_v, ax = 0, 0

        # Proportional gains
        delta_v += e_y * self.kp_e_y
        delta_v += e_psi * self.kp_e_psi
        ax = e_vx * self.kp_vx

        # Integral gains
        self.__e_y_integral += e_y * self.dt
        delta_v += self.__e_y_integral * self.ki_e_y

        # Differential gains
        e_yp = (e_y - self.__e_y_last) / self.dt
        self.__e_y_last = e_y
        delta_v += e_yp * self.kd_e_y

        # Feed forward
        ax += ref_node.a

        # TODO: feed forward for `delta_y` based on kappa is still
        #       missing.

        return delta_v, ax

    @staticmethod
    def __tracking_errors(ref_node, veh_x, veh_y, veh_psi, veh_vx):
        """Calculate tracking errors of vehicle to reference."""
        x_diff = veh_x-ref_node.x
        y_diff = veh_y-ref_node.y
        e_y = np.linalg.norm(
            np.array([x_diff, y_diff])
        )
        if not (np.sin(ref_node.psi + np.pi/2) * x_diff -
                np.cos(ref_node.psi + np.pi/2) * y_diff) > 0:
            e_y *= -1

        e_psi = heading_error(veh_psi, ref_node.psi)

        e_vx = ref_node.v - veh_vx

        return e_y, e_psi, e_vx

    def eval(self, **kwargs):
        """Apply PID control to simulation.

        Arguments
        ---------
        state_vector : ndarray
            Current system state.
        ref : Trajectory
            Current reference trajectory `ref`.

        Returns
        -------
        ctrl_out : ControlOut
            Contains `uk` and `err*`

        """
        xk = kwargs['state_vector']
        ref = kwargs['ref']

        veh_x, veh_y, veh_psi = xk.flatten()[0:3]
        veh_vx = xk.flatten()[4]

        ref = make_s_strictly_monotonic(ref)
        ref = make_psi_continuous(ref)

        # Calculate tracking errors
        ref_s = localize_on_trajectory(
            t=ref, p=Position(veh_x, veh_y)
        )
        ref_node = interpolate_node(ref, ref_s)
        e_y, e_psi, e_vx = self.__tracking_errors(
            ref_node, veh_x, veh_y, veh_psi, veh_vx
        )

        delta_v, ax = self.__pid_ffw_control(ref_node, e_y, e_psi, e_vx)

        return ControlOutput(
            np.array([[delta_v, ax]]).T, delta_v, ax, e_y, e_psi, e_vx, ref_s
        )

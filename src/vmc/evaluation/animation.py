"""Main module for animation tasks."""

from curses import COLOR_GREEN
import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter, sleep
from dataclasses import dataclass

from vmc.trajectories import Trajectory


C_BACKGROUND = '#101010'
C_FPS = '#AC2C24'
C_SIM_INFO = '#484848'
C_CTRL_INFO = '#484848'
C_VEHICLE = '#EE8F90'
C_REF = '#FADB75'
C_FULL_REF = '#EBE8E9'


def rotate_point(x, y, phi):
    """Rotate a point `(x,y)` by angle=`phi`."""
    return (
        x * np.cos(phi) - y * np.sin(phi),
        x * np.sin(phi) + y * np.cos(phi)
    )


@dataclass
class AnimationData:
    """Container to provide data for animation."""

    step: int = 0
    veh_x: float = 0
    veh_y: float = 0
    veh_psi: float = 0
    veh_vx: float = 0
    t: float = 0
    ref: Trajectory = 0
    e_y: float = 0
    e_psi: float = 0
    e_vx: float = 0
    delta_v: float = 0
    ax: float = 0
    full_ref: Trajectory = 0
    full_ref_length: float = 0
    ref_s: float = 0
    lap: int = 0
    laps_max: int = 0

    def assign_sim_data(self, t, step, ref, xk, ctrl, lap) -> None:
        """Assign data from simulation data container."""
        self.t = t
        self.step = step
        self.veh_x = xk[0][0]
        self.veh_y = xk[1][0]
        self.veh_psi = xk[2][0]
        self.veh_vx = xk[4][0]
        self.ref = ref
        self.e_y = ctrl.e_y
        self.e_psi = ctrl.e_psi
        self.e_vx = ctrl.e_vx
        self.delta_v = ctrl.delta_v
        self.ax = ctrl.ax
        self.lap = lap
        self.ref_s = ctrl.ref_s

        return self


class AnimateVehicle:
    """Main class to animate the motion of a vehicle along a trajectory."""

    vehicle_width = 4
    vehicle_height = 1.85

    show_fps = True
    fps_max = 250

    figure_size = (10, 8)
    figure_title = 'Trajectory tracking vehicle motion controller'
    x_label = 'X position in [m]'
    y_label = 'Y position in [m]'

    scaling = 250

    def __init__(self, dt, draw_rate=0.1):
        """Initialize class with needed information.

        Arguments
        ---------
        `dt` : float
            Sample time of simulation

        `draw_rate` : float
            Rate at which a new frame is drawn. Unfortunately updating
            a frame is computational expensive, so we do not draw every
            simulation step.
            First ideas: https://stackoverflow.com/questions/8955869/why-is-plotting-with-matplotlib-so-slow
        """
        self.dt = dt
        self.draw_rate = draw_rate

        self.vehicle_width = 4 * max(self.scaling/50, 1)
        self.vehicle_height = 1.85 * max(self.scaling/50, 1)

        self.__plt_vehicle = None
        self.__plt_ref = None
        self.__plt_scenario = None
        self.__anno_infos = None
        self.__anno_fps = None
        self.__anno_ref_s = None
        self.__backgrounds = None

        self.__create_figure()
        self.__fps_measure_time = 0
        self.__t_prev = 0

    def __determine_veh_rect(self) -> None:
        """Determine offsets from center to rectangle corner.

        Theses offsets are based on vehicle parameter `vehicle_width`
        and `vehicle_height`.
        """
        dx = self.vehicle_width/2
        dy = self.vehicle_height/2

        self.dx_rect = [-dx, dx, dx, -dx, -dx]
        self.dy_rect = [-dy, -dy, dy, dy, -dy]

    def __create_figure(self, full_screen=False) -> None:
        """Create/initialize figure and axes."""
        self.axes = list()

        self.figure, self.ax_main = plt.subplots(figsize=self.figure_size)
        self.axes.append(self.ax_main)


        self.figure.patch.set_facecolor(C_BACKGROUND)

        self.ax_main.set_facecolor(C_BACKGROUND)

        # Remove space around plot
        plt.gca().set_axis_off()
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
        )
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.axis("off")

        self.figure.show()
        self.figure.canvas.draw()

        if full_screen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

        plt.title(self.figure_title, fontsize=20)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        self.__determine_veh_rect()

    def __draw_scenario(self, full_ref) -> None:
        """Draw path of full scenario."""
        if not full_ref:
            return

        self.__plt_scenario, = self.ax_main.plot(
            full_ref.x, full_ref.y,
            color=C_FULL_REF, linewidth=1
        )
        self.ax_main.draw_artist(self.__plt_scenario)

    def __draw_vehicle(self, veh_x, veh_y, psi) -> None:
        """Draw a rectangle with a heading indicator as vehicle.

        Arguments
        ---------
        `X` : float
            global position of vehicle in x-direction
        `Y` : float
            global position of vehicle in y-direction
        `psi` : float
            heading angle of vehicle
            with: `east==0` and `west==pi` > counterclockwise
        """
        rotated_rect = [
            rotate_point(dx, dy, psi) for dx, dy in zip(
                self.dx_rect, self.dy_rect
            )
        ]
        x_rect = [veh_x+p[0] for p in rotated_rect]
        y_rect = [veh_y+p[1] for p in rotated_rect]

        rotated_head = rotate_point(self.vehicle_width/2, 0, psi)
        x_head = [veh_x, veh_x+rotated_head[0]]
        y_head = [veh_y, veh_y+rotated_head[1]]

        if self.__plt_vehicle is None:
            self.__plt_vehicle, = self.ax_main.plot(x_rect, y_rect, color=C_VEHICLE, linewidth=2)
            self.__plt_vehicle_dir, = self.ax_main.plot(x_head, y_head, color=C_VEHICLE)
        else:
            self.__plt_vehicle.set_xdata(x_rect)
            self.__plt_vehicle.set_ydata(y_rect)

            self.__plt_vehicle_dir.set_xdata(x_head)
            self.__plt_vehicle_dir.set_ydata(y_head)
        self.ax_main.draw_artist(self.__plt_vehicle)
        self.ax_main.draw_artist(self.__plt_vehicle_dir)

    def __draw_reference(self, ref) -> None:
        """Draw current reference for controller."""
        if not ref:
            return

        if self.__plt_ref is None:
            self.__plt_ref, = self.ax_main.plot(
                ref.x, ref.y,
                linewidth=3,
                color=C_REF
            )
        else:
            self.__plt_ref.set_xdata(ref.x)
            self.__plt_ref.set_ydata(ref.y)
        self.ax_main.draw_artist(self.__plt_ref)

    def __draw_controller_infos(self, ani_data) -> None:
        """Draw information in top right corner."""
        infos = [
            f'delta_v: {ani_data.delta_v/np.pi*180:.1f} deg\n',
            f'vx: {ani_data.veh_vx*3.6:.0f} km/h\n\n'
            f'e_y: {ani_data.e_y:.2f} m\n',
            f'e_psi: {ani_data.e_psi:+.4f} rad\n',
            f'e_vx: {ani_data.e_vx*3.6:.1f} km/h'
        ]
        info_string = "".join(infos)

        if self.__anno_infos is not None:
            self.__anno_infos.set(text=info_string)
        else:
            self.__anno_infos = self.ax_main.annotate(
                info_string,
                xy=(0.01, 0.85),
                xycoords="axes fraction",
                fontsize=14,
                weight="bold",
                color=C_CTRL_INFO,
                horizontalalignment="left",
                verticalalignment="top"
            )
        self.ax_main.draw_artist(self.__anno_infos)

    def __draw_sim_info(self, ani_data) -> None:
        """Draw fps in top left corner."""
        info_string = f't: {ani_data.t:.2f}s (x{self.real_time:.1f})\n'
        # if ctrl is not None:
        info_string += f's: {ani_data.ref_s:.0f}m / {ani_data.full_ref_length:.0f}m\n'
        info_string += f'Lap#: {ani_data.lap} / {ani_data.laps_max}'

        if self.__anno_ref_s is not None:
            self.__anno_ref_s.set(text=info_string)
        else:
            self.__anno_ref_s = self.ax_main.annotate(
                info_string,
                xy=(0.01, 0.99),
                xycoords="axes fraction",
                fontsize=14,
                weight="bold",
                color=C_SIM_INFO,
                horizontalalignment="left",
                verticalalignment="top"
            )
        self.ax_main.draw_artist(self.__anno_ref_s)

    def __draw_fps(self) -> None:
        """Draw fps in top left corner."""
        info_string = f'FPS: {self.fps:3.0f}'

        if self.__anno_fps is not None:
            self.__anno_fps.set(text=info_string)
        else:
            self.__anno_fps = self.ax_main.annotate(
                info_string,
                xy=(0.99, 0.99),
                xycoords="axes fraction",
                fontsize=8,
                weight="normal",
                color=C_FPS,
                horizontalalignment="right",
                verticalalignment="top"
            )
        self.ax_main.draw_artist(self.__anno_fps)

    def __rescale(self, veh_x, veh_y) -> None:
        """Rescale plot based on vehicle position."""
        self.ax_main.set_xlim([veh_x-self.scaling, veh_x+self.scaling])
        self.ax_main.set_ylim([veh_y-self.scaling, veh_y+self.scaling])

    def __calculate_fps(self, ani_data) -> None:
        """Calculate frames per second."""
        self.fps = 1 / (perf_counter() - self.__fps_measure_time)
        self.real_time = (ani_data.t - self.__t_prev) / (perf_counter() - self.__fps_measure_time)
        self.__fps_measure_time = perf_counter()
        self.__t_prev = ani_data.t

    def __cap_fps(self) -> None:
        """Limit frames per second."""
        current_fps = 1 / (perf_counter() - self.__fps_measure_time)
        if current_fps > self.fps_max:
            sleep(1/self.fps_max - 1/current_fps)

    def __draw_backgrounds(self) -> None:
        """Restore axes backgrounds after first draw."""
        if self.__backgrounds is None:
            self.__backgrounds = [self.figure.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
        else:
            for item in self.__backgrounds:
                self.figure.canvas.restore_region(item)

    def __render_axes(self) -> None:
        """Render updated axes."""
        for ax in self.axes:
            self.figure.canvas.blit(ax.bbox)

    def draw_next_frame(self, ani_data: AnimationData) -> None:
        """Draw new frame from available animation data."""
        if ani_data.step % int(self.draw_rate/self.dt) != 0:
            return

        self.__draw_backgrounds()

        self.__draw_scenario(ani_data.full_ref)
        self.__draw_reference(ani_data.ref)

        # Change heading back to north == 0.
        # See models for more information.
        veh_psi_transf = ani_data.veh_psi + np.pi/2
        self.__draw_vehicle(ani_data.veh_x, ani_data.veh_y, veh_psi_transf)

        self.__cap_fps()
        self.__calculate_fps(ani_data)
        self.__draw_fps()

        self.__draw_controller_infos(ani_data)
        self.__draw_sim_info(ani_data)
        self.__rescale(ani_data.veh_x, ani_data.veh_y)

        self.__render_axes()


if __name__ == "__main__":
    # Draw vehicle and make full rotation from east to east
    dt = 0.01
    t_end = 5

    ani = AnimateVehicle(dt=dt, draw_rate=0.1)
    ani.fps_max = 10

    ani_data = AnimationData()
    ani_data.veh_vx = 72/3.6

    for idx, t in enumerate(np.arange(0, t_end, dt)):
        ani_data.step = idx
        ani_data.t = t
        ani_data.veh_psi = 2*np.pi * t/t_end
        ani.draw_next_frame(ani_data)

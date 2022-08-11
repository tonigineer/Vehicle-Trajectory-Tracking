"""Main module for animation tasks."""

from sys import implementation
import numpy as np
import matplotlib.pyplot as plt


def rotate_point(x, y, phi):
    """Rotate a point `(x,y)` by angle=`phi`."""
    return (
        x * np.cos(phi) - y * np.sin(phi),
        x * np.sin(phi) + y * np.cos(phi)
    )


class AnimateVehicle:
    """Main class to animate the motion of a vehicle along a trajectory."""

    vehicle_width = 4
    vehicle_height = 1.85

    plt_vehicle = None

    figure_size = (10, 8)
    figure_title = 'Animation of vehicle motion'
    x_label = 'X position in [m]'
    y_label = 'Y position in [m]'

    scale_margin = 2
    scale_x_lim = (0, 100)
    scale_y_lim = (0, 100)

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
            TODO: should go better, have a deeper look here!
        """
        self.dt = dt
        self.draw_rate = draw_rate
        self.__create_figure()

    def __determine_veh_rect(self):
        """Determine offsets from center to rectangle corner.

        Theses offsets are based on vehicle parameter `vehicle_width`
        and `vehicle_height`.
        """
        dx = self.vehicle_width/2
        dy = self.vehicle_height/2

        self.dx_rect = [-dx, dx, dx, -dx, -dx]
        self.dy_rect = [-dy, -dy, dy, dy, -dy]

    def __create_figure(self, full_screen=False):
        """Create/initialize figure and axes."""
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=self.figure_size)

        if full_screen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

        plt.title(self.figure_title, fontsize=20)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        self.__determine_veh_rect()

    def __draw_vehicle(self, X, Y, psi):
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
        x_rect = [X+p[0] for p in rotated_rect]
        y_rect = [Y+p[1] for p in rotated_rect]

        rotated_head = rotate_point(self.vehicle_width/2, 0, psi)
        x_head = [X, X+rotated_head[0]]
        y_head = [Y, Y+rotated_head[1]]

        if self.plt_vehicle is None:
            self.plt_vehicle, = self.ax.plot(x_rect, y_rect, color='#323567')
            self.plt_vehicle_dir, = self.ax.plot(x_head, y_head, color='#a46773')
        else:
            self.plt_vehicle.set_xdata(x_rect)
            self.plt_vehicle.set_ydata(y_rect)

            self.plt_vehicle_dir.set_xdata(x_head)
            self.plt_vehicle_dir.set_ydata(y_head)

    def _rescale(self, type_='fixed-margin'):
        """Apply different methods to rescale axes.

        Arguments
        ---------
        `type_` : str
            `'fixed-margin'` rescale to vehicle with a certain margin

            `'fixed-size'` rescale fixed x and y limits
        """
        if type_ == 'fixed-margin':
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.axis('equal')
            self.ax.margins(self.scale_margin, self.scale_margin)
        elif type_ == 'fixed-size':
            self.ax.set_xlim(self.scale_x_lim)
            self.ax.set_ylim(self.scale_y_lim)
        else:
            raise NotImplementedError(f'Rescale type: {type_} not implemented yet.')

    def draw_next_frame(self, step, xk, uk):
        """Draw new frame from data.

        Arguments
        ---------
        `step` : int
            Simulation step, that is converted to time step via `self.dt`
        `xk` : list
            State vector of state space representation
        `uk` : list
            Input vector of state space representation
        """
        if step % int(self.draw_rate/self.dt) != 0:
            return

        X, Y, psi = np.array(xk)[0:3]
        delta_v = np.array(uk)[0]

        self.__draw_vehicle(X, Y, psi)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        self._rescale('fixed-margin')


if __name__ == "__main__":
    import time

    # Draw vehicle and make full rotation from east to east
    heading_angle = np.linspace(0, np.pi*2, 1000)
    ani = AnimateVehicle(dt=0.01, draw_rate=0.1)

    for idx, psi in enumerate(heading_angle):
        xk = [0, 0, psi]
        uk = [0, 0]
        ani.draw_next_frame(step=idx, xk=xk, uk=uk)
        time.sleep(0.01)

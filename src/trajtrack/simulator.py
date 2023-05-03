"""Framework to run a simulation for trajectory tracking."""

from time import perf_counter

import numpy as np
import pandas as pd

from trajtrack.scenario import Scenario
from trajtrack.vehicle_models import Vehicle, BicycleModel
from trajtrack.controller import Controller, ControlOutput
from trajtrack.planner import Tracks, Position, Reference

from trajtrack.visualization import Evaluation, ScatterEntry, SubplotLayout


class Simulator:
    """Wrapping class for whole simulation."""

    def __init__(self, scenario: Scenario,
                 vehicle: Vehicle,
                 controller: Controller):
        self.scenario = scenario
        self.vehicle = vehicle
        self.controller = controller

        self.simulation = Simulation(vehicle)

    def _simulate_step(self):
        """Evoke full workflow of one simulation step."""
        reference = self.scenario.update(
            self.simulation.vehicle_position
        )
        controller_output = self.controller.apply(
            self.simulation.vehicle_position,
            reference
        )
        self.simulation.update(
            controller_output,
            reference
        )
        self._show_verbose()

    def run(self):
        """Evoke main simulation loop wrapper."""
        self.simulation.prepare(
            self.scenario.initial_node
        )
        try:
            while not self.scenario.is_terminated:
                self._simulate_step()
        finally:
            self.simulation.post_processing()
            self.show_results()

    def _show_verbose(self):
        print(
            f'\r{self.simulation.step_duration*1000:0.5f} ms at vx: ' +
            f'{self.simulation.vehicle_state.flatten()[4]:0.1f} m/s',
            end=""
        )

    def show_results(self):
        """Visualize simulation results."""
        # First plot
        layout = SubplotLayout("Tracking")
        veh_x = np.array(self.sim_data['x'].values.tolist()).flatten()
        veh_y = np.array(self.sim_data['y'].values.tolist()).flatten()

        data = [
            ScatterEntry(veh_x, veh_y, "Vehicle path", "Position in x [m]", "Position in y [m]")
        ]

        Evaluation.subplots(data, layout, columns=1)

        # Second plot
        layout = SubplotLayout("Vehicle state")
        sim_t = np.array(self.sim_data['t'].values.tolist()).flatten()
        veh_vx = np.array(self.sim_data['vx'].values.tolist()).flatten()
        veh_beta = np.array(self.sim_data['beta'].values.tolist()).flatten()
        veh_psi = np.array(self.sim_data['psi'].values.tolist()).flatten()

        data = [
            ScatterEntry(sim_t, veh_vx, "Side slip angle", "Time [s]", "Side slip angle in [rad]"),
            ScatterEntry(sim_t, veh_beta, "Velocity", "Time [s]", "Velocity [m/s]"),
            ScatterEntry(sim_t, veh_psi, "Heading", "Time [s]", "Heading [rad]")
        ]

        Evaluation.subplots(data, layout, columns=2)

    @property
    def sim_data(self):
        """Return full simulation data."""
        return self.simulation.sim_data


class Simulation:
    """Class to handle simulation of vehicle.

    This include advancing simulation one step further and
    recording all simulation data.
    """

    dt = 0.1  # seconds

    _state = np.zeros([6, 1])  # x, y, psi, psip, vx, vy
    _t = 0.0
    _step_duration = 0.0

    # faster to store all data in list than pd.concat() every step
    _data_list = []
    _data = pd.DataFrame()

    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle

    def prepare(self, initial_node):
        """Initialize internal simulation variables."""
        x, y, psi, vx = initial_node[[0, 1, 3, 5]]
        self._state = np.array([x, y, psi, 0, vx, 0])
        self._t = 0
        self._data_list = []

        # assert self.dt == self.vehicle.model.dt

    def _update_time(self):
        """Increment internal sim time by `dt`."""
        self._t += self.dt

    def update(self,
               controller_output: ControlOutput,
               reference: Reference):
        """Apply control output and advance simulation by `dt`."""
        start = perf_counter()

        self._state = self.vehicle.apply(
            self._state, controller_output.to_array()
        )
        self._t += self.dt

        self._record_data(
            controller_output.to_array(), reference
        )

        self._step_duration = perf_counter() - start
        # print(f'\r{perf_counter() - start:0.6f}', end='')

    def _record_data(self, ctrl_out: np.ndarray, ref: np.ndarray):
        kappa = 1  # TODO: kappa from ref needed
        data = {
            't': self._t,
            'x': self._state[0],
            'y': self._state[1],
            'psi': self._state[2],
            'psip': self._state[3],
            'vx': self._state[4],
            'vy': self._state[5],
            'ctrl_delta_v': ctrl_out[0],
            'ctrl_ax': ctrl_out[1],
            'ay': self._state[4]*self._state[4] * kappa,
            'beta': np.tan(self._state[5]/self._state[4])
        }

        # Add reference
        # ref_attributes = ['x', 'y', 's', 'kappa', 'psi', 'vx', 'ax']
        # for k in range(ref.shape[0]):
        #     for i in range(ref.shape[2]):
        #         data[f'{ref_attributes[k]}{i}'] = ref[k][0][i]

        self._data_list.append(data)

    def post_processing(self):
        """Finish up recorded simulation data."""
        self._data = pd.DataFrame(
            self._data_list, index=np.arange(
                0, len(self._data_list) * self.dt, self.dt
            )
        )

    @property
    def vehicle_position(self):
        """Return current position of vehicle."""
        return Position(self._state[0], self._state[1])

    @property
    def vehicle_state(self):
        """Return current position of vehicle."""
        return self._state

    @property
    def step_duration(self):
        """Return current position of vehicle."""
        return self._step_duration

    @property
    def sim_data(self):
        """Return current position of vehicle."""
        return self._data


def development():
    """Test for development purposes."""
    scenario = Scenario(Tracks.Algarve_02G, settings={'number_laps': 1})
    vehicle = Vehicle(BicycleModel)
    controller = Controller()

    sim = Simulator(scenario=scenario, vehicle=vehicle, controller=controller)
    sim.run()

    sim.show_results()


if __name__ == "__main__":
    development()

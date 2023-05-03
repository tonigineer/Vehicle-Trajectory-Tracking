"""Basic examples of how to create simple simulation."""

from trajtrack import Simulator, Scenario, Controller, Vehicle
from trajtrack.controller import ControlOutput
from trajtrack.vehicle_models import BicycleModel
from trajtrack.planner import Tracks

import numpy as np


def custom_controller(steering_angle: np.float32, acceleration: np.float32):
    """Exemplary pseudo controller."""
    return ControlOutput(steering_angle=0, acceleration=0)


def coating_into_standstill():
    """Coating vehicle into standstill from initial reference."""
    scenario = Scenario(Tracks.Algarve_02G, settings={'number_laps': 1})
    vehicle = Vehicle(BicycleModel)
    controller = Controller()

    controller.control_function = custom_controller

    sim = Simulator(scenario=scenario, vehicle=vehicle, controller=controller)
    sim.run()

    sim.show_results()


if __name__ == "__main__":
    coating_into_standstill()

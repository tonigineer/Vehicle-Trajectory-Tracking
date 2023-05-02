"""Basic example of how to create simple simulation."""

from trajtrack import Vehicle, Controller

from trajtrack.controller import ControlOutput
from trajtrack.vehicle_models import FSVehSingleTrack


def custom_controller():
    """Exemplary pseudo controller."""
    return ControlOutput(steering_angle=0, acceleration=0)


def main():
    """Best practice main function for __main__ call."""
    vehicle = Vehicle(FSVehSingleTrack)
    controller = Controller()
    controller.control_function = custom_controller


if __name__ == "__main__":
    main()

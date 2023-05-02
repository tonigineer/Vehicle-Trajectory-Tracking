"""Definition of the basic controller interface."""

from dataclasses import dataclass
import numpy as np

from trajtrack.planner import Reference


@dataclass
class ControlOutput:
    """General container for controller output."""

    steering_angle: np.float32
    acceleration: np.float32

    def to_array(self):
        """Return as `np.ndarray`."""
        return np.array([self.steering_angle, self.acceleration])


class Controller:
    """Base class for trajectory tracking controller."""

    def __init__(self):
        pass

    def apply(self, current_state: np.ndarray, reference: np.ndarray):
        """Call `control function` to calculate control output."""
        return self.control_function(current_state, reference)

    def control_function(self, state: np.ndarray, ref: Reference):
        """Calculate controller output.

        Note: Supposed to be overwritten with the actual desired controller.
        """
        # TODO: Currently just zero is output.
        return ControlOutput(steering_angle=0, acceleration=1)

"""All the stuff to define the be simulated scenario."""

import numpy as np

from trajtrack.planner import OfflineTrackPlanner, Tracks, Position


class Scenario:
    """Class to handle the simulation environment."""

    REFERENCE_NUM_NODES = 20
    DEFAULT_SETTINGS = {
        'number_laps': 1
    }

    settings = DEFAULT_SETTINGS

    _is_terminated = False
    _lap = 0
    _last_localized_node_index = 0

    def __init__(self, track, settings=None):
        self.track = track
        self.planner = OfflineTrackPlanner(track)
        if settings:
            self.settings = settings

    def _update_laps(self):
        node_idx = self.planner.localized_node_index
        if self._last_localized_node_index > 0 and node_idx == 0:
            self._lap += 1
        self._last_localized_node_index = node_idx
        if self._lap > self.settings['number_laps']:
            self._is_terminated = True

    def update(self, vehicle_position: Position) -> None:
        """Update lap count and check termination.

        NOTE: `localized_node_index` is provided from Planner class.
        """
        ref = self.planner.get_reference(vehicle_position)
        self._update_laps()
        return ref

    @property
    def initial_node(self) -> np.ndarray:
        """Return tuple with initial x and y position and velocity in x."""
        idx = 0 if self.track.is_loop else 1
        return np.array(
            self.track.get_nodes(idx=idx, num=1)
        ).flatten()

    @property
    def is_terminated(self) -> bool:
        """Return if scenario is finished."""
        return self._is_terminated

    @property
    def current_lap(self) -> bool:
        """Return if scenario is finished."""
        return self._lap


def development():
    """Immediate tests for development."""
    scenario = Scenario(Tracks.Algarve_06G, settings=None)
    (init_x, init_y, init_vx) = scenario.initial_state
    print(init_x, init_y, init_vx)

    scenario.settings['number_laps'] = 5
    while True:

        scenario.update(Position(scenario.track.x[-2], scenario.track.y[-2]))
        scenario.update(Position(scenario.track.x[0], scenario.track.y[0]))
        print(scenario.current_lap)
        print(scenario.is_terminated)
        if scenario.is_terminated:
            break


if __name__ == "__main__":
    development()

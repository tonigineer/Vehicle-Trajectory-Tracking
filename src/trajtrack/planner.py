"""Planner module that provides reference to controller."""

import os
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np


NUM_ATTRIBUTES_PER_NODE = 7
TRACKS_DIR = Path(Path(__file__).parent.resolve(), 'resources/tracks')


@dataclass
class Position:
    """Position in 2D plane."""

    x: np.float32
    y: np.float32

    def as_array(self) -> np.ndarray:
        """Return Position as array."""
        return np.array([self.x, self.y])


@dataclass
class Track:
    """Provide container to store track data for scenario class."""

    name: str
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    vx: np.ndarray
    ax: np.ndarray

    def __post_init__(self):
        cls_fields: Tuple[Track, ...] = fields(self.__class__)
        for field in cls_fields:
            if issubclass(field.type, np.ndarray):
                setattr(
                    self, field.name,
                    getattr(self, field.name).flatten()
                )

    def show(self) -> None:
        """Show track in figure."""
        plt.figure()
        plt.plot(self.x, self.y)
        plt.show()

    def get_nodes(self, idx: int, num: int) -> np.ndarray:
        """Provide `array` with selected range of track nodes.

        Info: Indexing is wrapped around `num_nodes`.
        """
        assert idx < self.num_nodes
        assert idx + num < self.num_nodes or self.is_loop

        idc = np.array([range(idx, idx+num)])
        if self.is_loop:
            idc = np.mod(idc, self.num_nodes)

        return np.array([
            self.x[idc], self.y[idc], self.s[idc], self.psi[idc],
            self.kappa[idc], self.vx[idc], self.ax[idc]
        ])

    @property
    def is_loop(self):
        """Check if nodes are a loop.

        Loops are defined by the identical first and last nodes. Except
        for the traveled distance s, which represent the track length.
        """
        return self.x[0] == self.x[-1] and self.y[0] == self.y[-1]

    @property
    def length(self):
        """Get total length of track."""
        return self.s[-1]

    @property
    def num_nodes(self):
        """Get number of track nodes."""
        return self.x.shape[0]


def parse_json_track(filename) -> Track:
    """Load track data from `.json`."""
    with open(filename, 'r', encoding="UTF-8") as j:
        data = json.loads(j.read())

    return Track(
        os.path.basename(filename).replace('.json', ''),
        *[np.array([data['nodes'][i]]).flatten()
            for i in range(NUM_ATTRIBUTES_PER_NODE)]
    )


class Tracks:
    """Collection of exemplary tracks."""

    Algarve_02G = parse_json_track(
        Path(TRACKS_DIR,
             'Algarve_International_Circuit_02g_02g_128.json'))

    Algarve_06G = parse_json_track(
        Path(TRACKS_DIR,
             'Algarve_International_Circuit_03g_06g_130.json'))

    Hockenheim_06G = parse_json_track(
        Path(TRACKS_DIR,
             'Hockenheimring_04g_06g_117.json'))

    @classmethod
    def get_all_tracks(cls) -> List[Track]:
        """Provide a list with all exemplary tracks.

        TODO: Use loop over class attributes to provide tracks.
        """
        return [cls.Algarve_02G, cls.Algarve_06G, cls.Hockenheim_06G]


@dataclass
class Reference:
    """Selection of nodes."""

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    vx: np.ndarray
    ax: np.ndarray

    @property
    def num_nodes(self):
        """Number of nodes in reference."""
        return self.x.shape[0]

    def get_nodes(self, idx: int, num: int):
        """Return `num` nodes starting by `idx`.

        NOTE: Different to `Track.get_nodes`, no wrapping around!
        """
        assert idx + num < self.num_nodes, \
            f'Number of nodes exceeded with {idx+num}.'
        idc = np.array([x for x in range(idx, idx+num)])

        return np.array([
            self.x[idc], self.y[idc], self.s[idc], self.psi[idc],
            self.kappa[idc], self.vx[idc], self.ax[idc]
        ])


class OfflineTrackPlanner:
    """Emulation of Planning Layer of an Automated driving function."""

    REFERENCE_NUM_NODES = 20

    localized_node_index = 0

    def __init__(self, track: Track):
        self._track = track

    def _localization(self, position: Position) -> int:
        """Determine closest node to position."""
        # TODO: could be to expensive >> keeping track of index and increasing
        #       might be cheaper
        nodes = np.asarray(np.array([self._track.x, self._track.y]).T)
        dist_2 = np.sum((nodes - position.as_array().flatten())**2, axis=1)
        return np.argmin(dist_2)

    def get_reference(self, position: Position) -> Reference:
        """Return current reference nodes based on given position.

        Reference starts one node before `self._localization()` node.
        """
        idx = self.localized_node_index = self._localization(position)
        if self._track.is_loop:
            idx = (idx - 1) % self._track.num_nodes
        else:
            idx = np.max(0, idx - 1)
        return self._track.get_nodes(idx=idx, num=self.REFERENCE_NUM_NODES)


def development():
    """Test function for development purposes."""
    track = Tracks.Algarve_02G
    planner = OfflineTrackPlanner(track)
    ref = planner.get_reference(Position(x=1, y=1))
    print(ref)


if __name__ == "__main__":
    development()

import logging
import numpy as np
import pandas as pd
from morphio import Morphology
from morphio.mut import Morphology as MutableMorphology

from numpy.typing import NDArray

from .util import mvee, points_inside_ellipsoid

class Cell:
    """An astrocyte.
    
    Functions that don't obviously return something else allow chaining, e.g.
    `Cell().from_dict(cell_data)._find_branches().to_dict()`
    """
    def __init__(self,
                 morphology: Morphology = None,
                 ID: int = None,
                 logger: logging.Logger = None):
        self._logger = logger or logging.getLogger(__name__)

        self._morphology = morphology  # type: Morphology
        self._ID = ID  # type: int
        
        if type(morphology) == str:
            self.load_morphology_from_hdf(morphology)

        # caches
        self._section_depths = None  # type: dict[int, int]
        self._terminal_points_with_depth = None  # type: list[tuple[list, int]]
        self._ellipsoid = None  # type: tuple[NDArray, NDArray, NDArray]

    @property
    def morphology(self) -> Morphology:
        return self._morphology
    @morphology.setter
    def morphology(self, val: Morphology):
        if type(val) == MutableMorphology:
            raise TypeError("new morphology must be immutable")
        assert type(val) == Morphology, "new morphology must be of type morphio.Morphology"

        self._morphology = val
        # reset caches
        self._section_depths = None
        self._terminal_points_with_depth = None
        self._ellipsoid = None

    @property
    def ID(self) -> int:
        return self._ID
    @property
    def n_branchingPoints(self):
        """Number of branching points in the cell."""
        return sum(len(section.children) > 1 for section in self.morphology.iter())
    @property
    def section_depths(self) -> dict[int, int]:
        """
        Dictionary mapping section.id -> section-tree depth.
        """
        if self._section_depths is None:
            _section_depths = {}

            for section in self.morphology.iter():
                if section.is_root:
                    _section_depths[section.id] = 0
                else:
                    _section_depths[section.id] = _section_depths[section.parent.id] + 1

            self._section_depths = _section_depths

        return self._section_depths
    @property
    def terminal_points_with_depth(self) -> list[tuple[list, int]]:
        """
        Terminal section endpoints with their section-tree depth.
        """
        if self._terminal_points_with_depth is None:
            _terminal_points_with_depth = []

            for section in self.morphology.iter():
                if len(section.children) == 0 and len(section.points) > 0:
                    _terminal_points_with_depth.append(
                        (
                            np.asarray(section.points, dtype=float)[-1],
                            self.section_depths[section.id],
                        )
                    )

            self._terminal_points_with_depth = _terminal_points_with_depth

        return self._terminal_points_with_depth
    @property
    def ellipsoid(self):
        """Minimum volume encapsulating ellipsoid (MVEE) of the cell's filament points as a tuple (center, radii, rotation)."""
        if self._ellipsoid is None:
            self._set_ellipsoid()
        return self._ellipsoid

    def _set_ellipsoid(self, tolerance: float = 1e-8):
        """
        Sets the parameters of the minimum volume encapsulating ellipsoid (MVEE) of the cells morphology points.

        First, a rough ellipsoid is calculated from terminal points with sufficient
        section-tree depth. Then all morphology points are checked against the
        ellipsoid. Points outside the ellipsoid are added and the MVEE is recalculated.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for the ellipsoid containment check.
        """
        self._logger.info(f"Calculating ellipsoid for cell {self.ID}...")

        max_depth = max(depth for _, depth in self.terminal_points_with_depth)
        minimum_depth = max_depth / 2

        ellipse_points = np.asarray(
            [
                point
                for point, depth in self.terminal_points_with_depth
                if depth >= minimum_depth
            ],
            dtype=float,
        )
        if len(ellipse_points) < 4:
            raise ValueError(
                f"Not enough terminal points to calculate initial ellipsoid for cell {self.ID}. "
                f"Found {len(ellipse_points)} points with depth >= {minimum_depth}."
            )

        all_points = np.asarray(self.morphology.points, dtype=float)
        if len(all_points) == 0:
            raise ValueError(f"Cell {self.ID} has no morphology points.")

        # First rough calculation.
        center, radii, rotation = mvee(ellipse_points)

        # Refinement:
        # Add all morphology points that lie outside the current ellipsoid.
        inside_points_mask = points_inside_ellipsoid(
            all_points,
            center,
            radii,
            rotation,
            tolerance=tolerance
        )
        outside_points = all_points[~inside_points_mask]
        if len(outside_points) > 0:
            self._logger.info(
                f"Ellipsoid refinement for cell {self.ID}: "
                f"adding {len(outside_points)} outside points."
            )
            ellipse_points = np.vstack([ellipse_points, outside_points])
            center, radii, rotation = mvee(ellipse_points)

        self._ellipsoid = center, radii, rotation

        self._logger.info(f"Finished calculating ellipsoid for cell {self.ID}.")
        return self

    def save_morphology_to_hdf(self, path: str):
        """Saves the cell's morphology to an HDF5 file."""
        mutmorpho = MutableMorphology(self.morphology)  # only mutable morphologies can be written to file
        mutmorpho.write(path)
        return self

    def to_dict(self, version = "latest"):
        """Returns a dict containing the cell metadata excluding the morphology.

        Parameters
        ----------
        version : str
            Version of the export format. Consistent with `from_dict()`.
        """
        if version == "latest":
            version = "0.1"
        if version == "0.1":
            return {
                "version": version,
                "ID": self.ID,
                "section_depths": self._section_depths,
                "terminal_points_with_depth": self._terminal_points_with_depth,
                "ellipsoid": self._ellipsoid
            }
        else:
            raise ValueError(f"Unsupported dict export version: {version}")

    def load_morphology_from_hdf(self, path: str):
        """Loads the cell's morphology from an HDF5 file."""
        self._logger.debug(f"Loading morphology for cell {self.ID} from {path}")
        self.morphology = Morphology(path)
        return self

    def from_dict(self, data: dict):
        """Loads the cell metadata from a dict.

        Parameters
        ----------
        data : dict
            Contains the cell metadata and a version number consistent with `to_dict()`.
        """
        version = data["version"]
        if version == "0.1":
            self._ID = data["ID"]
            self._section_depths = data["section_depths"]
            self._terminal_points_with_depth = data["terminal_points_with_depth"]
            self._ellipsoid = data["ellipsoid"]
        else:
            raise ValueError(f"Unsupported version of dict import: {version}")
        return self

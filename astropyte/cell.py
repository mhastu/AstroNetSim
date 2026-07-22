import logging
import numpy as np
import pandas as pd
from morphio import Morphology
from morphio.mut import Morphology as MutableMorphology
from matplotlib import pyplot as plt

from numpy.typing import NDArray

from .util import mahalanobis_radius, mvee, points_inside_ellipsoid


class Cell:
    """An astrocyte.
    
    Functions that don't obviously return something else allow chaining, e.g.
    `Cell().from_dict(cell_data)._set_ellipsoid().to_dict()`
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

    def _set_ellipsoid(self, n_points: int = 50, n_refine: int = 10, debug_plot: bool = False) -> "Cell":
        """
        Sets the parameters of the minimum volume encapsulating ellipsoid (MVEE) of the cells morphology points.

        First, a rough ellipsoid is calculated from the `10*n_points` outermost terminal points (by depth).
        Then all morphology points are checked against the ellipsoid.
        At most `n_points` points outside the ellipsoid are added and the MVEE is recalculated.
        These steps are repeated for `n_refine` iterations or until all points are inside the ellipsoid.

        Parameters
        ----------
        n_points : int
            Number of outermost points to use for the ellipsoid calculation.
        n_refine : int
            Number of refinement iterations to perform. Each iteration adds points outside the current ellipsoid and recalculates the MVEE.
        debug_plot : bool
            If True, a debug plot is shown for each refinement iteration.
        """
        self._logger.info(f"Calculating ellipsoid for cell {self.ID}...")

        if debug_plot:
            from .plot import plot_mvee_debug_step  # would raise circular import if not inside function

        all_points = np.asarray(self.morphology.points, dtype=float)
        if len(all_points) < 4:
            raise ValueError(f"Cell {self.ID} has fewer than 4 points, cannot calculate ellipsoid.")

        if len(self.terminal_points_with_depth) > n_points*10:
            # use `n_points*10` outermost terminal points to get a rough estimate of the ellipsoid
            self._terminal_points_with_depth.sort(key=lambda x: x[1], reverse=True)
            ellipse_points = np.asarray(
                [point for point, _ in self.terminal_points_with_depth[:n_points*10]],
                dtype=float
            )
        else:
            ellipse_points = np.asarray(
                [point for point, _ in self.terminal_points_with_depth],
                dtype=float
            )  # fallback

        def mvee_tolerance(i: int) -> float:
            """Returns a tolerance value for the MVEE calculation based on the iteration number."""
            if i < 1:
                return 1e-2
            elif i < 2:
                return 1e-3
            else:
                return 1e-4
        def points_inside_ellipsoid_tolerance(i: int) -> float:
            """Returns a tolerance value for the points_inside_ellipsoid calculation based on the iteration number."""
            if i < 1:
                return 1e-1
            elif i < 2:
                return 1e-2
            else:
                return 1e-3
        for i in range(n_refine):
            center, radii, rotation = mvee(ellipse_points, tolerance=mvee_tolerance(i))
            inside_points_mask = points_inside_ellipsoid(
                all_points,
                center,
                radii,
                rotation,
                tolerance=points_inside_ellipsoid_tolerance(i)
            )
            
            outside_points = all_points[~inside_points_mask]
            if (len(outside_points) == 0) and (i >= 2):
                break  # all points are inside the ellipsoid and we have reached the smallest tolerance

            if i == n_refine - 1:
                self._logger.warning(
                    f"Ellipsoid refinement for cell {self.ID}: "
                    f"{len(outside_points)} points are still outside the ellipsoid after {n_refine} iterations."
                )
                break

            if len(outside_points) > n_points:  # more points outside than we want to add
                mahalanobis_radii = mahalanobis_radius(
                    outside_points,
                    center,
                    radii,
                    rotation
                )
                # Sort outside points (descending) by their Mahalanobis error to the ellipsoid surface
                sorted_outside_indices = np.argsort(np.abs(mahalanobis_radii - 1.0))[::-1]
                outermost_indices = sorted_outside_indices[:n_points]  # take the `n_points` outermost points
                new_points = outside_points[outermost_indices]
            else:
                new_points = outside_points

            # for performance: remove points that are far inside the ellipsoid
            far_inside_ellipse_points_mask = points_inside_ellipsoid(
                ellipse_points,
                center,
                radii,
                rotation,
                tolerance=0.1,
                relaxed=False
            )
            self._logger.debug(
                f"Ellipsoid refinement #{i+1} for cell {self.ID}: "
                f"Removing {np.sum(far_inside_ellipse_points_mask)} points that are far inside the ellipsoid."
            )
            ellipse_points = ellipse_points[~far_inside_ellipse_points_mask]

            self._logger.debug(
                f"Ellipsoid refinement #{i+1} for cell {self.ID}: "
                f"Adding {len(new_points)} points."
            )
            ellipse_points = np.vstack((ellipse_points, new_points))  # add new points to the ellipsoid calculation

            if debug_plot:
                plot_mvee_debug_step(i, center, radii, rotation,
                    inside_points=all_points[inside_points_mask],
                    outside_extra_points=outside_points[sorted_outside_indices[n_points:]] if len(outside_points) > n_points else np.empty((0, 3)),
                    outside_new_points=new_points
                )

        self._ellipsoid = center, radii, rotation

        self._logger.info(f"Finished calculating ellipsoid for cell {self.ID}.")
        if debug_plot:
            plot_mvee_debug_step(i, center, radii, rotation,
                inside_points=all_points[inside_points_mask],
                outside_extra_points=outside_points,
                outside_new_points=np.empty((0, 3))
            )
            plt.show()
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

import logging
import numpy as np
import pandas as pd

from .util import mvee

class Cell:
    """An astrocyte.
    
    Functions that don't obviously return something else allow chaining, e.g.
    `Cell().from_dict(cell_data)._find_branches().to_dict()`
    """
    def __init__(self, ID: int = None,
                 filamentPoints: np.ndarray = None,
                 filamentEdges: np.ndarray = None,
                 branchPositions: np.ndarray = None,
                 branchDiameters: np.ndarray = None,
                 logger: logging.Logger = None):
        self._logger = logger or logging.getLogger(__name__)

        self._ID = ID
        self._filamentPoints = filamentPoints
        self._filamentEdges = filamentEdges
        self._branchPositions = branchPositions
        if branchDiameters is not None:
            self._branchPositions.loc[:, "PtDiameter"] = branchDiameters.loc[:, "PtDiameter"]

        self._fine_branches = None  # type: dict[int, np.ndarray]
        self._rough_branches = None  # type: dict[int, pd.DataFrame]
        self._ellipsoid = None  # type: tuple[np.ndarray, np.ndarray, np.ndarray]

    @property
    def ID(self):
        return self._ID
    @property
    def filamentPoints(self):
        return self._filamentPoints
    @property
    def filamentEdges(self):    
        return self._filamentEdges
    @property
    def branchPositions(self):
        return self._branchPositions
    @property
    def n_branchingPoints(self):
        """Number of branching points in the cell."""
        return self._branchPositions.loc[self._branchPositions.loc[:, "Type"] == "Dendrite Branch"].shape[0]
    @property
    def fine_branches(self):
        """Fine branches of the cell as a dict with key = branch ID and value = np.ndarray of shape (n_points, 3) containing the points of the branch."""
        if not self._fine_branches:
            self._find_branches()
        return self._fine_branches
    @property
    def rough_branches(self):
        """Rough branches of the cell as a dict with key = branch ID and value = pd.DataFrame containing the branch data."""
        if not self._rough_branches:
            self._find_branches()
        return self._rough_branches
    @property
    def ellipsoid(self):
        """Minimum volume encapsulating ellipsoid (MVEE) of the cell's filament points as a tuple (center, radii, rotation)."""
        if self._ellipsoid is None:
            self._set_ellipsoid()
        return self._ellipsoid

    def _set_ellipsoid(self, minimum_depth: int = 10):
        """
        Sets the ellipsoid parameters for the cell.
        
        :param minimum_depth: Minimum depth of branching points to be considered for the ellipsoid calculation. Only branching points with a depth greater than or equal to this value will be used to calculate the ellipsoid.
        :type minimum_depth: int
        """
        if max(self._branchPositions.loc[:, "Depth"]) <= minimum_depth * 2:
            minimum_depth = max(self._branchPositions.loc[:, "Depth"])/2
        self._logger.info(f"Calculating ellipsoid for cell {self.ID} with minimum depth {minimum_depth}...")

        # take outermost points
        xp = self._branchPositions.loc[(self._branchPositions["Type"] == "Dendrite Terminal") & (self._branchPositions["Depth"] >= minimum_depth)]["PtPositionX"]
        yp = self._branchPositions.loc[(self._branchPositions["Type"] == "Dendrite Terminal") & (self._branchPositions["Depth"] >= minimum_depth)]["PtPositionY"]
        zp = self._branchPositions.loc[(self._branchPositions["Type"] == "Dendrite Terminal") & (self._branchPositions["Depth"] >= minimum_depth)]["PtPositionZ"]
        ellipse_points = np.array((xp, yp, zp)).T
        self._ellipsoid = mvee(ellipse_points)

        self._logger.info(f"Finished calculating ellipsoid for cell {self.ID}.")
        return self

    def _find_branches(self):
        """Finds the branches of the cell and stores them in self._fine_branches and self._rough_branches."""
        self._logger.info(f"Finding branches for cell {self.ID}...")
        self._fine_branches = {}
        self._rough_branches = {}

        # Find the corresponding filament points for each branch position
        pos = self._branchPositions[["PtPositionX", "PtPositionY", "PtPositionZ"]].copy().to_numpy() #Extract only Position Data
        filamentPointIndex_in_branchPosition = np.zeros(np.size(pos,0))
        for i, position in enumerate(pos):
            # Find the index of the closest filament point to the branch position
            filamentPointIndex_in_branchPosition[i] = int(np.argmin((position[0] - self._filamentPoints[:, 0])**2 + (position[1] - self._filamentPoints[:, 1])**2 + (position[2] - self._filamentPoints[:, 2])**2))

        indices = np.where(self.filamentEdges[:, 0] != self.filamentEdges[:, 1] - 1)[0]
        # first Branch
        startpoint = 0
        endpoint = indices[0]
        mergepoint = self.filamentEdges[0, 0]#
    
        # dict with key = mergepoint, keys = point pos, point ID
        self._fine_branches[0] = self._filamentPoints[startpoint:endpoint, :]
        self._rough_branches[0] = self._branchPositions.loc[(filamentPointIndex_in_branchPosition <= endpoint)].copy()
        #go over all mergepoints by indexes
        for i in range(len(indices)-1):
            mergepoint = self.filamentEdges[indices[i], 0]
            endpoint = indices[i+1]    
            startpoint = indices[i] + 1

            self._fine_branches[i+1] = np.vstack((self._filamentPoints[mergepoint], self._filamentPoints[startpoint:endpoint, :]))
            self._rough_branches[i+1] = self._branchPositions.loc[filamentPointIndex_in_branchPosition == mergepoint].copy()
            self._rough_branches[i+1] = pd.concat([self._rough_branches[i+1], self._branchPositions.loc[(filamentPointIndex_in_branchPosition <= endpoint) & (filamentPointIndex_in_branchPosition >= startpoint)]], ignore_index=True)

        self._logger.info(f"Finished finding branches for cell {self.ID}.")
        return self

    def to_dict(self, version = "latest"):
        """Returns a dict containing the cell data.
        
        :param version: Version of the export format. Consistent with `from_dict()`.
        """
        if version == "latest":
            version = "1.1"
        if version == "1.1":
            return {
                "version": version,
                "ID": self.ID,
                "filamentPoints": self._filamentPoints,
                "filamentEdges": self._filamentEdges,
                "branchPositions": self._branchPositions,
                "fine_branches": self._fine_branches,
                "rough_branches": self._rough_branches,
                "ellipsoid": self._ellipsoid
            }
        elif version == "1.0":
            return {
                "version": version,
                "ID": self.ID,
                "filamentPoints": self._filamentPoints,
                "filamentEdges": self._filamentEdges,
                "branches": self._branchPositions,
                "fine_branches": self._fine_branches,
                "rough_branches": self._rough_branches,
                "ellipsoid": self._ellipsoid
            }
        else:
            raise ValueError(f"Unsupported dict export version: {version}")

    def from_dict(self, data: dict):
        """Loads the cell data from a dict.

        :param data: A dict containing the cell data and a version number consistent with `to_dict()`.
        """
        version = data["version"]
        if version == "1.1":
            self._ID = data["ID"]
            self._filamentPoints = data["filamentPoints"]
            self._filamentEdges = data["filamentEdges"]
            self._branchPositions = data["branchPositions"]
            self._fine_branches = data["fine_branches"]
            self._rough_branches = data["rough_branches"]
            self._ellipsoid = data["ellipsoid"]
        elif version == "1.0":
            self._ID = data["ID"]
            self._filamentPoints = data["filamentPoints"]
            self._filamentEdges = data["filamentEdges"]
            self._branchPositions = data["branches"]
            self._fine_branches = data["fine_branches"]
            self._rough_branches = data["rough_branches"]
            self._ellipsoid = data["ellipsoid"]
        else:
            raise ValueError(f"Unsupported version of dict import: {version}")
        return self

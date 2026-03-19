import numpy as np

class Cell:
    """An astrocyte."""
    def __init__(self, ID: int = None,
                 filamentPoints: np.ndarray = None,
                 filamentEdges: np.ndarray = None,
                 branchPositions: np.ndarray = None,
                 branchDiameters: np.ndarray = None):
        self._ID = ID
        self._filamentPoints = filamentPoints
        self._filamentEdges = filamentEdges
        self._branchPositions = branchPositions
        self._branchDiameters = branchDiameters

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
    def branchDiameters(self):
        return self._branchDiameters

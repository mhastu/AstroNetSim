import os.path
from cell import Cell
import pandas as pd
from pymatreader import read_mat
import numpy as np

class Dataset:
    """A set of astrocytes."""
    def __init__(self, name: str = None):
        self._name = name
        self._path = None
        self._cells = {}  # type: dict[int, Cell]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path
    
    @property
    def cells(self):
        return self._cells
    
    def load(self, path: str):
        """Loads the dataset from the given path.
        Overwrites all previously loaded data.
        
        Parameters
        ----------
        path : str
            The path to the dataset directory. Assumes files to present:
            - diameterData.csv (Generated in Matlab)
            - positionData.csv (Generated in Matlab)
            - matlab_and_excel_data.mat
        """
        self._path = path
        branchDiameters = pd.read_csv(os.path.join(path, "diameterData.csv"))
        branchPositions = pd.read_csv(os.path.join(path, "positionData.csv"))
        matlabAndExcelData = read_mat(os.path.join(path, "matlab_and_excel_data.mat"))  # type: dict[str, np.ndarray]
        filamentPoints = matlabAndExcelData["vFilamentsPoints"]
        filamentEdges = matlabAndExcelData["vFilamentsEdges"]

        for cellID in range(len(filamentPoints)):
            cells_to_delete = []
            if len(filamentPoints[cellID]) <= 3:  # Delete Cells with less than 2 entries
                cells_to_delete.append(cellID)
                continue

            cell_index = 100000000 + cellID
            # position of branching points
            cell_branchPositions = branchPositions.loc[branchPositions.FilamentID == cell_index].sort_values("ID", ignore_index = True)
            # diameter at branching points
            cell_branchDiameters = branchDiameters.loc[branchDiameters.FilamentID == cell_index].sort_values("ID", ignore_index = True)

            self.cells[cellID] = Cell(ID = cellID,
                                      filamentPoints = filamentPoints[cellID],
                                      filamentEdges = filamentEdges[cellID],
                                      branchPositions = cell_branchPositions,
                                      branchDiameters = cell_branchDiameters)
        
        for cell in cells_to_delete:
            self._FilamentPoints = np.delete(self._FilamentPoints, (cell), axis=0)
            self._FilamentEdges = np.delete(self._FilamentEdges, (cell), axis=0)

    def _set_encapsulating_cuboid(self):
        min_x = min([np.min(self.cells[cellID].filamentPoints[:, 0]) for cellID in self.cells])
        max_x = max([np.max(self.cells[cellID].filamentPoints[:, 0]) for cellID in self.cells])
        min_y = min([np.min(self.cells[cellID].filamentPoints[:, 1]) for cellID in self.cells])
        max_y = max([np.max(self.cells[cellID].filamentPoints[:, 1]) for cellID in self.cells])
        min_z = min([np.min(self.cells[cellID].filamentPoints[:, 2]) for cellID in self.cells])
        max_z = max([np.max(self.cells[cellID].filamentPoints[:, 2]) for cellID in self.cells])

        self._encapsulating_cuboid = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

    def remove_edge_cells(self, offset: float = 2., mode: str = "hardlimit", limit: int = 20, threshold = None):
        assert mode in ["hardlimit", "percentage"], "Invalid mode. Must be either 'hardlimit' or 'percentage'."
        
        print("Cell Nr before removals: ", len(self.cells))

        min_x = self._encapsulating_cuboid[0][0]
        max_x = self._encapsulating_cuboid[0][1]
        min_y = self._encapsulating_cuboid[1][0]
        max_y = self._encapsulating_cuboid[1][1]
        min_z = self._encapsulating_cuboid[2][0]
        max_z = self._encapsulating_cuboid[2][1]

        for cellID, cell in self.cells.items():
            j = np.sum(cell.filamentPoints[:, 0] <= min_x + offset) + np.sum(cell.filamentPoints[:, 0] >= max_x - offset)
            j += np.sum(cell.filamentPoints[:, 1] <= min_y + offset) + np.sum(cell.filamentPoints[:, 1] >= max_y - offset)
            j += np.sum(cell.filamentPoints[:, 2] <= min_z + offset) + np.sum(cell.filamentPoints[:, 2] >= max_z - offset)

            if mode == "hardlimit":
                if j >= limit:
                    del self.cells[cellID]
            elif mode == "percentage":
                nr_points = len(cell[3][0])
                percent = j / nr_points * 100
                if percent >= limit:
                    del self.cells[cellID]
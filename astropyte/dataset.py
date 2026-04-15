import logging
import os.path
import pandas as pd
import numpy as np
import pickle
from pymatreader import read_mat

from .cell import Cell

class Dataset:
    """A set of astrocytes.
    
    Functions that don't obviously return something else allow chaining, e.g.
    `Dataset().from_matlab("path/to/dataset/").to_pickle("path/to/dataset.pkl")`
    """
    def __init__(self, name: str = None, logger: logging.Logger = None):
        self._logger = logger or logging.getLogger(__name__)

        assert name.isalnum(), "Dataset name must be alphanumeric."
        self._name = name
        self._path = None  # type: str
        self._cells = {}  # type: dict[int, Cell]

        self._encapsulating_cuboid = None  # type: list[tuple[float, float]]

    @property
    def name(self):
        return self._name
    @property
    def path(self):
        return self._path
    @property
    def cells(self):
        return self._cells
    @property
    def encapsulating_cuboid(self):
        if self._encapsulating_cuboid is None:
            self._set_encapsulating_cuboid()
        return self._encapsulating_cuboid

    def _set_encapsulating_cuboid(self):
        """
        Calculate the boundaries of the filament points of all cells in the dataset in x, y, and z direction.
        Sets self._encapsulating_cuboid.
        """
        min_x = min([np.min(self.cells[cellID].filamentPoints[:, 0]) for cellID in self.cells])
        max_x = max([np.max(self.cells[cellID].filamentPoints[:, 0]) for cellID in self.cells])
        min_y = min([np.min(self.cells[cellID].filamentPoints[:, 1]) for cellID in self.cells])
        max_y = max([np.max(self.cells[cellID].filamentPoints[:, 1]) for cellID in self.cells])
        min_z = min([np.min(self.cells[cellID].filamentPoints[:, 2]) for cellID in self.cells])
        max_z = max([np.max(self.cells[cellID].filamentPoints[:, 2]) for cellID in self.cells])

        self._encapsulating_cuboid = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
        return self
    
    def to_pickle(self, path: str, version = "latest"):
        """Saves the dataset to a pickle file.

        Parameters
        ----------
        path : str
            The path to the file to save the dataset to.
        version : str
            The version of the export format.
        """
        self._logger.info(f"Saving dataset to pickle file {path}...")
        if version == "latest":
            version = "1.1"
        if version in ["1.0", "1.1"]:
            data = {
                "version": version,
                "name": self._name,
                "path": self._path,
                "encapsulating_cuboid": self._encapsulating_cuboid,
                "cells": [cell.to_dict(version = version) for cell in self._cells.values()]
            }
        else:
            raise ValueError(f"Invalid export version: {version}")

        with open(path, "wb") as f:
            pickle.dump(data, f)
        self._logger.info(f"Finished saving dataset to pickle file {path}.")
        return self

    def from_pickle(self, path: str):
        """Loads the dataset from a pickle file.
        Overwrites all previously loaded data.
        
        Parameters
        ----------
        path : str
            The path to the file to load the dataset from.
        """
        self._logger.info(f"Loading dataset from pickle file {path}...")
        with open(path, "rb") as f:
            data = pickle.load(f)

        version = data["version"]
        if version in ["1.0", "1.1"]:
            self._name = data["name"] if "name" in data else None
            self._path = data["path"] if "path" in data else None
            self._encapsulating_cuboid = data["encapsulating_cuboid"] if "encapsulating_cuboid" in data else None
            self._cells = {
                cell_data["ID"]: Cell(logger = self._logger).from_dict(cell_data) for cell_data in data["cells"]
            }
        else:
            raise ValueError(f"Invalid import version: {version}")
        self._logger.info(f"Finished loading dataset from pickle file {path}. Loaded {len(self.cells)} cells.")
        return self

    def from_matlab(self, path: str, remove_edge_cells: bool = True, edge_cell_offset: float = 2., edge_cell_mode: str = "hardlimit", edge_cell_limit: int = 20, remove_artifact_cells: bool = False, artifact_cell_threshold: int = 1):
        """Loads the dataset from matlab and csv files in the given dataset directory.
        Overwrites all previously loaded data.
        
        Parameters
        ----------
        path : str
            The path to the dataset directory. Assumes files to be present:
            - diameterData.csv (Generated in Matlab)
            - positionData.csv (Generated in Matlab)
            - matlab_and_excel_data.mat
        """
        self._logger.info(f"Loading dataset from matlab directory {path}...")
        self._cells = {}
        self._encapsulating_cuboid = None

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

            self._cells[cellID] = Cell(ID = cellID,
                                      filamentPoints = filamentPoints[cellID],
                                      filamentEdges = filamentEdges[cellID],
                                      branchPositions = cell_branchPositions,
                                      branchDiameters = cell_branchDiameters,
                                      logger = self._logger)

        for cell in cells_to_delete:
            self._FilamentPoints = np.delete(self._FilamentPoints, (cell), axis=0)
            self._FilamentEdges = np.delete(self._FilamentEdges, (cell), axis=0)

        if remove_edge_cells:
            self.remove_edge_cells(offset = edge_cell_offset, mode = edge_cell_mode, limit = edge_cell_limit)
        if remove_artifact_cells:
            self.remove_artifact_cells(threshold = artifact_cell_threshold)
        
        self._logger.info(f"Finished loading dataset. Loaded {len(self.cells)} cells.")
        return self

    def remove_edge_cells(self, offset: float = 2., mode: str = "hardlimit", limit: int = 20):  # TODO: implement cell.is_edge_cell(offset, mode, limit)
        """
        Removes cells that have a certain number or percentage of their filament points within a certain distance to
        the boundaries of the encapsulating cuboid.
        
        :param offset: maximum allowed distance to the boundaries of the encapsulating cuboid
        :param mode: determines the meaning of the limit parameter. Must be either "hardlimit" or "percentage":
                     "hardlimit": removes cells with more than `limit` filament points within the `offset` to the boundaries of the encapsulating cuboid
                     "percentage": removes cells with more than `limit` percentage of their filament points within the `offset` to the boundaries of the encapsulating cuboid
        :param limit: see `mode`
        """
        self._logger.info("Removing edge cells...")
        assert mode in ["hardlimit", "percentage"], "Invalid mode. Must be either 'hardlimit' or 'percentage'."

        min_x = self.encapsulating_cuboid[0][0]
        max_x = self.encapsulating_cuboid[0][1]
        min_y = self.encapsulating_cuboid[1][0]
        max_y = self.encapsulating_cuboid[1][1]
        min_z = self.encapsulating_cuboid[2][0]
        max_z = self.encapsulating_cuboid[2][1]

        n_cells_before = len(self.cells)

        edge_cell_ids = []
        for cellID, cell in self.cells.items():
            j = np.sum(cell.filamentPoints[:, 0] <= min_x + offset) + np.sum(cell.filamentPoints[:, 0] >= max_x - offset)
            j += np.sum(cell.filamentPoints[:, 1] <= min_y + offset) + np.sum(cell.filamentPoints[:, 1] >= max_y - offset)
            j += np.sum(cell.filamentPoints[:, 2] <= min_z + offset) + np.sum(cell.filamentPoints[:, 2] >= max_z - offset)

            if mode == "hardlimit":
                if j >= limit:
                    edge_cell_ids.append(cellID)
            elif mode == "percentage":
                nr_points = len(cell.filamentPoints[:, 0])
                percent = j / nr_points * 100
                if percent >= limit:
                    edge_cell_ids.append(cellID)
        
        for cellID in edge_cell_ids:
            del self.cells[cellID]

        n_cells_after = len(self.cells.keys())
        self._logger.info(f"Removed {n_cells_before - n_cells_after} edge cells from original {n_cells_before} cells. Remaining cells: {n_cells_after}.")
        return self

    def remove_artifact_cells(self, threshold: int):  # TODO: implement cell.is_artifact(threshold)
        """Removes cells that have less than a certain number of branching points.

        :param threshold: Maximum number of branching points for a cell to be considered an artifact and removed.
        """
        self._logger.info("Removing artifact cells...")
        n_cells_before = len(self.cells)
        artifact_cell_ids = []
        for cellID, cell in self.cells.items():
            if cell.n_branchingPoints <= threshold:
                artifact_cell_ids.append(cellID)
        for cellID in artifact_cell_ids:
            del self.cells[cellID]
        n_cells_after = len(self.cells.keys())
        self._logger.info(f"Removed {n_cells_before - n_cells_after} artifact cells from original {n_cells_before} cells. Remaining cells: {n_cells_after}.")
        return self

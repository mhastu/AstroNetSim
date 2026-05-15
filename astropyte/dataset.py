import logging
import os.path
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from pymatreader import read_mat
from morphio import PointLevel, SectionType, SomaType
from morphio.mut import Morphology as MutableMorphology
from scipy.spatial import cKDTree

from numpy.typing import NDArray

from .cell import Cell
from .util import create_contour

def build_morphology_from_filaments(
        filamentPoints,
        filamentEdges,
        branchPositions,
        branchDiameters,
        *,
        root_index=None,
        root_diameter=1.0,
        section_type=SectionType.glia_process,
        atol=1e-3,
        logger=None
    ):
    """
    Convert filament point/edge data into a morphio.mut.Morphology.

    Parameters
    ----------
    filamentPoints : array-like, shape (N, 3)
        XYZ coordinates of all graph points.

    filamentEdges : array-like, shape (M, 2)
        Undirected edges, each row contains two point indices.

    branchPositions : array-like, shape (B, 3)
        Coordinates of branching points. Must match points in filamentPoints.

    branchDiameters : array-like, shape (B,)
        Diameter assigned to each branch point. All downstream sections starting
        at that branch point get this diameter.

    root_index : int or None
        Index of the root point. If None, an endpoint is chosen automatically.

    root_diameter : float
        Diameter used for the first section if the root is not a branch point.

    section_type : morphio.SectionType
        Section type to use. default: SectionType.glia_process

    atol : float
        Tolerance for matching branchPositions to filamentPoints.

    Returns
    -------
    morphio.mut.Morphology
    """
    if logger is None:
        logger = logging.getLogger("none")

    points = np.asarray(filamentPoints, dtype=float)
    edges = np.asarray(filamentEdges, dtype=int)
    branch_positions = np.asarray(branchPositions, dtype=float)
    branch_diameters = np.asarray(branchDiameters, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("filamentPoints must have shape (N, 3).")

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("filamentEdges must have shape (M, 2).")

    if branch_positions.ndim != 2 or branch_positions.shape[1] != 3:
        raise ValueError("branchPositions must have shape (B, 3).")

    if len(branch_positions) != len(branch_diameters):
        raise ValueError(f"branchPositions ({len(branch_positions)}) and branchDiameters ({len(branch_diameters)}) must have the same length.")

    n_points = len(points)

    if np.any(edges < 0) or np.any(edges >= n_points):
        raise ValueError("filamentEdges contains indices outside filamentPoints.")

    # ------------------------------------------------------------------
    # 1. Map branch positions back to filament point indices
    # ------------------------------------------------------------------
    logger.debug(f"-- mapping branch positions back to filament point indices")

    tree = cKDTree(points)

    distances, indices = tree.query(
        branch_positions,
        k=1,
        distance_upper_bound=atol,
    )

    unmatched = np.isinf(distances)

    if np.any(unmatched):
        i = int(np.where(unmatched)[0][0])
        closest_distance, closest_index = tree.query(branch_positions[i], k=1)

        raise ValueError(
            f"Branch position {branch_positions[i]} does not match any filament point "
            f"within tolerance {atol}. Closest distance: {closest_distance}, "
            f"closest point index: {closest_index}"
        )

    branch_diameter_by_index = {
        int(idx): float(diam)
        for idx, diam in zip(indices, branch_diameters)
    }

    branch_indices = set(branch_diameter_by_index.keys())

    # ------------------------------------------------------------------
    # 2. Build undirected adjacency
    # ------------------------------------------------------------------
    logger.debug(f"-- building unidirected adjacency")
    adjacency = defaultdict(list)

    for a, b in edges:
        a = int(a)
        b = int(b)

        if a == b:
            raise ValueError(f"Self-edge found at point {a}.")

        adjacency[a].append(b)
        adjacency[b].append(a)

    degrees = {i: len(adjacency[i]) for i in range(n_points)}

    # ------------------------------------------------------------------
    # 3. Choose a root if none is given
    # ------------------------------------------------------------------
    logger.debug(f"-- choosing root point")
    if root_index is None:
        endpoints = [i for i, d in degrees.items() if d == 1]

        if endpoints:
            root_index = endpoints[0]
        else:
            # Fallback for closed or unusual structures.
            root_index = 0
    else:
        root_index = int(root_index)

    if root_index < 0 or root_index >= n_points:
        raise ValueError("root_index is outside filamentPoints.")

    # A MorphIO section boundary is usually:
    # - root
    # - branch point
    # - endpoint
    # - any graph node with degree != 2
    def is_boundary_node(i):
        return (
            i == root_index
            or i in branch_indices
            or degrees.get(i, 0) != 2
        )

    morpho = MutableMorphology()

    soma_points, soma_diameters = create_contour(radius=5)  # inherited from AstrocyteHeterogeneity (Softcode)
    soma_center = points[root_index]
    morpho.soma.points = soma_points + soma_center
    morpho.soma.diameters = soma_diameters
    morpho.soma.type = SomaType.SOMA_SIMPLE_CONTOUR  # HDF5 does not support single point somas

    visited_edges = set()

    def edge_key(a, b):
        return tuple(sorted((int(a), int(b))))

    def section_diameter_starting_at(node, inherited_diameter):
        """
        Diameter rule:
        - If this node is a known branch point, use its branch diameter.
        - Otherwise inherit the upstream diameter.
        """
        if node in branch_diameter_by_index:
            return branch_diameter_by_index[node]
        return inherited_diameter

    def trace_until_boundary(start, nxt):
        """
        Start at a boundary node and walk through degree-2 nodes until the next
        boundary/end/branch point is reached.
        """
        path = [start, nxt]
        prev = start
        cur = nxt

        while not is_boundary_node(cur):
            candidates = [x for x in adjacency[cur] if x != prev]

            if len(candidates) != 1:
                break

            prev, cur = cur, candidates[0]
            path.append(cur)

        return path

    def add_sections_from_node(current_node, parent_node, parent_section, inherited_diameter):
        for nxt in adjacency[current_node]:
            if nxt == parent_node:
                continue

            key = edge_key(current_node, nxt)
            if key in visited_edges:
                continue

            path_indices = trace_until_boundary(current_node, nxt)

            # Mark all graph edges in this traced section as visited.
            for a, b in zip(path_indices[:-1], path_indices[1:]):
                visited_edges.add(edge_key(a, b))

            section_points = points[path_indices].tolist()

            diameter = section_diameter_starting_at(current_node, inherited_diameter)
            section_diameters = [diameter] * len(section_points)

            point_level = PointLevel(section_points, section_diameters)

            if parent_section is None:
                section = morpho.append_root_section(point_level, section_type)
            else:
                section = parent_section.append_section(point_level, section_type)

            end_node = path_indices[-1]

            # The diameter for downstream sections is determined by the end node
            # if it is a branch point; otherwise it remains inherited.
            downstream_diameter = section_diameter_starting_at(end_node, diameter)

            add_sections_from_node(
                current_node=end_node,
                parent_node=path_indices[-2],
                parent_section=section,
                inherited_diameter=downstream_diameter,
            )

    initial_diameter = section_diameter_starting_at(root_index, root_diameter)

    logger.debug(f"-- adding sections")
    add_sections_from_node(
        current_node=root_index,
        parent_node=None,
        parent_section=None,
        inherited_diameter=initial_diameter,
    )

    # sanity check: all edges should be consumed in a tree-like graph.
    original_edges = {edge_key(a, b) for a, b in edges}
    unvisited = original_edges - visited_edges

    if unvisited:
        raise ValueError(
            "Some edges were not converted. The graph may be disconnected, "
            "cyclic, or not reachable from root_index. "
            f"Number of unvisited edges: {len(unvisited)}"
        )

    return morpho

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
        all_points = np.vstack([
            np.asarray(cell.morphology.points)
            for cell in self.cells.values()
            if cell.morphology.n_points > 0
        ])

        min_xyz = np.min(all_points, axis=0)
        max_xyz = np.max(all_points, axis=0)

        self._encapsulating_cuboid = [
            (min_xyz[0], max_xyz[0]),
            (min_xyz[1], max_xyz[1]),
            (min_xyz[2], max_xyz[2]),
        ]
        return self
    
    def save(self, path: str, version = "latest", overwrite = False):
        """Saves the dataset to one HDF5 file per cell and a metadata pickle file in the given directory.

        Parameters
        ----------
        path : str
            The path to the directory to save the dataset to.
        version : str
            The version of the export format.
        overwrite : bool
            Whether to overwrite existing files. If False and files already exist, raises an error.
        """
        self._logger.info(f"Saving dataset to directory {path}...")
        if version == "latest":
            version = "0.1"
        if version == "0.1":
            data = {
                "version": version,
                "name": self._name,
                "path": self._path,
                "encapsulating_cuboid": self._encapsulating_cuboid,
                "cells": [cell.to_dict(version = version) for cell in self._cells.values()]
            }
        else:
            raise ValueError(f"Invalid export version: {version}")

        if not os.path.exists(path):
            os.makedirs(path)
        # only allow saving to an empty directory to prevent accidental overwriting
        if os.listdir(path):
            if overwrite:
                self._logger.info(f"Directory {path} is not empty. Overwriting existing files.")
            else:
                raise FileExistsError(f"Directory {path} is not empty. Set overwrite=True to overwrite existing files.")
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(data, f)
        for cell in self.cells.values():
            cell.save_morphology_to_hdf(os.path.join(path, f"cell_{cell.ID}.h5"))
        self._logger.info(f"Finished saving dataset to directory {path}.")
        return self

    def load(self, path: str):
        """Loads the dataset from the given directory containing a HDF5 file for each cell and a pickle file for metadata.
        Overwrites all previously loaded data.

        Parameters
        ----------
        path : str
            The path to the file to load the dataset from.
        """
        self._logger.info(f"Loading dataset from directory {path}...")
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)

        version = data["version"]
        if version == "0.1":
            self._name = data["name"] if "name" in data else None
            self._path = data["path"] if "path" in data else None
            self._encapsulating_cuboid = data["encapsulating_cuboid"] if "encapsulating_cuboid" in data else None
            self._cells = {
                # loading morphology resets caches like ellipsoid, so we load morphology first and then call from_dict to set the metadata and caches again
                cell_data["ID"]: Cell(ID = cell_data["ID"], logger = self._logger).load_morphology_from_hdf(os.path.join(path, f"cell_{cell_data['ID']}.h5")).from_dict(cell_data) for cell_data in data["cells"]
            }
        else:
            raise ValueError(f"Invalid import version: {version}")
        self._logger.info(f"Finished loading dataset from directory {path}. Loaded {len(self.cells)} cells.")
        return self

    def from_matlab(self, path: str, remove_edge_cells: bool = True, edge_cell_offset: float = 2., edge_cell_mode: str = "hardlimit", edge_cell_limit: int = 20, remove_artifact_cells: bool = False, min_sections: int = 1):
        """Loads the dataset from matlab and csv files in the given dataset directory.
        Overwrites all previously loaded data.
        
        Parameters
        ----------
        path : str
            The path to the dataset directory. Assumes files to be present:
            - diameterData.csv (Generated in Matlab)
            - positionData.csv (Generated in Matlab)
            - matlab_and_excel_data.mat
        remove_edge_cells : bool
            Whether cells that are too close to the overall dataset boundary (encapsulating cuboid) should be removed
        edge_cell_offset : float
            Maximum allowed distance to dataset encapsulating cuboid
        edge_cell_mode : str
            determines the meaning of the limit parameter. Must be either "hardlimit" or "percentage":
            - "hardlimit": removes cells with more than `limit` filament points within the `offset` to the boundaries of the encapsulating cuboid
            - "percentage": removes cells with more than `limit` percentage of their filament points within the `offset` to the boundaries of the encapsulating cuboid
        edge_cell_limit : int
            see `edge_cell_mode`
        remove_artifact_cells : bool
            Whether cells that are considered an artifact should be removed from the dataset
        min_sections : int
            Minimum number of sections a cell should have to not be considered an artifact
        """
        self._logger.info(f"Loading dataset from matlab directory {path}...")
        self._cells = {}
        self._encapsulating_cuboid = None
        self._path = path

        # load raw data
        branchDiameters = pd.read_csv(os.path.join(path, "diameterData.csv"))
        branchPositions = pd.read_csv(os.path.join(path, "positionData.csv"))
        matlabAndExcelData = read_mat(os.path.join(path, "matlab_and_excel_data.mat"))  # type: dict[str, NDArray]
        filamentPoints = matlabAndExcelData["vFilamentsPoints"]
        filamentEdges = matlabAndExcelData["vFilamentsEdges"]

        skipped_trivial_cells = 0
        skipped_cells_due_to_errors = 0
        for cellID in range(len(filamentPoints)):
            self._logger.info(f"Loading cell {cellID}...")
            try:
                cell_filamentPoints = np.array(filamentPoints[cellID])
                cell_filamentEdges = np.array(filamentEdges[cellID], dtype=int)

                # skip trivial cells
                if len(cell_filamentPoints) < 4:
                    skipped_trivial_cells += 1
                    continue

                filamentID = 100000000 + cellID  # by definition of dataset
                # position of branching points
                cell_branchPositions_df = branchPositions.loc[branchPositions.FilamentID == filamentID].sort_values("ID", ignore_index = True)
                # diameter at branching points
                cell_branchDiameters_df = branchDiameters.loc[branchDiameters.FilamentID == filamentID].sort_values("ID", ignore_index = True)

                cell_branchPositions = cell_branchPositions_df.copy()[
                    ["PtPositionX", "PtPositionY", "PtPositionZ"]
                ].to_numpy()
                cell_branchDiameters = cell_branchDiameters_df.copy()["PtDiameter"].to_numpy()

                morphology = build_morphology_from_filaments(filamentPoints=cell_filamentPoints,
                                                            filamentEdges=cell_filamentEdges,
                                                            branchPositions=cell_branchPositions,
                                                            branchDiameters=cell_branchDiameters,
                                                            logger=self._logger)
                
                self._logger.debug(f"-- making cell morphology immutable")
                # convert to immutable: more efficient at working with astrocyte
                # to change the morphology of the astrocyte again, the morphology
                # needs to be converted to morphio.mut.Morphology again.
                morphology = morphology.as_immutable()
                self._cells[cellID] = Cell(morphology, ID=cellID, logger=self._logger)
            except Exception as e:
                self._logger.error(f"Error loading cell {cellID}: {e}")
                skipped_cells_due_to_errors += 1
                continue
            self._logger.info(f"Cell {cellID} loaded.")

        self._logger.info(f"Skipped {skipped_trivial_cells} trivial cells with less than 4 filament points.")
        self._logger.info(f"Skipped {skipped_cells_due_to_errors} cells due to errors during loading.")

        if remove_edge_cells:
            self.remove_edge_cells(offset = edge_cell_offset, mode = edge_cell_mode, limit = edge_cell_limit)
        if remove_artifact_cells:
            self.remove_artifact_cells(min_sections = min_sections)

        self._logger.info(f"Finished loading dataset. Loaded {len(self.cells)} cells.")
        return self

    def remove_edge_cells(self, offset: float = 2., mode: str = "hardlimit", limit: int = 20):
        """
        Removes cells that have a certain number or percentage of their filament points within a certain distance to
        the boundaries of the encapsulating cuboid.
        
        Parameters
        ----------
        offset : float
            Maximum allowed distance to the boundaries of the encapsulating cuboid
        mode : str
            Determines the meaning of the limit parameter. Must be either "hardlimit" or "percentage":
            - "hardlimit": removes cells with more than `limit` filament points within the `offset` to the boundaries of the encapsulating cuboid
            - "percentage": removes cells with more than `limit` percentage of their filament points within the `offset` to the boundaries of the encapsulating cuboid
        limit : int
            see `mode`
        """
        self._logger.info("Removing edge cells...")
        assert mode in ["hardlimit", "percentage"], "Invalid mode. Must be either 'hardlimit' or 'percentage'."

        min_x, max_x = self.encapsulating_cuboid[0]
        min_y, max_y = self.encapsulating_cuboid[1]
        min_z, max_z = self.encapsulating_cuboid[2]

        n_cells_before = len(self.cells)
        edge_cell_ids = []

        for cellID, cell in self.cells.items():
            points = np.asarray(cell.morphology.points)
            if points.size == 0:
                continue

            j = np.sum(points[:, 0] <= min_x + offset)
            j += np.sum(points[:, 0] >= max_x - offset)

            j += np.sum(points[:, 1] <= min_y + offset)
            j += np.sum(points[:, 1] >= max_y - offset)

            j += np.sum(points[:, 2] <= min_z + offset)
            j += np.sum(points[:, 2] >= max_z - offset)

            if mode == "hardlimit":
                if j >= limit:
                    edge_cell_ids.append(cellID)
            elif mode == "percentage":
                percent = j / len(points) * 100
                if percent >= limit:
                    edge_cell_ids.append(cellID)
        
        for cellID in edge_cell_ids:
            del self.cells[cellID]

        n_cells_after = len(self.cells.keys())
        self._logger.info(f"Removed {n_cells_before - n_cells_after} edge cells from original {n_cells_before} cells. Remaining cells: {n_cells_after}.")
        return self

    def remove_artifact_cells(self, min_sections: int):
        """Removes cells that have too few sections from the dataset.

        Parameters
        ----------
        min_sections : int
            Minimum number of sections a cell should have to not be considered an artifact
        """
        self._logger.info("Removing artifact cells...")
        n_cells_before = len(self.cells)
        artifact_cell_ids = []
        for cellID, cell in self.cells.items():
            if len(cell.morphology.sections) < min_sections:
                artifact_cell_ids.append(cellID)
        for cellID in artifact_cell_ids:
            del self.cells[cellID]
        n_cells_after = len(self.cells.keys())
        self._logger.info(f"Removed {n_cells_before - n_cells_after} artifact cells from original {n_cells_before} cells. Remaining cells: {n_cells_after}.")
        return self

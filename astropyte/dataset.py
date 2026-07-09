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

def build_morphology_from_matlab_data(
        filamentPoints,
        filamentEdges,
        diameters,
        soma_index=0,
        section_type=SectionType.glia_process,
        atol=1e-3,
        logger=None
    ):
    """
    Convert filament point/edge data into a morphio.mut.Morphology. The soma is stored as a simple 2D contour with radius 5.

    Parameters
    ----------
    filamentPoints : array-like, shape (N, 3)
        XYZ coordinates of all graph points.

    filamentEdges : array-like, shape (M, 2)
        Undirected edges, each row contains two point indices.

    diameters : pandas.DataFrame with columns ["ID", "FilamentID", "Depth", "PtPositionX", "PtPositionY", "PtPositionZ", "PtDiameter"]
        Coordinates, depth and diameter at branching points. Must match points in filamentPoints.

    soma_index : int or None
        Index of the soma in filamentPoints. Default: 0.

    section_type : morphio.SectionType
        Section type to use. Default: SectionType.glia_process

    atol : float
        Tolerance for matching diameter positions to filamentPoints. Default: 1e-3

    logger : logging.Logger
        logger to use

    Returns
    -------
    morphio.mut.Morphology
    """
    # explanation of the algorithm:
    # 1. build adjacency (list all connected neighbors for each point). degree of a point = number of neighbors
    # 2. starting at the soma:
    #   - recursively trace through degree-2 nodes (using adjacency) until
    #     a boundary node is reached (degree != 2).
    #   - for each section (from boundary node to boundary node), interpolate diameters along the
    #     path using the coordinates and the branch diameters at the endpoints.

    # ------------------------------------------------------------------
    # 0. Argument validation
    # ------------------------------------------------------------------
    if logger is None:
        logger = logging.getLogger("none")

    nodes = np.asarray(filamentPoints, dtype=float)
    edges = np.asarray(filamentEdges, dtype=int)

    # remove invalid diameter entries (NaN coordinates, NaN diameter, negative diameter)
    valid_diameters_1_nan_coordinates = diameters.dropna(subset=["PtPositionX", "PtPositionY", "PtPositionZ"])
    n_invalid_nan_coordinates = len(diameters) - len(valid_diameters_1_nan_coordinates)
    if n_invalid_nan_coordinates > 0:
        logger.warning(f"Removed {n_invalid_nan_coordinates} invalid diameter entries (NaN coordinates).")
    valid_diameters_2_nan_diameter = valid_diameters_1_nan_coordinates.dropna(subset=["PtDiameter"])
    n_invalid_nan_diameter = len(valid_diameters_1_nan_coordinates) - len(valid_diameters_2_nan_diameter)
    if n_invalid_nan_diameter > 0:
        logger.warning(f"Removed {n_invalid_nan_diameter} invalid diameter entries (NaN diameter).")
    valid_diameters_3_negative = valid_diameters_2_nan_diameter[valid_diameters_2_nan_diameter["PtDiameter"] >= 0]
    n_invalid_negative = len(valid_diameters_2_nan_diameter) - len(valid_diameters_3_negative)
    if n_invalid_negative > 0:
        logger.warning(f"Removed {n_invalid_negative} invalid diameter entries (negative diameter).")
    diameters = valid_diameters_3_negative

    diameter_positions = diameters[["PtPositionX", "PtPositionY", "PtPositionZ"]].to_numpy(dtype=float)
    diameter_values = diameters["PtDiameter"].to_numpy(dtype=float)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("filamentPoints must have shape (N, 3).")

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("filamentEdges must have shape (M, 2).")
    
    if np.any(np.isnan(diameter_values)) or np.any(diameter_values < 0):
        raise ValueError("There must be a valid diameter for each diameter entry.")

    if np.any(edges < 0) or np.any(edges >= len(nodes)):
        raise ValueError("filamentEdges contains indices outside filamentPoints.")

    # ------------------------------------------------------------------
    # 1. Map diameter positions to filament points
    # ------------------------------------------------------------------
    logger.debug(f"-- mapping diameter positions to filament points")

    # search is much faster with cKDTree
    # cKDTree has nothing to do with the morphio.Morphology
    tree = cKDTree(nodes)

    queried_distances, queried_diameter_indices = tree.query(
        diameter_positions,
        k=1,
        distance_upper_bound=atol,
    )

    unmatched = np.isinf(queried_distances)

    if np.any(unmatched):
        i = int(np.where(unmatched)[0][0])
        closest_distance, closest_index = tree.query(diameter_positions[i], k=1)

        raise ValueError(
            f"Diameter position {diameter_positions[i]} does not match any filament point "
            f"within tolerance {atol}. Closest distance: {closest_distance}, "
            f"closest filament point index: {closest_index}"
        )

    diameter_by_node_index = {}
    diameter_row_by_node_index = {}  # needed for branchData validation

    for row_idx, (node_idx, diameter) in enumerate(zip(queried_diameter_indices, diameter_values)):
        node_idx = int(node_idx)
        if node_idx in diameter_by_node_index:
            raise ValueError("Multiple diameter positions map to the same filament point.")
        diameter_by_node_index[node_idx] = float(diameter)
        diameter_row_by_node_index[node_idx] = int(row_idx)

    # ------------------------------------------------------------------
    # 2. Build undirected adjacency
    # ------------------------------------------------------------------
    logger.debug(f"-- building undirected adjacency")
    adjacency = defaultdict(list)

    for a, b in edges:
        a = int(a)
        b = int(b)

        if a == b:
            raise ValueError(f"Self-edge found at point {a}.")

        adjacency[a].append(b)
        adjacency[b].append(a)

    # degree = number of neighbors for each point, used to identify endpoints and branch points
    degrees = {i: len(adjacency[i]) for i in range(len(nodes))}

    # validate that all diameter positions are at section boundary nodes
    for idx in diameter_by_node_index:
        if degrees.get(idx, 0) == 2:
            raise ValueError(f"Diameter position at index {idx} has degree 2.")
    # validate that all section boundary nodes in the graph have a corresponding diameter
    for idx in degrees:
        if degrees[idx] > 2 and idx not in diameter_by_node_index:
            raise ValueError(f"Filament point at index {idx} has degree {degrees[idx]} > 2 (i.e. branch point) but no corresponding diameter. Cannot interpolate diameters along sections without branch diameters at section boundaries.")
    # use a fallback diameter of 0 for all endpoints (degree 1) that do not have a corresponding point in diameterPositions
    missing_endpoint_diameters = [idx for idx in degrees if degrees[idx] == 1 and idx not in diameter_by_node_index]
    if missing_endpoint_diameters:
        logger.info(f"{len(missing_endpoint_diameters)} endpoints (degree 1) do not have a corresponding point in diameterPositions. Using fallback diameter 0 for these endpoints.")
        for idx in missing_endpoint_diameters:
            diameter_by_node_index[idx] = 0.0
            diameter_row_by_node_index[idx] = None  # no corresponding row in diameterPositions

    section_boundary_indices = set(i for i in degrees if degrees[i] != 2)
    if diameter_by_node_index.keys() != section_boundary_indices:
        raise ValueError("All section boundary nodes (degree != 2) must have a corresponding diameter.")

    # ------------------------------------------------------------------
    # 3. Build morphology
    # ------------------------------------------------------------------
    morpho = MutableMorphology()

    # create soma as a contour
    soma_points, soma_diameters = create_contour(radius=5)  # inherited from AstrocyteHeterogeneity (Softcode)
    soma_center = nodes[soma_index]
    morpho.soma.points = soma_points + soma_center
    morpho.soma.diameters = soma_diameters
    morpho.soma.type = SomaType.SOMA_SIMPLE_CONTOUR  # HDF5 does not support single point somas

    def is_boundary_node(i):
        """Returns True if the node at index `i` is a section boundary (has degree != 2)."""
        return degrees.get(i, 0) != 2

    def trace_until_boundary(start, nxt):
        """
        Start at a boundary node and walk through degree-2 nodes until the next
        boundary/end/branch point is reached.

        Parameters
        ----------
        start : int
            Index of the starting boundary node.
        nxt : int
            Index of the next node to visit (must be a neighbor of `start`).
        """
        path = [start, nxt]
        prev = start
        cur = nxt

        while not is_boundary_node(cur):
            candidates = [x for x in adjacency[cur] if x != prev]

            if len(candidates) != 1:
                break  # should not happen, since `is_boundary_node` depends on `degrees` which is built from `adjacency`

            prev, cur = cur, candidates[0]
            path.append(cur)

        return path

    def edge_key(a, b):
        """Returns immutable key defining an edge (tuple of 2 point indices)"""
        return tuple(sorted((int(a), int(b))))

    visited_edges = set()
    branch_records = []
    def add_sections_from_node(current_node, parent_node, parent_section, depth):
        """
        Recursively add sections to the morphology using `adjacency`.
        
        Parameters
        ----------
        current_node : int
            Index of the current node to process.
        parent_node : int or None
            Index of the parent node (the node from which we arrived at `current_node`).
        parent_section : Section or None
            The parent section to which the new sections will be appended. If `None`, the new sections will be appended to the root of the morphology.
        depth : int
            Topological depth of sections that start from `current_node`. Soma-connected sections have depth 1.
        """
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

            section_points = nodes[path_indices].tolist()

            # interpolate diameters along the path using the coordinates and the diameters at the endpoints
            section_distances = np.linalg.norm(np.diff(section_points, axis=0), axis=1)
            section_distances = np.insert(section_distances, 0, 0)
            interp_x = np.cumsum(section_distances)
            interp_xp = [0, interp_x[-1]]
            interp_yp = [diameter_by_node_index[path_indices[0]], diameter_by_node_index[path_indices[-1]]]
            section_diameters = np.interp(interp_x, interp_xp, interp_yp)

            point_level = PointLevel(section_points, section_diameters)

            if parent_section is None:
                section = morpho.append_root_section(point_level, section_type)
            else:
                section = parent_section.append_section(point_level, section_type)

            # first 3 columns of branchData (from astro_geometry.m)
            # {currentBranch, [startIdx, endIdx], branchDepth}
            branch_records.append({
                "path_indices": [int(i) for i in path_indices],
                "diameter_row_indices": [
                    diameter_row_by_node_index[path_indices[0]],
                    diameter_row_by_node_index[path_indices[-1]],
                ],
                "depth": int(depth),
            })

            end_node = path_indices[-1]

            add_sections_from_node(
                current_node=end_node,
                parent_node=path_indices[-2],
                parent_section=section,
                depth=depth + 1
            )

    # interpolate soma diameter if soma index has degree 2 (i.e. is not a branch point and therefore has no diameter specified)
    if soma_index not in diameter_by_node_index:
        neighbors = adjacency[soma_index]
        if len(neighbors) != 2:
            raise ValueError(f"Soma has no diameter specified and has degree {len(neighbors)}. Expected degree 2 to interpolate soma diameter.")
        neighbor1, neighbor2 = neighbors
        path1 = trace_until_boundary(soma_index, neighbor1)
        path2 = trace_until_boundary(soma_index, neighbor2)
        section1_points = nodes[path1].tolist()
        section2_points = nodes[path2].tolist()
        section1_length = np.sum(np.linalg.norm(np.diff(section1_points, axis=0), axis=1))
        section2_length = np.sum(np.linalg.norm(np.diff(section2_points, axis=0), axis=1))
        interp_x = [section1_length]
        interp_xp = [0, section1_length + section2_length]
        interp_yp = [diameter_by_node_index[path1[-1]], diameter_by_node_index[path2[-1]]]
        soma_diameter = np.interp(interp_x, interp_xp, interp_yp)[0]
        diameter_by_node_index[soma_index] = soma_diameter
        diameter_row_by_node_index[soma_index] = None  # no corresponding row in diameterPositions
        logger.info(f"Soma has degree 2, interpolated diameter as {soma_diameter}.")

    logger.debug(f"-- building morphology")
    add_sections_from_node(
        current_node=soma_index,
        parent_node=None,
        parent_section=None,
        depth=1
    )

    # Sort branch_records by depth to mirror astro_geometry.m's breadth-first traversal:
    # all soma-connected sections first, then their daughters, then deeper descendants.
    # Python's sort is stable, so adjacency/traversal order is preserved within each depth.
    branch_records.sort(key=lambda record: record["depth"])

    # sanity check: all edges should be consumed in a tree-like graph.
    original_edges = {edge_key(a, b) for a, b in edges}
    unvisited = original_edges - visited_edges

    if unvisited:
        raise ValueError(
            "Some edges were not converted. The graph may be disconnected or cyclic."
            f"Number of unvisited edges: {len(unvisited)}"
        )

    return morpho, branch_records

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
            metadata = {
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
            pickle.dump(metadata, f)
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
            The path to the directory to load the dataset from.
        """
        self._logger.info(f"Loading dataset from directory {path}...")
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        version = metadata["version"]
        if version == "0.1":
            self._name = metadata["name"] if "name" in metadata else None
            self._path = metadata["path"] if "path" in metadata else None
            self._encapsulating_cuboid = metadata["encapsulating_cuboid"] if "encapsulating_cuboid" in metadata else None
            self._cells = {
                # loading morphology resets caches like ellipsoid, so we load morphology first and then call from_dict to set the metadata and caches again
                cell_data["ID"]: Cell(ID = cell_data["ID"], logger = self._logger).load_morphology_from_hdf(os.path.join(path, f"cell_{cell_data['ID']}.h5")).from_dict(cell_data) for cell_data in metadata["cells"]
            }
        else:
            raise ValueError(f"Invalid import version: {version}")
        self._logger.info(f"Finished loading dataset from directory {path}. Loaded {len(self.cells)} cells.")
        return self

    def from_matlab(self, path: str,
                    cells_to_load: list = None,
                    validate_branch_records: bool = False,
                    remove_edge_cells: bool = True,
                    edge_cell_offset: float = 2.,
                    edge_cell_mode: str = "hardlimit",
                    edge_cell_limit: int = 20,
                    remove_artifact_cells: bool = False,
                    min_sections: int = 1):
        """Loads the dataset from matlab and csv files in the given dataset directory.
        Overwrites all previously loaded data.

        Parameters
        ----------
        path : str
            The path to the dataset directory. Assumes files to be present:
            - dataset_[dataset_name].mat (contains diameterData and positionData as tables, which can't be loaded in Python)
            - diameterData.csv (Generated in Matlab using `writetable(diameterData, '../datasets/\[_dataset name_\]/diameterData.csv')`)
            - positionData.csv (Generated in Matlab using `writetable(positionData, '../datasets/\[_dataset name_\]/positionData.csv')`)
        cells_to_load : list of int or None
            List of cell IDs to load. If None, loads all cells.
        validate_branch_records : bool
            Whether to validate the loaded morphology against branchData.mat.
            If True, a warning is given if the records do not match.
            Assumes <dataset directory>/cell_<dataset_name>_<cell_id>/branchData.mat exists for each cell.
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
        diameterValues = pd.read_csv(os.path.join(path, "diameterData.csv"))
        diameterPositions = pd.read_csv(os.path.join(path, "positionData.csv"))
        mat = read_mat(os.path.join(path, f"dataset_{self.name}.mat"))  # type: dict[str, NDArray]
        filamentPoints = mat["vFilamentsPoints"]
        filamentEdges = mat["vFilamentsEdges"]

        # Merge tables on ID
        # left join keeps all rows from positionData
        diameters = pd.merge(
            diameterPositions,
            diameterValues,
            on="ID",
            how="left",
            suffixes=("_pos", "_diam")
        )

        # Check whether Depth values match for rows where diameter data exists
        depth_mismatch = diameters[
            diameters["Depth_diam"].notna() &
            (diameters["Depth_pos"] != diameters["Depth_diam"])
        ]
        if not depth_mismatch.empty:
            raise ValueError(
                f"Depth values do not match for some rows:\n{depth_mismatch[['ID', 'Depth_pos', 'Depth_diam']]}"
            )
        else:
            diameters["Depth"] = diameters["Depth_pos"]
            diameters = diameters.drop(columns=["Depth_pos", "Depth_diam"])

        # Check FilamentID consistency
        filament_mismatch = diameters[
            diameters["FilamentID_diam"].notna() &
            (diameters["FilamentID_pos"] != diameters["FilamentID_diam"])
        ]
        if not filament_mismatch.empty:
            raise ValueError(
                f"FilamentID values do not match for some rows:\n{filament_mismatch[['ID', 'FilamentID_pos', 'FilamentID_diam']]}"
            )
        else:
            diameters["FilamentID"] = diameters["FilamentID_pos"]
            diameters = diameters.drop(columns=["FilamentID_pos", "FilamentID_diam"])

        skipped_trivial_cells = 0
        skipped_cells_due_to_errors = 0
        all_cell_ids = set(range(len(filamentPoints)))
        if cells_to_load is None:
            cells_to_load = all_cell_ids
        else:
            cells_to_load = set(cells_to_load)
        valid_cell_ids = cells_to_load & all_cell_ids
        if len(valid_cell_ids) < len(cells_to_load):
            invalid_cell_ids = cells_to_load - all_cell_ids
            self._logger.warning(f"Cell IDs in dataset range from 0 to {len(filamentPoints)-1}. Ignoring invalid cell IDs: {invalid_cell_ids}")
        cells_to_load = valid_cell_ids
        for cellID in cells_to_load:
            self._logger.info(f"Loading cell {cellID}...")
            try:
                cell_filamentPoints = np.array(filamentPoints[cellID])
                cell_filamentEdges = np.array(filamentEdges[cellID], dtype=int)

                # skip trivial cells
                if len(cell_filamentPoints) < 4:
                    skipped_trivial_cells += 1
                    continue

                filamentID = 100000000 + cellID  # by definition of dataset
                # position and diameter of branching points
                cell_diameters = diameters.loc[diameters.FilamentID == filamentID]

                morphology, branch_records = build_morphology_from_matlab_data(filamentPoints=cell_filamentPoints,
                                                               filamentEdges=cell_filamentEdges,
                                                               diameters=cell_diameters,
                                                               logger=self._logger)
                

                if validate_branch_records:
                    self._logger.debug(f"-- validating branch records")
                    branchData_path=os.path.join(path, f"cell_{self.name}_{cellID+1}", "branchData.mat")  # MATLAB is 1-indexed
                    try:
                        branchData_mat = read_mat(branchData_path)
                        branchData = branchData_mat["branchData"]
                        branchData_records = [
                            {
                                "path_indices": [int(i)-1 for i in row[0]],  # MATLAB is 1-indexed
                                "diameter_row_indices": [int(row[1][0])-1, int(row[1][1])-1],  # MATLAB is 1-indexed
                                "depth": int(row[2])
                            }
                            for row in branchData
                        ]
                        if branchData_records != branch_records:
                            self._logger.warning(f"Branch records from MATLAB do not match those generated in Python for cell {cellID}.")
                            self._logger.debug(f"In total {len(branch_records)} records in Python, {len(branchData_records)} records in MATLAB.")
                            not_in_mat = [r for r in branch_records if r not in branchData_records]
                            not_in_python = [r for r in branchData_records if r not in branch_records]
                            self._logger.debug(f"{len(not_in_mat)} branch records in Python but not in MATLAB: {not_in_mat if len(not_in_mat) < 10 else str(not_in_mat[:10]) + '...'}")
                            self._logger.debug(f"{len(not_in_python)} branch records in MATLAB but not in Python: {not_in_python if len(not_in_python) < 10 else str(not_in_python[:10]) + '...'}")
                        else:
                            self._logger.info("Branch records from MATLAB match those generated in Python.")
                    except FileNotFoundError:
                        self._logger.error(f"branchData.mat not found for cell {cellID}. Skipping validation of branch records.")
                    except Exception as e:
                        self._logger.error(f"{type(e).__name__} while validating branch records for cell {cellID}: {e}")
                        self._logger.debug(f"Traceback:\n", exc_info=True)


                self._logger.debug(f"-- making cell morphology immutable")
                # convert to immutable: more efficient at working with astrocyte
                # to change the morphology of the astrocyte again, the morphology
                # needs to be converted to morphio.mut.Morphology again.
                morphology = morphology.as_immutable()
                self._cells[cellID] = Cell(morphology, ID=cellID, logger=self._logger)
            except Exception as e:
                self._logger.error(f"{type(e).__name__} while loading cell {cellID}: {e}")
                self._logger.debug(f"Traceback for cell {cellID}:\n", exc_info=True)
                skipped_cells_due_to_errors += 1
                continue
            self._logger.info(f"Cell {cellID} loaded.")

        self._logger.info(f"Skipped {skipped_trivial_cells} trivial cells with less than 4 filament points.")
        self._logger.info(f"Skipped {skipped_cells_due_to_errors} cells due to errors during loading.")

        if len(self._cells) == 0:
            self._logger.error("No cells were loaded.")
            return self

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

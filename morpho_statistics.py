import logging
from astropyte.dataset import Dataset
from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] [%(name)s:%(levelname)s] %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def average_total_process_length(
    dataset: Dataset
) -> list[float]:
    """Calculate the average total process length of a dataset."""
    return sum([cell.total_process_length for cell in dataset.cells.values()]) / len(dataset.cells)

def average_volume(
    dataset: Dataset,
    method: str = "cones"
) -> float:
    """Calculate the average volume of a dataset."""
    if method == "cones":
        return sum([cell.volume_of_cones for cell in dataset.cells.values()]) / len(dataset.cells)
    elif method == "cylinders":
        return sum([cell.volume_of_cylinders for cell in dataset.cells.values()]) / len(dataset.cells)
    else:
        raise ValueError(f"Unknown volume calculation method: {method}")

def average_number_of_terminals(
    dataset: Dataset
) -> float:
    """Calculate the average number of terminals in a dataset."""
    return sum([cell.n_terminals for cell in dataset.cells.values()]) / len(dataset.cells)

def average_center_of_mass_to_soma_distance_normalized(
    dataset: Dataset
) -> float:
    """Calculate the average distance from the center of mass (CoM) to the soma in a dataset, normalized by total process length."""
    # CoM = mean of all node coordinates in the cell
    return sum([
        np.linalg.norm(np.mean(cell.morphology.points) - np.mean(cell.morphology.soma.points))/cell.total_process_length*1000
         for cell in dataset.cells.values()
        ]) / len(dataset.cells)

def main():
    for dataset_name in ["H00", "H01", "H02", "H04", "C10", "C11"]:
        dataset = Dataset(name=dataset_name, logger=logger)
        dataset.load("./datasets/" + dataset.name)

        logger.info(f"Average Cone Volume of cells in {dataset_name}: {average_volume(dataset, method='cones'):.2f} µm³")
        logger.info(f"Average Cylinder Volume of cells in {dataset_name}: {average_volume(dataset, method='cylinders'):.2f} µm³")
        #logger.info(f"Average Total Process Length of {dataset_name}: {average_total_process_length(dataset):.2f} µm")
        #logger.info(f"Average Number of Terminals of {dataset_name}: {average_number_of_terminals(dataset):.2f}")
        #logger.info(f"Average CoM to Soma Distance (normalized by total process length) of {dataset_name}: {average_center_of_mass_to_soma_distance_normalized(dataset):.4f} * 10⁻³")

if __name__ == "__main__":
    main()

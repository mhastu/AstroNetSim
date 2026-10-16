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

def main():
    for dataset_name in ["H01", "H02", "H04", "C10", "C11"]:
        dataset = Dataset(name=dataset_name, logger=logger)
        # dataset.from_matlab(f"datasets/{dataset_name}_matlab/")
        dataset.load("./datasets/" + dataset.name)

        volumes = [cell.volume for cell in dataset.cells.values()]

        v = sum(volumes) / len(dataset.cells)
        logger.info(f"{dataset_name}: {v}")

        f = plt.figure()
        ax = f.add_subplot(111)

        ax.scatter(np.random.rand(len(volumes)), volumes)

        plt.show()

        # cell = dataset.cells[cell_id]
        # fig = plt.figure()
        # plot_cell_rot(fig, cell, mode="skeleton", plot_ellipse=True)
        # plt.show()

        # dataset.save("./datasets/" + dataset.name)

if __name__ == "__main__":
    main()

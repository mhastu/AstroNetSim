import logging
import os
import random
import matplotlib.pyplot as plt

from astropyte.dataset import Dataset
from astropyte.plot import plot_cell

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] [%(name)s:%(levelname)s] %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

def main():
    for dataset_name in ["H00"]:
        dataset = Dataset(name=dataset_name, logger=logger)
        dataset.load("./datasets/" + dataset.name)

        fig = plt.figure()
        fig.suptitle("Cells of cluster 1")
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_title("Cell 50 of cluster 1")
        plot_cell(ax=ax1, cell=dataset.cells[50], mode="skeleton", ellipsoid=False, color="b")
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_title("Cell 151 of cluster 1")
        plot_cell(ax=ax2, cell=dataset.cells[151], mode="skeleton", ellipsoid=False, color="b")
        plt.show()

        # dataset.save("./datasets/" + dataset.name)

if __name__ == "__main__":
    main()

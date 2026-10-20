import logging
import os
import random
import matplotlib.pyplot as plt

from astropyte.dataset import Dataset
from astropyte.plot import plot_cell_rot

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] [%(name)s:%(levelname)s] %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

def main():
    for dataset_name in ["H00"]:
        dataset = Dataset(name=dataset_name, logger=logger)
        # dataset.from_matlab(f"datasets/{dataset_name}_matlab/")
        dataset.load("./datasets/" + dataset.name)
        for cell_id in dataset.cells:
            cell = dataset.cells[cell_id]
            cell._set_ellipsoid(debug_plot=True)

        dataset.save("./datasets/" + dataset.name)

if __name__ == "__main__":
    main()

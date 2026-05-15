import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from astropyte.dataset import Dataset
from astropyte.plot import plot_cell
from astropyte.util import points_inside_ellipsoid

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s:%(levelname)s] %(message)s")
logger = logging.getLogger("main")

def main():
    dataset = Dataset(name="H00", logger=logger)
    dataset.load("./datasets/" + dataset.name)
    
    checked_pairs = set()  # to avoid checking pairs twice
    for ellipsoid_cell in dataset.cells.values():
        for cell in dataset.cells.values():
            if cell.ID == ellipsoid_cell.ID:
                continue

            if (cell.ID, ellipsoid_cell.ID) in checked_pairs or (ellipsoid_cell.ID, cell.ID) in checked_pairs:
                continue

            checked_pairs.add((cell.ID, ellipsoid_cell.ID))

            if np.any(points_inside_ellipsoid(cell.morphology.points, *ellipsoid_cell.ellipsoid)):
                # plot_cell(ax, cell, plot_ellipse = False)
                # print the distance of the cell pairs 10 closest points
                tree1 = cKDTree(ellipsoid_cell.morphology.points)
                tree2 = cKDTree(cell.morphology.points)
                dists, _ = tree1.query(tree2.data, k=10)
                logger.info(f"Cell {cell.ID} in {ellipsoid_cell.ID}. Closest distances: {', '.join(f'{d:.2f}' for d in dists.min(axis=0))} um")

    # plt.show()

if __name__ == "__main__":
    main()

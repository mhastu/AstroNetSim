import numpy as np
import matplotlib.pyplot as plt

from .cell import Cell

def plot_cell_sections(ax: plt.Axes, cell: Cell, color: str = "r", annotate: bool = False):
    """
    Plots the sections of the cell as single cylinders on the given axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on.
    cell : Cell
        The cell to plot.
    color : str
        The color to plot the cell in.
    """
    for section in cell.morphology.iter():
        points = np.asarray(section.points)

        if len(points) < 2:
            continue

        start = points[0]
        end = points[-1]

        diameters = np.asarray(section.diameters)

        # representative section diameter
        linewidth = np.mean(diameters) if len(diameters) > 0 else 1.0

        alpha = 1.0 / (cell.section_depths[section.id] + 1)

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            "-",
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

        if annotate:
            midpoint = 0.5 * (start + end)
            ax.text(
                midpoint[0],
                midpoint[1],
                midpoint[2],
                f"({section.id}, d={cell.section_depths[section.id]})",
            )

def plot_cell_segments(ax: plt.Axes, cell: Cell, color: str = "r", annotate: bool = False):
    """
    Plots all segments of the cell on the given axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on.
    cell : Cell
        The cell to plot.
    color : str
        The color to plot the cell in.
    """
    for section in cell.morphology.iter():
        points = np.asarray(section.points)

        if len(points) < 2:
            continue

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        diameters = np.asarray(section.diameters)

        # MorphIO stores diameters per point
        # Use diameter of segment start point
        for i in range(len(points) - 1):

            linewidth = diameters[i] if len(diameters) > i else 1.0

            alpha = 1.0 / (cell.section_depths[section.id] + 1)

            ax.plot(
                x[i:i+2],
                y[i:i+2],
                z[i:i+2],
                "-",
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

        if annotate:
            p0 = points[0]

            ax.text(
                p0[0],
                p0[1],
                p0[2],
                f"({section.id}, d={cell.section_depths[section.id]})",
            )

def plot_points(ax: plt.Axes, points: np.ndarray, size: float = 0.5, color: str = "r"):
    """
    Plots the given points on the given axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on.
    points : np.ndarray
        The points to plot. Shape (N, 3).
    color : str
        The color to plot the points in.
    """
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=size,
        color=color
    )

def plot_cell_points(ax: plt.Axes, cell: Cell, color: str = "r"):
    """
    Plots all section points of the cell on the given axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on.
    cell : Cell
        The cell to plot.
    color : str
        The color to plot the cell in.
    """
    all_points = []

    # soma
    if cell.morphology.soma is not None:
        all_points.append(np.asarray(cell.morphology.soma.points))

    # sections
    for section in cell.morphology.iter():
        all_points.append(np.asarray(section.points))

    if len(all_points) > 0:
        pts = np.vstack(all_points)
        plot_points(ax, pts, color=color)

def plot_ellipsoid(ax: plt.Axes, center: np.ndarray, radii: np.ndarray, rotation: np.ndarray):
    """
    Plots an ellipsoid on the given axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on.
    center : np.ndarray
        The center of the ellipsoid.
    radii : np.ndarray
        The radii of the ellipsoid along each axis.
    rotation : np.ndarray
        The rotation matrix of the ellipsoid.
    color : str
        The color to plot the ellipsoid in.
    alpha : float
        The transparency of the ellipsoid.
    """
    # parameter grid
    u = np.linspace(0, 2 * np.pi, 60)  # shape (N,)
    v = np.linspace(0, np.pi, 60)  # shape (M,)

    # ellipsoid in local coordinates
    x = radii[0] * np.outer(np.cos(u), np.sin(v))  # shape (N, M)
    y = radii[1] * np.outer(np.sin(u), np.sin(v))  # shape (N, M)
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))  # shape (N, M)

    # rotate + translate
    xyz = np.stack((x, y, z), axis=-1)  # shape (N, M, 3)
    xyz_rot = xyz @ rotation.T + center  # shape (N, M, 3)

    x = xyz_rot[..., 0]  # shape (N, M)
    y = xyz_rot[..., 1]  # shape (N, M)
    z = xyz_rot[..., 2]  # shape (N, M)

    ax.plot_surface(
        x,
        y,
        z,
        rstride=3,
        cstride=3,
        color="b",
        linewidth=0.1,
        alpha=0.2,
        shade=True,
    )

    # principal axes
    point1 = np.array([0, 0, radii[2]])
    point2 = np.array([0, radii[1], 0])
    point3 = np.array([radii[0], 0, 0])

    point_rot1 = rotation @ point1 + center
    point_rot2 = rotation @ point2 + center
    point_rot3 = rotation @ point3 + center

    ax.plot(
        [center[0], point_rot1[0]],
        [center[1], point_rot1[1]],
        [center[2], point_rot1[2]],
        color="b",
    )

    ax.plot(
        [center[0], point_rot2[0]],
        [center[1], point_rot2[1]],
        [center[2], point_rot2[2]],
        color="g",
    )

    ax.plot(
        [center[0], point_rot3[0]],
        [center[1], point_rot3[1]],
        [center[2], point_rot3[2]],
        color="m",
    )

def plot_cell(ax: plt.Axes, cell: Cell, color: str = "r", plot_ellipse: bool = False, mode: str = "skeleton", annotate: bool = False):
    """Plots the cell on the given axis.
    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on.
    cell : Cell
        The cell to plot.
    color : str
        The color to plot the cell in.
    plot_ellipse : bool
        Whether to plot the minimum volume encapsulating ellipsoid (MVEE) of the cell.
    mode : str
        The mode to plot the cell in. Choose from "skeleton", "section", or "scatter".
        "skeleton": Plots sections as straight cylinders between the branching points of the cell.
        "segment": Plots all segments of the cell as cylinders.
        "scatter": Plots all section points of the cell as scatter points.
    annotate : bool
        Whether to annotate the branches with their ID and depth. Only works in "skeleton" mode.
    """
    if mode == "skeleton":
        plot_cell_sections(ax, cell, color, annotate)
    elif mode == "segment":
        plot_cell_segments(ax, cell, color, annotate)
    elif mode == "scatter":
        plot_cell_points(ax, cell, color)
    else:
        raise ValueError("Invalid mode. Choose from 'skeleton', 'section', or 'scatter'.")

    if plot_ellipse:
        # shapes (3,), (3,), (3, 3)
        center, radii, rotation = cell.ellipsoid
        plot_ellipsoid(ax, center, radii, rotation)

def plot_cell_rot(fig: plt.Figure, cell: Cell, mode: str = "skeleton", plot_ellipse: bool = True):
    """Plots the cell in different views with the option to plot the minimum volume encapsulating ellipsoid (MVEE) of the cell.
    Parameters
    ----------
    fig : plt.Figure
        The figure to plot on.
    cell : Cell
        The cell to plot.
    mode : str
        The mode to plot the cell in.
    plot_ellipse : bool
        Whether to plot the minimum volume encapsulating ellipsoid (MVEE) of the cell.
    """
    if mode == None:
        mode = "skeleton"

    ax1 = fig.add_subplot(221, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax2 = fig.add_subplot(222, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax3 = fig.add_subplot(223, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax4 = fig.add_subplot(224, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")

    plot_cell(ax1, cell, plot_ellipse = plot_ellipse, mode = mode)
    plot_cell(ax2, cell,  plot_ellipse = plot_ellipse, mode = mode)
    plot_cell(ax3, cell,  plot_ellipse = plot_ellipse, mode = mode)
    plot_cell(ax4, cell,  plot_ellipse = plot_ellipse, mode = mode)
    
    ax1.set_title("Iso view")
    ax2.set_title("View onto xz-plane")
    ax3.set_title("View onto xy-plane")
    ax4.set_title("View onto yz-plane")

    ax2.view_init(0, 90) #frontal
    ax2.set_yticks([])
    ax3.view_init(90, 90)
    ax3.set_zticks([])
    ax4.view_init(0, 0)
    ax4.set_xticks([])

def plot_mvee_debug_step(i: int,
                         center: np.ndarray, radii: np.ndarray, rotation: np.ndarray,
                         inside_points: np.ndarray, outside_new_points: np.ndarray, outside_extra_points: np.ndarray
                         ):
    fig = plt.figure(i)
    ax1 = fig.add_subplot(221, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax2 = fig.add_subplot(222, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax3 = fig.add_subplot(223, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax4 = fig.add_subplot(224, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")


    plot_points(ax1, inside_points, color="gray")
    plot_ellipsoid(ax1, center, radii, rotation)
    plot_points(ax1, outside_extra_points, color="b")
    plot_points(ax1, outside_new_points, color="r")

    plot_points(ax2, inside_points, color="gray")
    plot_ellipsoid(ax2, center, radii, rotation)
    plot_points(ax2, outside_extra_points, color="b")
    plot_points(ax2, outside_new_points, color="r")

    plot_points(ax3, inside_points, color="gray")
    plot_ellipsoid(ax3, center, radii, rotation)
    plot_points(ax3, outside_extra_points, color="b")
    plot_points(ax3, outside_new_points, color="r")

    plot_points(ax4, inside_points, color="gray")
    plot_ellipsoid(ax4, center, radii, rotation)
    plot_points(ax4, outside_extra_points, color="b")
    plot_points(ax4, outside_new_points, color="r")
    
    ax1.set_title("Iso view")
    ax2.set_title("View onto xz-plane")
    ax3.set_title("View onto xy-plane")
    ax4.set_title("View onto yz-plane")

    ax2.view_init(0, 90) #frontal
    ax2.set_yticks([])
    ax3.view_init(90, 90)
    ax3.set_zticks([])
    ax4.view_init(0, 0)
    ax4.set_xticks([])

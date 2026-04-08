import numpy as np
import matplotlib.pyplot as plt

from .cell import Cell

def plot_cell(ax: plt.Axes, cell: Cell, color: str = "r", plot_ellipse: bool = False, mode: str = "rough", annotate: bool = False):
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
        The mode to plot the cell in. Choose from "rough", "fine", or "scatter".
        "rough": Plots the rough branches of the cell.
        "fine": Plots the fine branches of the cell.
        "scatter": Plots the branching points of the cell as scatter points.
    annotate : bool
        Whether to annotate the branches with their ID and depth. Only works in "rough" mode.
    """

    if mode not in ["rough", "fine", "scatter"]:
        raise ValueError("Invalid mode. Choose from 'rough', 'fine', or 'scatter'.")
    
    if mode == "rough":
        for id, branch in cell.rough_branches.items():
            x = np.array(branch["PtPositionX"])
            y = np.array(branch["PtPositionY"])
            z = np.array(branch["PtPositionZ"])
            dia = np.array(branch["PtDiameter"])
            depth = np.array(branch["Depth"])

            #https://stackoverflow.com/questions/38079366/matplotlib-line3dcollection-multicolored-line-edges-are-jagged
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            for ii in range(len(x)-1):
                segii=segments[ii]
                ax.plot(segii[:,0],segii[:,1],segii[:,2], "-",color= color,linewidth=dia[ii], alpha = 1 / (depth[ii] + 1))
                if annotate:
                    if ii == 0:
                        ax.text(segii[0,0],segii[0,1],segii[0,2], str((id, depth[ii])))
                        pass
    elif mode == "fine":
        for id, branch in cell.fine_branches.items():
            x = np.array(branch[:, 0])
            y = np.array(branch[:, 1])
            z = np.array(branch[:, 2])

            #https://stackoverflow.com/questions/38079366/matplotlib-line3dcollection-multicolored-line-edges-are-jagged
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            for ii in range(len(x)-1):
                segii=segments[ii]
                ax.plot(segii[:,0],segii[:,1],segii[:,2], "-",color= color) #, alpha = 1 / (depth[ii] + 1)
    elif mode == "scatter":
        x = np.array(cell.BranchingPointsData["PtPositionX"])
        y = np.array(cell.BranchingPointsData["PtPositionY"])
        z = np.array(cell.BranchingPointsData["PtPositionZ"])
        depth = np.array(cell.BranchingPointsData["Depth"])
        ax.scatter(x, y, z) #, alpha = 1/(depth+ 1)
    
    if plot_ellipse:
        # rx, ry, rz, angle0, angle1, angle2, center0, center1, center2 = cell.ellipsoid
        center, radii, rotation = cell.ellipsoid

        # parameter grid
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)

        # ellipsoid in local coordinates
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # stack and transform all points at once
        xyz = np.stack((x, y, z), axis=-1)  # shape (60, 60, 3)
        xyz_rot = xyz @ rotation.T + center  # rotate + translate

        x = xyz_rot[..., 0]
        y = xyz_rot[..., 1]
        z = xyz_rot[..., 2]

        ax.plot_surface(x, y, z,  rstride=3, color = "b", cstride=3, linewidth=0.1, alpha=0.2, shade=True)

        #plot principal axes:
        point1 = np.array([0, 0, radii[2]])
        point2 = np.array([0, radii[1], 0])
        point3 = np.array([radii[0], 0, 0])

        point_rot1 = np.dot(point1, rotation) + center
        point_rot2 = np.dot(point2, rotation) + center
        point_rot3 = np.dot(point3, rotation) + center

        ax.plot([center[0], point_rot1[0]], [center[1], point_rot1[1]], [center[2], point_rot1[2]], color = "b")
        ax.plot([center[0], point_rot2[0]], [center[1], point_rot2[1]], [center[2], point_rot2[2]], color = "g")
        ax.plot([center[0], point_rot3[0]], [center[1], point_rot3[1]], [center[2], point_rot3[2]], color = "m")

def plot_cell_rot(fig: plt.Figure, cell: Cell, mode: str = "rough", plot_ellipse: bool = True):
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
    ax1 = fig.add_subplot(221, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax2 = fig.add_subplot(222, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax3 = fig.add_subplot(223, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax4 = fig.add_subplot(224, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")

    if mode == None:
        plot_cell(ax1, cell, plot_ellipse = plot_ellipse)
        plot_cell(ax2, cell, plot_ellipse = plot_ellipse)
        plot_cell(ax3, cell, plot_ellipse = plot_ellipse)
        plot_cell(ax4, cell, plot_ellipse = plot_ellipse)
    else:
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

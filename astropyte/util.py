import numpy as np
from numpy.typing import NDArray

def mvee(points: np.array, tolerance: float = 1e-4) -> tuple[np.array, np.array, np.array]:
    """Computes the minimum volume encapsulating ellipsoid (MVEE) of a set of points using Khachiyan's algorithm.

    Finds the parameters of the ellipse equation in "center form"
    `(x-center).T * A * (x-center) = 1`
    
    Parameters
    ----------
    points : np.array
        A 2D array of shape (n_points, n_dimensions) containing the points to be enclosed by the ellipsoid.
    tolerance : float
        The tolerance for the convergence of the algorithm.

    Returns
    -------
    center : np.array
        The center of the ellipsoid.
    radii : np.array
        The lengths of the semi-axes of the ellipsoid.
    rotation : np.array
        The rotation matrix of the ellipsoid.
    """
    # Implementation adapted from https://stackoverflow.com/a/14016898

    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    u = np.ones(N) / N

    err = tolerance + 1
    while err > tolerance:
        V = Q @ np.diag(u) @ Q.T
        M = np.diag(Q.T @ np.linalg.inv(V) @ Q)
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    center = points.T @ u
    A = np.linalg.inv((points - center).T @ np.diag(u) @ (points - center)) / d

    # Eigen-decomposition to get the axes and rotation
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    radii = 1.0 / np.sqrt(eigenvalues)
    rotation = eigenvectors

    return center, radii, rotation


def points_inside_ellipsoid(points, center, radii, rotation, tolerance: float = 1e-3, relaxed: bool = True) -> NDArray[np.bool_]:
    """
    Returns the mask (np.array(bool)) of all points inside the ellipsoid.

    A point is inside the ellipsoid if:
        `sum((((points - center) @ rotation) / radii) ** 2) < 1 +/- tolerance`

    Parameters
    ----------
    points : np.ndarray
        The points to check, shape (N, 3).
    center : np.ndarray
        The center of the ellipsoid, shape (3,).
    radii : np.ndarray
        The radii of the ellipsoid, shape (3,).
    rotation : np.ndarray
        The rotation matrix of the ellipsoid, shape (3, 3).
    tolerance : float
        The tolerance for determining if a point is inside the ellipsoid.
    relaxed : bool
        If False, the tolerance is applied in the opposite direction (i.e., points must be strictly inside the ellipsoid). Default True (points can be slightly outside the ellipsoid).
    """

    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    rotation = np.asarray(rotation, dtype=float)

    if np.any(radii <= 0):
        raise ValueError(
            f"Invalid ellipsoid radii: {radii}. "
            "All radii must be positive."
        )

    shifted = points - center

    # Transform points into the ellipsoid's local coordinate system.
    local = shifted @ rotation

    normalized = local / radii
    values = np.sum(normalized ** 2, axis=1)

    if not relaxed:
        tolerance = -tolerance
    return values < (1.0 + tolerance)


def create_contour(radius, point_count=20, line_width=0.25):
    """Create a contour from a radius at origin 0.

    Note: The contour is created in the xy plan.

    within the corpus of ASC files, the mean is 0.21 for the diameter.
    this diameter is used for displaying the width of the contour line in
    Neurolucida, and 0.25 seems to be a nice display value
    """
    points = np.zeros((point_count, 3))
    phase = 2 * np.pi / point_count * np.arange(point_count)
    points[:, 0] = radius * np.sin(phase)
    points[:, 1] = radius * np.cos(phase)
    diameters = np.repeat(line_width, point_count)

    return points, diameters

def mahalanobis_radius(points, center, radii, rotation):
    """
    Compute the Mahalanobis radius of one or more points relative to an ellipsoid.

    Parameters
    ----------
    points : (..., D) array
        One point or many points.
    center : (D,) array
    radii : (D,) array
    rotation : (D, D) array
        Rotation matrix returned by mvee().

    Returns
    -------
    r : (...) array
        Mahalanobis radius.
        r < 1 : inside
        r = 1 : on surface
        r > 1 : outside
    """
    points = np.asarray(points)

    # Transform into ellipsoid coordinates
    local = (points - center) @ rotation

    # Scale by the radii
    scaled = local / radii

    return np.linalg.norm(scaled, axis=-1)

import numpy as np

def mvee(points: np.array, tolerance: float = 1e-2):
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

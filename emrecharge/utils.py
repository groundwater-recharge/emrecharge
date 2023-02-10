import numpy as np
import numpy.ma as ma
import skfmm
from discretize import TensorMesh
from scipy.spatial import cKDTree as kdtree
from SimPEG import utils
from verde import distance_mask


def inverse_distance_interpolation(
    xy,
    values,
    dx=100,
    dy=100,
    x_pad=1000,
    y_pad=1000,
    power=0,
    epsilon=None,
    k_nearest_points=20,
    max_distance=4000.0,
):
    """
    Evaluating 2D inverse distance weighting interpolation
    for given (x, y) points and values.

    Inverse distance weight, w, can be written as:
        w = 1/(distance+epsilon)**power

    Parameters
    ----------
    xy : array_like
        Input array including (x, y) locations; (n_locations, 2)
    values: array_like
        Input array including values defined at (x, y) locations; (n_locations, )
    dx : int
        Size of the uniform grid in x-direction
    dy : int
        Size of the uniform grid in y-direction
    x_pad : float
        Length of padding in x-direction
    y_pad : float
        Length of padding in y-direction
    power: float
        Exponent used when evaluating inverse distance weight.
    epsilon: float
        A floor value used when evaluating inverse distance weight.
    k_nearest_points: int
        k-nearest-point used when evaluating inverse distance weight.
    max_distance: float
        A separation distance used to maks grid points away from the (x, y) locations.

    Returns
    -------


    """
    xmin, xmax = xy[:, 0].min() - x_pad, xy[:, 0].max() + x_pad
    ymin, ymax = xy[:, 1].min() - y_pad, xy[:, 1].max() + y_pad

    nx = int((xmax - xmin) / dx)
    ny = int((ymax - ymin) / dy)
    hx = np.ones(nx) * dx
    hy = np.ones(ny) * dy
    x = np.arange(nx) * dx + xmin
    y = np.arange(ny) * dy + ymin
    X, Y = np.meshgrid(x, y)

    tree = kdtree(xy)

    d, inds_idw = tree.query(np.c_[X.flatten(), Y.flatten()], k=int(k_nearest_points))
    if epsilon is None:
        epsilon = np.min([dx, dy])
    w = 1.0 / ((d + epsilon) ** power)
    values_idw = np.sum(w * values[inds_idw], axis=1) / np.sum(w, axis=1)
    mask_inds = ~distance_mask(
        (xy[:, 0], xy[:, 1]),
        maxdist=max_distance,
        coordinates=(X.flatten(), Y.flatten()),
    )
    #     values_idw[mask_inds] = np.nan
    values_idw = ma.masked_array(values_idw, mask=mask_inds)
    values_idw = values_idw.reshape(X.shape)
    return x, y, values_idw

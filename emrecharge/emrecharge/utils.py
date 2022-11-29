import numpy as np
import numpy.ma as ma
import pandas as pd
import skfmm
from discretize import TensorMesh
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree as kdtree
from SimPEG import utils
from verde import distance_mask


def find_locations_in_distance(xy_input, xy_output, distance=100.0):
    """
    Find indicies of locations of xy_output within a given separation distance
    from locations of xy_input.

    Parameters
    ----------

    xy_input: (*,2) array_like
        Input locations.
    xy_output: (*,2) array_like
        Ouput Locations where the indicies of the locations are sought.
    distance: float
        Separation distance used as a threshold

    Returns
    -------
    pts : (*,2) ndarray, float
        Sought locations.
    inds: (*,) ndarray, integer
        Sought indicies.
    """
    tree = kdtree(xy_output)
    out = tree.query_ball_point(xy_input, distance)
    temp = np.unique(out)
    inds = []
    for ind in temp:
        if ind != []:
            inds.append(ind)
    if len(inds) == 0:
        return None, None
    inds = np.unique(np.hstack(inds))
    pts = xy_output[inds, :]
    return pts, inds


def find_closest_locations(xy_input, xy_output):
    """
    Find indicies of the closest locations of xy_output from from locations of xy_input.

    Parameters
    ----------

    xy_input: (*,2) array_like
        Input locations.
    xy_output: (*,2) array_like
        Ouput Locations where the indicies of the locations are sought.

    Returns
    -------
    d : (*,) ndarray, float
        Closest distance.
    inds: (*,) ndarray, integer
        Sought indicies.
    """
    tree = kdtree(xy_output)
    d, inds = tree.query(xy_input)
    return d, inds


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


# TODO:
# Calcuate fraction for each lithologic unit
# Need to simplify this use volume avearging
def compute_fraction_for_aem_layer(hz, lith_data, unique_code):
    """
    Compute fraction of lithology in AEM layers

    Parameters
    ----------

    hz: (n_layer,) array_like
        Thickness of the AEM layers
    lith_data: pandas DataFrame including ['From', 'To', 'Code']
        Lithology logs
    unique_code: array_like
        uniuqe lithology code; n_code = unique.size

    Returns
    -------
    fraction : (n_layer, n_code) ndarray, float
        Fractoin of each lithologic code (or unit)
    """
    n_code = unique_code.size
    z_top = lith_data.From.values
    z_bottom = lith_data.To.values
    z = np.r_[z_top, z_bottom[-1]]
    code = lith_data.Code.values
    zmin = z_top.min()
    zmax = z_bottom.max()
    depth = np.r_[0.0, np.cumsum(hz)][:]
    z_aem_top = depth[:-1]
    z_aem_bottom = depth[1:]

    # assume lithology log always start with zero depth
    # TODO: at the moment, the bottom aem layer, which overlaps with a portion of the driller's log
    # is ignored.
    n_layer = (z_aem_bottom < zmax).sum()
    fraction = np.ones((hz.size, n_code)) * np.nan

    for i_layer in range(n_layer):
        inds_in = np.argwhere(
            np.logical_and(z >= z_aem_top[i_layer], z <= z_aem_bottom[i_layer])
        ).flatten()
        dx_aem = z_aem_bottom[i_layer] - z_aem_top[i_layer]
        if inds_in.sum() != 0:
            z_in = z[inds_in]
            dx_in = np.diff(z_in)
            code_in = code[inds_in[:-1]]
            if i_layer == 0:
                inds_bottom = inds_in[-1] + 1
                inds = np.r_[inds_in, inds_bottom]
                z_tmp = z[inds]
                dx_bottom = z_aem_bottom[i_layer] - z[inds_bottom - 1]
                dx = np.r_[dx_in, dx_bottom]
                code_bottom = code[inds_bottom - 1]
                code_tmp = np.r_[code_in, code_bottom]
            else:
                inds_bottom = inds_in[-1] + 1
                inds_top = inds_in[0] - 1
                inds = np.r_[inds_top, inds_in, inds_bottom]
                z_tmp = z[inds]
                dx_top = z[inds_top + 1] - z_aem_top[i_layer]
                dx_bottom = z_aem_bottom[i_layer] - z[inds_bottom - 1]
                dx = np.r_[dx_top, dx_in, dx_bottom]
                code_bottom = code[inds_bottom - 1]
                code_top = code[inds_top]
                code_tmp = np.r_[code_top, code_in, code_bottom]
        else:
            inds_top = np.argmin(abs(z - z_aem_top[i_layer]))
            inds_bottom = inds_top + 1
            inds = np.r_[inds_top, inds_bottom]
            z_tmp = z[inds]
            dx = np.r_[dx_aem]
            #     print (code[inds_top])
            code_tmp = np.r_[code[inds_top]]
        for i_code, unique_code_tmp in enumerate(unique_code):
            fraction[i_layer, i_code] = dx[code_tmp == unique_code_tmp].sum() / dx_aem
    return fraction


def rock_physics_transform_rk_2018(
    fraction_matrix,
    resistivity,
    n_bootstrap=10000,
    bounds=None,
    circuit_type="parallel",
):
    """
    Solve a linear inverse problem to compute resistivity values of each lithologic unit
    then bootstrap to generate resistivity distribution for each lithologic unit

    Parameters
    ----------

    fraction_matrix: array_like
        fraction of lithology in upscaled layers, size of the matrix is (n_layers x n_lithology)
    resistivity: array_like
        resistivity values in upscaled layers
    n_bootstrap: optional, int
        number of bootstrap iteration

    Returns
    -------

    resistivity_for_lithology: array_like
        bootstrapped resistivity values for each lithology, size of the matrix is (n_bootstrap, n_lithology)
    """

    if bounds is None:
        bounds = (0, np.inf)
    if circuit_type == "parallel":
        conductivity_for_lithology = []
        for ii in range(n_bootstrap):
            n_sample = int(resistivity.size)
            inds_rand = np.random.randint(0, high=resistivity.size - 1, size=n_sample)
            d = 1.0 / resistivity[inds_rand].copy()
            conductivity_for_lithology.append(
                lsq_linear(fraction_matrix[inds_rand, :], d, bounds=(bounds))["x"]
            )
        resistivity_for_lithology = 1.0 / np.vstack(conductivity_for_lithology)
    elif circuit_type == "series":
        resistivity_for_lithology = []
        for ii in range(n_bootstrap):
            n_sample = int(resistivity.size)
            inds_rand = np.random.randint(0, high=resistivity.size - 1, size=n_sample)
            d = resistivity[inds_rand].copy()
            resistivity_for_lithology.append(
                lsq_linear(fraction_matrix[inds_rand, :], d, bounds=(bounds))["x"]
            )
        resistivity_for_lithology = np.vstack(resistivity_for_lithology)
    return resistivity_for_lithology


def classify_cf(cf, threshold):
    """
    Classify coarse fraction values into two categories: low-K and high-K unit

    Parameters
    ----------

    cf: array_like, float
        coarse
    threshold: float
        a threshold value to classify coarse fraction values into low-K and high-K unit
    Returns
    -------

    resistivity_for_lithology: array_like
        bootstrapped resistivity values for each lithology, size of the matrix is (n_bootstrap, n_lithology)
    """
    binary_model = np.ones_like(cf)
    binary_model[cf < threshold] = 0.0
    return binary_model


# ==================================== RECHARGE METRICS ====================================#


def get_ready_for_fmm(mesh, xyz_water):
    """
    Extend mesh for running fast marching and find indices for the source.

    Parameters
    ----------

    mesh: object
        discretize TensorMesh object.
    xyz_water: array_like, (*,3)
        locations of water table.

    Returns
    -------

    mesh_fmm: array_like
        discretize TensorMesh object.
    inds_source: array_like, bool
        indicies of the source.
    """
    dx = mesh.hx.min()
    dy = mesh.hy.min()
    dz = mesh.hz.min()

    hx = np.r_[mesh.hx[0], mesh.hx, mesh.hx[-1]]
    hy = np.r_[mesh.hy[0], mesh.hy, mesh.hy[-1]]
    hz = np.r_[mesh.hz, mesh.hz[-1]]
    x0 = [mesh.x0[0] - mesh.hx[0], mesh.x0[1] - mesh.hx[1], mesh.x0[2]]
    dz = hz.min()
    mesh_fmm = TensorMesh([hx, hy, hz], x0=x0)
    xyz_wse_up = np.c_[xyz_water[:, :2], -xyz_water[:, 2]]
    xyz_wse_down = np.c_[xyz_water[:, :2], -xyz_water[:, 2] - dz]
    inds_below_wt_up = utils.surface2ind_topo(mesh_fmm, xyz_wse_up)
    inds_below_wt_down = ~utils.surface2ind_topo(mesh_fmm, xyz_wse_down)
    inds_source = inds_below_wt_down & inds_below_wt_up
    return mesh_fmm, inds_source


def run_fmm(binary_values, mesh_fmm, inds_source, vel_ratio=None):
    """
    Run fast marching to calculate distance from surface to water table.

    Parameters
    ----------

    binary_values: array_like (*,)

    mesh_fmm: object
        discretize TensorMesh object.
    xyz_water: array_like, (*,3)
        locations of water table.
    inds_source:
        indicies of the source.
    vel_ratio: None or float
        ratio of the infiltration rate between low-K and high-K unit.

    Returns
    -------

    shortest_distance: array_like, (*,*)
        2D map of the shortest distance.
    """

    nx, ny, nz = mesh_fmm.vnC
    values_tmp = np.random.randint(low=0, high=2, size=nx * ny * nz).reshape(
        (nx, ny, nz)
    )
    values_tmp[1:-1, 1:-1, :-1] = binary_values.reshape(
        (nx - 2, ny - 2, nz - 1), order="F"
    )
    values_tmp[:, :, -1] = np.ones((nx, ny))
    values_tmp = utils.mkvc(values_tmp)
    dx = mesh_fmm.hx[0]
    dy = mesh_fmm.hy[0]
    dz = mesh_fmm.hz[0]
    # source term
    phi = np.ones(mesh_fmm.nC)
    # speed (assume 1 m/s)
    speed = np.ones(mesh_fmm.nC) * 1
    inds_noflow = values_tmp == 0.0
    # mask indicies (I assumed that clay unit as a barrier)
    if vel_ratio is None:
        mask = np.zeros(mesh_fmm.nC, dtype=bool)
        mask[inds_noflow] = True
        phi = np.ma.MaskedArray(phi, mask)
        phi[inds_source] = -1
        d = skfmm.distance(
            phi.reshape(mesh_fmm.vnC, order="F").transpose((1, 0, 2)),
            dx=[dx, dy, dz],
        )
        shortest_distance = d.transpose((1, 0, 2))[1:-1, 1:-1, -1].flatten(order="F")

    else:
        mask = np.zeros(mesh_fmm.nC, dtype=bool)
        inds_noflow = values_tmp == 0.0
        if vel_ratio >= 1.0:
            raise Exception(
                "input vel_ratio is {:.1e}, this should be less than 1".format(
                    vel_ratio
                )
            )
        speed[inds_noflow] = vel_ratio
        phi = np.ma.MaskedArray(phi, mask)
        phi[inds_source] = -1
        # travel time (since we assumed 1m/s, t is same as distance)
        # t and d will be different when you assign variable velocity
        t = skfmm.travel_time(
            phi.reshape(mesh_fmm.vnC, order="F").transpose((1, 0, 2)),
            speed.reshape(mesh_fmm.vnC, order="F").transpose((1, 0, 2)),
            dx=[dx, dy, dz],
            order=1,
        )
        shortest_distance = t.transpose((1, 0, 2))[1:-1, 1:-1, -1].flatten(order="F")
    return shortest_distance


def get_distance_to_first_lowK(values, inds_below, z):
    inds_lowK = np.logical_or(values == 0, inds_below)
    z_inds = np.ones(inds_lowK.shape[0], dtype=int) * -1
    for ii in range(inds_lowK.shape[0]):
        tmp = np.argwhere(inds_lowK[ii, :])
        if tmp.size > 0:
            z_inds[ii] = tmp.max()
    shortest_distance = z[z_inds]
    return -shortest_distance

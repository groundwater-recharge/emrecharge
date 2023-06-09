import numpy as np
import numpy.ma as ma
import skfmm
from discretize import TensorMesh
from SimPEG import utils


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

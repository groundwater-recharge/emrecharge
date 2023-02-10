import numpy as np
import pandas as pd
from discretize import TensorMesh
from discretize.utils import volume_average
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree as kdtree

from .datasets import EMDataset
from .utils import inverse_distance_interpolation


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


def compute_fraction_for_aem_layer(hz, lith_data):
    """
    Compute fraction of lithology in AEM layers

    Parameters
    ----------

    hz: (n_layer,) array_like
        Thickness of the AEM layers
    lith_data: pandas DataFrame including ['From', 'To', 'Code']
        Lithology logs

    Returns
    -------
    fraction : (n_layer, 2) ndarray, float
        columns of fine and coarse fractions
    """
    z_top = lith_data.From.values
    z_bottom = lith_data.To.values
    z = np.r_[z_top, z_bottom[-1]]
    code = lith_data.Code.values
    zmin = z_top.min()
    zmax = z_bottom.max()
    depth = np.r_[0.0, np.cumsum(hz)][:]
    z_aem_top = depth[:-1]
    z_aem_bottom = depth[1:]
    fraction = np.zeros((len(hz), 2), dtype=float) * np.nan
    hz_lith = z_bottom - z_top
    inds_active = np.logical_and(depth[: len(hz)] >= zmin, depth[: len(hz)] <= zmax)
    hz_active = hz[inds_active]

    mesh_1d_aem = TensorMesh([hz_active])
    mesh_1d_lith = TensorMesh([hz_lith])
    P = volume_average(mesh_1d_lith, mesh_1d_aem)
    coarse_fraction_active = P @ lith_data["Code"].values
    fraction[inds_active, 1] = coarse_fraction_active
    fraction[inds_active, 0] = 1 - coarse_fraction_active
    return fraction


def generate_water_level_map(
    water_level_df: pd.DataFrame,
    em_data: EMDataset,
    dx=500,
    dy=500,
    max_distance=1000,
    k_nearest_points=200,
    water_level_contour_df=None,
    constant_water_level=30.0,
):
    lx = em_data.xy[:, 0].max() - em_data.xy[:, 0].min()
    ly = em_data.xy[:, 1].max() - em_data.xy[:, 1].min()
    x_pad = lx * 0.1
    y_pad = ly * 0.1

    if water_level_df.shape[0] == 0:
        wse_em = np.ones(em_data.num_soundings, dtype=float) * constant_water_level
        f_int_wse = NearestNDInterpolator(em_data.xy, wse_em)
    else:
        xy_wse = water_level_df[["UTMX", "UTMY"]].values
        wse = water_level_df["GSE_WSE"].values

        if water_level_contour_df is not None:
            xy_wse_contour = water_level_contour_df[["UTMX", "UTMY"]].values
            xy_wse = np.vstack((xy_wse, xy_wse_contour))
            wse = np.r_[wse, water_level_contour_df["GSE_WSE"].values]

        # linear interpolation
        f_int_wse = NearestNDInterpolator(xy_wse, wse)
        wse_em = f_int_wse(em_data.xy)

    x, y, wse_idw = inverse_distance_interpolation(
        em_data.xy,
        wse_em,
        dx=dx,
        dy=dy,
        max_distance=max_distance,
        k_nearest_points=k_nearest_points,
        power=0,
        x_pad=x_pad,
        y_pad=y_pad,
    )

    X, Y = np.meshgrid(x, y)

    df_water_table_grid = pd.DataFrame(
        data=np.c_[X.flatten(), Y.flatten(), wse_idw.data.flatten()],
        columns=["x", "y", "water_table"],
    )

    return dict(
        x=x,
        y=y,
        X=X,
        Y=X,
        wse_idw=wse_idw,
        wse_em=wse_em,
        df=df_water_table_grid,
        nn_interpolator=f_int_wse,
    )


def compute_colocations(
    distance_threshold: int, em_data: EMDataset, df_lithology: pd.DataFrame
):
    lithology_group = df_lithology.groupby("WELL_ID")
    df_lithology_collar = lithology_group[["UTMX", "UTMY"]].mean()
    xy_lithology = df_lithology_collar[["UTMX", "UTMY"]].values

    # find the well locations that are within the distance_threshold of any em suruvey location
    xy_lithology_colocated, inds_driller = find_locations_in_distance(
        em_data.xy, xy_lithology, distance=distance_threshold
    )

    # use each well location to lookup the cloest em survey location
    d_aem_colocated, inds_aem_colocated = find_closest_locations(
        xy_lithology_colocated, em_data.xy
    )

    # get the subset of co-located survey locations
    xy_aem_colocated = em_data.xy[inds_aem_colocated, :]

    # get the subset of co-located wells
    _, inds_lithology_colocated = find_closest_locations(xy_aem_colocated, xy_lithology)
    df_lithology_collar_colocated = df_lithology_collar.loc[
        df_lithology_collar.index[inds_lithology_colocated]
    ]

    # get names of the colocated wells
    well_names_colocated = df_lithology_collar_colocated.index.to_list()

    # there should always be the same number of co-ocated wells and survey locations
    assert inds_aem_colocated.size == inds_driller.size

    n_colocated = inds_aem_colocated.size
    mean_separation_distance = d_aem_colocated.mean()
    return dict(
        n_colocated=n_colocated,
        mean_separation_distance=mean_separation_distance,
        xy_em=xy_aem_colocated,
        lithology_collar=df_lithology_collar_colocated,
        well_names=well_names_colocated,
        inds_em=inds_aem_colocated,
        inds_lithology=inds_lithology_colocated,
    )

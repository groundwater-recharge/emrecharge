from .datasets import EMDataset
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as kdtree
from scipy.interpolate import NearestNDInterpolator
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



def generate_water_level_map(water_level_df: pd.DataFrame, em_data: EMDataset):
    xy_wse = water_level_df[['X', 'Y']].values
    wse = -water_level_df['TSZ'].values
    
    # nearest interpolation
    f_int_wse = NearestNDInterpolator(xy_wse, wse)
    wse_em = f_int_wse(em_data.xy)

    x, y, wse_idw = inverse_distance_interpolation(
        em_data.xy, wse_em, 
        dx=500, dy=500,
        max_distance=1000., k_nearest_points=200, power=0,
        x_pad=2000, y_pad=2000,
    )
    
    X, Y = np.meshgrid(x, y)
    # values_tmp = wse_idw[~wse_idw.mask]
    # x_vec_tmp = X[~wse_idw.mask]
    # y_vec_tmp = Y[~wse_idw.mask]
    
    df_water_table_grid = pd.DataFrame(data=np.c_[X.flatten(), Y.flatten(), wse_idw.data.flatten()], columns=['x', 'y', 'water_table'])
    
    return dict(
        x=x,
        y=y,
        X=X,
        Y=X,
        wse_idw=wse_idw,
        wse_em=wse_em,
        df=df_water_table_grid,
        nn_interpolator=f_int_wse
    )

def compute_colocations(distance_threshold: int, em_data: EMDataset, df_lithology: pd.DataFrame):
    lithology_group = df_lithology.groupby("WELL_ID")
    df_lithology_collar = lithology_group[['UTMX', 'UTMY']].mean()
    xy_lithology = df_lithology_collar[['UTMX', 'UTMY']].values

    # find the well locations that are within the distance_threshold of any em suruvey location
    xy_lithology_colocated, inds_driller = find_locations_in_distance(em_data.xy, xy_lithology, distance=distance_threshold)

    # use each well location to lookup the cloest em survey location
    d_aem_colocated, inds_aem_colocated = find_closest_locations(xy_lithology_colocated, em_data.xy)

    # get the subset of co-located survey locations
    xy_aem_colocated = em_data.xy[inds_aem_colocated,:]

    # get the subset of co-located wells
    _, inds_lithology_colocated = find_closest_locations(xy_aem_colocated, xy_lithology)
    df_lithology_collar_colocated = df_lithology_collar.loc[df_lithology_collar.index[inds_lithology_colocated]]

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
        inds_lithology=inds_lithology_colocated
    )
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear


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


def compute_interval(df: pd.DataFrame, n_bootstrap=1000):
    fraction_matrix_above = df[df["gse_wse"] >= df["top"]][
        ["f_fine", "f_coarse"]
    ].values
    resistivity_above = df[df["gse_wse"] >= df["top"]]["rho_aem"].values
    resistivity_for_lithology_above = rock_physics_transform_rk_2018(
        fraction_matrix_above, resistivity_above, n_bootstrap=n_bootstrap
    )
    return resistivity_for_lithology_above

    # fraction_matrix_below = df[df["gse_wse"] <= df["top"]][
    #     ["f_fine", "f_coarse"]
    # ].values
    # resistivity_below = df[df["gse_wse"] <= df["top"]]["rho_aem"].values
    # resistivity_for_lithology_below = rock_physics_transform_rk_2018(
    #     fraction_matrix_below, resistivity_below, n_bootstrap=n_bootstrap
    # )

    # return resistivity_for_lithology_above, resistivity_for_lithology_below


def compute_integral(df: pd.DataFrame, n_bootstrap=1000):
    def func(x):
        hz = x.bottom - x.top
        return 1.0 / ((1.0 / x.rho_aem * hz).sum() / hz.sum())

    def func_cf(x):
        hz = x.bottom - x.top
        return (x.f_coarse * hz).sum() / hz.sum()

    df_above = df[df["bottom"] <= df["gse_wse"]]
    group_above = df_above.groupby("well_names")

    # df_below = df[df["bottom"] >= df["gse_wse"]]
    # group_below = df_below.groupby("well_names")

    rho_int_above = group_above.apply(func).values
    # rho_int_below = group_below.apply(func).values

    cf_int_above = group_above.apply(func_cf).values
    # cf_int_below = group_below.apply(func_cf).values

    fraction_matrix_above = np.c_[1 - cf_int_above, cf_int_above]
    resistivity_for_lithology_above = rock_physics_transform_rk_2018(
        fraction_matrix_above, rho_int_above, n_bootstrap=n_bootstrap
    )

    # fraction_matrix_below = np.c_[1 - cf_int_below, cf_int_below]
    # resistivity_for_lithology_below = rock_physics_transform_rk_2018(
    #     fraction_matrix_below, rho_int_below, n_bootstrap=n_bootstrap
    # )

    # return resistivity_for_lithology_above, resistivity_for_lithology_below
    return resistivity_for_lithology_above


def from_sigma_to_fraction(sigma, sigma_fine, sigma_coarse):
    sigma_bounded = sigma.copy()
    sigma_bounded[sigma >= sigma_fine] = sigma_fine
    sigma_bounded[sigma <= sigma_coarse] = sigma_coarse
    fraction_coarse = (sigma_fine - sigma_bounded) / (sigma_fine - sigma_coarse)
    return fraction_coarse


"""
    Compute and return mapping functions given percentiles and resistivity range
    
    Parameters
    ----------
    
    df_rho_distribution: Pandas DataFrame
        dataframe comtaining the. lithology distributions
    percentile_fines: array_like
        mapping function percentiles for fines [low, mid, high]
    percentile_coarses: array_like
        mapping function percentiles for coarses [low, mid, high]
    rho_min: float
        minimum resistivity value
    rho_max: float
        maximum resistivity value    
    
        
    Returns
    -------
    
    Dictionary
    
    func_cf_aboves: coarse fraction interpolator for above the water table f(log10(rho))
    func_cf_belows: coarse fraction interpolator for below the water table f(log10(rho))
    rho: resistivity value array corresponding to fractions

"""


def compute_rho_to_cf_mappings(
    df_rock_physics, percentile_fines, percentile_coarses, rho_min, rho_max
):
    rho_fine_dominated_above = df_rock_physics["rho_fine_dominated_above"].values
    rho_coarse_dominated_above = df_rock_physics["rho_coarse_dominated_above"].values
    # rho_fine_dominated_below = df_rock_physics["rho_fine_dominated_below"].values
    # rho_coarse_dominated_below = df_rock_physics["rho_coarse_dominated_below"].values

    func_cf_aboves = []
    # func_cf_belows = []

    rho_tmp = np.logspace(np.log10(rho_min), np.log10(rho_max), 500)
    for ii in range(len(percentile_fines)):
        percentile_fine = percentile_fines[ii]
        percentile_coarse = percentile_coarses[::-1][ii]
        rho_fine_above = np.percentile(rho_fine_dominated_above, percentile_fine)
        rho_coarse_above = np.percentile(rho_coarse_dominated_above, percentile_coarse)
        # rho_fine_below = np.percentile(rho_fine_dominated_below, percentile_fine)
        # rho_coarse_below = np.percentile(rho_coarse_dominated_below, percentile_coarse)
        sigma_above_fine, sigma_above_coarse = (
            1.0 / rho_fine_above,
            1.0 / rho_coarse_above,
        )
        # sigma_below_fine, sigma_below_coarse = (
        #     1.0 / rho_fine_below,
        #     1.0 / rho_coarse_below,
        # )
        f_tmp_above = from_sigma_to_fraction(
            1.0 / rho_tmp, sigma_above_fine, sigma_above_coarse
        )
        # f_tmp_below = from_sigma_to_fraction(
        #     1.0 / rho_tmp, sigma_below_fine, sigma_below_coarse
        # )
        func_cf_above = interp1d(np.log10(rho_tmp), f_tmp_above)
        # func_cf_below = interp1d(np.log10(rho_tmp), f_tmp_below)

        func_cf_aboves.append(func_cf_above)
        # func_cf_belows.append(func_cf_below)

    # return dict(
    #     func_cf_aboves=func_cf_aboves, func_cf_belows=func_cf_belows, rho=rho_tmp
    # )
    return dict(func_cf_aboves=func_cf_aboves, rho=rho_tmp)

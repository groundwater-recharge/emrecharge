import json
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class EMDataset:
    filename: str
    df: pd.DataFrame

    def __init__(
        self, csv_filename: str, csv_filename_thickness: str, spatial_unit="m"
    ):
        self.filename = csv_filename
        self.filename_thickness = csv_filename_thickness
        self.df = pd.read_csv(
            self.filename, dtype={"LINE_NO": "O", "RECORD": "Int32"}
        ).sort_values("RECORD")
        self.df_thickness = pd.read_csv(self.filename_thickness)
        self.spatial_unit = spatial_unit
        if spatial_unit == "ft":
            self.spatial_conversion = 1.0 / 0.3048
        elif spatial_unit == "m":
            self.spatial_conversion = 1.0

    @property
    def header(self):
        return list(self.df.columns)

    @property
    def line(self):
        return self.df["LINE_NO"].values

    @property
    def timestamps(self):
        return self.df["RECORD"].values.to_numpy(dtype=int)

    @property
    def topography(self):
        return (
            self.df[["UTMX", "UTMY", "ELEVATION"]].values[:, :]
            * self.spatial_conversion
        )

    @property
    def depth_of_investigtaion(self):
        return self.df["DOI_STANDARD"].values * self.spatial_conversion

    @property
    def hz(self):
        hz = np.array(json.loads(self.df_thickness.THICKNESS[0]))
        return np.r_[hz, hz[-1]] * self.spatial_conversion

    @property
    def depth(self):
        return np.cumsum(np.r_[0, self.hz])

    @property
    def resistivity(self):
        if getattr(self, "_resistivity", None) is None:
            resistivity = []
            for string in self.df["MEASUREMENTS"]:
                resistivity.append(json.loads(string)["RHO"])
            self._resistivity = np.vstack(resistivity)
        return self._resistivity

    @property
    def minmax(self):
        flat = self.resistivity.flatten()
        return np.min(flat), np.max(flat)

    @property
    def num_soundings(self):
        return self.df.shape[0]

    @property
    def xy(self):
        return self.df[["UTMX", "UTMY"]].values * self.spatial_conversion

    @property
    def lines_xy(self):
        lines = []
        em_lines = dict()
        for g, data in self.df[["LINE_NO", "RECORD", "UTMX", "UTMY"]].groupby(
            "LINE_NO"
        ):
            lines.append(g)
            em_lines[g] = (
                data[["UTMX", "UTMY", "RECORD"]].values * self.spatial_conversion
            )
        return lines, em_lines

    @property
    def num_layers(self):
        return self.hz.size

    def get_resistivity_by_line(self, line_number: int):
        records = self.df[self.df["LINE_NO"] == line_number][["UTMX", "UTMY"]]
        inds_line = self.line == line_number
        xy = self.xy[inds_line, :]
        rho = self.resistivity[inds_line, :]
        doi = self.depth_of_investigtaion[inds_line]
        elevation = self.topography[inds_line, 2]
        start_x = xy[0, 0]
        end_x = xy[-1, 0]
        if start_x > end_x:
            rho = rho[::-1, :]
            xy = xy[::-1, :]

        delta = np.concatenate(
            [[0], (np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2) ** 0.5]
        )
        return rho, delta, xy, elevation, doi

    def get_binned_resistivity_by_line(
        self, line_number: int, n_bins: int, maximum_depth: float
    ):
        rho, delta, xy, elevation, doi = self.get_resistivity_by_line(line_number)
        distance = np.cumsum(delta)
        d_max = distance.max()
        distance_bins = np.linspace(0, d_max, n_bins + 1)
        inds = np.digitize(distance, distance_bins)
        depth_all = np.cumsum(np.r_[0.0, self.hz[:-1]])
        inds_above = depth_all < maximum_depth
        n_layer = inds_above.sum()
        rho_bins = np.zeros((n_bins, n_layer), dtype=float) * np.nan
        doi_bins = np.zeros(n_bins, dtype=float) * np.nan
        elevation_bins = np.zeros(n_bins, dtype=float) * np.nan
        for ii in range(n_bins):
            inds_tmp = inds == ii
            if inds_tmp.sum() > 0:
                rho_bins[ii, :] = rho[inds_tmp, :n_layer].mean(axis=0)
                doi_bins[ii] = doi[inds_tmp].mean(axis=0)
                elevation_bins[ii] = elevation[inds_tmp].mean(axis=0)
        distance_bins_center = (distance_bins[:-1] + distance_bins[1:]) * 0.5
        f_doi = interp1d(
            distance_bins_center[~np.isnan(doi_bins)],
            doi_bins[~np.isnan(doi_bins)],
            fill_value="extrapolate",
        )
        doi_bins_filled = f_doi(distance_bins_center)
        f_elevation = interp1d(
            distance_bins_center[~np.isnan(elevation_bins)],
            elevation_bins[~np.isnan(elevation_bins)],
            fill_value="extrapolate",
        )
        elevation_bins_filled = f_elevation(distance_bins_center)
        depth = np.cumsum(np.r_[0.0, self.hz[:n_layer]])
        return (
            distance,
            distance_bins,
            depth,
            rho_bins,
            elevation_bins_filled,
            doi_bins_filled,
        )

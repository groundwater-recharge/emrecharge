import json
from collections import namedtuple

import numpy as np
import pandas as pd


class EMDataset:
    filename: str
    df: pd.DataFrame

    def __init__(self, csv_filename: str, csv_filename_thickness: str):
        self.filename = csv_filename
        self.filename_thickness = csv_filename_thickness
        self.df = pd.read_csv(
            self.filename, dtype={"LINE_NO": "Int32", "RECORD": "Int32"}
        ).sort_values("RECORD")
        self.df_thickness = pd.read_csv(self.filename_thickness)

    @property
    def header(self):
        return list(self.df.columns)

    @property
    def line(self):
        return self.df["LINE_NO"].values.to_numpy(dtype=int)

    @property
    def timestamps(self):
        return self.df["RECORD"].values.to_numpy(dtype=int)

    @property
    def topography(self):
        return self.df[["UTMX", "UTMY", "ELEVATION"]].values[:, :]

    @property
    def hz(self):
        hz = np.array(json.loads(self.df_thickness.THICKNESS[0]))
        return np.r_[hz, hz[-1]]

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
        return self.df[["UTMX", "UTMY"]].values

    @property
    def lines_xy(self):
        lines = []
        em_lines = dict()
        for g, data in self.df[["LINE_NO", "RECORD", "UTMX", "UTMY"]].groupby(
            "LINE_NO"
        ):
            lines.append(g)
            em_lines[g] = data[["UTMX", "UTMY", "RECORD"]].values
        return lines, em_lines

    @property
    def num_layers(self):
        return self.hz.size

    def get_resistivity_by_line(self, line_number: int):
        records = self.df[self.df["LINE_NO"] == line_number][["UTMX", "UTMY"]]
        inds_line = self.df["LINE_NO"].values.to_numpy(dtype=int) == line_number
        xy = records.values[:, :]
        rho = self.resistivity[inds_line, :]
        delta = np.concatenate(
            [[0], (np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2) ** 0.5]
        )
        return rho, delta, xy

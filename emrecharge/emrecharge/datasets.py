from collections import namedtuple
import numpy as np
import pandas as pd

class EMDataset():
    filename: str;
    df: pd.DataFrame;
    
    def __init__(self, csv_filename: str):
        self.filename = csv_filename;
        self.df = pd.read_csv(self.filename, dtype={'LINE_NO': 'Int32', 'RECORD': 'Int32'}).sort_values('RECORD')
        
    @property
    def header(self):
        return list(self.df.columns)
    
    @property
    def columns(self):
        return namedtuple("columns",["thk","rho"])(
                np.array(self.header)[['THK' in h for h in self.header]],
                np.array(self.header)[['RHO' in h for h in self.header]]
        )
    
    @property
    def names(self):
        return namedtuple("names",["thk","rho"])(
            self.columns.thk[:int(len(self.columns.thk)/2)],
            self.columns.rho[:int(len(self.columns.thk)/2)+1]
        )
    
    @property
    def line(self):
        return self.df['LINE_NO'].values
    
    @property
    def timestamps(self):
        return self.df['TIMESTAMP'].values
    
    @property
    def topography(self):
        return self.df[['UTMX', 'UTMY', 'ELEVATION']].values[:, :]
    
    @property
    def hz(self):
        hz = np.unique(self.df[self.names.thk].values)
        return np.r_[hz, hz[-1]]
    
    @property
    def resistivity(self):
        return self.df[self.names.rho].values[:,:]

    @property
    def xy(self):
        return self.df[['UTMX', 'UTMY']].values

    @property
    def lines_xy(self):
        lines = []
        em_lines = dict()
        for g, data in self.df[['LINE_NO','RECORD','UTMX', 'UTMY']].groupby('LINE_NO'):
            lines.append(g)
            em_lines[g] = data[['UTMX', 'UTMY', 'RECORD']].values
        return lines, em_lines
    
    @property
    def num_layers(self):
        return self.hz.size
    
    def get_resistivity_by_line(self, line_number: int):
        records = self.df[self.df['LINE_NO'] == line_number][['UTMX','UTMY', *self.names.rho]]
        values = records.values[:, :]
        rho = values[:, 2:]
        xy = values[:, :2]
        delta = np.concatenate([[0],(np.diff(values[:, 0])**2 + np.diff(values[:, 1])**2)**.5])
        return rho, delta, xy

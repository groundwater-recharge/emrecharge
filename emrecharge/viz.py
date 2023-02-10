import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import properties
import scipy.sparse as sp
from discretize import TensorMesh
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree as kdtree
from SimPEG import utils


def setup_mpl():
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    mpl.rc("lines", linewidth=3)
    mpl.rc("axes", titlesize=12)
    mpl.rc("xtick", labelsize=10)
    mpl.rc("ytick", labelsize=10)


def set_mesh_1d(hz):
    return TensorMesh([hz], x0=[0])


class Stitched1DModel(properties.HasProperties):
    topography = properties.Array("topography (x, y, z)", dtype=float, shape=("*", "*"))

    physical_property = properties.Array("Physical property", dtype=float)

    line = properties.Array("Line", dtype=float, default=None)

    time_stamp = properties.Array("Time stamp", dtype=float, default=None)

    hz = properties.Array("Vertical thickeness of 1D mesh", dtype=float)

    def __init__(self, **kwargs):
        super(Stitched1DModel, self).__init__(**kwargs)
        warnings.warn("code under construction - API might change in the future")

    @property
    def n_sounding(self):
        if getattr(self, "_n_sounding", None) is None:
            self._n_sounding = self.topography.shape[0]
        return self._n_sounding

    @property
    def unique_line(self):
        if getattr(self, "_unique_line", None) is None:
            if self.line is None:
                raise Exception("line information is required!")
            self._unique_line = np.unique(self.line)
        return self._unique_line

    @property
    def xyz(self):
        if getattr(self, "_xyz", None) is None:
            xyz = np.empty((self.hz.size, self.topography.shape[0], 3), order="F")
            for i_xy in range(self.topography.shape[0]):
                z = -self.mesh_1d.vectorCCx + self.topography[i_xy, 2]
                x = np.ones_like(z) * self.topography[i_xy, 0]
                y = np.ones_like(z) * self.topography[i_xy, 1]
                xyz[:, i_xy, :] = np.c_[x, y, z]
            self._xyz = xyz
        return self._xyz

    @property
    def mesh_1d(self):
        if getattr(self, "_mesh_1d", None) is None:
            if self.hz is None:
                raise Exception("hz information is required!")
            self._mesh_1d = set_mesh_1d(np.r_[self.hz])
        return self._mesh_1d

    @property
    def mesh_3d(self):
        if getattr(self, "_mesh_3d", None) is None:
            print(">> Automatically generating a 3D mesh.")
            self._mesh_3d = self.get_3d_mesh()
        return self._mesh_3d

    @property
    def physical_property_matrix(self):
        if getattr(self, "_physical_property_matrix", None) is None:
            if self.physical_property is None:
                raise Exception("physical_property information is required!")
            self._physical_property_matrix = self.physical_property.reshape(
                (self.hz.size, self.n_sounding), order="F"
            )
        return self._physical_property_matrix

    @property
    def distance(self):
        if getattr(self, "_distance", None) is None:
            self._distance = np.zeros(self.n_sounding, dtype=float)
            for line_tmp in self.unique_line:
                ind_line = self.line == line_tmp
                xy_line = self.topography[ind_line, :2]
                distance_line = np.r_[
                    0, np.cumsum(np.sqrt((np.diff(xy_line, axis=0) ** 2).sum(axis=1)))
                ]
                self._distance[ind_line] = distance_line
        return self._distance

    def plot_section(
        self,
        i_layer=0,
        i_line=0,
        x_axis="x",
        show_layer=False,
        plot_type="contour",
        physical_property=None,
        clim=None,
        ax=None,
        cmap="viridis",
        ncontour=20,
        scale="log",
        show_colorbar=True,
        aspect=1,
        zlim=None,
        dx=20.0,
        invert_xaxis=False,
        alpha=0.7,
        pcolorOpts={},
    ):
        ind_line = self.line == self.unique_line[i_line]
        if physical_property is not None:
            physical_property_matrix = physical_property.reshape(
                (self.hz.size, self.n_sounding), order="F"
            )
        else:
            physical_property_matrix = self.physical_property_matrix

        if x_axis.lower() == "y":
            x_ind = 1
            xlabel = "Northing (m)"
        elif x_axis.lower() == "x":
            x_ind = 0
            xlabel = "Easting (m)"
        elif x_axis.lower() == "distance":
            xlabel = "Distance (m)"

        if ax is None:
            fig = plt.figure(figsize=(15, 10))
            ax = plt.subplot(111)

        if clim is None:
            vmin = np.percentile(physical_property_matrix, 5)
            vmax = np.percentile(physical_property_matrix, 95)
        else:
            vmin, vmax = clim

        if scale == "log":
            norm = LogNorm(vmin=vmin, vmax=vmax)
            vmin = None
            vmax = None
        else:
            norm = None

        ind_line = np.arange(ind_line.size)[ind_line]

        for i in ind_line:
            inds_temp = [i]
            if x_axis == "distance":
                x_tmp = self.distance[i]
            else:
                x_tmp = self.topography[i, x_ind]

            topo_temp = np.c_[x_tmp - dx, x_tmp + dx]
            out = ax.pcolormesh(
                topo_temp,
                -self.mesh_1d.nodes_x + self.topography[i, 2],
                physical_property_matrix[:, inds_temp],
                cmap=cmap,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                shading="auto",
                **pcolorOpts
            )

        if show_layer:
            ax.plot(
                x_tmp,
                self.topography[ind_line, 2] - self.mesh_1d.vectorCCx[i_layer],
                "--",
                lw=1,
                color="grey",
            )

        if show_colorbar:
            from mpl_toolkits import axes_grid1

            cb = plt.colorbar(out, ax=ax, fraction=0.01)
            cb.set_label("Conductivity (S/m)")

        ax.set_aspect(aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Elevation (m)")
        if zlim is not None:
            ax.set_ylim(zlim)

        if x_axis == "distance":
            xlim = (
                self.distance[ind_line].min() - dx,
                self.distance[ind_line].max() + dx,
            )
        else:
            xlim = (
                self.topography[ind_line, x_ind].min() - dx,
                self.topography[ind_line, x_ind].max() + dx,
            )
        if invert_xaxis:
            ax.set_xlim(xlim[1], xlim[0])
        else:
            ax.set_xlim(xlim)

        plt.tight_layout()

        if show_colorbar:
            return out, ax, cb
        else:
            return out, ax
        return ax

    def get_3d_mesh(
        self,
        dx=None,
        dy=None,
        dz=None,
        npad_x=0,
        npad_y=0,
        npad_z=0,
        core_z_length=None,
        nx=100,
        ny=100,
    ):
        xmin, xmax = self.topography[:, 0].min(), self.topography[:, 0].max()
        ymin, ymax = self.topography[:, 1].min(), self.topography[:, 1].max()
        zmin, zmax = self.topography[:, 2].min(), self.topography[:, 2].max()
        zmin -= self.mesh_1d.nodes_x.max()

        lx = xmax - xmin
        ly = ymax - ymin
        lz = zmax - zmin

        if dx is None:
            dx = lx / nx
            print((">> dx:%.1e") % (dx))
        if dy is None:
            dy = ly / ny
            print((">> dy:%.1e") % (dy))
        if dz is None:
            dz = np.median(self.mesh_1d.hx)

        nx = int(np.floor(lx / dx))
        ny = int(np.floor(ly / dy))
        nz = int(np.floor(lz / dz))

        if nx * ny * nz > 1e6:
            warnings.warn(
                ("Size of the mesh (%i) will greater than 1e6") % (nx * ny * nz)
            )
        hx = [(dx, npad_x, -1.2), (dx, nx), (dx, npad_x, -1.2)]
        hy = [(dy, npad_y, -1.2), (dy, ny), (dy, npad_y, -1.2)]
        hz = [(dz, npad_z, -1.2), (dz, nz)]

        zmin = self.topography[:, 2].max() - utils.meshTensor(hz).sum()
        self._mesh_3d = TensorMesh([hx, hy, hz], x0=[xmin, ymin, zmin])

        return self.mesh_3d

    @property
    def P(self):
        if getattr(self, "_P", None) is None:
            raise Exception("Run get_interpolation_matrix first!")
        return self._P

    def get_interpolation_matrix(self, npts=20, epsilon=None):
        tree_2d = kdtree(self.topography[:, :2])
        xy = utils.ndgrid(self.mesh_3d.vectorCCx, self.mesh_3d.vectorCCy)

        distance, inds = tree_2d.query(xy, k=npts)
        if epsilon is None:
            epsilon = np.min([self.mesh_3d.hx.min(), self.mesh_3d.hy.min()])

        w = 1.0 / (distance + epsilon) ** 2
        w = utils.sdiag(1.0 / np.sum(w, axis=1)) * (w)
        I = utils.mkvc(np.arange(inds.shape[0]).reshape([-1, 1]).repeat(npts, axis=1))
        J = utils.mkvc(inds)

        self._P = sp.coo_matrix(
            (utils.mkvc(w), (I, J)), shape=(inds.shape[0], self.topography.shape[0])
        )

        mesh_1d = TensorMesh([np.r_[self.hz[:-1], 1e20]])

        z = self.P * self.topography[:, 2]

        self._actinds = utils.surface2ind_topo(self.mesh_3d, np.c_[xy, z])

        Z = np.empty(self.mesh_3d.vnC, dtype=float, order="F")
        Z = self.mesh_3d.gridCC[:, 2].reshape(
            (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz), order="F"
        )
        ACTIND = self._actinds.reshape(
            (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz), order="F"
        )

        self._Pz = []

        # This part can be cythonized or parallelized
        for i_xy in range(self.mesh_3d.nCx * self.mesh_3d.nCy):
            actind_temp = ACTIND[i_xy, :]
            z_temp = -(Z[i_xy, :] - z[i_xy])
            self._Pz.append(mesh_1d.getInterpolationMat(z_temp[actind_temp]))

    def interpolate_from_1d_to_3d(self, physical_property_1d):
        physical_property_2d = self.P * (
            physical_property_1d.reshape((self.hz.size, self.n_sounding), order="F").T
        )
        physical_property_3d = (
            np.ones(
                (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz),
                order="C",
                dtype=float,
            )
            * np.nan
        )

        ACTIND = self._actinds.reshape(
            (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz), order="F"
        )

        for i_xy in range(self.mesh_3d.nCx * self.mesh_3d.nCy):
            actind_temp = ACTIND[i_xy, :]
            physical_property_3d[i_xy, actind_temp] = (
                self._Pz[i_xy] * physical_property_2d[i_xy, :]
            )

        return physical_property_3d


def generate_average_map(mesh, values, z_top, z_bottom, averaging_method="arithmetic"):
    """
    Calculate vertical average of values in the 3D grid.

    Parameters
    ----------

    mesh: object
        Discretize TensorMesh object
    values: (*,) array_like, float
        3D coarse fraction model
    z_top: float
        top of the integration interval
    z_bottom: float
        bottom of the integration interval
    averaging_method: string
        option for average method (arithematic, harmonic, logarithmic)
    Returns
    -------
    averaged_values : (*,2) ndarray, float
        averaged values.
    """
    nx, ny, nz = mesh.vnC
    # this assumes the uniform cell size
    dz = mesh.hz[0]
    z_values = mesh.cell_centers_z
    tmp_values = values.copy().reshape((nx * ny, nz), order="F")
    inds_tmp = np.logical_and(z_values > z_bottom, z_values < z_top)
    tmp_values[:, ~inds_tmp] = np.nan
    inds_nan = np.isnan(tmp_values)
    tmp_values[inds_nan] = 0.0
    thickness = (~inds_nan).astype(float).sum(axis=1) * dz
    if averaging_method == "arithmetic":
        averaged_values = (tmp_values * dz).sum(axis=1) / thickness
    elif averaging_method == "harmonic":
        averaged_values = 1.0 / ((1.0 / tmp_values * dz).sum(axis=1) / thickness)
    elif averaging_method == "logarithmic":
        averaged_values = np.exp((np.log(tmp_values) * dz).sum(axis=1) / thickness)
    return averaged_values


def plot_layer_model(
    sig, mesh, xscale="log", ax=None, showlayers=False, xlim=None, **kwargs
):
    """
    Plot Conductivity model for the layered earth model
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        z_grid = mesh.nodes_x
        n_sig = sig.size
        sigma = np.repeat(sig, 2)
        z = []
        for i in range(n_sig):
            z.append(np.r_[z_grid[i], z_grid[i + 1]])
        z = np.hstack(z)
        if xlim == None:
            sig_min = sig[~np.isnan(sig)].min() * 0.5
            sig_max = sig[~np.isnan(sig)].max() * 2
        else:
            sig_min, sig_max = xlim

        if xscale == "linear" and sig.min() == 0.0:
            if xlim == None:
                sig_min = -sig[~np.isnan(sig)].max() * 0.5
                sig_max = sig[~np.isnan(sig)].max() * 2

        if ax == None:
            plt.xscale(xscale)
            plt.xlim(sig_min, sig_max)
            plt.ylim(z.min(), z.max())
            plt.xlabel("Conductivity (S/m)", fontsize=14)
            plt.ylabel("Depth (m)", fontsize=14)
            plt.ylabel("Depth (m)", fontsize=14)
            if showlayers == True:
                for locz in z_grid:
                    plt.plot(
                        np.linspace(sig_min, sig_max, 100),
                        np.ones(100) * locz,
                        "b--",
                        lw=0.5,
                    )
            return plt.plot(sigma, z, "k-", **kwargs)

        else:
            ax.set_xscale(xscale)
            ax.set_xlim(sig_min, sig_max)
            ax.set_ylim(z.min(), z.max())
            ax.set_xlabel("Conductivity (S/m)", fontsize=14)
            ax.set_ylabel("Depth (m)", fontsize=14)
            if showlayers == True:
                for locz in z_grid:
                    ax.plot(
                        np.linspace(sig_min, sig_max, 100),
                        np.ones(100) * locz,
                        "b--",
                        lw=0.5,
                    )
            return ax.plot(sigma, z, "k-", **kwargs)


def set_mesh_1d(hz):
    return TensorMesh([hz], x0=[0])

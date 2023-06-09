import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import LineString, Polygon


# Get one of the contours from the plot.
def create_gpd_dataframe_from_contours(
    cs, input_crs, output_crs, spatial_conversion, dx, dy
):
    geometry = []
    index = []
    i_count = 0
    for cnt in cs.collections:
        n = len(cnt.get_paths())
        for ii in range(n):
            vs = cnt.get_paths()[ii].vertices
            vs = np.c_[vs[:, 0] - dx / 2, vs[:, 1] - dy / 2] / spatial_conversion
            polygon_geom = LineString(vs)
            geometry.append(polygon_geom)
            index.append(i_count)
            i_count += 1
    polyline = gpd.GeoDataFrame(index=index, crs=input_crs, geometry=geometry)
    polyline = polyline.to_crs(crs=output_crs)
    return polyline


def export_to_tif(data, dx, dy, xmin, ymin, crs, fname):
    transform = Affine.translation(xmin - dx / 2, ymin - dy / 2) * Affine.scale(dx, dy)
    tmp = data.copy()
    new_dataset = rasterio.open(
        fname,
        "w",
        driver="GTiff",
        height=tmp.shape[0],
        width=tmp.shape[1],
        count=1,
        dtype=tmp.dtype,
        crs=crs,
        transform=transform,
    )
    new_dataset.write(tmp[:, :], 1)
    new_dataset.close()

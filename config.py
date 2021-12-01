try:
    from skimage.graph import route_through_array
    import numpy as np
    import os
    from flusstools import geotools as gt
    from osgeo import gdal
    from osgeo import osr
    from osgeo import ogr
    from nothingbuthefunc import *
    from raster import Raster
    from shape import Shape
    import matplotlib.pyplot as plt
    import pandas as pd
    import logging
    import geopandas
    import glob
    import sys
    from time import perf_counter
    from pathlib import Path
    import rasterio as rio
    from rastertodataframe import raster_to_dataframe
    from scipy.interpolate import RegularGridInterpolator
    # import flusstools as ft



except:
    print("A error with package import occurred")





# #  Classes of the shovel probes with 4 classes using the whole GSD
#
# shovel_probes = ["kb08_4", "kb07_3", "kb07_2", "kb08_2", "kb13_1",
#                  "kb08_1", "kb07_1", "kb07_4", "kb1902_02",
#                  "kb08_3", "kb05_2", "kb05_3", "kb05_4", "kb1902_03", "kb05_1", "kb1902_01",
#                  "sand_1", "sand_2", "sand_0", "sand_4", "sand_5"]
#
# shovel_classes = [2, 2, 2, 1, 1,
#                   1, 3, 3, 3,
#                   1, 3, 2, 2, 3, 3, 3,
#                   4, 4, 4, 4, 4]
#
# labels =["Courser sed.","Medium sed.", "Finer sed.", "Sand"]

# Classes of the shovel probes with 4 classes corrected

shovel_probes = ["kb08_4", "kb07_3", "kb07_2", "kb08_2", "kb13_1",
                 "kb08_1", "kb07_1", "kb07_4", "kb1902_02",
                 "kb08_3", "kb05_2", "kb05_3", "kb05_4", "kb1902_03", "kb05_1", "kb1902_01",
                 "sand_1", "sand_2", "sand_0", "sand_4", "sand_5"]

shovel_classes = [3, 3, 3, 1, 1,
                  1, 2, 2, 2,
                  1, 2, 3, 3, 2, 3, 2,
                  4, 4, 4, 4, 4]

labels =["Cobble1","Cobble2", "Cobble3", "Sand"]

# Classes of the shovel probes with 5 classes

# shovel_probes = ["kb08_4", "kb07_3", "kb07_2", "kb08_2", "kb13_1",
#                  "kb08_1", "kb07_1", "kb07_4", "kb1902_02",
#                  "kb08_3", "kb05_2", "kb05_3",
#                  "kb05_4", "kb1902_03", "kb05_1", "kb1902_01",
#                  "sand_1", "sand_2", "sand_0", "sand_4", "sand_5"]
#
# shovel_classes = [1, 1, 1, 1, 1,
#                   2, 2, 2, 2,
#                   3, 3, 3,
#                   4, 4, 4, 4,
#                   5, 5, 5, 5, 5]

# labels = np.array(["Cobble1", "Cobble2", "Cobble3", "Cobble4", "Sand"])

# find the classes for in and out colmation
in_colmation = False

if in_colmation:
    file_shapes_in = Path(os.path.abspath(os.getcwd()) + "/colmation_shapes/inner_colmation_folder")
    df_in_classes = find_colm_classes_in(filename=file_shapes_in)
    df_in_classes.reset_index(drop=True, inplace=True)
else:
    file_shapes_out = Path(os.path.abspath(os.getcwd()) + "/colmation_shapes/out_colmation_folder")
    df_out_classes = find_colm_classes_out(filename=file_shapes_out)
    df_out_classes.reset_index(drop=True, inplace=True)



# Add random variable to the training set
add_random_predictor = False


print()
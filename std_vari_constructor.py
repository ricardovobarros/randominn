from config import *
import time
from pathlib import Path
from raster import Raster
import logging

tic = time.clock()

np. set_printoptions(threshold=np. inf)

# create a list with all raster files
local_raster_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/shovel/shovel_rasters_test/kb1902")
file_list_raster = find_files(local_raster_add)

# Nodatavalue to serve in the future to mask arrays and optimize computing time
nodatavalue= - 9999

# instantiate the raster object
for i, raster_add in enumerate(file_list_raster):
    # look if the file is a rgb image
    final_name = raster_add.split("\\")[-1].strip(".tif").split("_")[-1]
    if final_name == "rgb":
        # instantiate a raster file as a object
        raster = Raster(raster_add)

        # create raster with light intensity
        red, green, blue = raster.rgb_to_arrays(nodatavalue)
        intensity = compute_intensity(red, green, blue)

        # compute the correspondent "matrix radius" for the radius in the field
        pixel_width = raster.transform[1]
        radius_m = 0.5   # INPUT in meters

        # compute the std/variogram/color value of N pixels around in circular form
        # predictor_array = std_constructor_improv(radius_m, pixel_width, intensity, nodatavalue)
        predictor_array = variogram_constructor(radius_m, pixel_width, intensity,nodatavalue,raster)

        # analise distribution standard deviation
        plot_intensity_std(predictor_array, raster_add.split("\\")[-1])

        # build the name of the raster file
        name_raster = raster_add.split("\\")[-1].strip("rgb.tif") + "vari"

        # burn raster with one band savinf the standard deviation values
        folder = str(Path(local_raster_add))
        raster.burn(predictor_array, folder, "/" + name_raster)

# time lapse
toc = time.clock()
print(toc - tic)














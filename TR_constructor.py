from config import *
from pathlib import Path
from raster import Raster
nodatavalue=- 9999

# folder to save the raster and find
# create a list with all raster files
local_raster_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/colmation/in/kb19"
                                                       ""
                                                       "/rasters")
file_list_raster = find_files(local_raster_add)

for i, raster_add in enumerate(file_list_raster):
    final_name = raster_add.split("\\")[-1].strip(".tif").split("_")[-1]
    if final_name == "ddem":
        # create a object
        raster = Raster(raster_add)

        # remove all band of raster object and transform to array
        elevation_array = raster.band2array(nodatavalue)

        # compute the TRI for each pixel within a predefined radius
        pixel_width = raster.transform[1]
        pixel_radius = 0.4  # radius in meters

        # construct the tri array
        tri_array = tri_constructor(pixel_radius, pixel_width, elevation_array, nodatavalue)

        # build the name of the raster file
        name_raster = raster_add.split("\\")[-1].strip("ddem.tif") + "tri"

        # Burn rater back
        folder = str(Path(local_raster_add))
        raster.burn(tri_array, folder_path=folder, file_name=("/" + name_raster))




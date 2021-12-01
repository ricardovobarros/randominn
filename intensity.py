from config import *
from pathlib import Path
from raster import Raster

raster_add = str(Path(os.path.abspath(os.getcwd()) + "/data/LISD_experiment/rasters/light_1.tif"))
# folder = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse/grave_corse"))
# create a object
raster = Raster(raster_add)
# remove all band of raster object and transform to array
red, green, blue = raster.rgb_to_arrays()
# compute and show and save intensity intensity
intensity = compute_intensity(red, green, blue)
plot_intensity(intensity)

# # Regulate Intensity
# red, green, blue = shift_rgb(red, green, blue, shift=40)
# rgb_array = np.array([red, green, blue])
# rgb_array = set_rgb_boundery(rgb_array)

# Burn rater back

# raster.burn_rgb(rgb_array, folder_path=folder)

# raster_df = raster_to_dataframe(raster_add)


# ______import with rasterrio
# #open raster
# src= rio.open(raster_add)
#
# # read bands
# array = src.read()
# # convert to a DataFrame
# raster_df = pd.DataFrame()
# raster_df['red'] = array[0].ravel()
# raster_df['green'] = array[1].ravel()
# raster_df['blue'] = array[2].ravel()


print()

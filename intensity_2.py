from config import *
from pathlib import Path
from raster import Raster

raster_add_light = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse/kb8_light.tif"))
raster_add_dark = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse/kb08_dark.tif"))
folder = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse"))
# create a object
raster_light = Raster(raster_add_light)
raster_dark = Raster(raster_add_dark)
# remove all band of raster object and transform to array
red_l, green_l, blue_l= raster_light.rgb_to_arrays(np.nan)
red_d, green_d, blue_d = raster_dark.rgb_to_arrays(np.nan)
# compute and show and save intensity intensity
intensity_l = compute_intensity(red_l, green_l, blue_l)
intensity_d = compute_intensity(red_d, green_d, blue_d)

#correct dark intencity
factor = 1
shift = (np.nanmean(intensity_l) - np.nanmean(intensity_d))*factor

# plot the raster in their original light intensity
# plot_intensity(intensity_l)
# plot_intensity(intensity_d)

# Regulate Intensity
red_d, green_d, blue_d = shift_rgb(red_d, green_d, blue_d, shift=shift)
rgb_array = np.array([red_d, green_d, blue_d])
rgb_array = set_rgb_boundery(rgb_array)

# Plot after correction
# plot_intensity(intensity_l)
plot_intensity(compute_intensity(red_d, green_d, blue_d))

# Burn rater back

raster_dark.burn_rgb(rgb_array, folder_path=folder)

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

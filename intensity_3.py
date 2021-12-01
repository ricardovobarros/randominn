from config import *
from pathlib import Path
from raster import Raster

raster_add_coarse = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse/gravel_coarse.tif"))
raster_add_medium = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse/gravel_medium.tif"))
raster_add_fine = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse/gravel_fine.tif"))

folder = str(Path(os.path.abspath(os.getcwd()) + "/data/facies/rasters_diverse"))
# create a object
raster_coarse = Raster(raster_add_coarse)
raster_medium = Raster(raster_add_medium)
raster_fine = Raster(raster_add_fine)

# remove all band of raster object and transform to array
red_c, green_c, blue_c, band4_c = raster_coarse.rgb_to_arrays()
red_d, green_d, blue_d, band4_d = raster_medium.rgb_to_arrays()
red_f, green_f, blue_f, band4_f = raster_fine.rgb_to_arrays()

# compute and show and save intensity intensity
intensity_c = compute_intensity(red_c, green_c, blue_c)
intensity_m = compute_intensity(red_d, green_d, blue_d)
intensity_f = compute_intensity(red_f, green_f, blue_f)

#correct dark intencity
factor = 0.55
shift = (np.nanmean(intensity_c) - np.nanmean(intensity_m)) * factor

plot_intensity(intensity_c)
plot_intensity(intensity_m)
plot_intensity(intensity_f)

# # Regulate Intensity
# red_d, green_d, blue_d = shift_rgb(red_d, green_d, blue_d, shift=shift)
# rgb_array = np.array([red_d, green_d, blue_d])
# rgb_array = set_rgb_boundery(rgb_array)
#
# # Plot after correction
# plot_intensity(intensity_c)
# plot_intensity(compute_intensity(red_d, green_d, blue_d))

# Burn rater back

# raster_medium.burn_rgb(rgb_array, folder_path=folder)

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


print(np.nanstd(intensity_c))
print(np.nanstd(intensity_m))
print(np.nanstd(intensity_f))

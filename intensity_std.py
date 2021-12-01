from config import *
import time

tic = time.clock()
nodatavalue = -9999

# instantiate the raster object
local_raster_add = Path(os.path.abspath(os.getcwd()) +
                        "/data/facies/colmation/out/kb19/rasters")

file_list_raster = find_files(local_raster_add)
for i, raster_add in enumerate(file_list_raster):
    final_name = raster_add.split("\\")[-1].strip(".tif").split("_")[-1]
    if final_name == "rgb":
        raster = Raster(raster_add)

        # create raster with light intensity
        red, green, blue = raster.rgb_to_arrays(nodatavalue)
        intensity = compute_intensity(red, green, blue)

        # compute the correspondent "matrix radius" for the radius in the field
        pixel_width = raster.transform[1]
        radius_field = 0.3  # INPUT
        radius_matrix = radius_field / pixel_width

        # compute the std value of N pixels around in circular form
        # std_array = std_constructor(radius=radius_matrix, array=intensity)
        std_array = tri_constructor(radius_field, pixel_width, intensity, nodatavalue)

        # analise distribution standard deviation
        # plot_intensity_std(std_array, radius_field)

        # build the name of the raster file
        name_raster = raster_add.split("\\")[-1].strip("rgb.tif") + "lisd"

        # Burn rater back
        folder = str(Path(local_raster_add))
        raster.burn(std_array, folder_path=folder, file_name=("/" + name_raster))

# time lapse
toc = time.clock()
print(toc - tic)


def std_constructor(radius, array):
    """ Captures the neighbours and their memberships
    :param array: array A or B
    :param x: int, cell in x
    :param y: int, cell in y
    :return: np.array (float) membership of the neighbours (without mask), np.array (float) neighbours' cells (without mask)
    """
    std_array = np.empty((len(array), len(array[0])))

    std_array[:] = np.nan
    i = 0
    for y in array:
        j = 0

        for x in y:
            if ~np.isnan(array[i, j]):
                i1 = 0
                neigh_vector = []
                for k in array:
                    j1 = 0
                    for z in k:
                        distance = ((i - i1) ** 2 + (j - j1) ** 2) ** 0.5
                        if distance <= radius:
                            neigh_vector.append(array[i1, j1])
                        j1 += 1
                    i1 += 1
                std_array[i, j] = np.nanstd(neigh_vector)
            if j == len(array[0]) - 1:
                j = 0
            else:
                j += 1
        i += 1

    return std_array

"""
Nothing but variate many functions
"""
# from config import *
import matplotlib.pyplot as plt
from pathlib import Path
import numpy.ma as ma
import math
import pandas as pd
import skgstat as skg
import numpy as np
import glob
import geopandas
import os
from skimage.graph import route_through_array
from sklearn.model_selection import StratifiedKFold
import itertools
import collections
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score, make_scorer
from numpy import savetxt
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_path_array(raster_array, geo_transform, start_coord, stop_coord):
    # transform coordinates to array index
    start_index_x, start_index_y = coords2offset(geo_transform, start_coord[0], start_coord[1])
    stop_index_x, stop_index_y = coords2offset(geo_transform, stop_coord[0], stop_coord[1])

    # replace np.nan with max raised by an order of magnitude to exclude pixels from least cost
    raster_array[np.isnan(raster_array)] = np.nanmax(raster_array) * 10

    # create path and costs
    index_path, cost = route_through_array(raster_array, (start_index_y, start_index_x),
                                           (stop_index_y, stop_index_x),
                                           geometric=False, fully_connected=False)

    index_path = np.array(index_path).T
    path_array = np.zeros_like(raster_array)
    path_array[index_path[0], index_path[1]] = 1
    return path_array


def raster2line(raster_file_name, out_shp_fn, pixel_value):
    """
    Convert a raster to a line shapefile, where pixel_value determines line start and end points
    :param raster_file_name: STR of input raster file name, including directory; must end on ".tif"
    :param out_shp_fn: STR of target shapefile name, including directory; must end on ".shp"
    :param pixel_value: INT/FLOAT of a pixel value
    :return: None (writes new shapefile).
    """

    # calculate max. distance between points
    # ensures correct neighbourhoods for start and end pts of lines
    raster, array, geo_transform = gt.raster2array(raster_file_name)
    pixel_width = geo_transform[1]
    # max_distance = np.ceil(np.sqrt(2 * pixel_width**2))

    # _______ Ricardos change
    max_distance = pixel_width
    sum = np.sum(array)
    # _______

    # extract pixels with the user-defined pixel value from the raster array
    trajectory = np.where(array == pixel_value)
    if np.count_nonzero(trajectory) is 0:
        print("ERROR: The defined pixel_value (%s) does not occur in the raster band." % str(pixel_value))
        return None

    # convert pixel offset to coordinates and append to nested list of points
    points = []
    count = 0
    for offset_y in trajectory[0]:
        offset_x = trajectory[1][count]
        points.append(gt.offset2coords(geo_transform, offset_x, offset_y))
        count += 1

    # create multiline (write points dictionary to line geometry (wkbMultiLineString)
    multi_line = ogr.Geometry(ogr.wkbMultiLineString)
    for i in gt.itertools.combinations(points, 2):
        point1 = ogr.Geometry(ogr.wkbPoint)
        point1.AddPoint(i[0][0], i[0][1])
        point2 = ogr.Geometry(ogr.wkbPoint)
        point2.AddPoint(i[1][0], i[1][1])

        distance = point1.Distance(point2)
        if distance <= max_distance:
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(i[0][0], i[0][1])
            line.AddPoint(i[1][0], i[1][1])
            multi_line.AddGeometry(line)

    # write multiline (wkbMultiLineString2shp) to shapefile
    new_shp = gt.create_shp(out_shp_fn, layer_name="raster_pts", layer_type="line")
    lyr = new_shp.GetLayer()
    feature_def = lyr.GetLayerDefn()
    new_line_feat = ogr.Feature(feature_def)
    new_line_feat.SetGeometry(multi_line)
    lyr.CreateFeature(new_line_feat)

    # create projection file
    srs = gt.get_srs(raster)
    gt.make_prj(out_shp_fn, int(srs.GetAuthorityCode(None)))
    print("Success: Wrote %s" % str(out_shp_fn))


def compute_intensity(r, g, b):
    return 1 / 3 * (r + g + b)


def plot_intensity(array):
    # reshape array to work in the histogram
    array = array.reshape(array.size, 1)
    fig = plt.figure(figsize=(6.18, 3.82), dpi=150, facecolor="w", edgecolor="gray")
    axes = fig.add_subplot(1, 1, 1)
    axes.hist(array, 20)

    # text
    plt.ylim([0, 25])
    plt.xlim([160, 255])
    plt.text(180, 21, "LI mean = " + str(truncate(np.nanmean(array), 2)))
    plt.text(180, 19, "LI std = " + str(truncate(np.nanstd(array), 2)))
    plt.xlabel('LI [--]',size=12)
    plt.ylabel('number of pixels')
    plt.grid(axis="y")

    # text
    # plt.xlabel('intensity value')
    # plt.title('Histogram of Intensity $\mu=$' + str(truncate(np.nanmean(array), 2)) + '$,\ \sigma=$' + str(
    #     truncate(np.nanstd(array), 2)))
    # plt.text(100, 300,
    #          '$\mu=$' + str(truncate(np.nanmean(array), 2)) + '$,\ \sigma=$' + str(truncate(np.nanstd(array), 2)))
    # </editor-fold>
    # show graph
    plt.show()

    # save figure
    # plt.savefig(str(Path("geodata_example/plots/{} from {}")).format(type_, title))


def plot_distribution(array, label, riverbank, predictor):
    # reshape array to work in the histogram
    array = array.reshape(array.size, 1)
    fig = plt.figure(figsize=(6.18, 3.82), dpi=150, facecolor="w", edgecolor="gray")
    axes = fig.add_subplot(1, 1, 1)
    axes.hist(array, 20)

    # text
    plt.xlabel('predictor')
    plt.ylabel('number of pixels')
    plt.title('Riverbank ' + str(riverbank) + ' / label ' + str(label) + " / predictor " + str(predictor))

    # plt.text(100, 300,
    #          '$\mu=$' + str(truncate(np.nanmean(array), 2)) + '$,\ \sigma=$' + str(truncate(np.nanstd(array), 2)))
    # </editor-fold>
    # show graph
    plt.show()

    # save figure
    # plt.savefig(str(Path("geodata_example/plots/{} from {}")).format(type_, title))


def plot_intensity_std(array, radius):
    # reshape array to work in the histogram
    array = array.reshape(array.size, 1)
    fig = plt.figure(figsize=(6.18, 3.82), dpi=150, facecolor="w", edgecolor="gray")
    axes = fig.add_subplot(1, 1, 1)
    axes.hist(array, 20)

    # text
    # plt.ylim([0, 30])
    # plt.xlim([9, 13])
    plt.text(11.5, 25.2, "LISD mean = " + str(truncate(np.nanmean(array), 2)))
    plt.text(11.5, 23, "LISD std = " + str(truncate(np.nanstd(array), 2)))
    plt.xlabel('LISD [--]')
    plt.ylabel('number of pixels')
    plt.grid(axis="y")

    # plt.title(
    #     'Histogram of Intensity_std $\mu=$' + str(truncate(np.nanmean(array), 2)) + '$,\ \sigma of \sigma = $' + str(
    #         truncate(np.nanstd(array), 2)))
    # plt.text(100, 300,
    #          '$\mu=$' + str(truncate(np.nanmean(array), 2)) + '$,\ \sigma=$' + str(truncate(np.nanstd(array), 2)))
    # </editor-fold>
    # show graph
    # plt.savefig(str(Path("std_distributions/vari_fine_lm.png")))
    plt.show()

    # save figure
    # plt.savefig(str(Path("geodata_example/plots/{} from {}")).format(type_, title))


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


def shift_rgb(red, green, blue, shift):
    red += shift
    green += shift
    blue += shift
    return red, green, blue


def set_rgb_boundery(rgb_array):
    rgb_array = np.where(rgb_array > 255, 255, rgb_array)
    rgb_array = np.where(rgb_array < 0, 0, rgb_array)
    return rgb_array


def std_constructor_improv(radius_m, cellsize, array, nodatavalue):
    """ Captures the neighbours and their memberships
    :param radius_m: float, radius in meters to consider
    :param array: numpy array
    :return: np.array (float) membership of the neighbours (without mask), np.array (float) neighbours' cells (without mask)
    """
    # Calcultes the number of cells correspoding to the given radius in meters
    radius = int(np.around(radius_m / cellsize, 0))

    array = ma.masked_where(array == nodatavalue, array, copy=True)

    # Creates number array with same shape as the input array with filling values as the given nodatavalue
    std_array = np.full(np.shape(array), np.nan, dtype=np.float)
    # Loops through the numpy array
    for index, central in np.ndenumerate(array):
        if not array.mask[index]:
            # Determines the cells that are within the window given by the radius (in cells)
            x_up = max(index[0] - radius, 0)
            x_lower = min(index[0] + radius + 1, array.shape[0])
            y_up = max(index[1] - radius, 0)
            y_lower = min(index[1] + radius + 1, array.shape[1])

            # neighborhood array (window)
            neigh_array = array[x_up: x_lower, y_up: y_lower]

            # Distance (in cells) of all neighbours to the cell in x,y in analysis
            i, j = np.indices(neigh_array.shape)
            i = i.flatten() - (index[0] - x_up)
            j = j.flatten() - (index[1] - y_up)
            d = np.reshape((i ** 2 + j ** 2) ** 0.5, neigh_array.shape)

            # Unraveling of arrays as the order doesnt matter
            d = np.ravel(d)
            neigh_array = np.ravel(neigh_array)
            neigh_array_filtered = neigh_array[d <= radius]

            # Test to check if results are correct
            # np.savetxt('neigharrayignore.csv', neigh_array_filtered, delimiter=',')

            std_array[index] = np.nanstd(neigh_array_filtered)
    # std_array = ma.masked_equal(std_array, np.nan)
    # std_return = ma.masked_where(array == nodatavalue, std_array, copy=True)
    return std_array


def variogram_constructor(radius_m, cellsize, array, nodatavalue, raster):
    """ Captures the neighbours and their memberships
        :param raster: instantiated object raster
        :param nodatavalue: 9999
        :param cellsize: pixel size
        :param radius_m: float, radius in meters to consider
        :param array: numpy array
        :return: np.array (float) membership of the neighbours (without mask), np.array (float) neighbours' cells (without mask)
        """
    # Calculate the number of cells correspoding to the given radius in meters
    radius = int(np.around(radius_m / cellsize, 0))
    array = ma.masked_where(array == nodatavalue, array, copy=True)

    # get array coordinates
    intensity_df = raster.coord_dataframe(array.reshape(array.size, 1))

    # turn the frame in a three dimensional array
    position_x = intensity_df["x"].to_numpy().reshape(array.shape[0], array.shape[1])
    position_y = intensity_df["y"].to_numpy().reshape(array.shape[0], array.shape[1])

    # Creates number array with same shape as the input array with filling values as the given nodatavalue
    v_array = np.full(np.shape(array), np.nan, dtype=np.float)

    # Loops through the numpy array
    for index, central in np.ndenumerate(array):
        if not array.mask[index]:
            # Determines the cells that are within the window given by the radius (in cells)
            x_up = max(index[0] - radius, 0)
            x_lower = min(index[0] + radius + 1, array.shape[0])
            y_up = max(index[1] - radius, 0)
            y_lower = min(index[1] + radius + 1, array.shape[1])

            # neighborhood array (window)
            neigh_array = array[x_up: x_lower, y_up: y_lower]
            neigh_array_x = position_x[x_up: x_lower, y_up: y_lower]
            neigh_array_y = position_y[x_up: x_lower, y_up: y_lower]

            # Distance (in cells) of all neighbours to the cell in x,y in analysis
            i, j = np.indices(neigh_array.shape)
            i = i.flatten() - (index[0] - x_up)
            j = j.flatten() - (index[1] - y_up)
            d = np.reshape((i ** 2 + j ** 2) ** 0.5, neigh_array.shape)

            # Unraveling of arrays as the order doesnt matter
            d = np.ravel(d)
            neigh_array = np.ravel(neigh_array)
            neigh_array_x = np.ravel(neigh_array_x)
            neigh_array_y = np.ravel(neigh_array_y)
            neigh_array = neigh_array[d <= radius]
            neigh_array_x = neigh_array_x[d <= radius]
            neigh_array_y = neigh_array_y[d <= radius]

            # create dataframe of local variogram
            v_df = pd.DataFrame({"x": neigh_array_x, "y": neigh_array_y, "int": neigh_array})
            v_df = v_df.dropna(how='any', axis=0)

            # Instantiate variogram
            v_obj = skg.Variogram(np.array(v_df[["x", "y"]]),
                                  np.array(v_df[["int"]]).reshape(len(np.array(v_df[["int"]])))
                                  , fit_method='trf', model='exponential')  # reshape the array from (R,1) to (R,)

            # return lag values
            v_lag_value = v_obj.data(n=2)[1][1]  # return the last

            # plot semivariogram
            # v_obj.plot()

            # fill v_array fit variogram values for the defines lag(radius)
            v_array[index] = v_lag_value
            del v_obj
    # plot_intensity_std(v_array, radius_m)
    return v_array


def tri_constructor(radius_m, cellsize, array, nodatavalue):
    """
    :param cellsize: pixel width of the raster object
    :param radius_m: radius around pixel to be taken into account o compute TRI
    :param nodatavalue: value to be masked when printing raster
    :param raster: instatiated object which represents the raster
    :return: array with the TRI value for each pixel
    """
    # Calculate the number of cells corresponding to the given radius in meters
    radius = int(np.around(radius_m / cellsize, 0))
    array = ma.masked_where(array == nodatavalue, array, copy=True)

    # Creates number array with same shape as the input array with filling values as the given nodatavalue
    tri_array = np.full(np.shape(array), np.nan, dtype=np.float)

    # Loops through the numpy array
    for index, central in np.ndenumerate(array):
        if not array.mask[index]:
            # Determines the cells that are within the window given by the radius (in cells)
            x_up = max(index[0] - radius, 0)
            x_lower = min(index[0] + radius + 1, array.shape[0])
            y_up = max(index[1] - radius, 0)
            y_lower = min(index[1] + radius + 1, array.shape[1])

            # neighborhood array (window)
            neigh_array = array[x_up: x_lower, y_up: y_lower]

            # Distance (in cells) of all neighbours to the cell in x,y in analysis
            i, j = np.indices(neigh_array.shape)
            i = i.flatten() - (index[0] - x_up)
            j = j.flatten() - (index[1] - y_up)
            d = np.reshape((i ** 2 + j ** 2) ** 0.5, neigh_array.shape)

            # Unraveling of arrays as the order doesnt matter
            d = np.ravel(d)
            neigh_array = np.ravel(neigh_array)
            neigh_array = neigh_array[d <= radius]

            # Compute TRI for one pixel
            tri_pixel = np.nanstd(neigh_array)

            # fill tri_array fit variogram values for the defines lag(radius)
            tri_array[index] = tri_pixel

    return tri_array


def find_files(directory=None):
    """It finds all the .tif or .shp files inside a folder and
     create list of strings with their raw names

    :param directory: string of directory's address
    :return: list of strings from addresses of all files inside the directory
    """
    # Set up variables
    is_raster = False
    is_shape = False
    # raster_folder, shape_folder = verify_folders(directory)

    # terminate the code if there is no directory address
    if directory is None:
        print("Any directory was given")
        sys.exit()

    # Append / or / in director name if it does not have
    if not str(directory).endswith("/") and not str(directory).endswith("\\"):
        directory = Path(str(directory) + "/")

    # Find out if there is shape or raster file inside the folder
    try:
        for file_name in os.listdir(directory):
            if file_name.endswith('.tif'):
                is_raster = True
                break
            if file_name.endswith('.shp'):
                is_shape = True
                break
    except:
        print("Input directory {} was not found".format(directory))

    # Create a list of shape files or raster files names
    if is_shape:
        file_list = glob.glob(str(directory) + "/*.shp")
    elif is_raster:
        file_list = glob.glob(str(directory) + "/*.tif")
    else:
        print("There is no valid file inside the folder {}".format(directory))
        exit()
    return file_list


def find_probes_info(file_list):
    numbers_list = []
    predictors_names = []
    probes_names = []
    for i, file_raster in enumerate(file_list):
        numbers_list.append(int(file_raster.split(str(Path("/")))[-1].split("_")[1]))
        predictors_names.append(file_raster.split(str(Path("/")))[-1].split("_")[2].split(".")[0])
        probes_names.append(file_raster.split(str(Path("/")))[-1].split("_")[0] + "_" +
                            file_raster.split(str(Path("/")))[-1].split("_")[1])
    # # remove repeted string in the list
    # seen = set()
    # result = []
    # for item in probes_names:
    #     if item not in seen:
    #         seen.add(item)
    #         result.append(item)
    # arrange return values
    # probes_names = result
    probes_list = [numbers_list[0]]
    for i, p in enumerate(numbers_list): probes_list.append(p) if (i > 1 and p != numbers_list[i - 1]) else None
    probes_number = len(probes_list)
    predictors_number = int(len(numbers_list) / probes_number)
    predictors_names = predictors_names[0:predictors_number]
    return probes_number, predictors_number, predictors_names, probes_names


def sample_band(band_array, raster, radius):
    """

    :param band_array: array to be reduced
    :param raster: object raster to get pixel size
    :param radius: real radius in meters to be sampled
    :return:
    """
    # compute matrix distance
    pixel_width = raster.transform[1]
    matrix_radius = int(radius / pixel_width)
    index = (int(band_array.shape[0] / 2), int(band_array.shape[1] / 2))
    # find index of the m
    # Determines the cells that are within the window given by the radius (in cells)
    x_up = max(index[0] - matrix_radius, 0)
    x_lower = min(index[0] + matrix_radius + 1, band_array.shape[0])
    y_up = max(index[1] - matrix_radius, 0)
    y_lower = min(index[1] + matrix_radius + 1, band_array.shape[1])

    # neighborhood array (window)
    band_array = band_array[x_up: x_lower, y_up: y_lower]

    return band_array


def find_colm_classes_in(filename=""):
    file_list_raster = find_files(filename)
    df_shapes = pd.DataFrame()
    for shape in file_list_raster:
        df_shape_temp = geopandas.read_file(shape)
        if "Innere_Kol" in df_shape_temp.columns:
            df_shape_temp.rename(columns={'Innere_Kol': 'Innere_K_1'}, inplace=True)  # correct QGIS columns names

        # create a colunm with the name of the bank an ID
        bank_name = shape.split("\\")[-1].split("_")[0]
        df_shape_temp["name"] = bank_name + "_" + df_shape_temp["ID"].astype(str)
        df_shape_temp.drop(df_shape_temp.columns.difference(['name', 'Innere_K_1']), 1, inplace=True)

        df_shapes = df_shapes.append(df_shape_temp, ignore_index=True)
    return df_shapes


def find_colm_classes_out(filename=""):
    file_list_raster = find_files(filename)
    df_shapes = pd.DataFrame()
    for shape in file_list_raster:
        df_shape_temp = geopandas.read_file(shape)
        if "Stufe_AK" in df_shape_temp.columns:
            df_shape_temp.rename(columns={'Stufe_AK': 'AK'}, inplace=True)  # correct QGIS columns names
        if "STUFE_AK" in df_shape_temp.columns:
            df_shape_temp.rename(columns={'STUFE_AK': 'AK'}, inplace=True)  # correct QGIS columns names

        # create a colunm with the name of the bank an ID
        bank_name = shape.split("\\")[-1].split("_")[0]
        df_shape_temp["name"] = bank_name + "_" + df_shape_temp["i"].astype(str)
        df_shape_temp.drop(df_shape_temp.columns.difference(['name', 'AK']), 1, inplace=True)

        df_shapes = df_shapes.append(df_shape_temp, ignore_index=True)

    # Get names of indexes

    indexes = df_shapes.index[df_shapes['AK'] == "Flimz"].tolist()
    indexes.append(df_shapes.index[df_shapes['AK'] == "Bewuchs"].tolist()[0])

    # Delete these row indexes from dataFrame
    df_shapes.drop(indexes, inplace=True)
    return df_shapes


def exclude_bank(df=None, kiesbank=None, status=None):
    if df is None or kiesbank is None or status is None:
        print("Give the name of the bank correctly")
        quit()
    if kiesbank == "kb05" and status is "in":
        df = df[(df["name"] == "kb05_10") | (df["name"] == "kb05_11") |
                (df["name"] == "kb05_12") | (df["name"] == "kb05_13") |
                (df["name"] == "kb05_14") | (df["name"] == "kb05_15") |
                (df["name"] == "kb05_16") | (df["name"] == "kb05_17") |
                (df["name"] == "kb05_18") | (df["name"] == "kb05_19") |
                (df["name"] == "kb05_1") | (df["name"] == "kb05_20") |
                (df["name"] == "kb05_4") | (df["name"] == "kb05_5") |
                (df["name"] == "kb05_6") | (df["name"] == "kb05_7") |
                (df["name"] == "kb05_8") | (df["name"] == "kb05_9")
                ]
    if kiesbank == "kb05" and status is "out":
        df = df[(df["name"] != "kb05_10") & (df["name"] != "kb05_11") &
                (df["name"] != "kb05_12") & (df["name"] != "kb05_13") &
                (df["name"] != "kb05_14") & (df["name"] != "kb05_15") &
                (df["name"] != "kb05_16") & (df["name"] != "kb05_17") &
                (df["name"] != "kb05_18") & (df["name"] != "kb05_19") &
                (df["name"] != "kb05_1") & (df["name"] != "kb05_20") &
                (df["name"] != "kb05_4") & (df["name"] != "kb05_5") &
                (df["name"] != "kb05_6") & (df["name"] != "kb05_7") &
                (df["name"] != "kb05_3") & (df["name"] != "kb05_21") &
                (df["name"] != "kb05_8") & (df["name"] != "kb05_9")
                ]
    # kb07
    if kiesbank is "kb07" and status is "out":
        df = df[~df["name"].str.contains("kb07")]
    if kiesbank is "kb07" and status is "in":
        df = df[df["name"].str.contains("kb07")]
    # kb08
    if kiesbank is "kb08" and status is "out":
        df = df[~df["name"].str.contains("kb08")]
    if kiesbank is "kb08" and status is "in":
        df = df[df["name"].str.contains("kb08")]

    # kb13
    if kiesbank is "kb13" and status is "out":
        df = df[~df["name"].str.contains("kb13")]
    if kiesbank is "kb13" and status is "in":
        df = df[df["name"].str.contains("kb13")]

    # kb19
    if kiesbank is "kb19" and status is "out":
        df = df[~df["name"].str.contains("kb19")]
    if kiesbank is "kb19" and status is "in":
        df = df[df["name"].str.contains("kb19")]

    return df


def split_test_samples(df=None, split=None, list_drop=None):
    # take randomly a percentage of the samples
    df_save = df
    num_samples = len(df["name"].unique())
    num_samples_test = int(num_samples * split)
    samples = df["name"].unique().to_numpy().tolist()
    samples_test = []
    for i in range(1, num_samples_test + 1):
        samples_length = len(samples)
        rand_num = np.random.randint(0, samples_length)
        randsample = samples[rand_num]
        df = df[df["name"] != randsample]
        samples.remove(randsample)

        samples_test.append(randsample)

    # create test set
    X = pd.DataFrame()
    for test_sample in samples_test:
        X = X.append(df_save[df_save["name"] == test_sample])

    # # Split training set in a another way
    y_test = X["class"].values
    X.drop(list_drop, axis=1, inplace=True)
    columns = X.columns
    X_test = X[columns].values

    return df, X_test, y_test


def split_test_samples_force(df=None, split=None, list_drop=None):
    # find the number of test samples given a split percentage
    df_save = df  # save the original dataframe
    num_samples = len(df["name"].unique())
    num_samples_test = int(num_samples * split)

    # list all classes
    labels_length = len(df["class"].unique())

    # loop to create a test set that has all classes
    test_classes_length = 0
    while labels_length != test_classes_length:
        samples_test = []
        samples = df_save["name"].unique().to_numpy().tolist()
        df_loop = df_save
        for i in range(1, num_samples_test + 1):
            samples_length = len(samples)
            rand_num = np.random.randint(0, samples_length)
            randsample = samples[rand_num]
            df_loop = df_loop[df_loop["name"] != randsample]
            samples.remove(randsample)
            samples_test.append(randsample)
        # find out how many classes are inside the test set
        df_test = df_save[df_save["name"].isin(samples_test)]
        test_classes_length = len(df_test["class"].unique())
        # remove test samples from training set
    for i, lala in enumerate(samples_test):
        df = df[df.name != samples_test[i]]

    # create test set
    X = pd.DataFrame()
    for test_sample in samples_test:
        X = X.append(df_save[df_save["name"] == test_sample])

    # # Split training set in a another way
    y_test = X["class"].values
    X.drop(list_drop, axis=1, inplace=True)
    columns = X.columns
    X_test = X[columns].values

    return df, X_test, y_test


def correct_labels_incolmation(df=None, n_classes=None):
    # Remove velocity failure pixels
    df = df[df["velo"] != -9999]

    # Choose the number of classes between 2,5 or 8
    if df is None or n_classes is None:
        print("dataframe empty")
        quit()
    if n_classes == 5:
        df.loc[df["class"] == 1.5, "class"] = 1
        df.loc[df["class"] == 1.75, "class"] = 2
        df.loc[df["class"] == 2.25, "class"] = 2
        df.loc[df["class"] == 2.5, "class"] = 2
        df.loc[df["class"] == 2.75, "class"] = 3
        df.loc[df["class"] == 3.25, "class"] = 3
        df.loc[df["class"] == 3.5, "class"] = 3
        df.loc[df["class"] == 3.75, "class"] = 4
        df.loc[df["class"] == 4.25, "class"] = 4
        df.loc[df["class"] == 4.5, "class"] = 5
        df.loc[df["class"] == 4.75, "class"] = 5
    elif n_classes == 3:
        df.loc[df["class"] == 1.5, "class"] = 2
        df.loc[df["class"] == 1.75, "class"] = 2
        df.loc[df["class"] == 2.25, "class"] = 2
        df.loc[df["class"] == 2.5, "class"] = 2
        df.loc[df["class"] == 2.75, "class"] = 3
        df.loc[df["class"] == 3.25, "class"] = 3
        df.loc[df["class"] == 3.5, "class"] = 3
        df.loc[df["class"] == 3.75, "class"] = 4
        df.loc[df["class"] == 4.25, "class"] = 4
        df.loc[df["class"] == 4.5, "class"] = 4
        df.loc[df["class"] == 4.75, "class"] = 4
        df.loc[df["class"] == 5, "class"] = 4

    elif n_classes == 4:
        df.loc[df["class"] == 1.5, "class"] = 2
        df.loc[df["class"] == 1.75, "class"] = 2
        df.loc[df["class"] == 2.25, "class"] = 2
        df.loc[df["class"] == 2.5, "class"] = 3
        df.loc[df["class"] == 2.75, "class"] = 3
        df.loc[df["class"] == 3.25, "class"] = 3
        df.loc[df["class"] == 3.5, "class"] = 4
        df.loc[df["class"] == 3.75, "class"] = 4
        df.loc[df["class"] == 4.25, "class"] = 4
        df.loc[df["class"] == 4.5, "class"] = 5
        df.loc[df["class"] == 4.75, "class"] = 5
        df.loc[df["class"] == 5, "class"] = 5
    else:
        df.loc[df["class"] == 1.75, "class"] = 1.5
        df.loc[df["class"] == 2.25, "class"] = 2
        df.loc[df["class"] == 2.75, "class"] = 2.5
        df.loc[df["class"] == 3.25, "class"] = 3
        df.loc[df["class"] == 3.75, "class"] = 3.5
        df.loc[df["class"] == 4.25, "class"] = 4
        df.loc[df["class"] == 4.75, "class"] = 4.5

    return df


def correct_labels_outcolmation(df=None, modus_classes=None):
    # Remove velocity failure pixels
    df = df[df["velo"] != -9999]

    # Choose the number of classes between 2,5 or 8
    if df is None or modus_classes is None:
        print("dataframe empty")
        quit()

    #  push the classification to the extreme
    if modus_classes == 1:
        df.loc[df["class"] == 1.5, "class"] = 1
        df.loc[df["class"] == 2.5, "class"] = 3
    #  push the classification up
    if modus_classes == 2:
        df.loc[df["class"] == 1.5, "class"] = 2
        df.loc[df["class"] == 2.5, "class"] = 3

    #  push the classification down
    if modus_classes == 3:
        df.loc[df["class"] == 1.5, "class"] = 1
        df.loc[df["class"] == 2.5, "class"] = 2
    #  push the classification to the middle
    if modus_classes == 4:
        df.loc[df["class"] == 1.5, "class"] = 2
        df.loc[df["class"] == 2.5, "class"] = 2
    return df


def split_test_samples_banktopredict(df, riverbank, list_drop):
    # take randomly a percentage of the samples
    df = df[df["velo"] != -9999]  # correct velocities outliers
    df_save = df  # save the original dataframe

    # separate all the values that contains a specific kb
    df_test = df[df['name'].str.match(riverbank)]
    df_train = df[~df['name'].str.match(riverbank)]

    # Save test parameter and classes of test set in vectors
    y_test = df_test["class"].values
    df_test.drop(list_drop, axis=1, inplace=True)
    columns = df_test.columns
    X_test = df_test[columns].values

    # num_samples = len(df["name"].unique())
    # num_samples_test = int(num_samples * split)
    #
    # # list all classes
    # labels_length = len(df["class"].unique())
    #
    # # loop to create a test set that has all classes
    # test_classes_length = 0
    # while labels_length != test_classes_length:
    #     samples_test = []
    #     samples = df_save["name"].unique().to_numpy().tolist()
    #     df_loop = df_save
    #     for i in range(1, num_samples_test + 1):
    #         samples_length = len(samples)
    #         rand_num = np.random.randint(0, samples_length)
    #         randsample = samples[rand_num]
    #         df_loop = df_loop[df_loop["name"] != randsample]
    #         samples.remove(randsample)
    #         samples_test.append(randsample)
    #     # find out how many classes are inside the test set
    #     df_test = df_save[df_save["name"].isin(samples_test)]
    #     test_classes_length = len(df_test["class"].unique())
    #
    # # create test set
    # X = pd.DataFrame()
    # for test_sample in samples_test:
    #     X = X.append(df_save[df_save["name"] == test_sample])
    #
    # # # Split training set in a another way
    # y_test = X["class"].values
    # X.drop(list_drop, axis=1, inplace=True)
    # columns = X.columns
    # X_test = X[columns].values

    return df_train, X_test, y_test


def create_folders(n_folder, dataset):
    labeled_samples = dataset[["name", "class"]].groupby("name").mean().reset_index()
    folder_size = int(len(labeled_samples) / n_folder)  # minimum number of samples per folder
    samples_surplus = len(labeled_samples) % n_folder  # rest of exact division of folders
    X = labeled_samples.drop(["class"], axis=1)
    y = labeled_samples["class"]

    skf = StratifiedKFold(n_splits=n_folder)
    skf.split(X, y)
    test_set = []
    test_set_temp = []
    train_set = []
    train_set_temp = []
    i = 0
    for train, test in skf.split(X, y):
        # # print groups
        # print(labeled_samples.iloc[train].groupby("class")["name"].nunique())
        # print(labeled_samples.iloc[test].groupby("class")["name"].nunique())
        # print("----------------------------------------------------------------")

        # create a nested list with testing set of all folders
        for index in test:
            sample_name = labeled_samples.iloc[index]["name"]
            test_set_temp.append(sample_name)
        test_set.append(test_set_temp)
        test_set_temp = []
        # create nested list with training set for all folders2
        for index in train:
            sample_name = labeled_samples.iloc[index]["name"]
            train_set_temp.append(sample_name)
        train_set.append(train_set_temp)
        train_set_temp = []
    return train_set, test_set


def force_class_3(dft, dftr):
    cl3_n = None
    try:
        cl3_n = dftr.groupby("class")["name"].nunique()[3]
    except:
        pass

    if cl3_n == 3:
        df_cl3 = dftr.groupby("name").mean().loc[lambda dftr: dftr["class"] == 3]
        rand_sample = df_cl3.index[np.random.randint(2, size=1)[0]]
        df_temp = dftr[dftr["name"].str.contains(rand_sample)]  # save pixels of randsample
        sample_inds = dftr[dftr["name"].str.contains(rand_sample)].index
        dft = dft.append(df_temp)  # insert sample inside testing set
        dftr.drop(sample_inds, inplace=True)
        dft = dft.reset_index()
        dftr = dftr.reset_index()

    return dftr, dft


def k_fold_cross_validation(rfc, df_train, t_set,
                            v_set, n_folder,
                            validation_graphs,
                            max_depth):
    # Range of depth
    min_depth = 1

    # loop and fit a new model for different number of tree estimators
    error_train_general = []
    error_validation_general = []
    for k, train in enumerate(t_set):
        validate = v_set[k]

        # concatenate string key for str.contains
        train_key = train[0]
        for sample_name in train: train_key = train_key + str("|") + sample_name
        v_key = validate[0]
        for sample_name in validate: v_key = v_key + str("|") + sample_name

        # separate train predictor and labels
        X_train = df_train[df_train["name"].str.contains(train_key)][df_train.columns[1:-1]].values
        y_train = df_train[df_train["name"].str.contains(train_key)][df_train.columns[-1]].values

        # separate validation predictor and labels
        X_validate = df_train[df_train["name"].str.contains(v_key)][df_train.columns[1:-1]].values
        y_validate = df_train[df_train["name"].str.contains(v_key)][df_train.columns[-1]].values

        # applying standard scaling to normalize data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_validate = sc.fit_transform(X_validate)

        # Measure train and validation error for changing depth
        error_rate_validation = []
        error_rate_train = []
        for i in range(min_depth, max_depth + 1):
            # set the depth of the tree for the Train/validation pair
            rfc.set_params(max_depth=i)
            rfc.fit(X_train, y_train)

            # Use only one test set which was splitted before the loop
            test_error = 1 - accuracy_score(y_validate, rfc.predict(X_validate))
            train_error = 1 - rfc.score(X_train, y_train)
            error_rate_train.append((i, train_error))
            error_rate_validation.append((i, test_error))

        # save train and validation error for general analysis
        error_train_general.append(error_rate_train)
        error_validation_general.append(error_rate_validation)

    # take mean error of train and validation
    depth_n, error_train = compute_error_mean(array=error_train_general)
    depth_n, error_validation = compute_error_mean(array=error_validation_general)

    # compute de optimal depth that provides minimum error for validation
    depth_opt = np.where(error_validation == min(error_validation))[0][0] + 1

    # save error values for each k-fold
    arr_folds_valid = np.array(error_validation_general) \
        .reshape(len(error_validation_general) * len(error_validation_general[0]),
                 2)
    savetxt('validation/internal/error_folds_valid ' + datetime.now().strftime("(%H_%M)%d_%m_%Y") + ".csv"
            , arr_folds_valid, delimiter=',')
    arr_folds_train = np.array(error_train_general) \
        .reshape(len(error_train_general) * len(error_train_general[0]),
                 2)
    savetxt('validation/internal/error_folds_train ' + datetime.now().strftime("(%H_%M)%d_%m_%Y") + ".csv"
            , arr_folds_train, delimiter=',')

    # plot mean resultant of cross-validation
    plot_cross_validation(err_train=error_train,
                          err_validation=error_validation,
                          depths=depth_n)

    return depth_opt


def compute_error_mean(array):
    arr_sum = np.zeros(len(array[0]))
    for v_loop in array:
        v_loops_n, arr = zip(*v_loop)
        arr_sum = arr_sum + np.array(arr)
    arr_mean = arr_sum / len(array)

    return v_loops_n, arr_mean


def plot_cross_validation(err_train, err_validation, depths):
    # print array with mean errors
    print("train error: ", err_train)
    print("validation error: ", err_validation)

    # plot mean validation and train error
    x_train, y_train = depths, err_train
    xs_dum, y_validation = depths, err_validation
    fig, ax1 = plt.subplots()
    ax1.plot(x_train, y_validation, label="mean validation error", marker="o")
    ax1.plot(x_train, y_train, label="mean train error", marker="o")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.xlim(min_depth, max_depth)
    plt.xlabel("Random Forest depth")
    plt.ylabel("Error rate")
    plt.legend(loc="upper right")
    plt.show()
    pass


def train_and_test(rfc, df_train, df_test, confusion_matrix, save_model, depth, plot_importance, evalute_n_trees,
                   labels, n_max_trees):
    # separate train predictor and labels
    X_train = df_train[df_train.columns[1:-1]].values
    y_train = df_train[df_train.columns[-1]].values

    # separate validation predictor and labels
    X_test = df_test[df_train.columns[1:-1]].values
    y_test = df_test[df_train.columns[-1]].values

    # applying standard scaling to normalize data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # train the model
    rfc.set_params(max_depth=depth)
    rfc.fit(X_train, y_train)

    if confusion_matrix:
        plot_confusion_matrix(rfc, X_test, y_test, display_labels=labels, normalize="true")
        plt.show()

    if save_model:
        joblib.dump(rfc, "trained_rf/colm_in/model" +
                    datetime.now().strftime("(%H_%M)%d_%m_%Y")
                    + ".joblib")

    if plot_importance:
        ploter_importance(df=df_train, rfc=rfc)

    if evalute_n_trees:
        ploter_trees_evaluation(trees_max=n_max_trees, rfc=rfc,
                                X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test)
    pass


def ploter_importance(df, rfc):
    feature_names = list(df.columns[1:-1])
    importances = rfc.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in rfc.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_xlabel("Predictors")
    fig.tight_layout()
    plt.show()
    pass
3

def ploter_trees_evaluation(trees_max, rfc, X_train, y_train,
                            X_test, y_test):
    error_rate = []
    error_rate_train = []

    # Range of `n_estimators` values to expl ore.
    min_estimators = 2
    max_estimators = trees_max

    # loop and fit a new model for different number of tree estimators
    for i in range(min_estimators, max_estimators + 1):
        rfc.set_params(n_estimators=i)
        rfc.fit(X_train, y_train)

        # Use only one test set which was splitted before the loop
        test_error = 1 - accuracy_score(y_test, rfc.predict(X_test))
        train_error = 1 - rfc.score(X_train, y_train)
        error_rate.append((i, test_error))
        error_rate_train.append((i, train_error))

    # devide x and y axes
    xs, ys = zip(*error_rate)
    xsdum, ys_train = zip(*error_rate_train)
    fig, ax1 = plt.subplots()
    ax1.plot(xs, ys, label="test error", marker="o")
    ax1.plot(xs, ys_train, label="train error", marker="o")
    # plt.xlim(min_depth, max_depth)
    plt.xlabel("number of trees")
    plt.ylabel("error rate")
    plt.legend(loc="upper right")
    plt.show()
    pass


def modify_predictors_labels(df):
    df.rename(columns={"ddem": "DDEM",
                       "fa": "FA",
                       "lateral": "MA",
                       "lisd": "LISD",
                       "tr": "TR",
                       "tri": "TRI",
                       "velo": "FVI",
                       "wa": "WAP"}, inplace=True)
    return df

def modify_predictors_labels_shovel(df):
    df.rename(columns={"ddem": "DDEM",
                       "green": "FA",
                       "lateral": "MA",
                       "vari": "LISD",
                       "rgb": "TR",
                       "tri": "TRI",
                       "longitudinal": "FVI",
                       "wa": "WAP"}, inplace=True)
    return df
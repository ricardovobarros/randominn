from config import *
import glob

# loop trough all the predictors types

local_raster_file = Path(os.path.abspath(os.getcwd()) + "/data/facies/colmation/out/kb19/rasters")
file_list_raster = find_files(local_raster_file)
name_csv = str(local_raster_file).split("\\")[-2]
# find out how many probes there are
list = ["x", "y", "name"]
probes_number, predictors_number, predictors, probes_names = find_probes_info(file_list_raster)
df_train_temp = pd.DataFrame()
list.extend(predictors)
df_train = pd.DataFrame(columns=list)

for i, file_raster in enumerate(file_list_raster):
    raster = Raster(file_raster)
    band_array = raster.band2array()
    # band_array = sample_band(band_array=band_array, raster=raster, radius=0.5) # unfinished function to
    # resample the array
    df = raster.coord_dataframe(band_array.reshape(band_array.size, 1))

    # fill a dataframe temporarily with all values of a measurement
    if i % predictors_number == 0 and i != predictors_number * probes_number:
        df_train_temp["x"], df_train_temp["y"] = df["x"], df["y"]
        df_train_temp["name"] = probes_names[i]
    df_train_temp[str(predictors[i % predictors_number])] = df["z"]  # create a new column with predictor

    # When the pixel in the previous measurement end, fill a larger dataset with all values
    if (i + 1) % predictors_number == 0 and i != 0:
        df_train = df_train.append(df_train_temp)
        df_train_temp = pd.DataFrame()
df_train.dropna(how='any', axis=0, inplace=True)  # delete all rows that contains at least one nan value


# label out or inner colmation according with field classification
probes = df_in_classes if in_colmation else df_out_classes
probes_col_name = "Innere_K_1" if in_colmation else "AK"
list = df_train["name"].tolist()
for i, probe in enumerate(probes["name"]):
    if probe in df_train["name"].tolist():
        df_train.loc[df_train['name'] == probe, 'class'] = probes[probes_col_name][i]
df_train.dropna(how='any', axis=0, inplace=True)
df_train.to_csv("training_csv/out_colm/" + name_csv + ".csv", index=False)

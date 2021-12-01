from config import *
# # location for gransize distribution
# local_rasterin_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/shovel/shovel_rasters_test/kb1902/aspect_ratio")
# local_shape_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/shovel/shovel_facies/kb1902")
# local_rasterout_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/shovel/shovel_rasters_test/kb1902")

# location for inner colmation
local_rasterin_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/colmation/predictors/kb08")
local_shape_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/colmation/out/kb08/shapes")
local_rasterout_add = Path(os.path.abspath(os.getcwd()) + "/data/facies/colmation/out/kb08/rasters")


# Size of output raster
facie_radius = 1  # value in meters
pixel_size = 0.06961  # value in meters
board_factor = 1.1  # enlarge de size to ont cut out information
width_rasterout = int((facie_radius / pixel_size) * board_factor)
hight_rasterout = width_rasterout

# create two lists with all name of raster and shape files
file_list_raster = find_files(local_rasterin_add)
file_list_shapes = find_files(local_shape_add)

# for i, rasterin in enumerate(file_list_raster):
#     num = 0 if i < 5 else 1 if i < 10 else 2
#     predict = rasterin.split("_")[-1].strip(".tif")
#     rasterout = rasterin.split(".")[0] + predict + ".tif"
#     shape = file_list_shapes[num]
#     gdal.Warp(rasterout, rasterin, cutlineDSName=shape)

for i, rasterin in enumerate(file_list_raster):
    predictor = rasterin.split("_")[-1].split(".")[0]  # name of the predictor
    if predictor == "fa":
        for k, facie in enumerate(file_list_shapes):
            number_facie = facie.split("_")[-1].strip(".shp")  # number generated from vector split
            name_bank = str(local_shape_add).split("\\")[-2]  # folder of shapes must be named with the name of the bank
            is_sand = True if facie.split("\\")[-1].split("_")[0] == "FID" else False  # verify if the shape is sand
            name_bank = "sand" if is_sand else name_bank  # change the name of the bank to sand if the facie is sand
            name_raster_out = name_bank + "_" + number_facie + "_" + predictor + ".tif"  # name od the clipped raster
            rasterout = str(local_rasterout_add) + str(Path("/")) + name_raster_out
            gdal.Warp(destNameOrDestDS=rasterout, srcDSOrSrcDSTab=rasterin,
                      cutlineDSName=facie,
                      cropToCutline=True)  # clip rater


from config import *



# define raster input and out names to create least cost raster
in_raster_name = r"" + os.path.abspath('') + "/Inn_rasters/dem_EPSG32632_kb05.tif"
out_raster_name = r"" + os.path.abspath('') + "/Inn_rasters/dem_EPSG32632_kb07_least_cost_not_geo_full.tif"

# define raster input and shapefile output names to create thalweg line vector
target_shp_fn = r"" + os.path.abspath('') + "/Inn_shapes/dem_EPSG32632_kb19_02_least_cost_line.shp"
source_raster_fn = r"" + os.path.abspath('') + "/Inn_rasters/dem_EPSG32632_kb19_02_least_cost_not_geo_full.tif"

# define coordinates of points 1 and 2 (in EPSG:6418)
# # Whole Inn EPSG 32632
# point_1_coord = (751520, 5341515)
# point_2_coord = (768872.6, 5350425.3)
# # Piece Inn EPSG 32632
# point_1_coord = (755330.4, 5344911.2)
# point_2_coord = (756994.59, 5343623.03)
# kb05 EPSG 32632
point_1_coord = (753498.0, 5343219.2)
point_2_coord = (754323.8, 5342896.6)
# # kb07 EPSG 32632
# point_1_coord = (754413.4, 5342842.9)
# point_2_coord = (755216.8, 5343103.2)
# # kb08 EPSG 32632
# point_1_coord = (755324.2, 5344920.3)
# point_2_coord = (756178.2, 5344129.1)
# # kb13 EPSG 32632
# point_1_coord = (759313.3, 5345896.1)
# point_2_coord = (758136.7, 5346709.6)
# # kb19_1 EPSG 32632
# point_1_coord = (761482.9, 5348794.1)
# point_2_coord = (761871.1, 5348235.1)
# # kb19_2 EPSG 32632
# point_1_coord = (761893.5, 5348236.9)
# point_2_coord = (762060.0, 5348966.3)

# # get source raster (osgeo.gdal.Dataset), the raster as nd.array, and the geotransformation tuple
src_raster, raster_array, geo_transform = gt.raster2array(in_raster_name)
#
# # get the zeros-like array with least cost pixels = 1
# path_array = create_path_array(raster_array, geo_transform, point_1_coord, point_2_coord)
# #
# # get the spatial reference system of the input raster (slope-percent.tif)
# src_srs = gt.get_srs(src_raster)

## project the least cost path_array into a Byte (only zeros and ones) raster
# gt.create_raster(out_raster_name, path_array, epsg=int(src_srs.GetAuthorityCode(None)),
#                  rdtype=gdal.GDT_Byte, geo_info=geo_transform)


# Transform the raster into line
pixel_value = 1
raster2line(source_raster_fn, target_shp_fn, pixel_value)





"""Module designated to Raster class

Author : Ricardo Barros
"""

from config import *
from pathlib import Path
import numpy.ma as ma
import pandas as pd


class Raster:

    def __init__(self, raster_address="", driver="Gtiff", band=1):
        """A class used to represent a raster file (.tif)

        Attributes
        ----
        name: String of the files' name
        driver: Object Driver from Gdal
        dataset:Object Dataset from Gdal
        transform: Tuple with pixels information
        projection: String of DEMs' projection
        band: Object Band from Gdal

        Methods:
        ----
        get_band_array: creates a array with a given band and substitutes
        the values -9999 for np.nan
        coord_dataframe: creates a data frame with XYZ coordinates of pixels
        burn: burns a array into a raster (.tif)

        Parameters:
        ____
        :param raster_address: String local address of raster file
        :param driver: Type of driver to open the raster file
        :param band: Integer number of the raster's band
        """
        try:
            # self.name = raster_address.split(str(Path("/")))[-1].strip(".tif")
            self.driver = gdal.GetDriverByName(driver)
            self.dataset = gdal.Open(raster_address)
            self.transform = self.dataset.GetGeoTransform()
            self.projection = self.dataset.GetProjection()
            # self.band = self.dataset.GetRasterBand()
        except:
            logging.error("Raster file could not be read")
            pass

    def rgb_to_arrays(self, nodatavalue=np.nan):
        """Creates a array with a given band and substitutes
        the values -9999 for np.nan

        :return: Array of bands' float values
        :return: Vector (flat array of bands' float values)
        """
        red_array = self.dataset.GetRasterBand(1).ReadAsArray().astype("float")
        green_array = self.dataset.GetRasterBand(2).ReadAsArray().astype("float")
        blue_array = self.dataset.GetRasterBand(3).ReadAsArray().astype("float")
        band4 = self.dataset.GetRasterBand(4).ReadAsArray().astype("float")

        red_array[red_array == 0] = nodatavalue
        green_array[green_array == 0] = nodatavalue
        blue_array[blue_array == 0] = nodatavalue
        band4[band4 == 0] = nodatavalue

        return red_array, green_array, blue_array

    def band2array(self, nodatavalue=np.nan):
        """

        :param nodatavalue: Value with no value in the band array
        :return: first band of raster as array
        """
        band_array = self.dataset.GetRasterBand(1).ReadAsArray().astype("float")
        band_array[band_array == 0] = nodatavalue
        band_array = np.nan_to_num(band_array, nan=nodatavalue)
        band_array[band_array < - 100000000] = nodatavalue

        return band_array

    def get_band_array(self):
        """Creates a array with a given band and substitutes
        the values -9999 for np.nan

        :return: Array of bands' float values
        :return: Vector (flat array of bands' float values)
        """
        band_array = np.array(self.band.ReadAsArray())
        band_array[band_array == 0] = np.nan
        band_array_flat = band_array.reshape(band_array.size, 1)
        return band_array, band_array_flat

    def coord_dataframe(self, array):
        """Creates a data frame with XYZ coordinates of pixels

        :param array: flat array of bands' float values
        :return: data frame with columns [x,y,z]
        """
        table_xyz = gdal.Translate("dem.xyz", self.dataset)
        table_xyz = None
        df = pd.read_csv("dem.xyz", sep=" ", header=None)
        os.remove("dem.xyz")
        df.columns = ["x", "y", "z"]
        df[["z"]] = array
        return df

    def burn_rgb(self, array, transf="", folder_path="", file_name=""):
        """Creates a (.tif) file from a array and saves it with
        the same projection and transform information of the
        instantiated object.

        :param transf:
        :param array: Array nxm of Floats/Int
        :return:None
        """
        array = np.nan_to_num(array, nan=0.0)
        array = ma.masked_where(array == 0, array)
        array = np.ma.masked_invalid(array)

        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
        self.driver.Register()
        output_raster = self.driver.Create(
            str(folder_path) + str(file_name) + ".tif",
            xsize=array.shape[2],
            ysize=array.shape[1],
            bands=3, eType=gdal.GDT_Float32, options=options)
        output_raster.SetGeoTransform(self.transform)
        output_raster.SetProjection(self.projection)

        for band in range(1, 4):
            output_raster.GetRasterBand(band).WriteArray(array[band - 1, :, :])

        output_raster.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        output_raster.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        output_raster.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        # output_raster.SetNoDataValue(np.nan)

        # create tif file
        output_raster.FlushCache()

        del output_raster

        # except Exception:
        #     print("Raster file could not be created")

    def burn(self, array, folder_path="", file_name=""):
        """Creates a (.tif) file from a array and saves it with
            the same projection and transform information of the
            instantiated object.

            :param transf:
            :param array: Array nxm of Floats/Int
            :return:None
            """
        array = np.nan_to_num(array, nan=0.0)
        array = ma.masked_where(array == 0, array)
        array = np.ma.masked_invalid(array)

        try:
            self.driver.Register()
            output_raster = self.driver.Create(
                str(folder_path) + str(file_name) + ".tif",
                xsize=array.shape[1],
                ysize=array.shape[0],
                bands=1, eType=gdal.GDT_Float32)
            output_raster.SetGeoTransform(self.transform)
            output_raster.SetProjection(self.projection)
            output_raster_band = output_raster.GetRasterBand(1)

            output_raster_band.WriteArray(array)

            # output_raster_band.SetNoDataValue(np.nan)
            output_raster_band.FlushCache()

            output_raster_band = None
            output_raster = None
        except Exception:
            print("Raster file could not be created")

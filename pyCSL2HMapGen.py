#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""A simple script to extract & interpolate CSL2 height map from JAXA:AW3D30 data files.

Author: HomeOnMars
"""


# # Functions

# In[2]:


# dependencies: numpy, scipy, gdal, pyppng
import numpy as np
from numpy import pi
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from osgeo import gdal
import png
gdal.UseExceptions()


# In[3]:


def get_interpolator_AW3D30(
    filename    : str,
    method      : str  = 'linear',
    opened_data : dict = {},
    verbose     : bool = True,
) -> RegularGridInterpolator:
    """Find the elevation at a single point."""

    if filename not in opened_data.keys():
        # open file
        if verbose:
            print(f"Reading data from file {filename}...")
        file = gdal.Open(filename)
        trans_pars = file.GetGeoTransform()
        elev_xy = file.GetRasterBand(1).ReadAsArray()
        
        # get xy grid
        if trans_pars[2] != 0 or trans_pars[4] != 0:
            raise NotImplementedError("Input tiff has a twisted grid. Interpolation here has not been implemented. Add code plz!")
        #long_xy = np.fromfunction((lambda y, x: trans_pars[0] + x*trans_pars[1] + y*trans_pars[2]), elev_xy.shape)
        #lati_xy = np.fromfunction((lambda y, x: trans_pars[3] + x*trans_pars[4] + y*trans_pars[5]), elev_xy.shape)
        long_ax = np.fromfunction((lambda x: trans_pars[0] + x*trans_pars[1]), (elev_xy.shape[1],))
        lati_ax = np.fromfunction((lambda y: trans_pars[3] + y*trans_pars[5]), (elev_xy.shape[0],))
        
        interp  = RegularGridInterpolator((lati_ax, long_ax), elev_xy, method=method, bounds_error=False, fill_value=-1)
        
        # save data
        opened_data[filename] = interp
    interp = opened_data[filename]

    return interp


# In[4]:


def get_grid_coord(
    ilatis: np.ndarray,
    ilongs: np.ndarray,
    center_lati: float,
    center_long: float,
    NS_width_km: float,
    EW_width_km: float,
) -> float:
    """The function for np.fromfunction() to get the coordinates for our map's grid.
    """
    #  earth radius
    Rearth_km = 6378.1

    ans = np.zeros(ilatis.shape)
    nlati, nlong, _ = ilatis.shape

    # latitude
    NS_width_deg = NS_width_km / Rearth_km / pi * 180
    dlati = NS_width_deg / nlati
    #lati  = center_lati + dlati * (-nlati / 2. + 0.5 + ilatis)
    #  (inverted because images work in weird ways)
    lati  = center_lati + dlati * (nlati / 2. - 0.5 - ilatis)
    ans[:, :, 0] = lati[:, :, 0]
    
    # longtitude
    EW_width_deg = EW_width_km / (Rearth_km * np.cos(lati/180.*pi)) / pi * 180
    dlong = EW_width_deg / nlong
    long  = center_long + dlong * (-nlong / 2. + 0.5 + ilongs)
    ans[:, :, 1] = long[:, :, 1]
    
    return ans


# In[5]:


def interpolate_height_map_AW3D30(
    long        : float,
    lati        : float,
    tiffilenames: tuple[str],
    map_width_km: float = 57.344,    # 57.344 or 14.336 for CS2
    EW_width_km : float = None,
    NS_width_km : float = None,
    interp_method: str  = 'linear',
    opened_data : dict  = {},
    Rearth_km   : float = 6378.1,    #  earth radius in km
    nlati       : int   = 4096,    #  grid size
    nlong       : int   = 4096,    #  grid size
    verbose     : bool  = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate existing tiff files to get height map for CSL2.

    long, lat: float
        in Degrees.

    map_width_km: float
        width & length of the map.
        Will override EW_width_km and NS_width_km (width in East-West / North-South direction)
        57.344 (world map) or 14.336 (playable map) for CS2.

    opened_data: dict
        Don't touch this.
        Temperaroy buffer to store the data from files, so that we don't need to open multiple times.

    """

    # init
    if map_width_km is not None:
        EW_width_km = map_width_km
        NS_width_km = map_width_km

    # the answer we are looking for! i.e. elevation
    ans = np.full((nlati, nlong), -1, dtype=np.int16)

    # init long to be in [0, 360)
    long = long % 360

    # approximating grid
    coord = np.fromfunction(
        (lambda ilatis, ilongs, itypes: get_grid_coord(ilatis, ilongs, lati, long, NS_width_km, EW_width_km)),
        (nlati, nlong, 2),
    )
    
    for filename in tiffilenames:
        # update ans with tiles data from each tif file
        interp = get_interpolator_AW3D30(filename, method=interp_method, opened_data=opened_data, verbose=verbose)
        ans0 = interp(coord, method=interp_method)
        ans = np.where(ans0 >= 0, ans0, ans)

    # fixing the gaps between tiles (where ans==-1) by filling them the closest neighbour values
    ind = distance_transform_edt(ans < 0, return_distances=False, return_indices=True)
    ans = ans[tuple(ind)]
    
    return ans, coord


# In[6]:


def get_CSL_height_maps(
    long        : float,
    lati        : float,
    tiffilenames: tuple[str],
    cityname    : str   = None,
    scale       : float = 1.0,
    height_scale: float = 4096.,
    min_height  : float = 64.,
    interp_method: str  = 'linear',
    opened_data : dict  = {},
    verbose     : bool  = True,
):
    """Wrapper function to extract height map from data and save them to disk."""

    WORLDMAP_WIDTH_km = 57.344
    PLAYABLE_WIDTH_km = 14.336

    long = long % 360

    # step 1: get world map
    ans, _ = interpolate_height_map_AW3D30(
        long=long, lati=lati, tiffilenames=tiffilenames,
        map_width_km=WORLDMAP_WIDTH_km*scale, interp_method=interp_method,
        nlati=4096, nlong=4096, opened_data=opened_data, verbose=verbose)
    ans = ans * scale + min_height
    
    # sanity checks
    if np.count_nonzero(ans < 0):
        print("*   Warning: artifacts in worldmap image detected.")
    if ans.max() >= height_scale:
        print(f"*** Warning: maximum height = {ans.max()} is higher than height_scale.")
        height_scale = np.ceil(ans.max())+1
        print(f"\tSetting new height scale to be {height_scale}")
    
    img_arr = (ans / height_scale * 2**16).astype(np.uint16)

    if cityname is None:
        cityname = f"{long:.3f}_{lati:+.3f}"
        
    outfilename = f"worldmap_{cityname}.png"
    with open(outfilename, 'wb') as f:
        writer = png.Writer(width=img_arr.shape[1], height=img_arr.shape[0], bitdepth=16, greyscale=True)
        if verbose: print(f"Saving to {outfilename}")
        writer.write(f, img_arr)
    img_arr_orig = img_arr


    # step 2: get the height map
    ans, _ = interpolate_height_map_AW3D30(
        long=long, lati=lati, tiffilenames=tiffilenames,
        map_width_km=PLAYABLE_WIDTH_km*scale, interp_method=interp_method,
        nlati=4096, nlong=4096, opened_data=opened_data, verbose=verbose)
    ans = ans * scale + min_height
    
    # sanity checks
    if np.count_nonzero(ans < 0):
        print("*   Warning: artifacts in playable image detected.")
    if ans.max() >= height_scale:
        print(f"*** Warning: maximum height = {ans.max()} is higher than height_scale.")
        print(f"\tWill NOT do anything.")

    img_arr = (ans / height_scale * 2**16).astype(np.uint16)
    
    outfilename = f"playable_{cityname}.png"
    with open(outfilename, 'wb') as f:
        writer = png.Writer(width=img_arr.shape[1], height=img_arr.shape[0], bitdepth=16, greyscale=True)
        if verbose: print(f"Saving to {outfilename}")
        writer.write(f, img_arr)

    return img_arr_orig


# # Example

# In[7]:


# example

long=168.77
lati=-44.05
tiffilenames = [
    'ALPSMLC30_S044E168_DSM.tif',
    'ALPSMLC30_S045E168_DSM.tif',
    'ALPSMLC30_S044E169_DSM.tif',
    'ALPSMLC30_S045E169_DSM.tif',
]

img_arr = get_CSL_height_maps(
    long=long, lati=lati, tiffilenames=tiffilenames, scale=1.5)


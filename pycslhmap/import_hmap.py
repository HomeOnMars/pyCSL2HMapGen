#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""A simple script to extract & interpolate CSL2 height map from JAXA:AW3D30 data files.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# # Functions

# In[2]:


# dependencies: numpy, scipy, gdal, pypng
import numpy as np
from numpy import pi
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from osgeo import gdal
import png
gdal.UseExceptions()


# In[3]:


def get_interpolator_tiff(
    filename    : str,
    opened_data : dict = {},
    verbose     : bool = True,
) -> RegularGridInterpolator:
    """Find the elevation at a single point."""

    if filename not in opened_data.keys():
        # open file
        if verbose:
            print(f"Reading data from file {filename}...", end=' ')
        file = gdal.Open(filename)
        trans_pars = file.GetGeoTransform()
        elev_xy = file.GetRasterBand(1).ReadAsArray()
        
        # get xy grid
        if trans_pars[2] != 0 or trans_pars[4] != 0:
            raise NotImplementedError("Input tiff has a twisted grid. Interpolation here has not been implemented. Add code plz!")
        #long_xy = np.fromfunction((lambda y, x: trans_pars[0] + x*trans_pars[1] + y*trans_pars[2]), elev_xy.shape)
        #lati_xy = np.fromfunction((lambda y, x: trans_pars[3] + x*trans_pars[4] + y*trans_pars[5]), elev_xy.shape)
        long_ax = np.fromfunction((lambda x: trans_pars[0] + x*trans_pars[1]), (elev_xy.shape[1],)) % 360
        lati_ax = np.fromfunction((lambda y: trans_pars[3] + y*trans_pars[5]), (elev_xy.shape[0],))
        
        interp  = RegularGridInterpolator((lati_ax, long_ax), elev_xy, bounds_error=False, fill_value=-1)
        
        # save data
        opened_data[filename] = interp
    else:
        if verbose:
            print(f"Using data from file {filename}...", end=' ')
    interp = opened_data[filename]

    return interp


# In[4]:


def get_grid_coord(
    ilatis: np.ndarray,
    ilongs: np.ndarray,
    angle_rad  : float,
    center_lati: float,
    center_long: float,
    NS_width_km: float,
    EW_width_km: float,
) -> np.ndarray:
    """The function for np.fromfunction() to get the coordinates for our map's grid.
    """
    #  earth radius
    Rearth_km = 6378.1

    ans = np.zeros(ilatis.shape)
    nlati, nlong, _ = ilatis.shape

    # set ilatis to be 0 at the center (n for new)
    #    (ilatis inverted because images work in weird ways)
    ilatis_n =  nlati / 2. - 0.5 - ilatis
    ilongs_n = -nlong / 2. + 0.5 + ilongs

    # rotation (r for rotated)
    ilatis_r = np.sin(angle_rad) * ilongs_n + np.cos(angle_rad) * ilatis_n
    ilongs_r = np.cos(angle_rad) * ilongs_n - np.sin(angle_rad) * ilatis_n

    # latitude
    NS_width_deg = NS_width_km / Rearth_km / pi * 180
    dlati = NS_width_deg / nlati
    #lati  = center_lati + dlati * (-nlati / 2. + 0.5 + ilatis_r)
    #  (inverted because images work in weird ways)
    lati  = center_lati + dlati * ilatis_r
    
    # longtitude
    EW_width_deg = EW_width_km / (Rearth_km * np.cos(lati/180.*pi)) / pi * 180
    dlong = EW_width_deg / nlong
    long  = center_long + dlong * ilongs_r

    # write answer
    ans[:, :, 0] = lati[:, :, 0]
    ans[:, :, 1] = long[:, :, 1]
    
    return ans


# In[5]:


def interpolate_height_map_tiff(
    long        : float,
    lati        : float,
    tiffilenames: tuple[str],
    angle_deg   : float = 0.,
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

    angle_deg: float
        how many degrees we are rotating the map counter-clockwise.

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
    angle_rad = angle_deg / 180. * pi

    # the answer we are looking for! i.e. elevation
    ans = np.full((nlati, nlong), -1, dtype=np.int32)

    # init long to be in [0, 360)
    long = long % 360

    # approximating grid
    coord = np.fromfunction(
        (lambda ilatis, ilongs, itypes: get_grid_coord(ilatis, ilongs, angle_rad, lati, long, NS_width_km, EW_width_km)),
        (nlati, nlong, 2),
    )

    # trace how much of the map has been covered
    nhit_total = 0
    
    for filename in tiffilenames:
        # update ans with tiles data from each tif file
        interp = get_interpolator_tiff(filename, opened_data=opened_data, verbose=verbose)
        ans0 = interp(coord, method=interp_method)
        nhit = np.count_nonzero(ans0+1)
        nhit_total += nhit
        if nhit:
            if verbose: print(f"Hit ({nhit/ans.size*100: 6.2f}%).")
            ans = np.where(ans0 >= 0, ans0, ans)
        else:
            if verbose: print("Missed.")

    if verbose:
        print(f"Total {nhit_total/ans.size*100: 6.2f}% of the map has been covered.")
        if nhit_total/ans.size < 0.9973: # 99.73% is 3 sigma because why not
            print(
                f"*** Warning: a large portion of the map ({(1.-nhit_total/ans.size)*100: 6.2f}%)",
                "hasn't been covered by interpolation. Please consider download and add more map tiles data."
            )

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
    angle_deg   : float = 0.,
    map_scales  : float | tuple[float, float] = 1.0,
    height_scale: float = 4096.,
    min_height_m:  int  = 128,
    ocean_height:  int  = 64,
    smooth_shore_rad_m  : float= 448.,
    smooth_rad_m: float = 14.,
    interp_method       : str  = 'linear',
    opened_data : dict  = {},
    Rearth_km   : float = 6378.1,
    out_filepath: None|str = './out/',
    verbose     : bool  = True,
):
    """Wrapper function to extract height map from data and save them to disk.

    angle_deg: float
        how many degrees we are rotating the map counter-clockwise.
        
    map_scales: float | tuple[float, float]
        map_scales = real world size / game map size
        if tuple, it should be in format of (width scale, height scale).

    min_height_m: int
        Minimum height for *NON-OCEAN* area, in meters. Must >= 1.
        The ocean area will still have an height of ocean_height.

    smooth_shore_rad_m: float
        size of the smoothing kernel in in-sgame meters (using gaussian_filter), for shorelines.
        The function will first smooth the shoreline with this (smooth_shore_rad_m),
            then go through the smooth kernel again for the whole map using smooth_rad_m.
    
    smooth_rad_m: float
        size of the smoothing kernel in in-game meters (using gaussian_filter).
        Set to 0 to disable this.

    out_filepath: None|str
        Provide this to output the result to the specified folder.
    
    Rearth_km: float
        Earth radius in real-world km.
        Do NOT change this unless you are generating a map of Mars or something.
        
    """

    WORLDMAP_WIDTH_km = 57.344
    PLAYABLE_WIDTH_km = 14.336
    WORLDMAP_NRES = 4096
    PLAYABLE_NRES = 4096
    

    long = long % 360.
    angle_deg = angle_deg % 360.
    try:
        scale_w = map_scales[0]
        scale_h = 1./map_scales[1]
    except TypeError:
        scale_w = map_scales
        scale_h = map_scales
        

    # step 1: get world map
    if verbose: print(f"\n\tWorld map size ({WORLDMAP_WIDTH_km*scale_w:.3f} km)^2")
    ans, coord = interpolate_height_map_tiff(
        long=long, lati=lati, tiffilenames=tiffilenames, angle_deg=angle_deg,
        map_width_km=WORLDMAP_WIDTH_km*scale_w, interp_method=interp_method,
        nlati=WORLDMAP_NRES, nlong=WORLDMAP_NRES,
        opened_data=opened_data, Rearth_km=Rearth_km, verbose=verbose)
    
    #  get # of pixels for the smooth kernel radius
    smooth_shore_rad_pix = smooth_shore_rad_m / (1e3 * WORLDMAP_WIDTH_km / WORLDMAP_NRES)
    smooth_rad_pix       = smooth_rad_m       / (1e3 * WORLDMAP_WIDTH_km / WORLDMAP_NRES)
    #  smooth the shorelines- cap the height to min_height_m just in case
    ans_in_ocean         = (ans==0)    # np bool array, true if ocean, false if land
    ans = np.where(ans_in_ocean, ocean_height, ans * scale_h + min_height_m)    # re-scaled
    #  how close is a pixel in ocean to land
    if verbose: print("Smoothing.", end='')
    ans_shorelineness    = gaussian_filter(np.where(ans_in_ocean, 0., 1.), sigma=smooth_shore_rad_pix)
    #  smooth the shoreline
    if verbose: print('.', end='')
    ans_ocean_filtered   = gaussian_filter(
        np.where(ans_shorelineness<0.125, ocean_height, min_height_m),
        sigma=smooth_shore_rad_pix)
    ans = np.where(ans_in_ocean,
                   np.where(ans_ocean_filtered>min_height_m-1,
                            min_height_m-1,
                            ans_ocean_filtered),
                   ans)
    # final general smoothing
    if verbose: print('.', end='')
    ans = gaussian_filter(ans, sigma=smooth_rad_pix)
    if verbose: print(" Done.")
        
    
    # sanity checks
    if verbose and np.count_nonzero(ans < 0):
        print("*   Warning: artifacts in worldmap image detected.")
    if ans.max() >= height_scale:
        if verbose: print(f"*** Warning: maximum height = {ans.max()} is higher than height_scale.")
        height_scale = np.ceil(ans.max())+1
        if verbose: print(f"\tSetting new height scale to be {height_scale}")
    elif verbose:
        print(f"\tmaximum height = {ans.max()}")
    if verbose:
        print(
            f"\tCenter point at longtitude {long}, latitude {lati}\n",
            f"\tWorld Map longitude range from {np.min(coord[:, :, 1]): 10.6} to {np.max(coord[:, :, 1]): 10.6}\n",
            f"\t          latitude  range from {np.min(coord[:, :, 0]):+10.6} to {np.max(coord[:, :, 0]):+10.6}",
        )
    if verbose:
        print(f"\tSmoothing Kernel radius {smooth_shore_rad_pix:.2f} pixel (shore), {smooth_rad_pix:.2f} pixel (all)")
    
    img_arr = (ans / height_scale * 2**16).astype(np.uint16)

    if cityname is None:
        cityname = f"long{long:07.3f}_lati{lati:+07.3f}_angle{angle_deg:05.1f}_scale{scale_w:.2f}+{scale_h:.2f}"

    if out_filepath is not None:
        outfilename = f"{out_filepath}worldmap_{cityname}.png"
        with open(outfilename, 'wb') as f:
            writer = png.Writer(width=img_arr.shape[1], height=img_arr.shape[0], bitdepth=16, greyscale=True)
            if verbose: print(f"Saving to {outfilename}")
            writer.write(f, img_arr)
    img_arr_orig = img_arr


    
    # step 2: get the height map
    if verbose: print(f"\n\tPlayable map size ({PLAYABLE_WIDTH_km*scale_w:.3f} km)^2")
    ans, _ = interpolate_height_map_tiff(
        long=long, lati=lati, tiffilenames=tiffilenames, angle_deg=angle_deg,
        map_width_km=PLAYABLE_WIDTH_km*scale_w, interp_method=interp_method,
        nlati=PLAYABLE_NRES, nlong=PLAYABLE_NRES,
        opened_data=opened_data, Rearth_km=Rearth_km, verbose=verbose)
    
    #  get # of pixels for the smooth kernel radius
    smooth_shore_rad_pix = smooth_shore_rad_m / (1e3 * PLAYABLE_WIDTH_km / PLAYABLE_NRES)
    smooth_rad_pix       = smooth_rad_m       / (1e3 * PLAYABLE_WIDTH_km / PLAYABLE_NRES)
    #  smooth the shorelines- cap the height to min_height_m just in case
    ans_in_ocean         = (ans==0)    # np bool array, true if ocean, false if land
    ans = np.where(ans_in_ocean, ocean_height, ans * scale_h + min_height_m)    # re-scaled
    #  how close is a pixel in ocean to land
    if verbose: print("Smoothing.", end='')
    ans_shorelineness    = gaussian_filter(np.where(ans_in_ocean, 0., 1.), sigma=smooth_shore_rad_pix)
    #  smooth the shoreline
    if verbose: print('.', end='')
    ans_ocean_filtered   = gaussian_filter(
        np.where(ans_shorelineness<0.125, ocean_height, min_height_m),
        sigma=smooth_shore_rad_pix)
    ans = np.where(ans_in_ocean,
                   np.where(ans_ocean_filtered>min_height_m-1,
                            min_height_m-1,
                            ans_ocean_filtered),
                   ans)
    # final general smoothing
    if verbose: print('.', end='')
    ans = gaussian_filter(ans, sigma=smooth_rad_pix)
    if verbose: print(" Done.")
        
    
    # sanity checks
    if verbose and np.count_nonzero(ans < 0):
        print("*   Warning: artifacts in playable image detected.")
    if verbose and ans.max() >= height_scale:
        print(f"*** Warning: maximum height = {ans.max()} is higher than height_scale.")
        print(f"\tWill NOT do anything.")
    elif verbose:
        print(f"\tmaximum height = {ans.max()}")
    if verbose:
        print(f"\tSmoothing Kernel radius {smooth_shore_rad_pix:.2f} pixel (shore), {smooth_rad_pix:.2f} pixel (all)")
        
    img_arr = (ans / height_scale * 2**16).astype(np.uint16)

    if out_filepath is not None:
        outfilename = f"{out_filepath}playable_{cityname}.png"
        with open(outfilename, 'wb') as f:
            writer = png.Writer(width=img_arr.shape[1], height=img_arr.shape[0], bitdepth=16, greyscale=True)
            if verbose: print(f"Saving to {outfilename}")
            writer.write(f, img_arr)

    if verbose:
        print("\n\tAll Done.\n")

    return img_arr_orig, coord


# # Example

# In[7]:


if __name__ == '__main__':
    
    # Example 1
    
    # download the relevant data from https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d_e.htm
    #    (or some other sources, I don't care)
    #    If you download from JAXA, you will need to register an account and read their terms of service
    #    after downloading, put them in the ./raw/ folder, and supply the file path (incl. file name) here
    #    they will be used to interpolate the elevations in the respective areas of the image.
    #    if you see a patch of the image is constant at minimal height-1,
    #    then you haven't downloaded & added the data of that patch. Probably.
    tiffilenames = [
        'raw/ALPSMLC30_N063W017_DSM.tif',
        'raw/ALPSMLC30_N064W016_DSM.tif',
        'raw/ALPSMLC30_N064W017_DSM.tif',
    ]
    
    # Parameters explanation
    #  angle_deg is the degrees the map will be rotated
    #  map_scales=(1.5, 1.2) means stretching the width of the map to 1:1.5
    #    (i.e. mapping real world 1.5*57.344km to game 57.344km)
    #    while stretching the heights to 1:1.2
    img_arr, coord = get_CSL_height_maps(
        long=-16.000, lati=+64.185, angle_deg=30., tiffilenames=tiffilenames, map_scales=(1.125, 1.0))


# In[ ]:





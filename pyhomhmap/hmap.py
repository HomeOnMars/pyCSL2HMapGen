#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""A class to handle height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""

# dependencies: numpy, scipy, pypng
from typing import Self
import numpy as np
from numpy import pi
from numpy import typing as npt
#from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates #, distance_transform_edt, gaussian_filter
import png




class HMap:
    """Height Map

    Instance Variables
    ------------------
    data       : (self._npix_xy)-shaped numpy array (np.float64)
        Contains height map data in meters.
        MUST be 2D with len in every dim a multiple of 8

    _ndim      : int = 2
        dimensions. Should always be 2.

    _npix_xy   : tuple[int, int]
        Data shape. i.e. number of pixel in each dim.

    _npix_xy_8, _npix_xy_4 : tuple[int, int]
        == int(self._npix_xy / 8)
        Cached for fast access.
        
    _map_width : tuple[float, float]
        map width in meters (i.e. whatever unit self.data is in).
        It's 57344. for CSL2 world map, and 14336. for CSL2 playable area.
        (Because 57344 = 3.5*4*4096)
        
    

        
    see self.__init__() and self.normalize() for details.
    -------------------------------------------------------------------------------
    """

    def __init__(
        self,
        data : npt.ArrayLike = np.zeros((4096, 4096), dtype=np.float64),
        # note: 57344 = 3.5*4*4096 is the CSL2 worldmap width
        map_width : float|tuple[float, float] = 57344.,
    ):
        # variables
        
        self.data  : npt.NDArray[np.float64] = np.array(data, dtype=np.float64)
        # note: will normalize float into tuple of floats later
        self._map_width : tuple[float, float] = map_width

        
        # do things
        
        self.normalize()


    
    def normalize(self) -> Self:
        
        # variables
        
        # no of pixel: defining map resolution
        self._npix_xy   : tuple[int, int] = self.data.shape
        self._npix_xy_8 : tuple[int, int] = tuple([
            int(npix_i/8) for npix_i   in self._npix_xy  ])
        self._npix_xy_4 : tuple[int, int] = tuple([
            npix_i_8 * 2  for npix_i_8 in self._npix_xy_8])
        self._ndim  : int = len(self._npix_xy)

        try:
            len(self._map_width)
        except TypeError:
            self._map_width = tuple([
                self._map_width for i in range(len(self._npix_xy))])
            
        # safety checks
        assert self._ndim == 2
        assert np.all(np.array(self._npix_xy_8) * 8 == np.array(self._npix_xy))
        
        return self



    def load_png(
        self,
        filename     : str,
        map_width    : float|tuple[float, float],    # 57344. wm / 14336. pa
        height_scale : float = 4096.,
        verbose      : bool = True,
    ) -> Self:
        """Load height map from a png file.
        
        Parameters
        ----------
        filename: str
            path to the file (incl. extentions)
            
        map_width: float or [float, float]
            [x, y]-widths of the map in meters.

        height_scale: float
            max height in meters storable in the data, i.e. the scale of height.
        
        """
        npix_x, npix_y, pixels, meta = png.Reader(filename=filename).read_flat()
        bit_depth : int = meta['bitdepth']
        if verbose and bit_depth != 16:
            print(f"**  Warning: Unexpected {bit_depth = }")
        pixels = np.array(
            pixels).reshape( (npix_x, npix_y)
        ) * (height_scale / 2**bit_depth)
        self.__init__(data=pixels, map_width=map_width)
        
        return self
        
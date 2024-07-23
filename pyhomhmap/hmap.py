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
import matplotlib as mpl
from matplotlib import pyplot as plt




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

    z_seabed : float
        Seabed height in meters.
        Defines the minimum height of self.data

    z_sealvl : float
        Sea level height in meters.
        Defines the height of the ocean.
        Every pixel in self.data below self.z_sealvl is considered in the sea.

        
    see self.__init__() and self.normalize() for details.
    ---------------------------------------------------------------------------
    """

    def __init__(
        self,
        data : npt.ArrayLike = np.zeros((4096, 4096), dtype=np.float64),
        # note: 57344 = 3.5*4*4096 is the CSL2 worldmap width
        map_width : float|tuple[float, float] = 57344.,
        z_seabed : float = 64.,
        z_sealvl : float = 128.,
    ):
        # variables
        
        self.data  : npt.NDArray[np.float64] = np.array(data, dtype=np.float64)
        # note: will normalize float into tuple of floats later
        self._map_width : tuple[float, float] = map_width
        self.z_seabed = z_seabed
        self.z_sealvl = z_sealvl

        
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
        z_seabed     : float = 64.,
        z_sealvl     : float = 128.,
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
            max height in meters storable in the data,
            i.e. the scale of the height.
        
        """
        npix_x, npix_y, pixels, meta = png.Reader(
            filename=filename).read_flat()
        bit_depth : int = meta['bitdepth']
        if verbose and bit_depth != 16:
            print(f"**  Warning: Unexpected {bit_depth = }")
        pixels = np.array(
            pixels).reshape( (npix_x, npix_y)
        ) * (height_scale / 2**bit_depth)
        self.__init__(
            data      = pixels,
            map_width = map_width,
            z_seabed  = z_seabed,
            z_sealvl  = z_sealvl,
        )
        
        return self



    def plot(
        self,
        figsize : tuple[int, int] = (8, 6),
        norm : None|str|mpl.colors.Normalize = 'default',
        z_sealvl: None|float = None,
    ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Return a plot of the data.
        """

        # init
        if z_sealvl is None:
            z_sealvl = self.z_sealvl
        if norm == 'default':
            norm = mpl.colors.Normalize(vmin=self.z_seabed - z_sealvl)

        # plot things
        fig, ax = plt.subplots(figsize=figsize)
        cax  = ax.imshow(self.data - z_sealvl, norm=norm)
        cmap = fig.colorbar(cax)
        ax.set_title(
            "Height Map\n" +
            f"(Seabed: {self.z_seabed:.0f}m above zero for self.data," +
            f"{z_sealvl - self.z_seabed:.0f}m below sea)")
        cmap.set_label('Meters above sea level')

        # update tick labels
        tick_locs = tuple([
            np.linspace(0, self._npix_xy[i], 8, dtype=np.int64)
            for i in range(2)
        ])
        tick_vals = tuple([
            tick_locs[0] / self._npix_xy[0] * self._map_width[0],
            (1. - tick_locs[1] / self._npix_xy[1]) * self._map_width[1],
        ])
        tick_labels = tuple([
            [f'{tick_val:.0f}' for tick_val in tick_vals[i]]
            for i in range(2)
        ])
        tick_labels[0][-1] = f"{    tick_labels[0][-1]}\nSE"
        tick_labels[1][ 0] = f"NW\n{tick_labels[1][ 0]}"
        tick_labels[1][-1] = f"{    tick_labels[1][-1]}\nSW"
        ax.set_xticks(tick_locs[0])
        ax.set_xticklabels(tick_labels[0])
        ax.set_yticks(tick_locs[1])
        ax.set_yticklabels(tick_labels[1])

        return fig, ax
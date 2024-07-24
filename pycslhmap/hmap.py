#!/usr/bin/env python
# coding: utf-8

"""A class to handle height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies

from typing import Self
import numpy as np
from numpy import pi
from numpy import typing as npt
from scipy.ndimage import map_coordinates
import png
import matplotlib as mpl
from matplotlib import pyplot as plt




class HMap:
    """Generic Height Map.

    Instance Variables
    ------------------
    
    Public:
    
    data       : (self._npix_xy)-shaped numpy array (np.float64)
        Contains height map data in meters.
        MUST be:
            1) 2D,
            2) postive in every pixel,
            3) # of pixels per row and per column being a multiple of 8.

    z_seabed : float
        Seabed height in meters. Must be positive.
        Defines the minimum height of self.data

    z_sealvl : float
        Sea level height in meters. Must be positive.
        Defines the height of the ocean.
        Every pixel in self.data below self.z_sealvl is considered in the sea.


    Private:
        
    _ndim      : int = 2
        dimensions. Should always be 2.

    _npix_xy   : tuple[int, int]
        Data shape. i.e. number of pixel in each dim.

    _npix_xy_8 : tuple[int, int]
        == int(self._npix_xy / 8)
        Cached for fast access.
        
    _map_width : tuple[float, float]
        map width in meters (i.e. whatever unit self.data is in).
        It's 57344. for CSL2 world map, and 14336. for CSL2 playable area.
        (Because 57344 = 3.5*4*4096)


        
    see self.__init__() and self.normalize() for details.
    ---------------------------------------------------------------------------
    """

    def __init__(
        self,
        data : npt.ArrayLike = np.zeros((256, 256), dtype=np.float64),
        # note: 57344 = 3.5*4*4096 is the CSL2 worldmap width
        map_width : float|tuple[float, float],
        z_seabed  : float = 64.,
        z_sealvl  : float = 128.,
    ):
        # variables
        
        self.data  : npt.NDArray[np.float64] = np.array(data, dtype=np.float64)
        # note: will normalize float into tuple of floats later
        self._map_width : tuple[float, float] = map_width
        self.z_seabed   : float = z_seabed
        self.z_sealvl   : float = z_sealvl
        
        # vars yet to be set
        self._ndim      : int             = 2
        self._npix_xy   : tuple[int, int] = (0, 0)
        self._npix_xy_8 : tuple[int, int] = (0, 0)

        
        # do things
        self.normalize()


    
    def normalize(self) -> Self:
        """Resetting parameters and do safety checks."""
        
        # variables
        
        # no of pixel: defining map resolution
        self._npix_xy   = self.data.shape
        self._npix_xy_8 = tuple([int(npix_i/8) for npix_i in self._npix_xy])
        self._ndim      = len(self._npix_xy)

        try:
            len(self._map_width)
        except TypeError:
            self._map_width = tuple([
                self._map_width for i in range(self._ndim)])
            
        # safety checks
        assert self._ndim == 2
        assert np.all(np.array(self._npix_xy_8) * 8 == np.array(self._npix_xy))
        assert self.z_seabed >= 0
        assert self.z_sealvl >= 0
        
        return self


    
    def __repr__(self):
        return f"""Height map object:

# Meta data
    Pixels shape  : {self.data.shape = } | {self._npix_xy = }
    Map Widths    : NS/y {self._map_width[0]:.2f},\
    EW/x {self._map_width[1]:.2f},\
    with {len(self._map_width) = }

# Height data insight
    Average height: {np.average(self.data):.2f} +/- {np.std(self.data):.2f}
    Height  range : [{np.min(self.data):.2f}, {np.max(self.data):.2f}]
    Seabed height : {self.z_seabed = :.2f}
    Sea level     : {self.z_sealvl = :.2f}
        """



    def __str__(self):
        return self.__repr__()

    
    
    def load_png(
        self,
        filename     : str,
        map_width    : float|tuple[float, float],    # 57344. wm / 14336. pa
        height_scale : float = 4096.,
        z_seabed     : float = 64.,
        z_sealvl     : float = 128.,
        verbose      : bool  = True,
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
        if verbose:
            print(f"Loading height map data from file '{filename}'.", end='')
        npix_y, npix_x, pixels, meta = png.Reader(
            filename=filename).read_flat()
        bit_depth : int = meta['bitdepth']

            
        if verbose: print(".", end='')
        pixels = np.array(
            pixels).reshape( (npix_x, npix_y)
        ) * (height_scale / 2**bit_depth)
        
        if verbose: print(".", end='')
        self.__init__(
            data      = pixels,
            map_width = map_width,
            z_seabed  = z_seabed,
            z_sealvl  = z_sealvl,
        )
        
        if verbose:
            print(f" Done.\n\n{self.__str__()}")

        if verbose and bit_depth != 16:
            print(f"**  Warning: Unexpected {bit_depth = }")
        
        return self
    


    def load_csl_hmap(
        self,
        map_name     : None|str,
        map_type     : str   = 'worldmap', # 'worldmap' or 'playable'
        dir_path     : None|str = './out/',
        height_scale : float = 4096.,
        z_seabed     : float = 64.,
        z_sealvl     : float = 128.,
        verbose      : bool  = True,
        **kwargs,
    ) -> Self:
        """Loading a Cities Skylines 2 Height Map.
        
        Parameters
        ----------
        map_name : str
            hmap file has the name of
            f"{dir_path}{map_type}_{map_name}.png"
            
        dir_path  : str
            Input directory path (i.e. filepath prefix)

        map_type  : 'worldmap' or 'playable'
        ...
        
        """
        if dir_path is None: dir_path = './'
        if map_name  is None: map_name  = ''
            
        filename = f"{dir_path}{map_type}_{map_name}.png"

        if   map_type in {'worldmap'}:
            map_width = 57344.
        elif map_type in {'playable'}:
            map_width = 14336.

        return self.load_png(
            filename  = filename,
            map_width = map_width,
            height_scale = height_scale,
            z_seabed  = z_seabed,
            z_sealvl  = z_sealvl,
            verbose   = verbose,
            **kwargs,
        )
    


    def save_png(
        self,
        filename     : str,
        bit_depth    : int = 16,
        height_scale : None|float = 4096.,
        compression  : int = 9,    # maximum compression
        verbose      : bool = True,
    ) -> Self:
        """Save to a png file."""

        if height_scale is None:
            height_scale = np.max(self.data) + 1

        self.normalize()

        # safety check
        if verbose:
            nbad_pixels = np.count_nonzero(self.data < self.z_seabed)
            noverflowed = np.count_nonzero(self.data > height_scale)
            if nbad_pixels:
                print(
                    f"\n**  Warning: Data have {nbad_pixels} "
                    + f"({nbad_pixels/self.data.size*100:6.1f} %) "
                    + "bad pixels where data < seabed height.\n"
                    + "These pixels will be replaced by seabed height "
                    + f"{self.z_seabed = }"
                )
            if noverflowed:
                print(
                    f"\n**  Warning: Data have {noverflowed} "
                    + f"({noverflowed/self.data.size*100:6.1f} %) "
                    + f"overflowed pixels where data > height scale "
                    + f"{height_scale = }.\n"
                    + "These pixels will be replaced by maximum height "
                    + f"{(2**bit_depth - 1) / 2**bit_depth * height_scale = }"
                )
            if nbad_pixels or noverflowed:
                print(self)
        
        ans_dtype = np.uint64
        if   bit_depth == 16:
            ans_dtype = np.uint16
        elif bit_depth == 24 or bit_depth == 32:
            ans_dtype = np.uint32
        elif bit_depth == 8:
            ans_dtype = np.uint8
        elif verbose:
            print(
                f"**  Warning: Unknown {bit_depth = }. " +
                "Are you sure it's not supposed to be 16?"
            )

        
        if verbose:
            print(f"Saving height map data to file '{filename}'.", end='')

        
        # convert from float to uint, for saving
        ans = np.where(
            self.data >= height_scale,
            2**bit_depth - 1,    # overflowed
            (np.where(
                self.data < self.z_seabed,
                self.z_seabed,   # bad pixel
                self.data,       # good data
            )) / (height_scale / 2**bit_depth),
        ).astype(ans_dtype)

        if verbose: print(f'.', end='')
        
        with open(filename, 'wb') as f:
            writer = png.Writer(
                width=ans.shape[1], height=ans.shape[0],
                bitdepth=bit_depth, greyscale=True,
                compression=compression,
            )
            if verbose: print(f'.', end='')
            writer.write(f, ans)
            if verbose: print(f" Done.")

        return self



    def save_csl_hmap(
        self,
        map_name     : None|str,
        map_type     : str   = 'worldmap', # 'worldmap' or 'playable'
        dir_path     : None|str = './out/',
        height_scale : None|float = 4096.,
        compression  : int = 9,    # maximum compression
        verbose      : bool = True,
        **kwargs,
    ) -> Self:
        """Save to Cities Skylines 2 compatible Height Map.

        
        Parameters
        ----------
        map_name : str
            hmap file has the name of
            f"{dir_path}{map_type}_{map_name}.png"
            
        dir_path  : str
            Input directory path (i.e. filepath prefix)

        map_type  : 'worldmap' or 'playable'
        ...
        """
        filename = f"{dir_path}{map_type}_{map_name}.png"
        bit_depth = 16
        return self.save_png(
            filename     = filename,
            bit_depth    = 16,
            height_scale = height_scale,
            compression  = compression,
            verbose = verbose,
            **kwargs,
        )


    
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
            f"(Seabed: {self.z_seabed:.0f}m above zero point; " +
            f"{z_sealvl - self.z_seabed:.0f}m below sea)")
        cmap.set_label('Meters above sea level')

        # update tick labels
        tick_locs = tuple([
            np.linspace(0, self._npix_xy[i], 8, dtype=np.int64)
            for i in range(2)
        ])
        tick_vals = tuple([
            (1. - tick_locs[0] / self._npix_xy[0]) * self._map_width[0],
                  tick_locs[1] / self._npix_xy[1]  * self._map_width[1],
        ])
        tick_labels = tuple([
            [f'{tick_val:.0f}' for tick_val in tick_vals[i]]
            for i in range(2)
        ])
        tick_labels[0][ 0] = f"NW\n{tick_labels[0][ 0]}"
        tick_labels[0][-1] = f"{    tick_labels[0][-1]}\nSW"
        tick_labels[1][-1] = f"{    tick_labels[1][-1]}\nSE"
        ax.set_yticks(tick_locs[0])
        ax.set_yticklabels(tick_labels[0])
        ax.set_xticks(tick_locs[1])
        ax.set_xticklabels(tick_labels[1])

        return fig, ax
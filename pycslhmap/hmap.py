#!/usr/bin/env python
# coding: utf-8

"""A class to handle height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies
from .hmap_util import (
    _ind_to_pos, _pos_to_ind_f, _pos_to_ind_d,
    _erode_rainfall_init, _erode_rainfall_evolve,
)
from .hmap_util_cuda import _erode_rainfall_init_cuda

from typing import Self

from numba import jit, prange
import numpy as np
from numpy import typing as npt
import matplotlib as mpl
from matplotlib import pyplot as plt
import png
from scipy.ndimage import map_coordinates



#-----------------------------------------------------------------------------#
#    Classes
#-----------------------------------------------------------------------------#


class HMap:
    """Generic Height Map.


    Instance Variables
    ------------------
    
    Public:
    
    data       : (self.npix_xy)-shaped numpy array (np.float32)
        Contains height map data in meters.
        MUST be:
            1) 2D,
            2) postive in every pixel,

    z_min : float
        Seabed height in meters. Must be positive.
        Defines the minimum height of self.data

    z_sea : float
        Sea level height in meters. Must be positive.
        Defines the height of the ocean.
        Every pixel in self.data below self.z_sea is considered in the sea.
        
    ndim : int = 2
        [Read-only]
        Data Dimensions. Should always be 2.

    npix_xy : tuple[int, int]
        [Read-only]
        Data shape. i.e. number of pixel in each dim.

    map_widxy : tuple[float, float]
        [Read-only]
        Map width in meters (i.e. whatever unit self.data is in).
        It's 57344. for CSL2 world map, and 14336. for CSL2 playable area.
        (Because 57344 = 3.5*4*4096)

    pix_widxy : tuple[float, float]
        Pixel width in meters (i.e. whatever unit self.data is in).
        It's (14, 14) for CSL2 world map, and (3.5, 3.5) for CSL2 playable.


    Private:
    _map_widxy : tuple[float, float]


        
    see self.__init__() and self.normalize() for details.
    ---------------------------------------------------------------------------
    """

    

    #-------------------------------------------------------------------------#
    #    Meta: Initialize
    #-------------------------------------------------------------------------#

    
    def __init__(
        self,
        data : Self|npt.ArrayLike = np.zeros((256, 256), dtype=np.float32),
        map_width : None|float|tuple[float, float] = None,
        pix_width : None|float|tuple[float, float] = 1.,
        z_min: float = 0.,
        z_sea: float = 0.,
        use_data_meta: bool  = True,
    ):
        """Init.

        map_width, pix_width: None | float | tuple[float, float]
            Supply either.
            map_width refers to the width of the whole map;
            pix_width refers to the width of a single pixel.
            if both are provided, pix_width will be ignored.

        use_data_meta : bool
            If true and data is of type Self or HMap,
                will copy the metadata in it
                instead of the supplied parameters.
        """
        
        # init
        if isinstance(data, HMap):
            if use_data_meta:
                map_width = data._map_widxy
                z_min  = data.z_min
                z_sea  = data.z_sea
            data = data.data.copy()
        data = np.array(data, dtype=np.float32)
        map_width = self._get_map_wid_from_pix_wid(
            data.shape, map_width, pix_width)
                
                
        # variables
        
        self.data: npt.NDArray[np.float32] = data
        # note: will normalize float into tuple of floats later
        self._map_widxy: tuple[float, float] = map_width
        self.z_min: float = z_min
        self.z_sea: float = z_sea

        
        # do things
        self.normalize()


    
    @property
    def npix_xy(self) -> tuple[int, int]:
        """Data shape. i.e. number of pixel in each dim."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Data dimensions. Should always be 2."""
        return self.data.ndim

    @property
    def map_widxy(self) -> tuple[float, float]:
        """Map width in meters (i.e. whatever unit self.data is in).
        
        It's 57344 for CSL2 world map, and 14336 for CSL2 playable area.
        (Because 57344 = 3.5*4*4096).
        """
        return self._map_widxy

    @property
    def pix_widxy(self) -> tuple[float, float]:
        """Pixel width in meters (i.e. whatever unit self.data is in).
        
        It's (14, 14) for CSL2 world map, and (3.5, 3.5) for CSL2 playable.
        """
        return tuple([
            map_wid / npix
            for map_wid, npix in zip(self.map_widxy, self.npix_xy)])
        


    @staticmethod
    def _get_map_wid_from_pix_wid(
        data_shape: tuple[int, int],
        map_width : None|float|tuple[float, float] = None,
        pix_width : None|float|tuple[float, float] = 1.,
    ) -> tuple[float, float]:
        """Normalize map width from map_width and/or pix_width.

        Returns map_width as a tuple.
        """
        
        len_map_width : int = 0
        try: len_map_width = len(map_width)  # tuple?
        except TypeError: len_map_width = 0  # No.
            
        len_pix_width : int = 0
        try: len_pix_width = len(pix_width)  # tuple?
        except TypeError: len_pix_width = 0  # No.
            
        # set each element to pixel width if it's None
        if pix_width is None: pix_width = 1.
        map_width_new = np.zeros(len(data_shape), dtype=np.float32)
        for i_mw in range(len(data_shape)):
            mw = None
            if i_mw < len_map_width:
                mw  = map_width[i_mw]
            elif len_map_width == 0:
                mw  = map_width
            if mw is None:
                mw = data_shape[i_mw]
                if i_mw < len_pix_width and pix_width[i_mw] is not None:
                    mw *= pix_width[i_mw]
                elif len_pix_width == 0 and pix_width is not None:
                    mw *= pix_width
            map_width_new[i_mw] = mw
        map_width = tuple(map_width_new)
        
        return map_width


    
    def normalize(self, verbose:bool=True) -> Self:
        """Resetting parameters and do safety checks."""
        
        try: len(self._map_widxy)
        except TypeError:
            self._map_widxy = tuple([
                self._map_widxy for i in range(self.ndim)])
            
        # safety checks
        assert self.ndim  == 2
        assert self.z_sea >= self.z_min
        
        return self

    

    #-------------------------------------------------------------------------#
    #    Meta: magic methods
    #-------------------------------------------------------------------------#

    
    def __repr__(self):
        return f"""Height map object:

# Meta data
    Pixels shape  : {self.data.shape = } | {self.npix_xy = }
    Map Widths    : NS/y {self._map_widxy[0]:.2f},\
    WE/x {self._map_widxy[1]:.2f},\
    with {len(self._map_widxy) = }

# Height data insight
    Average height: {np.average(self.data):.2f} +/- {np.std(self.data):.2f}
    Height  range : [{np.min(self.data):.2f}, {np.max(self.data):.2f}]
    Seabed height : {self.z_min = :.2f}
    Sea level     : {self.z_sea = :.2f}
        """



    def __str__(self):
        return self.__repr__()

    

    #-------------------------------------------------------------------------#
    #    Meta: Copying
    #-------------------------------------------------------------------------#

    
    def copy(self) -> Self:
        """Returns a new copy."""
        return HMap(self)

    def copy_zeros_like(self) -> Self:
        """Returns a new obj with same meta data as self, but zeros in data."""
        # *** optimization not included.
        ans = self.copy()
        ans.data = np.zeros_like(self.data)
        return ans

    def copy_meta_only(self) -> Self:
        """Returns a new obj with same meta data as self, but None in data."""
        # *** optimization not included.
        ans = self.copy()
        ans.data = np.zeros((8, 8), dtype=np.float32)
        return ans
    


    #-------------------------------------------------------------------------#
    #    Meta: Mapping coordinates
    #-------------------------------------------------------------------------#


    def pos_to_ind_f(
        self, pos: tuple[float, float],
    ) -> tuple[float, float]:
        """Mapping position to indexes.
        
        e.g. For a 4096**2 14336m wide map,
            it maps [-7168., 7168.] -> [-0.5, 4095.5]
        """
        return (
            _pos_to_ind_f(pos[0], self._map_widxy[0], self.npix_xy[0]),
            _pos_to_ind_f(pos[1], self._map_widxy[1], self.npix_xy[1]),
        )

    def pos_to_ind_d(
        self, pos: tuple[float, float],
        verbose: bool = True,
    ) -> tuple[int, int]:
        """Mapping position to indexes.
        
        e.g. For a 4096**2 14336m wide map,
            it maps [-7168., 7168.] -> [0, 4095]
        """
        ans = [
            _pos_to_ind_d(pos[i], self._map_widxy[i], self.npix_xy[i])
            for i in self.ndim
        ]
        for i in self.ndim:
            # safety check
            if ans[i] < 0:
                ans[i] = 0
                if verbose:
                    print("*   Warning: HMap.pos_to_ind_d():"
                        + f"Input lies outside the map."
                        + f"Answer reset to {ans[i]}")
            elif ans[i] >= self.npix_xy[i]:
                ans[i] = self.npix_xy[i] - 1
                if verbose:
                    print("*   Warning: HMap.pos_to_ind_d():"
                        + f"Input lies outside the map."
                        + f"Answer reset to {ans[i]}")
        return tuple(ans)

    def ind_to_pos(
        self, ind: tuple[int, int]|tuple[float, float],
    ) -> tuple[float, float]:
        """Mapping indexes to position.
        
        e.g. For a 4096**2 14336m wide map,
            it maps [0, 4095] -> [-7168 + 3.5/2, 7168 - 3.5/2]
        """
        return (
            _ind_to_pos(ind[0], self._map_widxy[0], self.npix_xy[0]),
            _ind_to_pos(ind[1], self._map_widxy[1], self.npix_xy[1]),
        )
        
    
    #-------------------------------------------------------------------------#
    #    I/O
    #-------------------------------------------------------------------------#

    
    def load_png(
        self,
        filename : str,
        map_width: float|tuple[float, float],    # 57344. wm / 14336. pa
        z_min: float = 64.,
        z_sea: float = 128.,
        z_max: float = 4096.,
        dtype: type  = np.float32,
        verbose: bool= True,
    ) -> Self:
        """Load height map from a png file.
        
        Parameters
        ----------
        filename: str
            path to the file (incl. extentions)
            
        map_width: float or [float, float]
            [x, y]-widths of the map in meters.

        z_max: float
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
        ) * (z_max / 2**bit_depth)
        
        if verbose: print(".", end='')

        #self.__init__(pixels, map_width=map_width,
        #              z_min=z_min, z_sea=z_sea,)
        self.data  = np.array(pixels, dtype=dtype)
        self._map_widxy = map_width
        self.z_min = z_min
        self.z_sea = z_sea
        self.normalize()
        
        
        if verbose:
            print(f" Done.\n\n{self.__str__()}")

        if verbose and bit_depth != 16:
            print(f"**  Warning: Unexpected {bit_depth = }")
        
        return self
    


    def save_png(
        self,
        filename : str,
        bit_depth: int = 16,
        z_max: None|float = 4096.,
        compression: int = 9,    # maximum compression
        verbose: bool = True,
    ) -> Self:
        """Save to a png file."""

        if z_max is None:
            z_max = np.max(self.data) + 1

        self.normalize()

        # safety check
        if verbose:
            nbad_pixels = np.count_nonzero(self.data < self.z_min)
            noverflowed = np.count_nonzero(self.data > z_max)
            if nbad_pixels:
                print(
                    f"\n**  Warning: Data have {nbad_pixels} "
                    + f"({nbad_pixels/self.data.size*100:6.1f} %) "
                    + "bad pixels where data < seabed height.\n"
                    + "These pixels will be replaced by seabed height "
                    + f"{self.z_min = }"
                )
            if noverflowed:
                print(
                    f"\n**  Warning: Data have {noverflowed} "
                    + f"({noverflowed/self.data.size*100:6.1f} %) "
                    + f"overflowed pixels where data > height scale "
                    + f"{z_max = }.\n"
                    + "These pixels will be replaced by maximum height "
                    + f"{(2**bit_depth - 1) / 2**bit_depth * z_max = }"
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
            self.data >= z_max,
            2**bit_depth - 1,    # overflowed
            (np.where(
                self.data < self.z_min,
                self.z_min,   # bad pixel
                self.data,       # good data
            )) / (z_max / 2**bit_depth),
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



    #-------------------------------------------------------------------------#
    #    Plotting
    #-------------------------------------------------------------------------#


    def plot(
        self,
        fig : None|mpl.figure.Figure = None,
        ax  : None|mpl.axes.Axes     = None,
        figsize : tuple[int, int] = (8, 6),
        norm    : None|str|mpl.colors.Normalize = 'default',
        add_cbar: bool = True,
        z_sea: None|float = None,
        **kwargs,
    ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Return a plot of the data.

        Parameters
        ----------
        fig, ax:
            if either are None, will generate new figure.
        ...
        """

        # init
        if z_sea is None:
            z_sea = self.z_sea
        if norm == 'default':
            norm = mpl.colors.Normalize(vmin=self.z_min - z_sea)

        # plot things
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cax  = ax.imshow(self.data - z_sea, norm=norm, **kwargs)
        if add_cbar:
            cmap = fig.colorbar(cax)
            cmap.set_label('Meters above sea level')
        ax.set_title(
            "Height Map\n" +
            f"(Seabed: {self.z_min:.0f}m above zero point; " +
            f"{z_sea - self.z_min:.0f}m below sea)")

        # update tick labels
        tick_locs = tuple([
            np.linspace(0, self.npix_xy[i], 9, dtype=np.int64)
            for i in range(2)
        ])
        tick_vals = tuple([
            (0.5 - tick_locs[0] / self.npix_xy[0]      ) * self._map_widxy[0],
            (      tick_locs[1] / self.npix_xy[1] - 0.5) * self._map_widxy[1],
        ])
        tick_labels = tuple([
            [f'{tick_val:.0f}' for tick_val in tick_vals[i]]
            for i in range(2)
        ])
        tick_labels[0][ 0] = f"NW\n{tick_labels[0][ 0]}"
        tick_labels[0][-1] = f"{    tick_labels[0][-1]}\n\n\nSW     "
        tick_labels[1][-1] = f"{    tick_labels[1][-1]}\nSE"
        ax.set_yticks(tick_locs[0])
        ax.set_yticklabels(tick_labels[0])
        ax.set_xticks(tick_locs[1])
        ax.set_xticklabels(tick_labels[1])

        return fig, ax



    def plot_3D(
        self,
        fig : None|mpl.figure.Figure = None,
        ax  : None|mpl.axes.Axes     = None,
        figsize : tuple[int, int] = (8, 6),
        add_cbar: bool = True,
        z_sea: None|float = None,
        rotate_azim_deg: float = 30.,
        **kwargs,
    ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Return a 3D surface plot.
        
        The plot can be rotated using ax.view_init(azim=[angle_in_degrees])

        Parameters
        ----------
        ...
        rotate_azim_deg: float
            rotate the plot w.r.t. z axis
        ...
        """
        if z_sea is None:
            z_sea = self.z_sea
        if fig is None:
            fig = plt.figure(figsize = figsize)
        if ax is None:
            ax = fig.add_subplot(projection='3d')

        # figure out coordinates
        x_coord  = np.linspace(
            _ind_to_pos(
                0,                  self._map_widxy[0], self.npix_xy[0]),
            _ind_to_pos(
                self.npix_xy[0]-1, self._map_widxy[0], self.npix_xy[0]),
            self.npix_xy[0], endpoint=True)
        y_coord  = np.linspace(
            _ind_to_pos(
                0,                  self._map_widxy[1], self.npix_xy[1]),
            _ind_to_pos(
                self.npix_xy[1]-1, self._map_widxy[1], self.npix_xy[1]),
            self.npix_xy[1], endpoint=True)
        xy_coords= np.stack(
            np.meshgrid(x_coord, y_coord, indexing='ij'), axis=0)

        # plot
        cax = ax.plot_surface(
            xy_coords[0], xy_coords[1], self.data - z_sea,
            cmap=mpl.cm.coolwarm,
        )
        if add_cbar:
            cmap = fig.colorbar(cax)
            cmap.set_label('Meters above sea level')
        ax.set_title("Height Map")
        
        # update tick labels
        tick_locs = (
            np.linspace(x_coord[0], x_coord[-1], 9, dtype=np.int64),
            np.linspace(y_coord[0], y_coord[-1], 9, dtype=np.int64),
        )
        tick_vals = tick_locs
        tick_labels = tuple([
            [f'{tick_val:.0f}' for tick_val in tick_vals[i]]
            for i in range(2)
        ])
        tick_labels[0][ 0] = f"{tick_labels[0][ 0]}  W"
        tick_labels[0][-1] = f"{tick_labels[0][-1]}  E"
        tick_labels[1][ 0] = f"{tick_labels[1][ 0]}  N"
        tick_labels[1][-1] = f"{tick_labels[1][-1]}  S"
        ax.set_yticks(tick_locs[0])
        ax.set_yticklabels(tick_labels[0])
        ax.set_xticks(tick_locs[1])
        ax.set_xticklabels(tick_labels[1])
        
        ax.view_init(azim=30)
        return fig, ax



    #-------------------------------------------------------------------------#
    #    Resampling
    #-------------------------------------------------------------------------#


    def resample(
        self,
        new_npix_xy  : tuple[int, int], #= (256, 256),
        nslim_in_ind : None|tuple[float, float] = None, #= (0., 256.),
        welim_in_ind : None|tuple[float, float] = None, #= (0., 256.),
        interp_order : int = 3,
        interp_mode  : str = 'constant',
        z_min: None|float = None,
        verbose: bool = True,
        **kwargs,
    ) -> Self:
        """Return a new obj with resampled HMap.
        
        I.e. with different resolution/bounrdary/etc.
        
        Using scipy.ndimage.map_coordinates().
        Extra keywords will passed onto them.


        Parameters
        ----------
        nslim_in_ind, welim_in_ind: tuple[float, float]
            NS/x and WE/y limits in the index space: (Begin, End)
            End not inclusive.
            e.g. nslim_in_ind = [128, 256], welim_in_ind = [0, 128]
                will select the left bottom part (1/4) of the image
                if the original data is in shape of (256, 256).
                i.e. elements with index of [128:256, 0:128]
            Do NOT Supply negative number (won't work).
            if None, will use (0, self.npix_xy[0]) or (0, self.npix_xy[1]).

        new_npix_xy: tuple[int, int]
            new HMap resolution in x and y.
        
        interp_order: int
            The order of the spline interpolation,
            used by scipy.ndimage.map_coordinates().

        z_min: None|float
            min val of the hmap. Used when extrapolating.
            if None, will use the value stored in self.
        """

        # init
        if z_min is None: z_min = self.z_min
        if nslim_in_ind is None: nslim_in_ind = (0, self.npix_xy[0])
        if welim_in_ind is None: welim_in_ind = (0, self.npix_xy[1])

        nslim_npix = abs(nslim_in_ind[1] - nslim_in_ind[0])    # x width
        welim_npix = abs(welim_in_ind[1] - welim_in_ind[0])    # y width
        # get coord
        edges_in_ind = [
            nslim_npix / new_npix_xy[0] / 2.,
            welim_npix / new_npix_xy[1] / 2.,
        ]
        if nslim_in_ind[0] > nslim_in_ind[1]:
            edges_in_ind[0] *= -1
        if welim_in_ind[0] > welim_in_ind[1]:
            edges_in_ind[1] *= -1
        x_coord  = np.linspace(
            nslim_in_ind[0] - 0.5 + edges_in_ind[0],
            nslim_in_ind[1] - 0.5 - edges_in_ind[0],
            new_npix_xy[0], endpoint=True)
        y_coord  = np.linspace(
            welim_in_ind[0] - 0.5 + edges_in_ind[1],
            welim_in_ind[1] - 0.5 - edges_in_ind[1],
            new_npix_xy[1], endpoint=True)
        
        # do interp
        xy_coords= np.stack(
            np.meshgrid(x_coord, y_coord, indexing='ij'), axis=0)
        ans = HMap(self.copy_meta_only())
        ans.data = map_coordinates(
            self.data, xy_coords,
            order=interp_order, mode=interp_mode, cval=z_min, **kwargs)
        ans._map_widxy = (
            self._map_widxy[0] * nslim_npix / self.npix_xy[0],
            self._map_widxy[1] * welim_npix / self.npix_xy[1],
        )
        ans.normalize()
        return ans


    
    def rescale(
        self,
        new_scale    : float|tuple[float, float, float],
        new_center_ip: tuple[float, float] = (0., 0.),
        interp_order : int = 3,
        interp_mode  : str = 'nearest',
        z_min_new: None|float = None,
        z_sea_new: None|float = None,
        verbose: bool = True,
        **kwargs,
    ) -> Self:
        """Re-scale the HMap such that sea level stay the same.

        Note:
            z < self.z_sea will scale differently regardless of new scale,
            so that self.z_sea -> z_sea_new and self.z_min -> z_min_new
        
        Parameters
        ----------
        new_scale: float|tuple[float, float, float]
            if tuple, provide scales in order of [z, NS/x, WE/y]
            Zoomed out level (New : Old = 1 : ?)
            i.e. the new 1 meter is the old ? meter
            
        new_center_ip: tuple[float, float]
            The center of the new hmap is at the old hmap coordinates of..?
            ip = in_pos (i.e. in physical space,
            i.e. in meters instead of indexes)
        
        interp_order: int
            The order of the spline interpolation,
            used by scipy.ndimage.map_coordinates().

        z_min_new: None|float
            Elevate the HMap to the New seabed height.

        z_sea_new: New sea level.
            Heights will be increased so old sea level meets the new one.
        """

        # normalize input parameters
        if z_min_new is None: z_min_new = self.z_min
        if z_sea_new is None: z_sea_new = self.z_sea
        try: len(new_scale)
        except TypeError: new_scale = [new_scale]
        if len(new_scale) < self.ndim+1:
            new_scale = tuple([
                new_scale[i] if i < len(new_scale) else new_scale[-1]
                for i in range(self.ndim+1)])

        # ii = in_ind (i.e. in index space)
        # + 0.5 because how self.resample assumes things
        lim_center_ii = np.array(self.pos_to_ind_f(new_center_ip)) + 0.5
        lim_half_wid_ii = np.array([
            npix / 2. * scale
            for npix, scale in zip(self.npix_xy, new_scale[1:])])
        lim_left_ii  = lim_center_ii - lim_half_wid_ii
        lim_right_ii = lim_center_ii + lim_half_wid_ii
        
        res = self.resample(
            new_npix_xy  = self.npix_xy,
            nslim_in_ind = (lim_left_ii[0], lim_right_ii[0]),
            welim_in_ind = (lim_left_ii[1], lim_right_ii[1]),
            interp_order = interp_order,
            interp_mode  = interp_mode,
            z_min        = z_min_new,
            verbose      = verbose,
            **kwargs,
        )
        ans = self.copy()   # force using self's meta data
        # normalize
        if np.isclose(self.z_sea, self.z_min):
            ans.data = (res.data - self.z_min)/new_scale[0] + z_sea_new
        else:
            # bs_scale_inv: below sea scale inversed
            bs_scale_inv = (z_sea_new - z_min_new) / (self.z_sea - self.z_min)
            ans.data = np.where(
                res.data < z_sea_new,
                (res.data - self.z_min)*bs_scale_inv + z_min_new,
                (res.data - self.z_sea)/new_scale[0] + z_sea_new,
            )
        ans.data  = np.where(ans.data < z_min_new, z_min_new, ans.data)
        ans.z_sea = z_sea_new
        ans.z_min = z_min_new
        ans.normalize()
        
        return ans
    
    
    
    #-------------------------------------------------------------------------#
    #    Erosion
    #-------------------------------------------------------------------------#


    def erode(self, **kwargs) -> Self:
        """Do Erosion!
        
        Inspired by Sebastian Lague's Erosion code (
            See https://github.com/SebLague/Hydraulic-Erosion
        ) who in turn was inspired by Hans Theobald Beyer's Bachelor's Thesis
            'Implementation of a method for hydraulic erosion'.
        """
        
        raise NotImplementedError
        
    

#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
#!/usr/bin/env python
# coding: utf-8

"""A class to handle height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies

from typing import Self
import random

from numba import jit, prange
import numpy as np
from numpy import pi
from numpy import typing as npt
import matplotlib as mpl
from matplotlib import pyplot as plt
import png
from scipy.ndimage import map_coordinates



#-----------------------------------------------------------------------------#
#    Functions
#-----------------------------------------------------------------------------#


@jit(nopython=True)
def _norm(
    v_x: float, v_y: float, v_z: float,
) -> float:
    """Get the norm of a vector."""
    return (v_x**2 + v_y**2 + v_z**2)**0.5

    

@jit(nopython=True)
def _hat(
    v_x: float, v_y: float, v_z: float,
    factor: float = 1.0,
) -> tuple[float, float, float]:
    """Get the directions of a vector as a new unit vector.

    If both input are zero, will return zero vector.
    """
    v = _norm(v_x, v_y, v_z) #(v_x**2 + v_y**2 + v_z**2)**0.5
    if v:
        return v_x/v*factor, v_y/v*factor, v_z/v*factor
    else:
        return 0., 0., 0.
    


@jit(nopython=True)
def _pos_to_ind_f(
    pos: float,
    map_wid: float,
    npix: int,
) -> float:
    """Mapping position to indexes.
    
    e.g. For a 4096**2 14336m wide map,
        it maps [-7168., 7168.] -> [-0.5, 4095.5]
    """
    return (0.5 + pos / map_wid) * npix - 0.5



@jit(nopython=True)
def _pos_to_ind_d(
    pos: float,
    map_wid: float,
    npix: int,
) -> int:
    """Mapping position to indexes.
    
    e.g. For a 4096**2 14336m wide map,
        it maps [-7168., 7168.] -> [0, 4095]

    Warning: No safety checks.
    """
    #return (0.5 + pos / map_wid) * npix - 0.5    # actual
    # note: int maps -0.? to 0 as well,
    #  so we needn't be concerned with accidentally mapping to -1
    ans = int((0.5 + pos / map_wid) * npix)
    if ans == npix: ans = npix - 1    # in case pos == map_wid/2. exactly
    return ans



@jit(nopython=True)
def _ind_to_pos(
    ind: int|float|npt.NDArray[int]|npt.NDArray[float],
    map_wid: float,
    npix: int,
) -> float|npt.NDArray[float]:
    """Mapping indexes to position.
    
    e.g. For a 4096**2 14336m wide map,
        it maps [0, 4095] -> [-7168. + 3.5/2, 7168. - 3.5/2]
    """
    #return (-map_wid + map_wid/npix)/2. + map_wid/npix*ind
    return (-0.5 + (0.5 + ind)/npix) * map_wid



@jit(nopython=True)
def _get_z_and_dz(
    pos_x: float,
    pos_y: float,
    data : npt.NDArray[np.float64],
    map_widxy: tuple[int, int],
) -> tuple[float, float, float]:
    """Get height and gradients at specified position in physical units.

    pos_xy in physical units, within range of [-map_widxy/2., map_widxy/2.]

    Returns: z, dz_dx, dz_dy
    Note that dz_dx is $ \\frac{\\partial z}{\\partial x} $
        i.e. partial derivative
    """

    # init
    map_wid_x, map_wid_y = map_widxy
    npix_x, npix_y = data.shape
    assert npix_x >= 2 and npix_y >= 2    # otherwise interpolation will break
    # coord in unit of indexes
    ind_x = _pos_to_ind_f(pos_x, map_wid_x, npix_x)
    ind_y = _pos_to_ind_f(pos_y, map_wid_y, npix_y)
    # closest indexes
    i_x_m = int(ind_x)    # m for minus
    i_y_m = int(ind_y)
    # allowing extrapolation
    if i_x_m < 0:
        i_x_m = 0
    elif i_x_m >= npix_x - 1:
        i_x_m = npix_x - 2
    if i_y_m < 0:
        i_y_m = 0
    elif i_y_m >= npix_y - 1:
        i_y_m = npix_y - 2
    # distance frac
    tx = ind_x - i_x_m
    ty = ind_y - i_y_m


    # 2D linear interpolate to find z
    z = (
        (  1.-tx) * (1.-ty) * data[i_x_m,   i_y_m  ]
        +     tx  * (1.-ty) * data[i_x_m+1, i_y_m  ]
        + (1.-tx) *     ty  * data[i_x_m,   i_y_m+1]
        +     tx  *     ty  * data[i_x_m+1, i_y_m+1]
    )

    # estimate the gradient with linear interpolation along the other axis
    #    not the most scientifically accurate but it will do
    dz_dx = (
        (1.-ty) * (data[i_x_m+1, i_y_m  ] - data[i_x_m,   i_y_m  ])
        +   ty  * (data[i_x_m+1, i_y_m+1] - data[i_x_m,   i_y_m+1])
    ) / (map_wid_x / npix_x)

    dz_dy = (
        (1.-tx) * (data[i_x_m  , i_y_m+1] - data[i_x_m,   i_y_m  ])
        +   tx  * (data[i_x_m+1, i_y_m+1] - data[i_x_m+1, i_y_m  ])
    ) / (map_wid_y / npix_y)
    
    return z, dz_dx, dz_dy



@jit(nopython=True)
def _erode_raindrop_once(
    data: npt.NDArray[np.float64],
    map_widxy: tuple[int, int],
    z_seabed: float,
    z_sealvl: float,
    max_steps_per_drop: int   = 65536,
    energy_conserv_fac: float = 0.,
    friction_coeff    : float = .01,
    g : float = 9.8,
):
    """Erosion.

    Best keep data in same resolution in both x and y.
    
    Parameters
    ----------
    ...
    max_steps_per_drop: int
        maximum steps (i.e. distance) the rain drop is allowed to travel.
        
    energy_conserv_fac : float
        Switch between energy-conserved mode and momentum-conserved mode
            when the raindrop changes its direction.
         0. for momentum conservation, 1. for energy conservation.
         
    friction_coeff : float
        friction coefficient. should be within 0. <= friction_coeff < 1.
            Physics says we cannot assign friction coeff to a liquid.
            Well let's pretend we can because it's all very approximate.
            
    g : gravitational accceleration in m/s^2
    ...
    """

    raise NotImplementedError
    
    paths = np.zeros_like(data, dtype=np.int64)


    if True:

        
        # step 1: Generate a raindrop at random locations
        
        npix_x, npix_y = data.shape
        map_wid_x, map_wid_y = map_widxy
        # boundaries- [-map_wid_x_b, map_wid_x_b]
        map_wid_x_b = map_wid_x * (0.5 - 0.5/npix_x)
        map_wid_y_b = map_wid_y * (0.5 - 0.5/npix_y)
        #assert map_wid_x == map_wid_y
        
        # i for in index unit; no i for in physical unit
        # distance per step (regardless of velocity)
        ds_xy : float = (map_wid_x/npix_x + map_wid_y/npix_y)/2.
        ds    : float = ds_xy
        dt    : float = 0.
        # position
        p_x : float = random.uniform(-map_wid_x_b, map_wid_x_b)
        p_y : float = random.uniform(-map_wid_y_b, map_wid_y_b)
        x_i : int = _pos_to_ind_d(p_x, map_wid_x, npix_x)
        y_i : int = _pos_to_ind_d(p_y, map_wid_y, npix_y)
        # direction (i.e. each step)
        d_x : float = random.uniform(-ds_xy, ds_xy)
        d_y : float = (ds_xy**2 - d_x**2)**0.5 * (random.randint(0, 1)*2 - 1)
        # velocity
        v   : float = 0.
        v_x : float = 0.
        v_y : float = 0.
        v_z : float = 0.
        # sediment content
        c   : float = 0.

        # debug
        a   : float = 0.
    
        for s in range(max_steps_per_drop):
        
            # step 2: Simulate physics for the droplet to slide downhill
    
            # init
            x_i = _pos_to_ind_d(p_x, map_wid_x, npix_x)
            y_i = _pos_to_ind_d(p_y, map_wid_y, npix_y)
            paths[x_i, y_i] += 1
            p_z, dz_dx, dz_dy = _get_z_and_dz(p_x, p_y, data, map_widxy)
            # step / velocity direction
            d_x, d_y, _ = _hat(d_x, d_y, 0., ds_xy)
            d_z = dz_dx * d_x + dz_dy * d_y
            ds  = _norm(d_x, d_y, d_z)
            # gradient direction of z - b for nabla
            b_x, b_y, _ = _hat(dz_dx, dz_dy, 0.)
            b_z = dz_dx * b_x + dz_dy * b_y
            b_x, b_y, b_z = _hat(b_x, b_y, b_z)
            # velocity
            if energy_conserv_fac:
                # conserve energy somewhat
                v_old_x, v_old_y, v_old_z = v_x, v_y, v_z
                v_x, v_y, v_z = _hat(d_x, d_y, d_z, v)
                v_x = (1.- energy_conserv_fac)*v_old_x + energy_conserv_fac*v_x
                v_y = (1.- energy_conserv_fac)*v_old_y + energy_conserv_fac*v_y
                v_z = (1.- energy_conserv_fac)*v_old_z + energy_conserv_fac*v_z
            # accelerations
            g_x, g_y, g_z = _hat(b_x, b_y, b_z, -b_z*g)
            # note: b_z/_norm(b_x, b_y, b_z) because
            #    the other two get cancelled out by ground's support force
            # friction (v**2/ds = v/dt)
            #    i.e. friction does not move things on its own
            f = min(friction_coeff * g, v**2/ds)    
            a_x = g_x - f * d_x / ds
            a_y = g_y - f * d_y / ds
            a_z = g_z - f * d_z / ds
            v_xy = _norm(v_x, v_y, 0.)
            a_xy = _norm(a_x, a_y, 0.)
            # get time step
            if a_xy:
                dt  = ((v_xy**2 + 2 * a_xy * ds_xy)**0.5 - v_xy) / a_xy
            elif v_xy:
                dt  = ds / v
            else:
                # droplet stuck - terminate
                break
            # update position / direction
            p_x += v_x * dt + a_x * dt**2 / 2
            p_y += v_y * dt + a_y * dt**2 / 2
            v_x += a_x * dt
            v_y += a_y * dt
            v_z += a_z * dt
            v    = _norm(v_x, v_y, v_z)

            # check
            if (   p_x <= -map_wid_x_b
                or p_x >=  map_wid_x_b
                or p_y <= -map_wid_y_b
                or p_y >=  map_wid_y_b):
                # droplet out of bounds- terminate
                break


    
    return paths, x_i, y_i, s, v, a, ds, dt, v_x * dt + a_x * dt**2 / 2, v_y * dt + a_y * dt**2 / 2



#-----------------------------------------------------------------------------#
#    Classes
#-----------------------------------------------------------------------------#



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
        
    _map_widxy : tuple[float, float]
        map width in meters (i.e. whatever unit self.data is in).
        It's 57344. for CSL2 world map, and 14336. for CSL2 playable area.
        (Because 57344 = 3.5*4*4096)


        
    see self.__init__() and self.normalize() for details.
    ---------------------------------------------------------------------------
    """

    def __init__(
        self,
        data : Self|npt.ArrayLike = np.zeros((256, 256), dtype=np.float64),
        map_width : float|tuple[float, float] = 3584.,   # = 14*256
        z_seabed  : float = 64.,
        z_sealvl  : float = 128.,
        use_data_meta: bool  = True,
    ):
        """Init.

        use_data_meta : bool
            If true and data is of type Self or HMap,
                will copy the metadata in it
                instead of the supplied parameters.
        """
        
        # init
        if isinstance(data, HMap):
            if use_data_meta:
                map_width = data._map_widxy
                z_seabed  = data.z_seabed
                z_sealvl  = data.z_sealvl
            data = data.data.copy()
            
                
        # variables
        
        self.data  : npt.NDArray[np.float64] = np.array(data, dtype=np.float64)
        # note: will normalize float into tuple of floats later
        self._map_widxy : tuple[float, float] = map_width
        self.z_seabed   : float = z_seabed
        self.z_sealvl   : float = z_sealvl
        
        # vars yet to be set
        self._ndim      : int             = 2
        self._npix_xy   : tuple[int, int] = (0, 0)

        
        # do things
        self.normalize()


    
    def normalize(self, verbose:bool=True) -> Self:
        """Resetting parameters and do safety checks."""
        
        # variables
        
        # no of pixel: defining map resolution
        self._npix_xy   = self.data.shape
        self._ndim      = len(self._npix_xy)

        try:
            len(self._map_widxy)
        except TypeError:
            self._map_widxy = tuple([
                self._map_widxy for i in range(self._ndim)])
            
        # safety checks
        assert self._ndim == 2
        assert self.z_seabed >= 0
        assert self.z_sealvl >= 0
        
        return self


    
    def __repr__(self):
        return f"""Height map object:

# Meta data
    Pixels shape  : {self.data.shape = } | {self._npix_xy = }
    Map Widths    : NS/y {self._map_widxy[0]:.2f},\
    WE/x {self._map_widxy[1]:.2f},\
    with {len(self._map_widxy) = }

# Height data insight
    Average height: {np.average(self.data):.2f} +/- {np.std(self.data):.2f}
    Height  range : [{np.min(self.data):.2f}, {np.max(self.data):.2f}]
    Seabed height : {self.z_seabed = :.2f}
    Sea level     : {self.z_sealvl = :.2f}
        """



    def __str__(self):
        return self.__repr__()

    

    #-------------------------------------------------------------------------#
    #    Meta
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
        ans.data = np.zeros((8, 8), dtype=np.float64)
        return ans
    

    
    #-------------------------------------------------------------------------#
    #    I/O
    #-------------------------------------------------------------------------#

    
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

        #self.__init__(pixels, map_width=map_width,
        #              z_seabed=z_seabed, z_sealvl=z_sealvl,)
        self.data = np.array(pixels, dtype=np.float64)
        self._map_widxy = map_width
        self.z_seabed   = z_seabed
        self.z_sealvl   = z_sealvl
        self.normalize()
        
        
        if verbose:
            print(f" Done.\n\n{self.__str__()}")

        if verbose and bit_depth != 16:
            print(f"**  Warning: Unexpected {bit_depth = }")
        
        return self
    


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
        z_sealvl: None|float = None,
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
        if z_sealvl is None:
            z_sealvl = self.z_sealvl
        if norm == 'default':
            norm = mpl.colors.Normalize(vmin=self.z_seabed - z_sealvl)

        # plot things
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cax  = ax.imshow(self.data - z_sealvl, norm=norm, **kwargs)
        if add_cbar:
            cmap = fig.colorbar(cax)
            cmap.set_label('Meters above sea level')
        ax.set_title(
            "Height Map\n" +
            f"(Seabed: {self.z_seabed:.0f}m above zero point; " +
            f"{z_sealvl - self.z_seabed:.0f}m below sea)")

        # update tick labels
        tick_locs = tuple([
            np.linspace(0, self._npix_xy[i], 9, dtype=np.int64)
            for i in range(2)
        ])
        tick_vals = tuple([
            (0.5 - tick_locs[0] / self._npix_xy[0]      ) * self._map_widxy[0],
            (      tick_locs[1] / self._npix_xy[1] - 0.5) * self._map_widxy[1],
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
        z_sealvl: None|float = None,
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
        if z_sealvl is None:
            z_sealvl = self.z_sealvl
        if fig is None:
            fig = plt.figure(figsize = figsize)
        if ax is None:
            ax = fig.add_subplot(projection='3d')

        # figure out coordinates
        x_coord  = np.linspace(
            _ind_to_pos(
                0,                  self._map_widxy[0], self._npix_xy[0]),
            _ind_to_pos(
                self._npix_xy[0]-1, self._map_widxy[0], self._npix_xy[0]),
            self._npix_xy[0], endpoint=True)
        y_coord  = np.linspace(
            _ind_to_pos(
                0,                  self._map_widxy[1], self._npix_xy[1]),
            _ind_to_pos(
                self._npix_xy[1]-1, self._map_widxy[1], self._npix_xy[1]),
            self._npix_xy[1], endpoint=True)
        xy_coords= np.stack(
            np.meshgrid(x_coord, y_coord, indexing='ij'), axis=0)

        # plot
        cax = ax.plot_surface(
            xy_coords[0], xy_coords[1], self.data - z_sealvl,
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
        z_seabed     : None|float = None,
        verbose      : bool = True,
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
            if None, will use (0, self._npix_xy[0]) or (0, self._npix_xy[1]).

        new_npix_xy: tuple[int, int]
            new HMap resolution in x and y.
        
        interp_order: int
            The order of the spline interpolation,
            used by scipy.ndimage.map_coordinates().

        z_seabed: None|float
            min val of the hmap. Used when extrapolating.
            if None, will use the value stored in self.
        """

        # init
        if z_seabed     is None: z_seabed     = self.z_seabed
        if nslim_in_ind is None: nslim_in_ind = (0, self._npix_xy[0])
        if welim_in_ind is None: welim_in_ind = (0, self._npix_xy[1])

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
            self.data, xy_coords, order=interp_order, cval=z_seabed, **kwargs)
        ans._map_widxy = (
            self._map_widxy[0] * nslim_npix / self._npix_xy[0],
            self._map_widxy[1] * welim_npix / self._npix_xy[1],
        )
        ans.normalize()
        return ans
        
    

    #-------------------------------------------------------------------------#
    #    Erosion
    #-------------------------------------------------------------------------#


    def erode(self, **kwargs) -> Self:
        """Do Erosion!
        
        Inspired by Sebastian Lague's Erosion code (
            See https://github.com/SebLague/Hydraulic-Erosion
        ) who was in turn inspired by Hans Theobald Beyer's Bachelor's Thesis
            'Implementation of a method for hydraulic erosion'.
        """
        
        raise NotImplementedError
        
    

    #-------------------------------------------------------------------------#
    #    End
    #-------------------------------------------------------------------------#
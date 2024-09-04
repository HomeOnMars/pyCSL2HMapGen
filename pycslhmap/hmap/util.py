#!/usr/bin/env python
# coding: utf-8

"""Functions for handling height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies
#from ..util import VerboseType

from numba import jit
import numpy as np
from numpy import typing as npt



#-----------------------------------------------------------------------------#
#    Short functions for vectors
#-----------------------------------------------------------------------------#


@jit(nopython=True, fastmath=True)
def _minabs(a, b):
    """Return the elem with minimum abs value."""
    return a if abs(a) < abs(b) else b



@jit(nopython=True, fastmath=True)
def _norm(
    v_x: float, v_y: float, v_z: float = 0.0,
) -> float:
    """Get the norm of a vector."""
    return (v_x**2 + v_y**2 + v_z**2)**0.5

    

@jit(nopython=True, fastmath=True)
def _hat(
    v_x: float, v_y: float, v_z: float,
    factor: float = 1.0,
) -> tuple[float, float, float]:
    """Get the directions of a vector as a new unit vector.

    If all v input are zero, will return zero vector.
    ---------------------------------------------------------------------------
    """
    v = _norm(v_x, v_y, v_z) #(v_x**2 + v_y**2 + v_z**2)**0.5
    if v:
        return v_x/v*factor, v_y/v*factor, v_z/v*factor
    else:
        return 0., 0., 0.
    


@jit(nopython=True, fastmath=True)
def _pos_to_ind_f(
    pos: float,
    map_wid: float,
    npix: int,
) -> float:
    """Mapping position to indexes.
    
    e.g. For a 4096**2 14336m wide map,
        it maps [-7168., 7168.] -> [-0.5, 4095.5]
    ---------------------------------------------------------------------------
    """
    return (0.5 + pos / map_wid) * npix - 0.5



@jit(nopython=True, fastmath=True)
def _pos_to_ind_d(
    pos: float,
    map_wid: float,
    npix: int,
) -> int:
    """Mapping position to indexes.
    
    e.g. For a 4096**2 14336m wide map,
        it maps [-7168., 7168.] -> [0, 4095]

    Warning: No safety checks.
    ---------------------------------------------------------------------------
    """
    #return (0.5 + pos / map_wid) * npix - 0.5    # actual
    # note: int maps -0.? to 0 as well,
    #  so we needn't be concerned with accidentally mapping to -1
    ans = int((0.5 + pos / map_wid) * npix)
    if ans == npix: ans = npix - 1    # in case pos == map_wid/2. exactly
    return ans



@jit(nopython=True, fastmath=True)
def _ind_to_pos(
    ind: int|float|npt.NDArray[int]|npt.NDArray[float],
    map_wid: float,
    npix: int,
) -> float|npt.NDArray[float]:
    """Mapping indexes to position.
    
    e.g. For a 4096**2 14336m wide map,
        it maps [0, 4095] -> [-7168 + 3.5/2, 7168 - 3.5/2]
    ---------------------------------------------------------------------------
    """
    #return (-map_wid + map_wid/npix)/2. + map_wid/npix*ind
    return (-0.5 + (0.5 + ind)/npix) * map_wid



@jit(nopython=True, fastmath=True)
def _get_z_and_dz(
    pos_x: float,
    pos_y: float,
    data : npt.NDArray[np.float32],
    map_widxy: tuple[int, int],
) -> tuple[float, float, float]:
    """Get height and gradients at specified position in physical units.

    Map shape must >= 2x2.
    pos_xy in physical units, within range of [-map_widxy/2., map_widxy/2.]

    Returns: z, dz_dx, dz_dy
    Note that dz_dx is $ \\frac{\\partial z}{\\partial x} $
        i.e. partial derivative
    ---------------------------------------------------------------------------
    """

    # init
    map_wid_x, map_wid_y = map_widxy
    npix_x, npix_y = data.shape
    #assert npix_x >= 2 and npix_y >= 2    # otherwise interpolation will break
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



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
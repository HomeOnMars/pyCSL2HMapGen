#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CPU version of .cuda codes. Incomplete. No longer supported.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (built-in)
from typing import Self, Callable

# imports (3rd party)
from numba import jit, prange
import numpy as np
from numpy import typing as npt

# imports (my libs)
from .cuda import (
    CAN_CUDA,
)
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)


#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Init
#-----------------------------------------------------------------------------#


@jit(nopython=True, fastmath=True, parallel=True)
def erode_rainfall_init_sub_nbjit(
    soils: npt.NDArray[np.float32],
    edges: npt.NDArray[np.float32],
    z_range: np.float32,
) -> tuple[npt.NDArray[np.float32], int]:
    """Numba version of the sub process for rainfall erosion init.

    Filling the basins.
    
    Parameters
    ----------
    ...
    z_range: np.float32
        z_range == z_max - z_min

    ---------------------------------------------------------------------------
    """
    
    npix_x, npix_y = soils.shape[0]-2, soils.shape[1]-2
    
    # - fill basins -
    # (lakes / sea / whatev)
    zs = edges.copy()
    zs[1:-1, 1:-1] = z_range    # first fill, then drain
    # note: zs' edge elems are fixed
    n_cycles : int = 0    # debug
    still_working_on_it: bool = True
    while still_working_on_it:
        n_cycles += 1
        zs_new = np.empty_like(zs)    # *** potential for optimization?
        for i in prange(1, npix_x+1):
            for j in range(1, npix_y+1):
                z_new = min(
                    zs[i-1, j],
                    zs[i+1, j],
                    zs[i, j-1],
                    zs[i, j+1],
                )
                zs_new[i, j] = max(z_new, soils[i, j])
        still_working_on_it = np.any(zs_new[1:-1, 1:-1] < zs[1:-1, 1:-1])
        zs[1:-1, 1:-1] = zs_new[1:-1, 1:-1]
    return zs, n_cycles



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
#!/usr/bin/env python
# coding: utf-8

"""GPU-accelerated functions.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies

from typing import Self

from numba import jit, prange, cuda
import numpy as np
from numpy import typing as npt

try:
    has_cuda_gpu = cuda.detect()
except Exception as e:
    print(f"**  Warning: Error during initializing GPU acceleration:\n\t{e}")
    has_cuda_gpu = False

if has_cuda_gpu:
    print("\nNote   :\n\t"
          + "Cuda supported devices found."
          + " GPU-accelerated functions available.\n")
else:
    print("\n*   Warning:\n\t"
          + "NO Cuda supported devices found."
          + " GPU-accelerated functions UNAVAILABLE.\n")



#-----------------------------------------------------------------------------#
#    Erosion: Rainfall
#-----------------------------------------------------------------------------#


@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_init_cuda(
    data : npt.NDArray[np.float64],    # ground level
    spawners: npt.NDArray[np.float64],
    pix_widxy: tuple[float, float],
    z_min: float,
    z_sea: float,
    z_max: float,
    sed_cap_fac: float = 1.0,
    sed_initial: float = 0.0,
    erosion_eff: float = 1.0,
):
    """Initialization for Rainfall erosion.
    
    data: (npix_x, npix_y)-shaped numpy array
        initial height.

    spawners: (npix_x, npix_y)-shaped numpy array
        Constant level water spawners height (incl. ground)
        use np.zeros_like(data) as default input.

    z_sea: float
        Sea level.
        *** Warning: z_sea = 0 will disable sea level mechanics ***

    Returns
    -------
    ...
    edges: (npix_x+2, npix_y+2)-shaped numpy array
        Constant river source. acts as a spawner.
        By default, will init any zero elements as sea levels at edges.
    ... 
    """

    raise NotImplementedError("Cuda version of this func not yet complete.")
    

    npix_x, npix_y = data.shape

    # - init ans arrays -
    
    # adding an edge
    soils = np.zeros((npix_x+2, npix_y+2))
    #aquas = np.zeros_like(soils)

    # init soils
    soils[1:-1, 1:-1] = data
    soils[ 0,   1:-1] = data[ 0]
    soils[-1,   1:-1] = data[-1]
    soils[1:-1,    0] = data[:, 0]
    soils[1:-1,   -1] = data[:,-1]
    soils[ 0, 0] = min(soils[ 0, 1], soils[ 1, 0])
    soils[-1, 0] = min(soils[-1, 1], soils[-2, 0])
    soils[ 0,-1] = min(soils[ 0,-2], soils[ 1,-1])
    soils[-1,-1] = min(soils[-1,-2], soils[-2,-1])
    soils -= z_min
    soils = np.where(soils <= 0., 0., soils)

    # init edges (i.e. const lvl water spawners)
    z_edge = z_sea - z_min
    edges = np.empty_like(soils)
    edges[1:-1, 1:-1] = spawners
    edges[ 0, 1:-1] = np.where(spawners[   0], spawners[   0], z_edge)
    edges[-1, 1:-1] = np.where(spawners[  -1], spawners[  -1], z_edge)
    edges[1:-1,  0] = np.where(spawners[:, 0], spawners[:, 0], z_edge)
    edges[1:-1, -1] = np.where(spawners[:,-1], spawners[:,-1], z_edge)
    edges[0, 0], edges[-1, 0] = z_edge, z_edge
    edges[0,-1], edges[-1,-1] = z_edge, z_edge

    # init aquas
    aquas = np.where(edges > soils, edges - soils, 0.)
    
    # - fill basins -
    # (lakes / sea / whatev)
    zs = np.full_like(soils, z_max) 
    zs = aquas + soils    # actual heights (water + ground)
    zs[1:-1, 1:-1] = z_max - z_min    # first fill, then drain
    # note: zs' edge elems are fixed
    n_cycles = 0    # debug
    still_working_on_it: bool = True
    while still_working_on_it:
        still_working_on_it = False
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

    aquas[1:-1, 1:-1] = (zs - soils)[1:-1, 1:-1]

    ekins = np.zeros_like(soils)
    sedis = np.zeros_like(soils) # is zero because speed is zero
    
    return soils, aquas, ekins, sedis, edges, n_cycles



@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_get_capas_cuda(
    zs   : npt.NDArray[np.float64],
    aquas: npt.NDArray[np.float64],
    ekins: npt.NDArray[np.float64],
    pix_widxy: tuple[float, float],
    sed_cap_fac: float = 1.0,
    v_cap: float = 16.,
) -> npt.NDArray[np.float64]:
    """Get sediment capacity.

    Parameters
    ----------
    ...
    sed_cap_fac : float
        Sediment capacity factor of the river.
        Limits the maximum of the sediemnt capacity.
        
    v_cap: float
        Characteristic velocity for sediment capacity calculations, in m/s.
        Used to regulate the velocity in capas calc,
        So its influence flatten out when v is high.
    """

    raise NotImplementedError("Cuda version of this func not yet complete.")
    
    npix_x, npix_y = zs.shape[0]-2, zs.shape[1]-2
    pix_wid_x, pix_wid_y = pix_widxy

    capas = np.zeros_like(zs)
    
    for i in prange(1, npix_x+1):
        for j in prange(1, npix_y+1):
            aq = aquas[i, j]
            if aq:
                z  = zs[i, j]
                ek = ekins[i, j]
                # average velocity (regulated to 0. < slope < 1.)
                v_avg = (6.*ek/aq)**0.5/2.
                v_fac = np.sin(np.atan(v_avg/v_cap))
                # get slope (but regulated to 0. < slope < 1.)
                dz_dx = (zs[i+1, j] - zs[i-1, j]) / (pix_wid_x*2)
                dz_dy = (zs[i, j+1] - zs[i, j-1]) / (pix_wid_y*2)
                slope = np.sin(np.atan((dz_dx**2 + dz_dy**2)**0.5))
                
                capas[i, j] = sed_cap_fac * aq * v_fac * slope

    return capas
    


#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
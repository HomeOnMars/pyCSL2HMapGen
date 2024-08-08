#!/usr/bin/env python
# coding: utf-8

"""GPU-accelerated functions.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies

from typing import Self

from numba import jit, prange, cuda, float32
import numpy as np
from numpy import typing as npt

try:
    CAN_CUDA = cuda.detect()
except Exception as e:
    print(f"**  Warning: Error during initializing GPU acceleration:\n\t{e}")
    CAN_CUDA = False

if CAN_CUDA:
    print("\nNote   :\n\t"
          + "Cuda supported devices found."
          + " GPU-accelerated functions available.\n")
else:
    print("\n*   Warning:\n\t"
          + "NO Cuda supported devices found."
          + " GPU-accelerated functions UNAVAILABLE.\n")



#-----------------------------------------------------------------------------#
#    Constants
#-----------------------------------------------------------------------------#

# Threads per block - controls shared memory usage for GPU
# The block will have (CUDA_TPB, CUDA_TPB)-shaped threads
# Example see https://numba.pydata.org/numba-doc/dev/cuda/examples.html#matrix-multiplication
CUDA_TPB : int = 16
CUDA_TPB_PLUS_2: int = CUDA_TPB+2



#-----------------------------------------------------------------------------#
#    Erosion: Rainfall
#-----------------------------------------------------------------------------#


@cuda.jit(fastmath=True)
def _erode_rainfall_init_cuda_sub(zs, soils, is_changed):
    """CUDA GPU-accelerated sub process.

    Input data type: cuda.cudadrv.devicearray.DeviceNDArray
    """
    
    # - define shared data structure -
    # (shared by the threads in the same block)
    # Note: the 4 corners will be undefined.
    # Note: the shared array 'shape' arg
    #    must take integer literals instead of integer
    sarr_zs = cuda.shared.array(
        shape=(CUDA_TPB_PLUS_2, CUDA_TPB_PLUS_2), dtype=float32)

    nx_p2, ny_p2 = zs.shape

    # - get thread coordinates -
    i, j = cuda.grid(2)
    # add 1 to account for the edges in the data
    i += 1
    j += 1
    if i + 1 >= nx_p2 or j + 1 >= ny_p2:
        # do nothing if out of bound
        return
    # add 1 to account for the edges in the data
    ti = cuda.threadIdx.x + 1
    tj = cuda.threadIdx.y + 1

    # - preload data -
    soil = soils[i, j]
    sarr_zs[ti, tj] = zs[i, j]
    # load edges
    if ti == 0:
        sarr_zs[ti-1, tj] = zs[i-1, j]
    if ti == CUDA_TPB-1 or i+2 == nx_p2:
        sarr_zs[ti+1, tj] = zs[i+1, j]
    if tj == 0:
        sarr_zs[ti, tj-1] = zs[i, j-1]
    if tj == CUDA_TPB-1 or j+2 == ny_p2:
        sarr_zs[ti, tj+1] = zs[i, j+1]
    cuda.syncthreads()

    # - do math -
    z_new = min(
        sarr_zs[ti-1, tj],
        sarr_zs[ti+1, tj],
        sarr_zs[ti, tj-1],
        sarr_zs[ti, tj+1],
    )
    z_new = max(z_new, soil)
    #sarr_zs[ti, tj] = z_new
    
    # - write data back -
    zs[i, j] = z_new
    if z_new < sarr_zs[ti, tj]:
        is_changed[0] = True




#@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_init_cuda(
    data : npt.NDArray[np.float32],    # ground level
    spawners: npt.NDArray[np.float32],
    pix_widxy: tuple[float, float],
    z_min: np.float32,
    z_sea: np.float32,
    z_max: np.float32,
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
    
    print("** Warning: Cuda version of this func is currently broken.")
    
    npix_x, npix_y = data.shape
    z_min = np.float32(z_min)
    z_sea = np.float32(z_sea)
    z_max = np.float32(z_max)
    # tpb: threads per block
    cuda_tpb_shape = (int(CUDA_TPB), int(CUDA_TPB))
    # bpg: blocks per grid
    cuda_bpg_shape = (
        (npix_x + cuda_tpb_shape[0] - 1) // cuda_tpb_shape[0],
        (npix_y + cuda_tpb_shape[1] - 1) // cuda_tpb_shape[1],
    )
    print(cuda_bpg_shape)

    # - init ans arrays -
    
    # adding an edge
    soils = np.zeros((npix_x+2, npix_y+2), dtype=np.float32)
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
    soils = np.where(soils <= np.float32(0.), np.float32(0.), soils)

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
    aquas = np.where(edges > soils, edges - soils, np.float32(0.))
    
    # - fill basins -
    # (lakes / sea / whatev)
    zs = edges.copy()
    zs[1:-1, 1:-1] = z_max - z_min    # first fill, then drain
    # note: zs' edge elems are fixed
    n_cycles = 0    # debug
    # - CUDA GPU-acceleration -
    is_changed_cuda = cuda.to_device(np.ones(1, dtype=np.bool_))
    zs_cuda = cuda.to_device(zs)
    soils_cuda = cuda.to_device(soils)
    while is_changed_cuda[0]:
        is_changed_cuda[0] = False
        n_cycles += 1
        _erode_rainfall_init_cuda_sub[cuda_bpg_shape, cuda_tpb_shape](
                zs_cuda, soils_cuda, is_changed_cuda)
        cuda.synchronize()
        # debug
        if n_cycles % 100 == 0:
            print(n_cycles, cuda_bpg_shape, cuda_tpb_shape)
        if n_cycles > 1500:
            break

    zs = zs_cuda.copy_to_host()
    aquas[1:-1, 1:-1] = (zs - soils)[1:-1, 1:-1]
    ekins = np.zeros_like(soils)
    sedis = np.zeros_like(soils) # is zero because speed is zero
    
    return soils, aquas, ekins, sedis, edges, n_cycles



@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_get_capas_cuda(
    zs   : npt.NDArray[np.float32],
    aquas: npt.NDArray[np.float32],
    ekins: npt.NDArray[np.float32],
    pix_widxy: tuple[float, float],
    sed_cap_fac: float = 1.0,
    v_cap: float = 16.,
) -> npt.NDArray[np.float32]:
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
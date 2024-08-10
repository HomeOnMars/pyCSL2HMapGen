#!/usr/bin/env python
# coding: utf-8

"""GPU-accelerated functions.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies

from typing import Self

from numba import jit, prange, cuda, float32, bool_
import numpy as np
from numpy import typing as npt

has_cuda : bool = False
CAN_CUDA : bool = False
try:
    has_cuda = cuda.detect()
    CAN_CUDA = cuda.is_available()
except Exception as e:
    print(f"**  Warning: Error during initializing GPU acceleration:\n\t{e}")
    CAN_CUDA = False

if CAN_CUDA:
    print(f"\nNote   : {__name__}:\n\t"
          + "Cuda supported devices and drivers found."
          + " GPU-accelerated functions available.")
elif has_cuda:
    print(f"\n**  Warning: {__name__}:\n\t"
          + "Cuda supported devices found, but drivers library NOT found."
          + " GPU-accelerated functions UNAVAILABLE.")
else:
    print(f"\n*   Warning: {__name__}:\n\t"
          + "NO Cuda supported devices found."
          + " GPU-accelerated functions UNAVAILABLE.")



#-----------------------------------------------------------------------------#
#    Constants
#-----------------------------------------------------------------------------#

# Threads per block - controls shared memory usage for GPU
# The block will have (CUDA_TPB, CUDA_TPB)-shaped threads
# Example see https://numba.pydata.org/numba-doc/dev/cuda/examples.html#matrix-multiplication
CUDA_TPB : int = 16
CUDA_TPB_P2: int = CUDA_TPB+2



#-----------------------------------------------------------------------------#
#    Device Functions
#-----------------------------------------------------------------------------#


@cuda.jit(device=True)
def _device_read_sarr_with_edges(
    in_arr, out_sarr,
    i, j, ti, tj,
):
    """Read data from global memory into shared array.

    Assuming data has an 'edge',
        i.e. extra row & column at both the beginning and the end.
        So ti, tj should be within 1 <= ti <= CUDA_TPB
        ans i,  j should be within 1 <=  i <= nx_p2 - 2
    """
    # nx_p2 means n pixel at x direction plus 2
    nx_p2, ny_p2 = in_arr.shape
    out_sarr[ti, tj] = in_arr[i, j]
    # load edges
    if ti == 1:
        out_sarr[ti-1, tj] = in_arr[i-1, j]
    if ti == CUDA_TPB or i+2 == nx_p2:
        out_sarr[ti+1, tj] = in_arr[i+1, j]
    if tj == 1:
        out_sarr[ti, tj-1] = in_arr[i, j-1]
    if tj == CUDA_TPB or j+2 == ny_p2:
        out_sarr[ti, tj+1] = in_arr[i, j+1]
    return



#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Init
#-----------------------------------------------------------------------------#


@cuda.jit(fastmath=True)
def _erode_rainfall_init_sub_cuda_sub(zs, soils, is_changed):
    """CUDA GPU-accelerated sub process.

    Input data type: cuda.cudadrv.devicearray.DeviceNDArray
    """
    
    # - define shared data structure -
    # (shared by the threads in the same block)
    # Note: the 4 corners will be undefined.
    # Note: the shared array 'shape' arg
    #    must take integer literals instead of integer
    sarr_zs = cuda.shared.array(
        shape=(CUDA_TPB_P2, CUDA_TPB_P2), dtype=float32)
    # flags:
    #    0: has_changes_in_this_thread_block
    sarr_flags = cuda.shared.array(shape=(1,), dtype=bool_)

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
    _device_read_sarr_with_edges(zs, sarr_zs, i, j, ti, tj)
    if ti == 1 and tj == 1:
        sarr_flags[0] = False
    cuda.syncthreads()

    # - do math -
    not_done = False
    for ki in range(cuda.blockDim.x + cuda.blockDim.y):
        # level the lake height within the block
        z_new = min(
            sarr_zs[ti-1, tj],
            sarr_zs[ti+1, tj],
            sarr_zs[ti, tj-1],
            sarr_zs[ti, tj+1],
        )
        z_new = max(z_new, soil)
        if z_new < sarr_zs[ti, tj]:
            not_done = True
            sarr_zs[ti, tj] = z_new
        cuda.syncthreads()
    
    # - write data back -
    zs[i, j] = z_new
    if not_done:
        sarr_flags[0] = True
    cuda.syncthreads()
    if ti == 1 and tj == 1 and sarr_flags[0] and not is_changed[0]:
        is_changed[0] = True



def _erode_rainfall_init_sub_cuda(
    soils: npt.NDArray[np.float32],
    edges: npt.NDArray[np.float32],
    z_range: np.float32,
) -> tuple[npt.NDArray[np.float32], int]:
    """CUDA version of the sub process for rainfall erosion init.

    Filling the basins.
    
    Parameters
    ----------
    ...
    z_range: np.float32
        z_range == z_max - z_min
    """
    npix_x, npix_y = soils.shape[0]-2, soils.shape[1]-2
    # tpb: threads per block
    cuda_tpb_shape = (int(CUDA_TPB), int(CUDA_TPB))
    # bpg: blocks per grid
    cuda_bpg_shape = (
        (npix_x + cuda_tpb_shape[0] - 1) // cuda_tpb_shape[0],
        (npix_y + cuda_tpb_shape[1] - 1) // cuda_tpb_shape[1],
    )

    # - fill basins -
    # (lakes / sea / whatev)
    zs = edges.copy()
    zs[1:-1, 1:-1] = z_range    # first fill, then drain
    # note: zs' edge elems are fixed
    n_cycles = 0    # debug
    # - CUDA GPU-acceleration -
    is_changed_cuda = cuda.to_device(np.ones(1, dtype=np.bool_))
    zs_cuda = cuda.to_device(zs)
    soils_cuda = cuda.to_device(soils)
    while is_changed_cuda[0]:
        is_changed_cuda[0] = False
        n_cycles += 1
        _erode_rainfall_init_sub_cuda_sub[cuda_bpg_shape, cuda_tpb_shape](
                zs_cuda, soils_cuda, is_changed_cuda)
        cuda.synchronize()

    zs = zs_cuda.copy_to_host()
    return zs, n_cycles



#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Evolve
#-----------------------------------------------------------------------------#


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



_erode_rainfall_evolve_sub_cuda = None


#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
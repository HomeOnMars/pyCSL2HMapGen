#!/usr/bin/env python
# coding: utf-8

"""GPU-accelerated functions.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (built-in)
from typing import Self

# imports (3rd party)
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

# imports (my libs)
from ..util import (
    _LOAD_ORDER, #now,
    VerboseType,
)
from .defaults import (
    _ErosionStateDataDtype, ErosionStateDataType,
    _ErosionStateDataExtendedDtype, ErosionStateDataExtendedType,
)
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)



#-----------------------------------------------------------------------------#
#    Constants
#-----------------------------------------------------------------------------#


# Threads per block - controls shared memory usage for GPU
# The block will have (CUDA_TPB, CUDA_TPB)-shaped threads
# Example see
# https://numba.pydata.org/numba-doc/dev/cuda/examples.html#matrix-multiplication
# 2 <= CUDA_TPB <= 32
CUDA_TPB : int = 16
CUDA_TPB_P2: int = CUDA_TPB+2

N_ADJ_P1 : int = 4+1    # number of adjacent cells +1 (+1 for the origin cell)
# Adjacent cells location offsets
#    matches _device_is_at_edge_k().
#    Do NOT change the first 5 rows.
ADJ_OFFSETS : tuple = (
    # i,  j
    ( 0,  0),
    (-1,  0),
    ( 1,  0),
    ( 0, -1),
    ( 0,  1),
)

_ZERO_STAT : npt.NDArray = np.zeros((1,), dtype=_ErosionStateDataDtype)



#-----------------------------------------------------------------------------#
#    Device Functions: Memory
#-----------------------------------------------------------------------------#


@cuda.jit(device=True, fastmath=True)
def _device_get_coord() -> tuple[int, int, int, int]:
    # - get thread coordinates -
    i, j = cuda.grid(2)
    # add 1 to account for the edges in the data
    i += 1; j += 1
    # add 1 to account for the edges in the data
    ti = cuda.threadIdx.x + 1
    tj = cuda.threadIdx.y + 1
    return i, j, ti, tj



@cuda.jit(device=True, fastmath=True)
def _device_is_at_edge_k(
    k, nx_p2, ny_p2,    # in
    i, j, ti, tj,       # in
) -> bool:
    """Test if it is at k-th edge.

    k-th edge matches ADJ_OFFSETS constant.
    """
    if   k == 1 and ti == 1:
        return True
    elif k == 2 and (ti == CUDA_TPB or i+2 == nx_p2):
        return True
    elif k == 3 and tj == 1:
        return True
    elif k == 4 and (tj == CUDA_TPB or j+2 == ny_p2):
        return True
    return False



@cuda.jit(device=True, fastmath=True)
def _device_read_sarr_with_edges(
    # out
    out_sarr,
    # in
    in_arr,
    i, j, ti, tj,
):
    """Read data from global memory into shared array.

    Assuming data has an 'edge',
        i.e. extra row & column at both the beginning and the end.
        So ti, tj should be within 1 <= ti <= CUDA_TPB
        ans i,  j should be within 1 <=  i <= nx_p2 - 2

    WARNING: the 4 corners will not be loaded.
    
    ---------------------------------------------------------------------------
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


@cuda.jit(device=True, fastmath=True)
def _device_init_sarr_with_edges(
    # out
    out_sarr,
    # in
    init_value,
    nx_p2, ny_p2,
    i, j, ti, tj,
):
    """Read data from global memory into shared array.

    Assuming data has an 'edge',
        i.e. extra row & column at both the beginning and the end.
        So ti, tj should be within 1 <= ti <= CUDA_TPB
        ans i,  j should be within 1 <=  i <= nx_p2 - 2

    nx_p2 means n pixel at x direction plus 2
    WARNING: the 4 corners will not be loaded.
    
    ---------------------------------------------------------------------------
    """
    out_sarr[ti, tj] = init_value
    # load edges
    if ti == 1:
        out_sarr[ti-1, tj] = init_value
    if ti == CUDA_TPB or i+2 == nx_p2:
        out_sarr[ti+1, tj] = init_value
    if tj == 1:
        out_sarr[ti, tj-1] = init_value
    if tj == CUDA_TPB or j+2 == ny_p2:
        out_sarr[ti, tj+1] = init_value
    return
    


#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Init
#-----------------------------------------------------------------------------#


@cuda.jit(fastmath=True)
def _erode_rainfall_init_sub_cuda_sub(
    zs_cuda,     # in/out
    soils_cuda,  # in
    is_changed,  # out
):
    """CUDA GPU-accelerated sub process.

    Input data type: cuda.cudadrv.devicearray.DeviceNDArray

    ---------------------------------------------------------------------------
    """
    
    # - define shared data structure -
    # (shared by the threads in the same block)
    # Note: the 4 corners will be undefined.
    # Note: the shared array 'shape' arg
    #    must take integer literals instead of integer
    zs_sarr = cuda.shared.array(
        shape=(CUDA_TPB_P2, CUDA_TPB_P2), dtype=float32)
    # flags_cuda:
    #    0: has_changes_in_this_thread_block
    flags_sarr = cuda.shared.array(shape=(1,), dtype=bool_)

    # - get thread coordinates -
    nx_p2, ny_p2 = zs_cuda.shape
    i, j = cuda.grid(2)
    # add 1 to account for the edges in the data
    i += 1; j += 1
    if i + 1 >= nx_p2 or j + 1 >= ny_p2:
        # do nothing if out of bound
        return
    # add 1 to account for the edges in the data
    ti = cuda.threadIdx.x + 1
    tj = cuda.threadIdx.y + 1

    # - preload data -
    soil = soils_cuda[i, j]
    _device_read_sarr_with_edges(zs_sarr, zs_cuda, i, j, ti, tj)
    if ti == 1 and tj == 1:
        flags_sarr[0] = False
    cuda.syncthreads()

    # - do math -
    not_done = False
    for ki in range(cuda.blockDim.x + cuda.blockDim.y):
        # level the lake height within the block
        z_new = min(
            zs_sarr[ti-1, tj],
            zs_sarr[ti+1, tj],
            zs_sarr[ti, tj-1],
            zs_sarr[ti, tj+1],
        )
        z_new = max(z_new, soil)
        if z_new < zs_sarr[ti, tj]:
            not_done = True
            zs_sarr[ti, tj] = z_new
        cuda.syncthreads()
    
    # - write data back -
    zs_cuda[i, j] = z_new
    if not_done:
        flags_sarr[0] = True
    cuda.syncthreads()
    if ti == 1 and tj == 1 and flags_sarr[0] and not is_changed[0]:
        # reduce writing to global memory as much as possible
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
        
    ---------------------------------------------------------------------------
    """

    # - init cuda -
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
    zs_cuda = cuda.to_device(np.ascontiguousarray(zs))
    soils_cuda = cuda.to_device(np.ascontiguousarray(soils))
    while is_changed_cuda[0]:
        is_changed_cuda[0] = False
        n_cycles += 1
        _erode_rainfall_init_sub_cuda_sub[cuda_bpg_shape, cuda_tpb_shape](
                zs_cuda, soils_cuda, is_changed_cuda)
        cuda.synchronize()

    # - return -
    zs = zs_cuda.copy_to_host()
    return zs, n_cycles



#-----------------------------------------------------------------------------#
#    Device Functions: Physics
#-----------------------------------------------------------------------------#


@cuda.jit(device=True, fastmath=True)
def _device_add_stats(
    stat : _ErosionStateDataDtype,    # in/out
    stat_add: _ErosionStateDataDtype, # in
):
    """Adding stat_add to stat."""
    stat['soil'] += stat_add['soil']
    stat['sedi'] += stat_add['sedi']
    stat['aqua'] += stat_add['aqua']
    stat['ekin'] += stat_add['ekin']
    return



@cuda.jit(device=True, fastmath=True)
def _device_get_z_and_h(
    stat : _ErosionStateDataDtype,  # in
) -> tuple[float32, float32]:
    """Get total height z and fluid height h from stat."""
    h = stat['sedi'] + stat['aqua']
    return stat['soil'] + h, h



@cuda.jit(device=True, fastmath=True)
def _device_normalize_stat(
    stat : _ErosionStateDataDtype,  # in/out
    z_res: float32,  # in
):
    """Normalize stat var.
    
    return sedi to soil if all water evaporated etc.
    """
    if stat['aqua'] < z_res:
        # water all evaporated.
        stat['aqua'] = 0
        stat['soil'] += stat['sedi']
        stat['sedi'] = 0
        stat['ekin'] = 0
    return



@cuda.jit(device=True, fastmath=True)
def _device_move_fluid(
    # out
    d_stats_local,
    # in
    stats_local,
    zero_stat,
    z_res: float32,
    flow_eff: float32,
    rho_soil_div_aqua: float32,
    g: float32,
):
    """Move fluids (a.k.a. water (aqua) + sediments (sedi)).

    Init and save the changes to d_stats_local.
    """

    # - init -
    d_hs_local = cuda.local.array(N_ADJ_P1, dtype=float32)  # amount of flows
    hws_local  = cuda.local.array(N_ADJ_P1, dtype=float32)  # weights of amount
    zs_local   = cuda.local.array(N_ADJ_P1, dtype=float32)  # z
    
    stat = stats_local[0]
    z0, h0 = _device_get_z_and_h(stat)
    zs_local[0] = z0
    hws_local[0] = 0

    for k in range(N_ADJ_P1):
        d_stats_local[k] = zero_stat

    if not h0:    # stop if no water
        return
    
    # init d_hs_local
    for k in range(N_ADJ_P1):
        d_hs_local[k] = float32(0.)
        
    # new movement logic (simplified)
    # move the water to only the lowest adjacent cell
    d_z_max = z_res    # maximum height difference
    k_max = 0
    for k in range(1, N_ADJ_P1):
        zk, _ = _device_get_z_and_h(stats_local[k])
        zs_local[k] = zk
        if z0 - zk > d_z_max:
            d_z_max = z0 - zk
            k_max = k
    if k_max:    # can flow
        d_h_k = min(d_z_max / 2, h0)
        d_h_tot = d_h_k
        d_hs_local[0]     = -d_h_tot
        d_hs_local[k_max] =  d_h_k

    # parse amount of fluid into amount of sedi, aqua, ekin
    if d_h_tot:
        # only do things if something actually moved
        
        # factions that flows away:
        d_se_fac = stat['sedi'] / h0
        d_aq_fac = stat['aqua'] / h0  # should == 1 - d_se_fac
        # # kinetic energy always flows fully away with current
        # d_ek_fac_flow = flow_eff
        d_ek_fac_flow = d_h_tot / h0
        d_ek_fac_g = (
            d_aq_fac + rho_soil_div_aqua * d_se_fac) * g / 2
        
        # kinetic energy gain from gravity
        ek_tot_from_g = float32(0.)
        for k in range(1, N_ADJ_P1):
            d_h_k = d_hs_local[k]
            zk = zs_local[k]
            ek_tot_from_g += d_h_k * (2*(z0 - zk) - d_h_k)
        ek_tot_from_g = (#max(
            d_ek_fac_g * (ek_tot_from_g - d_h_tot**2)#,
            # safety cap *** BELOW IS GIVING AWAY FREE ENERGY ***
            # *** Fix this ***
            #float32(0.),
        )
        if ek_tot_from_g < 0: ek_tot_from_g = np.nan    # debug

        # calc changes from above
        for k in range(N_ADJ_P1):
            d_h_k = d_hs_local[k]
            if d_h_k:
                #d_stats_local[k]['soil'] = 0
                d_stats_local[k]['sedi'] = d_se_fac * d_h_k
                d_stats_local[k]['aqua'] = d_aq_fac * d_h_k
                # gravitational energy gain
                # Note: a column of fluid of height from h0 to h1 has
                #    gravitational energy per area per density of
                #    g * (h1**2 - h0**2)  / 2.
                if k:
                    d_ek = (
                        ek_tot_from_g + d_ek_fac_flow * stat['ekin']
                    ) * d_h_k / d_h_tot
                else:
                    # kinetic energy flow only (no g gain)
                    # remove d_h_k / d_h_tot (which == -1.)
                    #    to avoid precision loss
                    #    resulting in negative stat['ekin']
                    d_ek = -d_ek_fac_flow * stat['ekin']
                d_stats_local[k]['ekin'] = d_ek
                # *** Fix kinetic energy here! ***
                # currently generates single high point stat['ekin']
                # for some reason
    return
    


#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Evolve
#-----------------------------------------------------------------------------#


@cuda.jit(fastmath=True)
def _erode_rainfall_evolve_cuda_final(
    # in/out
    stats_cuda,
    d_stats_cuda,
    # in
    edges_cuda,
    i_layer_read: int,
    z_res: float32,
):
    """Finalizing by adding back the d_stats_cuda to stats_cuda."""
    # *** Pending optimization ***
    #    - no need to store the entire grid, just the edges
    
    # - define shared data structure -
    zero_stat_cuda = cuda.const.array_like(_ZERO_STAT)
    zero_stat = zero_stat_cuda[0]
    
    # - get thread coordinates -
    nx_p2, ny_p2, _ = stats_cuda.shape
    i, j , ti, tj = _device_get_coord()

    # - preload data -
    stat = stats_cuda[i, j, i_layer_read]
    edge = edges_cuda[i, j]
    # add back at edges
    for k in range(1, N_ADJ_P1):
        if _device_is_at_edge_k(k, nx_p2, ny_p2, i, j, ti, tj):
            # add everything, just in case
            for n in range(d_stats_cuda.shape[-1]):
                if edge['soil'] < 0:    # no boundary condition here
                    _device_add_stats(stat, d_stats_cuda[i, j, n])
                # reset
                d_stats_cuda[i, j, n] = zero_stat
            _device_normalize_stat(stat, z_res)
            break
    # write back
    stats_cuda[i, j, i_layer_read] = stat
    


@cuda.jit(fastmath=True)
def _erode_rainfall_evolve_cuda_sub(
    # in/out
    stats_cuda, edges_cuda, flags_cuda, d_stats_cuda,
    # in
    i_layer_read: int,
    z_max: float32,
    z_res: float32,
    evapor_rate : float32,
    flow_eff    : float32,
    rho_soil_div_aqua: float32,
    g: float32,
):
    """Evolving 1 step.

    stats_cuda: (nx_p2, ny_p2, 2)-shaped
    edges_cuda: (nx_p2, ny_p2)-shaped
    flags_cuda: (1,)-shaped
    d_stats_cuda: (nx_p2, ny_p2, 2)-shaped
    
    flags_cuda:
        0: Completed without error?
    z_max:
        z_min is assumed to be zero.
    ---------------------------------------------------------------------------
    """
    
    # - define shared data structure -
    # (shared by the threads in the same block)
    # Note: the 4 corners will be undefined.
    # Note: the shared array 'shape' arg
    #    must take integer literals instead of integer
    stats_sarr = cuda.shared.array(
        shape=(CUDA_TPB_P2, CUDA_TPB_P2), dtype=_ErosionStateDataDtype)
    flags_sarr = cuda.shared.array(shape=(1,), dtype=bool_)

    # 5 elems **for** this [i, j] location **from** adjacent locations:
    #    0:origin, 1:pp, 2:pm, 3:mp, 4:mm
    d_stats_sarr = cuda.shared.array(
        shape=(CUDA_TPB_P2, CUDA_TPB_P2, N_ADJ_P1),
        dtype=_ErosionStateDataDtype)
    # for init
    zero_stat_cuda = cuda.const.array_like(_ZERO_STAT)
    zero_stat = zero_stat_cuda[0]

    # 5 elems **of/for** adjacent locations **from** this [i, j] location
    stats_local = cuda.local.array(N_ADJ_P1, dtype=_ErosionStateDataDtype)
    d_stats_local = cuda.local.array(N_ADJ_P1, dtype=_ErosionStateDataDtype)
    
    # - get thread coordinates -
    nx_p2, ny_p2, _ = stats_cuda.shape
    i, j , ti, tj = _device_get_coord()
    i_layer_write = 1 - i_layer_read
    is_at_edge: bool_ = False
    if i+1 >= nx_p2 or j+1 >= ny_p2:
        # do nothing if out of bound
        return

    # - preload data -
    # load shared
    _device_read_sarr_with_edges(
        stats_sarr, stats_cuda[:, :, i_layer_read], i, j, ti, tj)
    if ti == 1 and tj == 1:
        flags_sarr[0] = False
    # init shared temp
    for k in range(N_ADJ_P1):
        _device_init_sarr_with_edges(
            d_stats_sarr[:, :, k],
            zero_stat, nx_p2, ny_p2, i, j, ti, tj)
    # load local
    stat = stats_sarr[ti, tj]
    edge = edges_cuda[ i,  j]
    
    # - rain & evaporate -
    # cap rains to the maximum height
    stat['aqua'] = min(stat['aqua'] - evapor_rate, z_max)
    _device_normalize_stat(stat, z_res)
    stats_sarr[ti, tj] = stat
    # do the edges too
    for k in range(1, N_ADJ_P1):
        if _device_is_at_edge_k(k, nx_p2, ny_p2, i, j, ti, tj):
            is_at_edge = True
            stats_temp = stats_sarr[
                ti + ADJ_OFFSETS[k][0],
                tj + ADJ_OFFSETS[k][1]]
            stats_temp['aqua'] = min(stats_temp['aqua'] - evapor_rate, z_max)
            _device_normalize_stat(stats_temp, z_res)
            stats_sarr[
                ti + ADJ_OFFSETS[k][0],
                tj + ADJ_OFFSETS[k][1]] = stats_temp
    
    cuda.syncthreads()

    # load local
    for k in range(N_ADJ_P1):
        stats_local[k] = stats_sarr[
            ti + ADJ_OFFSETS[k][0],
            tj + ADJ_OFFSETS[k][1]]

    # - move water -
    _device_move_fluid(
        d_stats_local,
        stats_local, zero_stat, z_res,
        flow_eff, rho_soil_div_aqua, g,
    )
    
    for k in range(N_ADJ_P1):
        d_stats_sarr[
            ti + ADJ_OFFSETS[k][0],
            tj + ADJ_OFFSETS[k][1],
            k
        ] = d_stats_local[k]
    
    # *** Add code here! ***

    cuda.syncthreads()
    
    # - write data back -
    # summarize
    if edge['soil'] < 0:
        for k in range(N_ADJ_P1):
            _device_add_stats(stat, d_stats_sarr[ti, tj, k])
    else:
        # disgard changes and apply boundary conditions
        stat = edge
    # write back at edges
    if is_at_edge:
        for k in range(1, N_ADJ_P1):
            if _device_is_at_edge_k(k, nx_p2, ny_p2, i, j, ti, tj):
                d_stats_cuda[
                    i + ADJ_OFFSETS[k][0],
                    j + ADJ_OFFSETS[k][1],
                    (k-1)//2
                ] = d_stats_local[k]
    else:
        _device_normalize_stat(stat, z_res)
        # otherwise,
        #    wait for _erode_rainfall_evolve_cuda_final(...) for normalization
    # write back
    stats_cuda[i, j, i_layer_write] = stat
    

def _erode_rainfall_evolve_cuda(
    n_step: int,
    stats : ErosionStateDataType,
    edges : ErosionStateDataType,
    npix_xy: tuple[int, int],
    z_max: np.float32,
    z_res: np.float32,
    evapor_rate: np.float32,
    flow_eff: np.float32,
    rho_soil_div_aqua: float32,
    g: float32,
    # ...
    verbose: VerboseType = True,
    **kwargs,
) -> ErosionStateDataType:
    """Do rainfall erosion- evolve through steps.

    'kwargs' are not used.

    z_res:
        must > 0.
    ---------------------------------------------------------------------------
    """

    # - init cuda -
    npix_x, npix_y = npix_xy
    nx_p2, ny_p2 = stats.shape
    # tpb: threads per block
    cuda_tpb_shape = (int(CUDA_TPB), int(CUDA_TPB))
    # bpg: blocks per grid
    cuda_bpg_shape = (
        (npix_x + cuda_tpb_shape[0] - 1) // cuda_tpb_shape[0],
        (npix_y + cuda_tpb_shape[1] - 1) // cuda_tpb_shape[1],
    )

    # add 2 layers of stats: one for reading, one for writing
    # (to avoid interdependence between threads)
    stats_cuda = cuda.to_device(np.stack((stats, stats), axis=-1))
    i_layer_read: int = 0    # 0 or 1; the other one is for writing 
    edges_cuda = cuda.to_device(edges)
    # for flags_cuda def, see _erode_rainfall_evolve_cuda_sub doc string.
    flags_cuda = cuda.to_device(np.ones(1, dtype=np.bool_))
    # d_stats_cuda: for caching tempeorary results from the edges
    # last dim has 2 elems, for i and j direction
    # assuming CUDA_TPB >= 2
    d_stats_cuda = cuda.to_device(np.zeros(
        (nx_p2, ny_p2, 2), dtype=_ErosionStateDataDtype))
    
    # - run -
    for s in range(n_step):
        # *** add more sophisticated non-uniform rain code here! ***
        _erode_rainfall_evolve_cuda_sub[cuda_bpg_shape, cuda_tpb_shape](
            stats_cuda, edges_cuda, flags_cuda, d_stats_cuda, i_layer_read,
            z_max, z_res, evapor_rate, flow_eff, rho_soil_div_aqua, g,
        )
        cuda.synchronize()
        i_layer_read = 1 - i_layer_read
        # add back edges
        _erode_rainfall_evolve_cuda_final[cuda_bpg_shape, cuda_tpb_shape](
            stats_cuda, d_stats_cuda, edges_cuda, i_layer_read, z_res,
        )
        cuda.synchronize()
    
    # - return -
    stats = stats_cuda[:, :, i_layer_read].copy_to_host()
    print("WARNING: *** Cuda version of this func not yet complete. ***")
    return stats



#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: drafts
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
        
    ---------------------------------------------------------------------------
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
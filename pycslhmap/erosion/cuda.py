#!/usr/bin/env python
# coding: utf-8

"""GPU-accelerated functions.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (built-in)
from typing import Self
import math

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
# The block will have (CUDA_TPB_X, CUDA_TPB_Y)-shaped threads
# Example see
# https://numba.pydata.org/numba-doc/dev/cuda/examples.html#matrix-multiplication
# 4 <= CUDA_TPB <= 32
CUDA_TPB_X : int = 16
CUDA_TPB_Y : int = 16

# Adjacent cells location offsets
#    matches _is_at_inner_edge_k_cudev().
#    Do NOT change the first 5 rows.
#    Every even row and the next odd row must be polar opposite.
ADJ_OFFSETS : tuple = (
    # i,  j
    ( 0,  0),
    (-1,  0),
    ( 1,  0),
    ( 0, -1),
    ( 0,  1),
)
# number of adjacent cells +1 (+1 for the origin cell)
N_ADJ_P1 : int = 5    # == len(ADJ_OFFSETS)



#-----------------------------------------------------------------------------#
#    Functions: Cuda managements
#-----------------------------------------------------------------------------#


def get_cuda_bpg_tpb(nx, ny) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return Cuda thread/block configurations.

    tpb: threads per block
    bpg: blocks per grid
    """
    cuda_tpb = (CUDA_TPB_X, CUDA_TPB_Y)
    cuda_bpg = (
        # -2 because we are not using the edge of the block
        (nx-2 + cuda_tpb[0]-2 - 1) // (cuda_tpb[0]-2),
        (ny-2 + cuda_tpb[1]-2 - 1) // (cuda_tpb[1]-2),
    )
    return cuda_bpg, cuda_tpb



#-----------------------------------------------------------------------------#
#    Device Functions: Memory
#-----------------------------------------------------------------------------#


@cuda.jit(device=True, fastmath=True)
def _get_coord_cudev() -> tuple[int, int, int, int]:
    # - get thread coordinates -
    # blockDim-2 because we are only calculating in the center of the block
    i = cuda.threadIdx.x + cuda.blockIdx.x * (cuda.blockDim.x-2)
    j = cuda.threadIdx.y + cuda.blockIdx.y * (cuda.blockDim.y-2)
    ti = cuda.threadIdx.x
    tj = cuda.threadIdx.y
    return i, j, ti, tj



@cuda.jit(device=True, fastmath=True)
def _is_at_inner_edge_k_cudev(
    k, nx, ny,     # in
    i, j, ti, tj,  # in
) -> bool:
    """Test if thread is at k-th edge of the block.

    k-th edge matches ADJ_OFFSETS constant.

    ---------------------------------------------------------------------------
    """
    if (   k == 1 and ti == 1
        or k == 2 and (ti == CUDA_TPB_X-2 or i == nx-2)
        or k == 3 and tj == 1
        or k == 4 and (tj == CUDA_TPB_Y-2 or j == ny-2)
       ):
        return True
    return False



@cuda.jit(device=True, fastmath=True)
def _is_at_outer_edge_cudev(
    nx, ny, i, j, ti, tj,  # in
) -> bool:
    """Test if thread is at the edge of the block.

    See also _is_at_inner_edge_k_cudev(...).

    ---------------------------------------------------------------------------
    """
    if (   ti == 0
        or ti == CUDA_TPB_X-1 or i == nx-1
        or tj == 0
        or tj == CUDA_TPB_Y-1 or j == ny-1
       ):
        return True
    return False



@cuda.jit(device=True, fastmath=True)
def _is_outside_map_cudev(
    nx, ny, i, j, # in
) -> bool:
    """Test if the cell is outside the map."""
    return True if i >= nx or j >= ny else False



#-----------------------------------------------------------------------------#
#    Device Functions: Stats calc
#-----------------------------------------------------------------------------#


@cuda.jit(device=True, fastmath=True)
def _set_stat_zero_cudev(
    stat: _ErosionStateDataDtype,    # in/out
):
    """Init stat to zero."""
    stat['soil'] = 0
    stat['sedi'] = 0
    stat['aqua'] = 0
    stat['p_x' ] = 0
    stat['p_y' ] = 0
    stat['ekin'] = 0
    return


@cuda.jit(device=True, fastmath=True)
def _add_stats_cudev(
    stat : _ErosionStateDataDtype,    # in/out
    stat_add: _ErosionStateDataDtype, # in
    edge : _ErosionStateDataDtype,    # in
) -> _ErosionStateDataDtype:
    """Adding stat_add to stat if edge is not set, else reset to edge."""
    
    if math.isnan(edge['soil']): stat['soil'] += stat_add['soil']
    else: stat['soil'] = edge['soil']
        
    if math.isnan(edge['sedi']): stat['sedi'] += stat_add['sedi']
    else: stat['sedi'] = edge['sedi']
        
    if math.isnan(edge['aqua']): stat['aqua'] += stat_add['aqua']
    else: stat['aqua'] = edge['aqua']
        
    if math.isnan(edge['p_x' ]): stat['p_x' ] += stat_add['p_x']
    else: stat['p_x' ] = edge['p_x']
        
    if math.isnan(edge['p_y' ]): stat['p_y' ] += stat_add['p_y']
    else: stat['p_y' ] = edge['p_y']
    
    if math.isnan(edge['ekin']): stat['ekin'] += stat_add['ekin']
    else: stat['ekin'] = edge['ekin']

    return stat



@cuda.jit(device=True, fastmath=True)
def _get_z_and_h_cudev(
    stat : _ErosionStateDataDtype,  # in
) -> tuple[float32, float32]:
    """Get total height z and fluid height h from stat."""
    h = stat['sedi'] + stat['aqua']
    return stat['soil'] + h, h



@cuda.jit(device=True, fastmath=True)
def _normalize_stat_cudev(
    stat : _ErosionStateDataDtype,  # in/out
    edge : _ErosionStateDataDtype,  # in
    z_res: float32,  # in
) -> _ErosionStateDataDtype:
    """Normalize stat var.
    
    return sedi to soil if all water evaporated etc.

    ---------------------------------------------------------------------------
    """
    if stat['aqua'] < z_res:
        # water all evaporated.
        stat['aqua'] = 0 if math.isnan(edge['aqua']) else edge['aqua']
        if math.isnan(edge['soil']) and math.isnan(edge['sedi']):
            stat['soil'] += stat['sedi']
        stat['sedi'] = 0 if math.isnan(edge['sedi']) else edge['sedi']
        stat['p_x' ] = 0 if math.isnan(edge['p_x' ]) else edge['p_x' ]
        stat['p_y' ] = 0 if math.isnan(edge['p_y' ]) else edge['p_y' ]
        stat['ekin'] = 0 if math.isnan(edge['ekin']) else edge['ekin']
    return stat



#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Init
#-----------------------------------------------------------------------------#


@cuda.jit(fastmath=True)
def _erode_rainfall_init_sub_cuda_sub(
    is_changed,     # out
    zs_cuda,        # in/out
    soils_cuda,     # in
    edges_zs_cuda,  # in
):
    """CUDA GPU-accelerated sub process.

    Input data type: cuda.cudadrv.devicearray.DeviceNDArray

    ---------------------------------------------------------------------------
    """
    
    # - define shared data structure -
    # (shared by the threads in the same block)
    # Note: the shared array 'shape' arg
    #    must take integer literals instead of integer
    zs_sarr = cuda.shared.array(shape=(CUDA_TPB_X, CUDA_TPB_Y), dtype=float32)
    # flags_cuda:
    #    0: has_changes_in_this_thread_block
    flags_sarr = cuda.shared.array(shape=(1,), dtype=bool_)

    # - get thread coordinates -
    nx, ny = zs_cuda.shape
    i, j, ti, tj = _get_coord_cudev()

    if _is_outside_map_cudev(nx, ny, i, j):
        return

    # - preload data -
    soil = soils_cuda[i, j]
    edge_z = edges_zs_cuda[i, j]
    zs_sarr[ti, tj] = zs_cuda[i, j] if math.isnan(edge_z) else edge_z
    
    if ti == 1 and tj == 1:
        flags_sarr[0] = False

    if _is_at_outer_edge_cudev(nx, ny, i, j, ti, tj):
        return
        
    cuda.syncthreads()

    # - do math -
    done = True
    for ki in range(CUDA_TPB_X + CUDA_TPB_Y - 4):
        # level the lake height within the block
        if math.isnan(edge_z):    # only do things if not fixed
            z_new = min(
                zs_sarr[ti-1, tj],
                zs_sarr[ti+1, tj],
                zs_sarr[ti, tj-1],
                zs_sarr[ti, tj+1],
            )
            z_new = max(z_new, soil)
            if z_new < zs_sarr[ti, tj]:
                done = False
                zs_sarr[ti, tj] = z_new
        cuda.syncthreads()
    
    # - write data back -
    zs_cuda[i, j] = z_new
    if not done:
        flags_sarr[0] = True
    cuda.syncthreads()
    if ti == 1 and tj == 1 and flags_sarr[0] and not is_changed[0]:
        # reduce writing to global memory as much as possible
        is_changed[0] = True



def _erode_rainfall_init_sub_cuda(
    soils: npt.NDArray[np.float32],
    edges_zs: npt.NDArray[np.float32],
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
    nx, ny = soils.shape
    cuda_bpg, cuda_tpb = get_cuda_bpg_tpb(nx, ny)
    # tpb: threads per block

    # - fill basins -
    # (lakes / sea / whatev)
    zs = np.where(
        np.isnan(edges_zs),
        z_range,
        edges_zs,
    )
    # note: zs' edge elems are fixed
    n_cycles = 0    # debug
    # - CUDA GPU-acceleration -
    is_changed_cuda = cuda.to_device(np.ones(1, dtype=np.bool_))
    zs_cuda = cuda.to_device(np.ascontiguousarray(zs))
    soils_cuda = cuda.to_device(np.ascontiguousarray(soils))
    edges_zs_cuda = cuda.to_device(np.ascontiguousarray(edges_zs))
    while is_changed_cuda[0]:
        is_changed_cuda[0] = False
        n_cycles += 1
        _erode_rainfall_init_sub_cuda_sub[cuda_bpg, cuda_tpb](
            is_changed_cuda, zs_cuda, soils_cuda, edges_zs_cuda)
        cuda.synchronize()

    # - return -
    zs = zs_cuda.copy_to_host()
    return zs, n_cycles



#-----------------------------------------------------------------------------#
#    Device Functions: Physics
#-----------------------------------------------------------------------------#


@cuda.jit(device=True, fastmath=True)
def _get_d_hs_cudev(
    # out
    d_hs_local,
    # in
    h0: float32,
    zs_local,
    z_res: float32,
    flow_eff: float32,
) -> float32:
    """Get fluid movements plan.

    *   Note: Need to expand this if more rows are added to ADJ_OFFSETS!

    Return: d_h_tot
    
    ---------------------------------------------------------------------------
    """
    z0 = zs_local[0]
    d_h_tot = float32(0.)
    k = 0
    for ki in range(N_ADJ_P1//2):
        d_h_k1 = z0 - zs_local[2*ki+1]
        d_h_k2 = z0 - zs_local[2*ki+2]
        if d_h_k1 > d_h_k2:
            k = 2*ki+1
            d_hs_local[2*ki+2] = float32(0.)
        else:
            k = 2*ki+2
            d_hs_local[2*ki+1] = float32(0.)
        zk = zs_local[k]
        d_h_k = min((z0 - zk)/3*flow_eff, h0/2)
        if d_h_k < z_res: d_h_k = float32(0.)
        d_hs_local[k] = d_h_k
        d_h_tot += d_h_k

    d_hs_local[0] = -d_h_tot
    return d_h_tot



@cuda.jit(device=True, fastmath=True)
def _move_fluid_cudev(
    # out
    d_stats_local,
    # in
    stats_local,
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
    z0, h0 = _get_z_and_h_cudev(stat)
    zs_local[0] = z0
    hws_local[0] = 0

    for k in range(N_ADJ_P1):
        _set_stat_zero_cudev(d_stats_local[k])

    if not h0:    # stop if no water
        return

    # # init zs_local
    for k in range(1, N_ADJ_P1):
        zk, _ = _get_z_and_h_cudev(stats_local[k])
        zs_local[k] = zk

    # get d_hs_local
    d_h_tot = _get_d_hs_cudev(
        d_hs_local,  # out
        h0, zs_local, z_res, flow_eff,   # in
    )

    # parse amount of fluid into amount of sedi, aqua, ekin
    if d_h_tot: # only do things if something actually moved
        
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
        ek_tot_from_g = d_ek_fac_g * (ek_tot_from_g - d_h_tot**2)
        #if ek_tot_from_g < 0: ek_tot_from_g = np.nan    # debug

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
    # *** Pending optimization/overhaul ***
    #    - should not store the entire grid, just the edges
    #    - *** warning: potential racing condition unfixed
    
    # - get thread coordinates -
    nx, ny, _ = stats_cuda.shape
    
    i, j , ti, tj = _get_coord_cudev()

    # - preload data -
    stat = stats_cuda[i, j, i_layer_read]
    edge = edges_cuda[i, j]
    # add back at edges
    for k in range(1, N_ADJ_P1):
        if _is_at_inner_edge_k_cudev(k, nx, ny, i, j, ti, tj):
            # add everything, just in case
            for n in range(d_stats_cuda.shape[-1]):
                _add_stats_cudev(stat, d_stats_cuda[i, j, n], edge)
                # reset
                _set_stat_zero_cudev(d_stats_cuda[i, j, n])
            _normalize_stat_cudev(stat, edge, z_res)
            break
    # write back
    stats_cuda[i, j, i_layer_read] = stat
    


@cuda.jit(fastmath=True)
def _erode_rainfall_evolve_cuda_sub(
    # in/out
    stats_cuda, edges_cuda, d_stats_cuda,
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

    stats_cuda: (nx, ny, 2)-shaped
    edges_cuda: (nx, ny)-shaped
    d_stats_cuda: (nx, ny, 2)-shaped
    
    z_max:
        z_min is assumed to be zero.
    ---------------------------------------------------------------------------
    """
    
    # - define shared data structure -
    # (shared by the threads in the same block)
    # Note: the shared array 'shape' arg
    #    must take integer literals instead of integer
    stats_sarr = cuda.shared.array(
        shape=(CUDA_TPB_X, CUDA_TPB_Y), dtype=_ErosionStateDataDtype)
    edges_sarr = cuda.shared.array(
        shape=(CUDA_TPB_X, CUDA_TPB_Y), dtype=_ErosionStateDataDtype)

    # 5 elems **for** this [i, j] location **from** adjacent locations:
    #    0:origin, 1:pp, 2:pm, 3:mp, 4:mm
    d_stats_sarr = cuda.shared.array(
        shape=(CUDA_TPB_X, CUDA_TPB_Y, N_ADJ_P1),
        dtype=_ErosionStateDataDtype)

    # 5 elems **of/for** adjacent locations **from** this [i, j] location
    stats_local = cuda.local.array(N_ADJ_P1, dtype=_ErosionStateDataDtype)
    d_stats_local = cuda.local.array(N_ADJ_P1, dtype=_ErosionStateDataDtype)
    
    # - get thread coordinates -
    nx, ny, _ = stats_cuda.shape
    i, j, ti, tj = _get_coord_cudev()
    i_layer_write = 1 - i_layer_read
    if _is_outside_map_cudev(nx, ny, i, j):
        return

    # - preload data -
    # init shared temp
    for k in range(N_ADJ_P1):
        _set_stat_zero_cudev(d_stats_sarr[ti, tj, k])
    # load local
    stats_sarr[ti, tj] = stats_cuda[i, j, i_layer_read]
    stat = stats_sarr[ti, tj]    # by reference
    edges_sarr[ti, tj] = edges_cuda[i, j]
    edge = edges_sarr[ti, tj]
    not_at_outer_edge : bool_ = not _is_at_outer_edge_cudev(
        nx, ny, i, j, ti, tj)
    
    # - rain & evaporate -
    # cap rains to the maximum height
    stat['aqua'] = min(stat['aqua'] - evapor_rate, z_max)
    stat = _normalize_stat_cudev(stat, edge, z_res)
    stats_sarr[ti, tj] = stat
    
    cuda.syncthreads()

    if not_at_outer_edge:
        # load local
        for k in range(N_ADJ_P1):
            stats_local[k] = stats_sarr[
                ti + ADJ_OFFSETS[k][0],
                tj + ADJ_OFFSETS[k][1]]

        # - move water -
        _move_fluid_cudev(
            d_stats_local,
            stats_local, z_res,
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
    for k in range(N_ADJ_P1):
        # could use optimization
        stat = _add_stats_cudev(stat, d_stats_sarr[ti, tj, k], edge)
    if not_at_outer_edge:
        stat = _normalize_stat_cudev(stat, edge, z_res)
    else:
        # otherwise,
        #    wait for _erode_rainfall_evolve_cuda_final(...) for normalization
        pass
    for k in range(1, N_ADJ_P1):
        if _is_at_inner_edge_k_cudev(k, nx, ny, i, j, ti, tj):
            d_stats_cuda[
                i + ADJ_OFFSETS[k][0],
                j + ADJ_OFFSETS[k][1],
                (k-1)//2
            ] = d_stats_local[k]

    # write back
    if not_at_outer_edge:
        stats_cuda[i, j, i_layer_write] = stat
    

def _erode_rainfall_evolve_cuda(
    n_step: int,
    stats : ErosionStateDataType,
    edges : ErosionStateDataType,
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
    nx, ny = stats.shape
    cuda_bpg, cuda_tpb = get_cuda_bpg_tpb(nx, ny)

    # add 2 layers of stats: one for reading, one for writing
    # (to avoid interdependence between threads)
    stats_cuda = cuda.to_device(np.stack((stats, stats), axis=-1))
    i_layer_read: int = 0    # 0 or 1; the other one is for writing 
    edges_cuda = cuda.to_device(edges)
    # d_stats_cuda: for caching tempeorary results from the edges
    # last dim has 2 elems, for i and j direction
    # assuming CUDA_TPB >= 2
    d_stats_cuda = cuda.to_device(np.zeros(
        (nx, ny, 2), dtype=_ErosionStateDataDtype))
    
    # - run -
    for s in range(n_step):
        # *** add more sophisticated non-uniform rain code here! ***
        _erode_rainfall_evolve_cuda_sub[cuda_bpg, cuda_tpb](
            stats_cuda, edges_cuda, d_stats_cuda, i_layer_read,
            z_max, z_res, evapor_rate, flow_eff, rho_soil_div_aqua, g,
        )
        cuda.synchronize()
        i_layer_read = 1 - i_layer_read
        # add back edges
        _erode_rainfall_evolve_cuda_final[cuda_bpg, cuda_tpb](
            stats_cuda, d_stats_cuda, edges_cuda, i_layer_read, z_res,
        )
        cuda.synchronize()
    
    # - return -
    stats = stats_cuda[:, :, i_layer_read].copy_to_host()
    print("WARNING: *** Cuda version of this func not yet complete. ***")
    return stats



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
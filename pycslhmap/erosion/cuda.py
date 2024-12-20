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
from numba import cuda, float32, bool_
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
    ErosionStateDataDtype, ErosionStateDataType,
)
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)



#-----------------------------------------------------------------------------#
#    Constants
#-----------------------------------------------------------------------------#


# WARNING: DO NOT CHANGE THE BELOW ALL-CAPS VARIABLES DURING RUN TIME!!!


# Threads per block - controls shared memory usage for GPU
# The block will have (CUDA_TPB_X, CUDA_TPB_Y)-shaped threads
# Example see
# https://numba.pydata.org/numba-doc/dev/cuda/examples.html#matrix-multiplication
# 4 <= CUDA_TPB <= 32
CUDA_TPB_X : int = 8
CUDA_TPB_Y : int = 32

# Adjacent cells location offsets
#    matches _get_outer_edge_k_cudev().
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
N_ADJ_P1 : int = 5  # == len(ADJ_OFFSETS)
                    # must be odd number

_NAN_STATS : npt.NDArray = np.full((1,), np.nan, dtype=ErosionStateDataDtype)



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
def _get_outer_edge_idim_cudev(
    nx, ny, i, j, ti, tj,  # in
) -> int:
    """Get which outer edge is the thread on.

    if none or in the corner, return -1.
    """
    on_x_edge = (ti == 0 or ti == CUDA_TPB_X-1 or i == nx-1)
    on_y_edge = (tj == 0 or tj == CUDA_TPB_Y-1 or j == ny-1)
    if   on_x_edge and not on_y_edge:
        return 0
    elif not on_x_edge and on_y_edge:
        return 1
    else:
        return -1



@cuda.jit(device=True, fastmath=True)
def _get_outer_edge_k_cudev(
    nx, ny, i, j, ti, tj,  # in
) -> int:
    """Get which outer edge is the thread on.

    if none or in the corner, return -1.
    """
    on_x_edge = (ti == 0 or ti == CUDA_TPB_X-1 or i == nx-1)
    on_y_edge = (tj == 0 or tj == CUDA_TPB_Y-1 or j == ny-1)
    if   not on_y_edge and (ti == 0):
        return 1
    elif not on_y_edge and (ti == CUDA_TPB_X-1 or i == nx-1):
        return 2
    elif not on_x_edge and (tj == 0):
        return 3
    elif not on_x_edge and (tj == CUDA_TPB_Y-1 or j == ny-1):
        return 4
    else:
        return -1



@cuda.jit(device=True, fastmath=True)
def _is_at_outer_edge_cudev(
    nx, ny, i, j, ti, tj,  # in
) -> bool:
    """Test if thread is at the edge of the block.

    See also _is_in_inner_center_cudev(...).

    ---------------------------------------------------------------------------
    """
    if (   (ti == 0)
        or (ti == CUDA_TPB_X-1 or i == nx-1)
        or (tj == 0)
        or (tj == CUDA_TPB_Y-1 or j == ny-1)
       ):
        return True
    return False



@cuda.jit(device=True, fastmath=True)
def _is_in_inner_center_cudev(
    nx, ny, i, j, ti, tj,  # in
) -> bool:
    """Test if thread is at the inner center ([2:-2, 2:-2])."""
    if (    (ti > 1 and ti < CUDA_TPB_X-2 and i < nx-2)
        and (tj > 1 and tj < CUDA_TPB_Y-2 and j < ny-2)
       ):
        return True
    return False



@cuda.jit(device=True, fastmath=True)
def _is_outside_map_cudev(
    nx, ny, i, j, # in
) -> bool:
    """Test if the cell is outside the map."""
    return True if i >= nx or j >= ny else False



@cuda.jit(device=True, fastmath=True)
def _get_tkij_cudev(
    ti, tj, k,  # in
) -> tuple[float32, float32]:
    """Get thread index of k-th adjacent cells."""
    return ti + ADJ_OFFSETS[k][0], tj + ADJ_OFFSETS[k][1]



@cuda.jit(device=True, fastmath=True)
def _get_l_cudev(
    k, lx, ly,  # in
) -> float32:
    """Get pixel width (i.e. step width) at k-th direction"""
    # *** Update below if ADJ_OFFSETS were updated! ***
    if k==0: return np.nan    # debug
    return lx if (k==1 or k==2) else ly



@cuda.jit(device=True, fastmath=True)
def _get_sign_cudev(x) -> float32:
    """Get the sign of x. (1 if zero/nan)"""
    return float32(-1) if x < 0 else float32(1)



@cuda.jit(device=True, fastmath=True)
def _signed_sqrt_cudev(x) -> float32:
    """Signed sqrt."""
    return -math.sqrt(-x) if x < 0 else math.sqrt(x)


#-----------------------------------------------------------------------------#
#    Device Functions: Stats calc
#-----------------------------------------------------------------------------#


@cuda.jit(device=True, fastmath=True)
def _set_stat_zero_cudev(
    stat: ErosionStateDataDtype,    # in/out
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
    # in/out
    stat: ErosionStateDataDtype,
    # in
    stat_add: ErosionStateDataDtype,
    edge: ErosionStateDataDtype,
    rho_sedi: float32,
    g_eff   : float32,
) -> ErosionStateDataDtype:
    """Adding stat_add to stat if edge is not set, else reset to edge.

    ---------------------------------------------------------------------------
    """
    rhoh2_old = (    # for energy calc
        _get_rhoh2_cudev(stat, rho_sedi)+ _get_rhoh2_cudev(stat_add, rho_sedi))

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

    if math.isnan(edge['ekin']):
        stat['ekin'] += stat_add['ekin']
        # add back gravitational energy loss
        #    when putting fluid on top of each other
        rhoh2_new = _get_rhoh2_cudev(stat, rho_sedi)
        stat['ekin'] += (rhoh2_old - rhoh2_new) * g_eff / 2
    else: stat['ekin'] = edge['ekin']

    return stat



@cuda.jit(device=True, fastmath=True)
def _get_z_cudev(
    stat : ErosionStateDataDtype,  # in
) -> float32:
    """Get total height z from stat."""
    return stat['soil'] + stat['sedi'] + stat['aqua']



@cuda.jit(device=True, fastmath=True)
def _get_h_cudev(
    stat : ErosionStateDataDtype,  # in
) -> float32:
    """Get fluid height h from stat."""
    return stat['sedi'] + stat['aqua']



@cuda.jit(device=True, fastmath=True)
def _get_m_cudev(
    stat : ErosionStateDataDtype,  # in
    rho_sedi: float32,     # in
) -> float32:
    """Get mass per rhoS (in unit of water-equivalent height).

    Always positive.
    """
    return max(rho_sedi * stat['sedi'] + stat['aqua'], float32(0))



@cuda.jit(device=True, fastmath=True)
def _get_p_cudev(
    stat : ErosionStateDataDtype,  # in
) -> float32:
    """Get momentum per rhoS."""
    return math.sqrt(stat['p_x']**2 + stat['p_y']**2)



@cuda.jit(device=True, fastmath=True)
def _get_v_cudev(
    # in
    stat : ErosionStateDataDtype,
    z_res: float32,
    rho_sedi: float32,
    v_cap: float32,
) -> float32:
    """Get velocity (in unit of m/s; >= 0).

    ---------------------------------------------------------------------------
    """
    m = _get_m_cudev(stat, rho_sedi)
    return min(max(
        _signed_sqrt_cudev(
            stat['p_x']**2 + stat['p_y']**2 + 2 * m * stat['ekin']) / m,
        # limit result in between 0 and v_cap
        0), v_cap) if m >= z_res else float32(0)



@cuda.jit(device=True, fastmath=True)
def _get_rho_cudev(
    # in
    stat : ErosionStateDataDtype,
    rho_sedi : float32,
) -> float32:
    """Get density."""
    return (
        rho_sedi * stat['sedi'] + stat['aqua']) / (stat['sedi'] + stat['aqua'])



@cuda.jit(device=True, fastmath=True)
def _get_rhoh2_cudev(
    # in
    stat : ErosionStateDataDtype,
    rho_sedi : float32,
) -> float32:
    """Get density times h^2. Useful for gravitational energy calculations.

    ---------------------------------------------------------------------------
    """
    # rho = (rho_sedi*stat['sedi']+stat['aqua']) / (stat['sedi']+stat['aqua'])
    return (
        rho_sedi * stat['sedi'] + stat['aqua']) * (stat['sedi'] + stat['aqua'])



@cuda.jit(device=True, fastmath=True)
def _get_zfg_cudev(
    stat : ErosionStateDataDtype,  # in
) -> float32:
    """Get height z' for gradient calc."""
    return stat['soil'] + stat['sedi'] + stat['aqua']

    

@cuda.jit(device=True, fastmath=True)
def _normalize_stat_cudev(
    # in/out
    stat : ErosionStateDataDtype,
    # in
    edge : ErosionStateDataDtype,
    z_max: float32,
    z_res: float32,
    rho_sedi: float32,
) -> ErosionStateDataDtype:
    """Normalize stat var.
    
    return sedi to soil if all water evaporated etc.

    ---------------------------------------------------------------------------
    """
    
    if stat['soil'] < 0:
        # Fix over erosion by force depositing sediments
        stat['sedi'] += stat['soil']
        stat['soil'] = 0

    if _get_z_cudev(stat) > z_max:
        # Fix height overflow
        stat['soil'] = min(stat['soil'], z_max)
        stat['sedi'] = min(stat['sedi'], z_max - stat['soil'])
        stat['aqua'] = min(
            stat['aqua'],
            # cap rains to the maximum height
            z_max - (stat['soil'] + max(stat['sedi'], float32(0))),
        )
    
    if stat['aqua'] < z_res:
        # water all evaporated.
        stat['aqua'] = 0 if math.isnan(edge['aqua']) else edge['aqua']
        if math.isnan(edge['soil']) and math.isnan(edge['sedi']):
            stat['soil'] += stat['sedi']
        stat['sedi'] = 0 if math.isnan(edge['sedi']) else edge['sedi']
        stat['p_x' ] = 0 if math.isnan(edge['p_x' ]) else edge['p_x' ]
        stat['p_y' ] = 0 if math.isnan(edge['p_y' ]) else edge['p_y' ]
        stat['ekin'] = 0 if math.isnan(edge['ekin']) else edge['ekin']
    else:
        # normalize energy & momentum
        if not math.isnan(edge['ekin']):
            stat['ekin'] = edge['ekin']
        else:
            p2 = stat['p_x']**2 + stat['p_y']**2
            if p2 > 0:
                # attempt to put energy back into momentum
                m_2 = _get_m_cudev(stat, rho_sedi) * 2
                dp2 = m_2 * stat['ekin']
                if p2 < -dp2:
                    # cap the removal of energy so p doesn't go negative
                    stat['ekin'] = (p2 + dp2) / m_2 if m_2 > z_res else 0
                    dp2 = -p2
                else:
                    stat['ekin'] = 0
                p_fac = math.sqrt(max(1+dp2/p2, 0))
                if math.isnan(edge['p_x']): stat['p_x'] *= p_fac
                if math.isnan(edge['p_y']): stat['p_y'] *= p_fac
    
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

    cuda.syncthreads()

    if _is_at_outer_edge_cudev(nx, ny, i, j, ti, tj):
        return
        
    # - do math -
    z_new = zs_sarr[ti, tj]
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



def erode_rainfall_init_sub_cuda(
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
def _get_capa_cudev(
    # in
    stat : ErosionStateDataDtype,
    v0_capped: float32,
    slope: float32,
    v_cap: float32,
    capa_fac: float32,
    capa_fac_v: float32,
    capa_fac_slope: float32,
    capa_fac_slope_cap: float32,
) -> float32:
    """Get sediment capacity.

    Assumes 0 < v0_capped < v_cap
        And 0 < slope < capa_fac_slope_cap

    ---------------------------------------------------------------------------
    """
    capa = capa_fac * stat['aqua']
    if capa_fac_v:     # speed-based multiplier
        capa *= (v_cap + capa_fac_v*v0_capped) / ((1+capa_fac_v)*v_cap)
    if capa_fac_slope: # slope-based multiplier
        capa *= max(
            (capa_fac_slope_cap + capa_fac_slope*slope
             ) / ((1+capa_fac_slope)*capa_fac_slope_cap),
            float32(0),  # prevent capa going negative
        )
    return capa


@cuda.jit(device=True, fastmath=True)
def _get_d_p_from_g_cudev(
    # in
    stats_sarr,
    ti, tj, k,  # int
    stat, z0, h0, m0, p2_0,
    d_h_k, grad_x, grad_y,
    lx, ly, g_eff,
) -> tuple[float32, float32, float32]:
    """Calc momentum gain from proposed movement d_h_k.
    
    Used in _move_fluid_cudev().

    note: the sign of grad_x, grad_y should encodes direction of the slope
        rather than downward/upward. Upward should be capped at 0.

    ---------------------------------------------------------------------------
    """
    fac_k = d_h_k / h0    # move factor
    dd_p_x_k, dd_p_y_k = float32(0), float32(0)
    ek_from_g_k = float32(0)
    if fac_k > 0:    # only do things if movement are proposed
        tki, tkj = _get_tkij_cudev(ti, tj, k)
        zk = _get_z_cudev(stats_sarr[tki, tkj])
        # calc momemtum gain
        
        # momentum gain (from gravity) squared (per move factor squared)
        #    assuming all kinetic eneregy translated to momentum
        #    (p_from_g)**2 == 2 * d_m_k * ek_from_g_k
        #        where d_m_k = m0 * fac_k,
        #        ek_from_g_k = m0 * g_eff * fac_k * (z0 - zk - d_h_k)
        #    divide both sides by (fac_k)**2 and we have
        #d_p2_from_g_pf2k = 2 * m0**2 * g_eff * (z0 - zk - d_h_k) #, i.e.
        if m0:
            d_p2_from_g_pf2k = 2 * m0**2 * g_eff * (z0 - zk - d_h_k)
            ek_from_g_k = d_p2_from_g_pf2k / (2 * m0) * fac_k
        d_p2_k_pf2k = p2_0 + d_p2_from_g_pf2k
        grad2_k = (max(
            _get_zfg_cudev(stat) - _get_zfg_cudev(stats_sarr[tki, tkj]),
            0) / _get_l_cudev(k, lx, ly))**2
        # Summing up,
        #    with d_p2_from_g distributed on x/y axis according to the slope,
        # *** add more rows if ADJ_OFFSETS were expanded ***
        # dd_p_?_k now store directions in form of +/-1
        dd_p_x_k, dd_p_y_k = float32(1), float32(1)
        if   k == 1 or k == 2:
            grad2 = grad2_k + grad_y**2
            fac = (grad2_k/grad2) if grad2 else float32(1)
            dd_p_x_k = _get_sign_cudev(ADJ_OFFSETS[k][0])
            dd_p_y_k = _get_sign_cudev(grad_y)
        elif k == 3 or k == 4:
            grad2 = grad2_k + grad_x**2
            # fac is about x, so do (1-*)
            fac = (1 - grad2_k/grad2) if grad2 else float32(1)
            dd_p_x_k = _get_sign_cudev(grad_x)
            dd_p_y_k = _get_sign_cudev(ADJ_OFFSETS[k][1])
            
        # add momentum from gravity
        dd_p_x_k *= _signed_sqrt_cudev(d_p2_from_g_pf2k  * fac)  * fac_k
        dd_p_y_k *= _signed_sqrt_cudev(d_p2_from_g_pf2k*(1-fac)) * fac_k

    return dd_p_x_k, dd_p_y_k, ek_from_g_k



@cuda.jit(device=True, fastmath=True)
def _move_fluid_cudev(
    # in/out
    d_stats_sarr,
    # in
    stats_sarr,
    ti, tj,
    not_at_outer_edge : bool_,
    z_res : float32,
    lx    : float32, ly    : float32,
    grad_x: float32, grad_y: float32,
    flow_eff  : float32,
    rho_sedi  : float32,
    v_cap     : float32,
    v_damping : float32,
    g_eff     : float32,
    erosion_eff   : float32,
    erosion_brush : npt.NDArray[np.float32],
    capa_fac      : float32,
    capa_fac_v    : float32,
    capa_fac_slope: float32,
    capa_fac_slope_cap: float32,
):
    """Move fluids (a.k.a. water (aqua) + sediments (sedi)).

    Overwrite the changes to d_stats_sarr (but does not init it).

    Warning: Do not use cuda.syncthreads() in device functions,
        since we are returning early in some cases.

    ---------------------------------------------------------------------------
    """

    # - init -
    
    # 5 elems **of/for** adjacent locations **from** this [i, j] location
    d_hs_local = cuda.local.array(N_ADJ_P1, dtype=float32)  # amount of flows

    
    if not not_at_outer_edge:
        return
    
    stat = stats_sarr[ti, tj]
    z0, h0 = _get_z_cudev(stat), _get_h_cudev(stat)
    p_x, p_y = stat['p_x'], stat['p_y']

    if not h0:    # stop if no water
        return


    # - move fluids -
    #--------------------------------------------------------------------------

    # init temp vars
    rho0 = _get_rho_cudev(stat, rho_sedi)
    m0 = _get_m_cudev(stat, rho_sedi)
    p2_0 = p_x**2 + p_y**2    # momentum squared


    # -- get d_hs_local (positive, == -d_hs_local[0])

    # --- step 1: turning
    #    Done in _erode_rainfall_evolve_cuda_sub() before
        
    # --- step 2: init vars
    d_h_tot = float32(0)
    p0 = math.sqrt(p2_0)
    v0_capped = _get_v_cudev(stat, z_res, rho_sedi, v_cap)
    # h0_p_tot & h0_g_k: fraction of the h0 reserved
    #    for momentum- & gravity- based movement per adjacent cell
    # Warning: h0_p_tot + h0_g_k * (N_ADJ_P1-1) < h0 if v0 < v_cap!
    # However, all h0_p_tot will be gone
    h0_p_tot = h0 * (float32(1) - flow_eff) * (v0_capped / v_cap)
    # note for h0_g_k: at least 1/N_ADJ_P1 part of it is reserved to stay
    #    the rest may stay too based on terrain
    h0_g_k = h0 * flow_eff / float32(N_ADJ_P1-1)
    # momentum gain (from kinetic energy debt; negative)
    d_p2_from_ek_pf2k = 2 * m0 * stat['ekin']    # _pf2k means "per (fac_k)**2"
    ek_gain_paid = float32(0)    # Kinetic energy gain of the system
    for k in range(1, N_ADJ_P1):
        tki, tkj = _get_tkij_cudev(ti, tj, k)
        zk = _get_z_cudev(stats_sarr[tki, tkj])
        d_h_p_k, d_h_g_k = float32(0), float32(0)    # dh from p and from g
        d_p_x_k, d_p_y_k = float32(0), float32(0)    # dp in x and y axis
        
        # --- step 3: momentum-based movements
        if p2_0 and h0_p_tot:
            if   (k == 1 and p_x < 0) or (k == 2 and p_x > 0):
                d_h_p_k = (p_x**2/p2_0) * h0_p_tot
            elif (k == 3 and p_y < 0) or (k == 4 and p_y > 0):
                d_h_p_k = (p_y**2/p2_0) * h0_p_tot
            else:
                d_h_p_k = float32(0)
            if d_h_p_k:    # set momentum transfer
                d_p_x_k = p_x * (d_h_p_k / h0)
                d_p_y_k = p_y * (d_h_p_k / h0)

        
        # --- step 4: gravity-based movements
        #    (always allowed)
        d_h_g_k = min(max((z0 - zk)/2, 0), h0_g_k)

        
        # --- step 5: calc momentum changes
        d_h_k = d_h_p_k + d_h_g_k
        dd_p_x_k, dd_p_y_k, ek_from_g_k = _get_d_p_from_g_cudev(
            stats_sarr, ti, tj, k, stat, z0, h0, m0, p2_0,
            d_h_k, grad_x, grad_y, lx, ly, g_eff)
        # temporarily store d_p_x_k as dd_p_x_k
        dd_p_x_k += d_p_x_k; dd_p_y_k += d_p_y_k
        if d_h_k > 0 and (    # total energy for the moved part must >= 0
            # note: it's d_p_x_k, not dd_p_x_k,
            #    because that part has already been counted in ek_from_g_k
            d_p_x_k **2 + d_p_y_k**2 + 2 * rho0 * d_h_k * ek_from_g_k < 0
            ):
            # reject momentum-based movement
            d_h_p_k = float32(0)
            # cancel out momentum
            if k == 1 or k == 2:
                d_stats_sarr[ti, tj, 0]['p_x'] -= d_p_x_k #*2
            else:
                d_stats_sarr[ti, tj, 0]['p_y'] -= d_p_y_k #*2
            # re-calculate
            d_h_k = d_h_p_k + d_h_g_k
            dd_p_x_k, dd_p_y_k, ek_from_g_k = _get_d_p_from_g_cudev(
                stats_sarr, ti, tj, k, stat, z0, h0, m0, p2_0,
                d_h_k, grad_x, grad_y, lx, ly, g_eff)
        # confirm adding momentum
        d_p_x_k = dd_p_x_k; d_p_y_k = dd_p_y_k
                
        # now subtract the energy debt
        if d_p2_from_ek_pf2k:
            fac_k = d_h_k / h0    # move factor
            d_p2_k = d_p_x_k**2 + d_p_y_k**2
            d_p2_paid_back = min(    # positive
                -d_p2_from_ek_pf2k*fac_k**2,
                d_p2_k,    # upper limit
            )
            fac = 1-_signed_sqrt_cudev(d_p2_paid_back / d_p2_k)
            if math.isfinite(fac):
                d_p_x_k *= fac
                d_p_y_k *= fac
                if m0 and fac_k:
                    d_stats_sarr[ti, tj, 0]['ekin'] += (
                        d_p2_paid_back / (2*m0*fac_k))

        # --- step 6: summing up
        d_stats_sarr[tki, tkj, k]['p_x'] = d_p_x_k
        d_stats_sarr[tki, tkj, k]['p_y'] = d_p_y_k
        d_stats_sarr[ti, tj, 0]['p_x'] -= d_p_x_k
        d_stats_sarr[ti, tj, 0]['p_y'] -= d_p_y_k
        ek_gain_paid += ek_from_g_k    # kinetic energy added from gravity
        d_hs_local[k] = d_h_k    # d_h_p_k + d_h_g_k
        d_h_tot += d_h_k
    d_hs_local[0] = -d_h_tot


    # -- parse amount of fluid into amount of sedi, aqua, ekin
    if d_h_tot: # only do things if something actually moved
        
        # factions that flows away:
        d_se_fac = stat['sedi'] / h0
        d_aq_fac = stat['aqua'] / h0  # should == 1 - d_se_fac
        
        # making sure energy is conserved
        ek_gain_actual = float32(0)
        for k in range(1, N_ADJ_P1):
            tki, tkj = _get_tkij_cudev(ti, tj, k)
            zk = _get_z_cudev(stats_sarr[tki, tkj])
            d_h_k = d_hs_local[k]
            ek_gain_actual += d_h_k * (2*(z0 - zk) - d_h_k)
        ek_gain_actual = (d_aq_fac + rho_sedi * d_se_fac) * g_eff / 2 * (
            ek_gain_actual - d_h_tot**2)
        d_stats_sarr[ti, tj, 0]['ekin'] += ek_gain_actual - ek_gain_paid

        # calc changes from above
        for k in range(N_ADJ_P1):
            tki, tkj = _get_tkij_cudev(ti, tj, k)
            d_h_k = d_hs_local[k]
            if d_h_k:
                d_stats_sarr[tki, tkj, k]['sedi'] = d_se_fac * d_h_k
                d_stats_sarr[tki, tkj, k]['aqua'] = d_aq_fac * d_h_k
                # Note: kinetic energy change from adjacent cells
                # have already been incorporated into the p momentum transfer
    
    
    # - erode -
    #--------------------------------------------------------------------------
    if erosion_eff:    # h0 > 0 must be True from before
        # note: upward slopes are ignored
        slope = min(math.sqrt(grad_x**2 + grad_y**2), capa_fac_slope_cap)
        # get slope (of the soil surface instead of fluid surface)
        # if d_h_tot:
        #     # averaging slope, weighted by movement
        #     # note: slope can be negative
        #     for k in range(1, N_ADJ_P1):
        #         tki, tkj = _get_tkij_cudev(ti, tj, k)
        #         d_h_k = d_hs_local[k]
        #         slope += d_h_k * (
        #             stat['soil'] - stats_sarr[tki, tkj]['soil']
        #         ) / _get_l_cudev(k, lx, ly)
        #     slope /= d_h_tot
        # get capa
        capa = _get_capa_cudev(
            stat, v0_capped, slope, v_cap,
            capa_fac, capa_fac_v, capa_fac_slope, capa_fac_slope_cap)
        # do erosion / deposition
        if capa > stat['sedi']:    # erode
            # get erosion amount for this cell
            # dd_se_ref: erodable dirt from soil to sedi
            #    here dd_se_ref > 0
            dd_se_ref = min(
                erosion_eff * (capa - stat['sedi']),
                stat['soil'],    # cannot dredge through bedrock
            )
            soil_min_new = stat['soil'] - dd_se_ref * erosion_brush[0]
            # apply erosion brush
            for k in range(N_ADJ_P1):
                tki, tkj = _get_tkij_cudev(ti, tj, k)
                dd_se = min(
                    dd_se_ref * erosion_brush[k],
                    # cannot dredge through bedrock
                    stats_sarr[tki, tkj]['soil'],
                    # do not dredge lower than the central cell
                    max(0, stats_sarr[tki, tkj]['soil'] - soil_min_new),
                )
                d_stats_sarr[tki, tkj, k]['sedi'] += dd_se
                d_stats_sarr[tki, tkj, k]['soil'] -= dd_se
        else:    # deposit
            # only deposit the sediments on this cell
            # to make sure holes can be easily filled
            # dd_se: dirt converted from soil to sedi
            #    here dd_se < 0
            #    erosion_eff should be <= 1.
            dd_se = erosion_eff * (capa - stats_sarr[ti, tj]['sedi'])
            d_stats_sarr[ti, tj, 0]['sedi'] += dd_se
            d_stats_sarr[ti, tj, 0]['soil'] -= dd_se
        
    return
    


#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Evolve
#-----------------------------------------------------------------------------#


@cuda.jit(fastmath=True)
def _erode_rainfall_evolve_cuda_sub(
    # in/out
    stats_cuda, edges_cuda, d_stats_cuda,
    # in
    i_layer_read: int,
    z_max: float32,
    z_res: float32,
    lx: float32,
    ly: float32,
    evapor_rate   : float32,
    flow_eff      : float32,
    turning_gradref   : float32,
    rho_sedi : float32,
    v_cap         : float32,
    v_damping     : float32,
    g_eff         : float32,
    erosion_eff   : float32,
    erosion_brush : npt.NDArray[np.float32],
    capa_fac : float32,
    capa_fac_v: float32,
    capa_fac_slope: float32,
    capa_fac_slope_cap: float32,
):
    """Evolving 1 step.

    stats_cuda: (nx, ny, 2)-shaped
    edges_cuda: (nx, ny)-shaped
    d_stats_cuda: (nx, ny, 2)-shaped
    
    z_max:
        z_min is assumed to be zero.

    code here is mostly memory management.
    See _move_fluid_cudev(...) for actual physics part of the code.
        
    ---------------------------------------------------------------------------
    """
    
    # - define shared data structure -
    # (shared by the threads in the same block)
    # Note: the shared array 'shape' arg
    #    must take integer literals instead of integer
    stats_sarr = cuda.shared.array(
        shape=(CUDA_TPB_X, CUDA_TPB_Y), dtype=ErosionStateDataDtype)
    edges_sarr = cuda.shared.array(
        shape=(CUDA_TPB_X, CUDA_TPB_Y), dtype=ErosionStateDataDtype)

    # 5 elems **for** this [i, j] location **from** adjacent locations:
    #    0:origin, 1:pp, 2:pm, 3:mp, 4:mm
    d_stats_sarr = cuda.shared.array(
        shape=(CUDA_TPB_X, CUDA_TPB_Y, N_ADJ_P1),
        dtype=ErosionStateDataDtype)

    # # for disregarding edge
    # nan_stat_cuda = cuda.const.array_like(_NAN_STATS)
    # edge_nan = nan_stat_cuda[0]

    
    # - get thread coordinates -
    nx, ny, _ = stats_cuda.shape
    i, j, ti, tj = _get_coord_cudev()
    i_layer_write = 1 - i_layer_read
    for k in range(N_ADJ_P1):
        _set_stat_zero_cudev(d_stats_sarr[ti, tj, k])

    if _is_outside_map_cudev(nx, ny, i, j):
        return

    # - preload data -
    stats_sarr[ti, tj] = stats_cuda[i, j, i_layer_read]
    stat = stats_sarr[ti, tj]    # by reference
    edges_sarr[ti, tj] = edges_cuda[i, j]
    edge = edges_sarr[ti, tj]
    not_at_outer_edge : bool_ = not _is_at_outer_edge_cudev(
        nx, ny, i, j, ti, tj)

    
    # - rain & evaporate -
    v_damping_fac = float32(1)
    if evapor_rate > 0 and stat['aqua'] > z_res:
        # rain doesn't remove kinetic energy but evaporation does
        v_damping_fac -= evapor_rate / stat['aqua']
    # rain / evaporate
    stat['aqua'] = stat['aqua'] - evapor_rate
    # damping momentum / kinetic energy
    v_damping_fac -= v_damping
    if stat['aqua']:
        if not math.isnan(edge['p_x']):
            stat['p_x'] *= v_damping_fac
        if not math.isnan(edge['p_y']):
            stat['p_y'] *= v_damping_fac
        if not math.isnan(edge['ekin']) and stat['ekin'] > 0:
            stat['ekin'] *= v_damping_fac**2
            
    # - normalize -
    stat = _normalize_stat_cudev(stat, edge, z_max, z_res, rho_sedi)

    cuda.syncthreads()
    
    # - turning -
    # figuring out the local gradient (based on soil height)
    sign_x = -1 if (_get_zfg_cudev(  stats_sarr[ti-1, tj])
                    < _get_zfg_cudev(stats_sarr[ti+1, tj])) else 1
    sign_y = -1 if (_get_zfg_cudev(  stats_sarr[ti, tj-1])
                    < _get_zfg_cudev(stats_sarr[ti, tj+1])) else 1
    # note: the sign of grad_x, grad_y encodes direction of the slope
    #    rather than downward/upward. Upward is capped at 0.
    grad_x = sign_x * max(
        _get_zfg_cudev(stat) - _get_zfg_cudev(stats_sarr[ti+sign_x, tj]),
        0) / lx
    grad_y = sign_y * max(
        _get_zfg_cudev(stat) - _get_zfg_cudev(stats_sarr[ti, tj+sign_y]),
        0) / ly
    # turning v direction based on local gradient
    p0 = _get_p_cudev(stat)
    if p0:
        # set the scale of the gradient vector
        if turning_gradref:
            # re-align momentum based on turning
            stat['p_x'] += grad_x / turning_gradref * p0
            stat['p_y'] += grad_y / turning_gradref * p0
        elif grad_x or grad_y:
            # re-align only if there is local gradient guide
            stat['p_x'], stat['p_y'] = grad_x, grad_y
        # renormalize p back to p0
        fac = p0 / math.sqrt(stat['p_x']**2 + stat['p_y']**2)    # p0 / p_new
        if math.isfinite(fac):
            stat['p_x'] *= fac
            stat['p_y'] *= fac

    # write results (just in case)
    stats_sarr[ti, tj] = stat

    cuda.syncthreads()

    # - move water -
    _move_fluid_cudev(
        # out
        d_stats_sarr,
        # in
        stats_sarr, ti, tj, not_at_outer_edge, z_res, lx, ly, grad_x, grad_y,
        flow_eff, rho_sedi, v_cap, v_damping, g_eff,
        erosion_eff, erosion_brush, capa_fac, capa_fac_v,
        capa_fac_slope, capa_fac_slope_cap,
    )

    cuda.syncthreads()
    
    # - write data back -
    # summarize
    if not_at_outer_edge:
        for k in range(N_ADJ_P1):
            # could use optimization
            stat = _add_stats_cudev(
                stat, d_stats_sarr[ti, tj, k], edge, rho_sedi, g_eff)
        # write back
        stats_cuda[i, j, i_layer_write] = stat
    else:
        # saving boundary values aside
        # *** Warning: Re-write this if ADJ_OFFSETS is changed! ***
        # Remember that k=0 is the origin pixel, k=1...4 are the adjacent ones
        idim = _get_outer_edge_idim_cudev(nx, ny, i, j, ti, tj)
        if idim >= 0:
            k = _get_outer_edge_k_cudev(nx, ny, i, j, ti, tj)
            d_stats_cuda[i, j, idim] = d_stats_sarr[ti, tj, k]





@cuda.jit(device=True, fastmath=True)
def _erode_rainfall_evolve_cuda_final_sub(
    # in/out
    stats_cuda,
    d_stats_cuda,
    # in
    edges_cuda,
    nx, ny, i, j,
    i_layer_read: int,
    idim: int,
    rho_sedi: float32,
    g_eff   : float32,
):
    if not _is_outside_map_cudev(nx, ny, i, j):
        # preload data
        stat = stats_cuda[i, j, i_layer_read]
        edge = edges_cuda[i, j]
        # sum
        _add_stats_cudev(stat, d_stats_cuda[i, j, idim], edge, rho_sedi, g_eff)
        _set_stat_zero_cudev(d_stats_cuda[i, j, idim])
        # write back
        stats_cuda[i, j, i_layer_read] = stat
    return



@cuda.jit(fastmath=True)
def _erode_rainfall_evolve_cuda_final(
    # in/out
    stats_cuda,
    d_stats_cuda,
    # in
    edges_cuda,
    i_layer_read: int,
    z_res: float32,
    rho_sedi: float32,
    g_eff   : float32,
):
    """Finalizing by adding back the d_stats_cuda to stats_cuda.

    BlockDim is assumed to be 1.
    *** Warning: Re-write this if ADJ_OFFSETS is changed! ***

    ---------------------------------------------------------------------------
    """
    # *** Pending optimization/overhaul ***
    #    - should not store the entire grid, just the edges
    
    # - get thread coordinates -
    nx, ny, _ = stats_cuda.shape
    p = cuda.grid(1)

    # Do x first
    j = p
    for ki in range(nx//(CUDA_TPB_X-2)):
        i = ki * (CUDA_TPB_X-2)
        _erode_rainfall_evolve_cuda_final_sub(
            stats_cuda, d_stats_cuda,
            edges_cuda, nx, ny, i, j, i_layer_read, 0,
            rho_sedi, g_eff)
        i += 1
        _erode_rainfall_evolve_cuda_final_sub(
            stats_cuda, d_stats_cuda,
            edges_cuda, nx, ny, i, j, i_layer_read, 0,
            rho_sedi, g_eff)
    # the edge on the other side
    i = nx-1
    _erode_rainfall_evolve_cuda_final_sub(
        stats_cuda, d_stats_cuda,
        edges_cuda, nx, ny, i, j, i_layer_read, 0,
        rho_sedi, g_eff)
    # Now do y
    i = p
    for ki in range(ny//(CUDA_TPB_Y-2)):
        j = ki * (CUDA_TPB_Y-2)
        _erode_rainfall_evolve_cuda_final_sub(
            stats_cuda, d_stats_cuda,
            edges_cuda, nx, ny, i, j, i_layer_read, 1,
            rho_sedi, g_eff)
        j += 1
        _erode_rainfall_evolve_cuda_final_sub(
            stats_cuda, d_stats_cuda,
            edges_cuda, nx, ny, i, j, i_layer_read, 1,
            rho_sedi, g_eff)
    # the edge on the other side
    j = ny-1
    _erode_rainfall_evolve_cuda_final_sub(
        stats_cuda, d_stats_cuda,
        edges_cuda, nx, ny, i, j, i_layer_read, 1,
        rho_sedi, g_eff)


def erode_rainfall_evolve_cuda(
    steps_config: None|int|list[dict|int],
    stats : ErosionStateDataType,
    edges : ErosionStateDataType,
    z_max : float32,
    z_res : float32,
    pix_widxy : tuple[float32, float32],
    pars  : dict,
    verbose: VerboseType = True,
    **kwargs,
) -> ErosionStateDataType:
    """Do rainfall erosion- evolve through steps.

    'kwargs' are not used.

    z_res:
        must > 0
    ---------------------------------------------------------------------------
    """

    # - init cuda -
    nx, ny = stats.shape
    lx, ly = pix_widxy
    cuda_bpg, cuda_tpb = get_cuda_bpg_tpb(nx, ny)

    cuda_tpb_final = CUDA_TPB_X * CUDA_TPB_Y
    cuda_bpg_final = (max(nx, ny) + cuda_tpb_final - 1) // cuda_tpb_final

    # add 2 layers of stats: one for reading, one for writing
    # (to avoid interdependence between threads)
    stats_cuda = cuda.to_device(np.stack((stats, stats), axis=-1))
    i_layer_read: int = 0    # 0 or 1; the other one is for writing 
    edges_cuda = cuda.to_device(edges)
    # d_stats_cuda: for caching tempeorary results from the edges
    # last dim has 2 elems, for i and j direction
    # assuming CUDA_TPB >= 2
    d_stats_cuda = cuda.to_device(np.zeros(
        (nx, ny, 2), dtype=ErosionStateDataDtype))
    
    
    # - init pars -
    # steps_config
    if steps_config is None: steps_config = [{}]
    try: iter(steps_config)    # is list?
    except TypeError:
        pars['n_step']['value'] = steps_config
        steps_config = [{}]
    erosion_brush = cuda.to_device(pars['erosion_brush']['value'])
    n_step_tot: int = 0
    
    # - run -
    for step_config in steps_config:
        # update parameters
        if isinstance(step_config, dict):
            for k, v in step_config.items():
                pars[k]['value'] = v
                if k == 'erosion_brush':
                    erosion_brush = cuda.to_device(
                        pars['erosion_brush']['value'])
        else:  # assume int
            if verbose:
                print("*   Warning: instead of inputing int (assumed),"
                      + "please consider input steps_config as list of dict"
                      + "that updates self.pars for clarity.")
            pars['n_step']['value'] = step_config
            
        pars_v = {k: v['value'] for k, v in pars.items()}
        # dt = min(lx, ly) / pars_v['v_cap']    # time step
        
        # run loops
        n_step = pars_v['n_step']
        for s in range(n_step):
            n_step_tot += 1
            # *** add more sophisticated non-uniform rain code here! ***
            cuda.synchronize()
            _erode_rainfall_evolve_cuda_sub[cuda_bpg, cuda_tpb](
                stats_cuda, edges_cuda, d_stats_cuda,
                i_layer_read, z_max, z_res, lx, ly,
                pars_v['evapor_rate'],
                pars_v['flow_eff'],
                pars_v['turning_gradref'],
                pars_v['rho_sedi'],
                pars_v['v_cap'],
                pars_v['v_damping'],
                pars_v['g_eff'],
                pars_v['erosion_eff'],
                erosion_brush,
                pars_v['capa_fac'],
                pars_v['capa_fac_v'],
                pars_v['capa_fac_slope'],
                pars_v['capa_fac_slope_cap'],
            )
            cuda.synchronize()
            i_layer_read = 1 - i_layer_read
            # add back edges
            _erode_rainfall_evolve_cuda_final[cuda_bpg_final, cuda_tpb_final](
                stats_cuda, d_stats_cuda, edges_cuda, i_layer_read, z_res,
                pars_v['rho_sedi'], pars_v['g_eff'])
    
    # - return -
    cuda.synchronize()
    stats = stats_cuda[:, :, i_layer_read].copy_to_host()
    print("WARNING: *** Cuda version of this func not yet complete. ***")
    return stats, n_step_tot



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
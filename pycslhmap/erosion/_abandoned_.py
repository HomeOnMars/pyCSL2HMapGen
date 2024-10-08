#!/usr/bin/env python
# coding: utf-8

"""Abandoned codes- for archive purposes only.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (built-in)
from typing import Self
import random

# imports (3rd party)
from numba import jit, prange, cuda
import numpy as np
from numpy import typing as npt

# imports (my libs)
from ..hmap.util import (
    _minabs, _norm, _hat,
    _ind_to_pos, _pos_to_ind_f, _pos_to_ind_d,
    _get_z_and_dz,
)
from .nbjit import _erode_rainfall_init_sub_nbjit
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)



#-----------------------------------------------------------------------------#
#    Erosion: Rainfall (Abandoned codes)
#-----------------------------------------------------------------------------#


@cuda.jit(device=True, fastmath=True)
def _get_d_hs_cudev(
    # out
    d_hs_local,
    # in
    stat,
    stats_sarr,
    ti, tj,
    z_res: float32,
    flow_eff: float32,
) -> float32:
    """Get fluid movements plan.

    *   Note: Need to expand this if more rows are added to ADJ_OFFSETS!

    Return: d_h_tot
    
    ---------------------------------------------------------------------------
    """
    z0, h0 = _get_z_cudev(stat), _get_h_cudev(stat)
    d_h_tot = float32(0.)
    k = 0
    for ki in range(N_ADJ_P1//2):
        tki, tkj = _get_tkij_cudev(ti, tj, 2*ki+1)
        zk = _get_z_cudev(stats_sarr[tki, tkj])
        d_h_k1 = z0 - zk
        tki, tkj = _get_tkij_cudev(ti, tj, 2*ki+2)
        zk = _get_z_cudev(stats_sarr[tki, tkj])
        d_h_k2 = z0 - zk
        if d_h_k1 > d_h_k2:
            k = 2*ki+1
            d_hs_local[2*ki+2] = float32(0.)
        else:
            k = 2*ki+2
            d_hs_local[2*ki+1] = float32(0.)
        tki, tkj = _get_tkij_cudev(ti, tj, k)
        zk = _get_z_cudev(stats_sarr[tki, tkj])
        d_h_k = min((z0 - zk)/3*flow_eff, h0/2)
        if d_h_k < z_res: d_h_k = float32(0.)
        d_hs_local[k] = d_h_k
        d_h_tot += d_h_k

    d_hs_local[0] = -d_h_tot
    return d_h_tot



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



# with complex moving mechanic that fails
@cuda.jit(device=True, fastmath=True)
def _device_move_fluid(
    # out
    d_stats_local,
    # in
    stats_local,
    zero_stat,
    z_res: float32,
    flow_eff: float32,
    rho_sedi: float32,
    g_eff: float32,
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

    # - flow weights -
    # init zs_local and hws_local
    n_flowable = 0
    d_z_max = float32(0.)     # maximum height difference
    for k in range(1, N_ADJ_P1):
        zk, _ = _device_get_z_and_h(stats_local[k])
        zs_local[k] = zk
        # hws_local: weight of the fluid to be moved
        #    since fluid speed at given h is propotional to sqrt(h),
        #    the total water moved at given time span
        #    towards a given direction should be prop to h**1.5
        hw_k = (
            max(min(z0 - zk, h0), float32(0.))**float32(1.5)
            if z0 - zk >= z_res else float32(0.)
        )
        hws_local[k] = hw_k
        if hw_k:
            n_flowable += 1
            d_z_max = max(d_z_max, z0 - zk)
            
    if not n_flowable:    # stop if nothing can actually move
        return
        
    # init d_hs_local
    for k in range(N_ADJ_P1):
        d_hs_local[k] = float32(0.)

    # get d_hs_local
    #    i.e. how much fluid is flowing to adjacent cells
    z0_now = z0    # tmp storage of the height for parts of flowable fluids
    d_h_tot = float32(0.)
    for _ in range(n_flowable):
        
        # find the minimum height drop cell
        k_min = 0  # index at minimum height
        d_z_min = d_z_max     # minimum height difference
        d_h_min = float32(0.) # amount of flowable water for this part
        hw_tot = float32(0.)
        hw_at_k_min = float32(0.)
        for k in range(1, N_ADJ_P1):
            hw_k = hws_local[k]
            if hw_k:
                d_z = z0_now - zs_local[k] - d_hs_local[k]
                # hws_local: weight of the fluid to be moved
                #    since fluid speed at given h is propotional to sqrt(h),
                #    the total water moved at given time span
                #    towards a given direction should be prop to h**1.5
                hw_k = (
                    max(min(d_z, h0 - d_h_tot), float32(0.))**float32(1.5)
                    if d_z >= z_res else float32(0.)
                )
                hws_local[k] = hw_k
                hw_tot += hw_k    # re-calibrate
                if  d_z_min > d_z and d_z > 0:
                    d_z_min = d_z
                    k_min = k
                    hw_at_k_min = hw_k
                    d_h_min = min(d_z_min, h0 - d_h_tot)
                
        # calc the flows for this part of the flow
        d_h_now = float32(0.)  # sum of d_h_now_k
        hw_fac  = hw_tot + hw_at_k_min
        if not hw_fac: break  # safety check
        for k in range(1, N_ADJ_P1):
            hw_k = hws_local[k]
            if hw_k:    # actually can flow to
                d_h_now_k = d_h_min * hw_k / hw_fac * flow_eff
                if d_h_now_k < z_res: d_h_now_k = float32(0.)
                d_hs_local[k] += d_h_now_k
                d_h_now += d_h_now_k
        z0_now -= d_h_now
        d_h_tot += d_h_now
        hws_local[k_min] = float32(0.)
    #d_h_tot = z0 - z0_now
    d_hs_local[0] = -d_h_tot
    

    # parse amount of fluid into amount of sedi, aqua, ekin
    if d_h_tot:
        # only do things if something actually moved
        
        # factions that flows away:
        d_se_fac = stat['sedi'] / h0
        d_aq_fac = stat['aqua'] / h0  # should == 1 - d_se_fac
        # kinetic energy always flows fully away with current
        d_ek_fac_flow = flow_eff
        d_ek_fac_g = (
            d_aq_fac + rho_sedi * d_se_fac) * g_eff / 2
        
        # kinetic energy gain from gravity
        ek_tot_from_g = float32(0.)
        for k in range(1, N_ADJ_P1):
            d_h_k = d_hs_local[k]
            zk = zs_local[k]
            ek_tot_from_g += d_h_k * (2*(z0 - zk) - d_h_k)
        ek_tot_from_g = max(
            d_ek_fac_g * (ek_tot_from_g - d_h_tot**2),
            # safety cap *** BELOW IS GIVING AWAY FREE ENERGY ***
            # *** Fix this ***
            float32(0.),
        )
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
                #    g_eff * (h1**2 - h0**2)  / 2.
                if k:
                    d_ek = (
                        ek_tot_from_g + d_ek_fac_flow * stat['ekin']
                    ) * d_h_k / d_h_tot
                else:
                    # kinetic energy flow only (no g_eff gain)
                    # remove d_h_k / d_h_tot (which == -1.)
                    #    to avoid precision loss
                    #    resulting in negative stat['ekin']
                    d_ek = -d_ek_fac_flow * stat['ekin']
                d_stats_local[k]['ekin'] = d_ek
                # *** Fix kinetic energy here! ***
                # currently generates single high point stat['ekin']
                # for some reason
    return



def _erode_rainfall_init(
    data : npt.NDArray[np.float32],    # ground level
    spawners: npt.NDArray[np.float32],
    z_config: tuple[np.float32, np.float32, np.float32, np.float32],
    sub_func: Callable = _erode_rainfall_init_sub_nbjit,
):
    """Initialization for Rainfall erosion.

    Parameters
    ----------
    data: (npix_x, npix_y)-shaped numpy array
        initial height.

    spawners: (npix_x, npix_y)-shaped numpy array
        Constant level water spawners height (incl. ground)
        use np.zeros_like(data) as default input.

    z_config: tuple((z_min, z_sea, z_max, z_res))
        Minimum height allowed / Sea level / Maximum height allowed.
        *** Warning: z_sea = 0 will disable sea level mechanics ***
        
    sub_func: function
        Provide the function for the sub process.
        Choose between _erode_rainfall_init_sub_nbjit()   (CPU)
            and        _erode_rainfall_init_sub_cuda() (GPU)

    Returns
    -------
    ...
    edges: (npix_x+2, npix_y+2)-shaped numpy array
        Constant river source. acts as a spawner.
        By default, will init any zero elements as sea levels at edges.
    ... 

    ---------------------------------------------------------------------------
    """
    
    npix_x, npix_y = data.shape
    z_config = np.asarray(z_config, dtype=np.float32)
    z_min, z_sea, z_max, z_res = z_config

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
    zs, n_cycles = sub_func(soils, edges, z_range=z_max-z_min)
    
    # fix data
    aquas[1:-1, 1:-1] = (zs - soils)[1:-1, 1:-1]
    ekins = np.zeros_like(soils)
    sedis = np.zeros_like(soils) # is zero because speed is zero
    
    return soils, aquas, ekins, sedis, edges, n_cycles









#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Evolve (nbjit)
#-----------------------------------------------------------------------------#


@jit(nopython=True, fastmath=True)
def _erode_rainfall_get_slope_dz(
    z_mi: np.float32, z_ne1: np.float32, z_ne2: np.float32,
    slope_facs: tuple[np.float32, np.float32],
) -> np.float32:
    """Return the dz used for slope calc"""
    downhill_fac, uphill_fac = slope_facs
    dz1 = z_mi - z_ne1
    dz1 *= downhill_fac if dz1 > 0 else uphill_fac
    dz2 = z_mi - z_ne2
    dz2 *= downhill_fac if dz2 > 0 else uphill_fac
    return max(abs(dz1), abs(dz2))
    


@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_get_capas(
    zs   : npt.NDArray[np.float32],
    aquas: npt.NDArray[np.float32],
    ekins: npt.NDArray[np.float32],
    sedis: npt.NDArray[np.float32],
    pix_widxy: tuple[np.float32, np.float32],
    sed_cap_fac: np.float32,
    slope_facs : tuple[np.float32, np.float32],
    v_cap: np.float32,
) -> npt.NDArray[np.float32]:
    """Get sediment capacity.

    Parameters
    ----------
    ...
    sed_cap_fac : float
        Sediment capacity factor of the river.
        Limits the maximum of the sediemnt capacity.

    slope_facs: tuple((downhill_fac, uphill_fac))
        factor to used in the slope calculation. should be 0. < fac < 1.
        if downhill_fac > upfill_fac, will make more gently sloped hills;
        else, will make more cliffs

    v_cap: float
        Characteristic velocity for sediment capacity calculations, in m/s.
        Used to regulate the velocity in capas calc,
        So its influence flatten out when v is high.
        
    ---------------------------------------------------------------------------
    """
    npix_x, npix_y = zs.shape[0]-2, zs.shape[1]-2
    pix_wid_x, pix_wid_y = pix_widxy

    capas = np.zeros_like(zs)
    
    for i in prange(1, npix_x+1):
        for j in prange(1, npix_y+1):
            aq = aquas[i, j]
            if not np.isclose(aq, 0.):
                z  = zs[i, j]
                ek = ekins[i, j]
                se = sedis[i, j]
                # average velocity (regulated to 0. < slope < 1.)
                v_avg = (6.*ek/aq)**0.5/2.
                v_fac = np.sin(np.atan(v_avg/v_cap))
                # get slope (but regulated to 0. < slope < 1.)
                dz_dx = _erode_rainfall_get_slope_dz(
                    zs[i, j], zs[i+1, j], zs[i-1, j], slope_facs,
                ) / (pix_wid_x*2)
                dz_dy = _erode_rainfall_get_slope_dz(
                    zs[i, j], zs[i, j+1], zs[i, j-1], slope_facs,
                ) / (pix_wid_y*2)
                slope = np.sin(np.atan((dz_dx**2 + dz_dy**2)**0.5))

                capas[i, j] = sed_cap_fac * (aq-se) * v_fac * slope

    return capas
    


@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_evolve_sub_nb(
    soils: npt.NDArray[np.float32],
    aquas: npt.NDArray[np.float32],
    ekins: npt.NDArray[np.float32],
    sedis: npt.NDArray[np.float32],
    edges: npt.NDArray[np.float32],
    pix_widxy: tuple[np.float32, np.float32],
    z_config : tuple[np.float32, np.float32, np.float32, np.float32],
    flow_eff   : np.float32,
    visco_kin_range: tuple[np.float32, np.float32],
    sed_cap_fac: np.float32,
    erosion_eff: np.float32,
    diffuse_eff: np.float32,
    hole_depth : np.float32,
    slope_facs : tuple[np.float32, np.float32],
    v_cap: np.float32,
    g_eff : np.float32,
):
    """Numba version of the sub process for rainfall erosion evolution.

    See _erode_rainfall_evolve for parameters info.
    ---------------------------------------------------------------------------
    """
    # - init -
    # remember len(soils) is npix+2 because we added edges
    N_ADJ : int = 4    # number of adjacent cells
    npix_x, npix_y = soils.shape[0]-2, soils.shape[1]-2
    pix_wid_x, pix_wid_y = pix_widxy
    _, _, _, z_res = z_config
    visco_kin_aqua, visco_kin_soil = visco_kin_range
    # the boundary will not be changed
    edges_inds_x, edges_inds_y = np.where(edges)
    edges_n = len(edges_inds_x)


    # - update height and gradient -
    zs = soils + aquas

    # - init -
    soils_dnew = np.zeros_like(soils)
    aquas_dnew = np.zeros_like(soils)
    ekins_dnew = np.zeros_like(soils)
    sedis_dnew = np.zeros_like(soils)

    capas = _erode_rainfall_get_capas(
        zs, aquas, ekins, sedis, pix_widxy, sed_cap_fac, slope_facs, v_cap)
    
    for i in range(1, npix_x+1):
        for j in range(1, npix_y+1):
            if aquas[i, j] >= z_res: # only do things if there is water
                
                # - init -

                # optimization required
                # idea: replace array with individual variables?
                # idea: change to a more approximate moving method?
                # idea: cuda GPU-acceleration?
                
                dzs = np.empty(N_ADJ)    # altitude change
                wms = np.zeros(N_ADJ)    # water moved
                eks = np.zeros(N_ADJ)    # kinetic energy gained
                z  = zs[i, j]
                aq = aquas[i, j]
                ek = ekins[i, j]
                se = sedis[i, j]
                ca = capas[i, j]
                soil = soils[i, j]
                # arbitrarily define the array as
                dzs[0] = zs[i-1, j]
                dzs[1] = zs[i+1, j]
                dzs[2] = zs[i, j-1]
                dzs[3] = zs[i, j+1]
                dzs -= z

                
                # - move water -

                wms_w = np.where(    # wms_w: water level difference
                    dzs >= 0, 0.,    # water will not flow upward
                    -dzs)
                wrc = 0.    # water removed from center cell
                n = float(N_ADJ)
                inds = np.argsort(wms_w)
                for ik, k in enumerate(inds):
                    # smallest dropped height first, then higher ones
                    # required water to be moved for balancing the height:
                    # wmfe: water moved to each cells
                    wmfe = wms_w[k] / (1. + n)
                    wrc_d = wmfe * n    # more water to be removed
                    if wrc + wrc_d >= aq:
                        # cannot give more than have
                        wrc_d = aq - wrc
                        wrc = aq
                        wmfe = wrc_d / n
                        for jk in range(ik, N_ADJ):
                            kj = inds[jk]
                            wms[kj] += wmfe * flow_eff
                            eks[kj] += wmfe*g_eff*flow_eff*(
                                wms_w[kj] + (wrc_d + wmfe)*flow_eff/2.
                            )
                            #wms_w[kj] -= wms_w[k]
                        break
                    # otherwise
                    wrc += wrc_d
                    # optimizing below:
                    # wms[inds[ik:]] += wmfe
                    # eks[inds[ik:]] += wmfe*g_eff*(wms_w[inds[ik:]] - wms_w[k]/2)
                    # wms_w[inds[ik:]] -= wms_w[k]
                    for jk in range(ik, N_ADJ):
                        kj = inds[jk]
                        wms[kj] += wmfe * flow_eff
                        eks[kj] += wmfe*g_eff*flow_eff*(
                            wms_w[kj] - (wrc_d + wmfe)*flow_eff/2.
                        )
                        wms_w[kj] -= wms_w[k]    # for next
                    n -= 1
                wrc = np.sum(wms)    # re-normalize

                ek_d = wrc / aq * ek
                se_d = wrc / aq * se
                if not np.isclose(wrc, 0.):
                    # move water
                    aquas_dnew[i,   j] -= wrc
                    aquas_dnew[i-1, j] += wms[0]
                    aquas_dnew[i+1, j] += wms[1]
                    aquas_dnew[i, j-1] += wms[2]
                    aquas_dnew[i, j+1] += wms[3]
                    # transfer kinetic energy
                    # will update ekins later
                    #ekins_dnew[i,   j] -= ek_d
                    ekins_dnew[i-1, j] += ek_d * (wms[0] / wrc) + eks[0]
                    ekins_dnew[i+1, j] += ek_d * (wms[1] / wrc) + eks[1]
                    ekins_dnew[i, j-1] += ek_d * (wms[2] / wrc) + eks[2]
                    ekins_dnew[i, j+1] += ek_d * (wms[3] / wrc) + eks[3]
                    # transfer sediments
                    sedis_dnew[i,   j] -= se_d
                    sedis_dnew[i-1, j] += se_d * (wms[0] / wrc)
                    sedis_dnew[i+1, j] += se_d * (wms[1] / wrc)
                    sedis_dnew[i, j-1] += se_d * (wms[2] / wrc)
                    sedis_dnew[i, j+1] += se_d * (wms[3] / wrc)

                # update ekin with the friction from viscosity
                # friction from viscosity
                #    Energy loss W_f = $ \\frac{\\partial^2 f}{\\partial x \\partial z} s dx dz $
                #        where the force per cross-section \\frac{\\partial^2 f}{\\partial x \\partial z} = mu * v(z) / z,
                #        with mu being the dynamic viscosity.
                #    So, W_f = s**2 * mu * vmax, thus
                #        W_f/rho/s**2 = (mu/rho)*vmax = (mu/rho) * sqrt(6*ek/aq)
                #        where mu/rho is the kinetic viscosity of water.
                # Note: we already know that the aq != 0
                aq_mi = aq - wrc    # mi: me (this pixel)
                se_mi = se + se_d
                if not np.isclose(aq_mi, 0):
                    visco_kin = max(
                        visco_kin_aqua,    # make sure it doesn't go negative
                        # getting muddy water viscosity
                        ((aq_mi - se_mi) * visco_kin_aqua + se_mi * visco_kin_soil) / aq_mi)
                    ek_d += visco_kin * (6*(ek-ek_d)/aq_mi)**0.5
                ek_d = min(ek_d, ek)    # make sure ekins don't go negative
                ekins_dnew[i, j] -= ek_d
                ek_mi = ek + ek_d    # mi: me (this pixel)

                
                # - diffusion -
                
                # ne: neightbour
                aq_ne = aquas[i+1, j]
                # aq_sl: shared water level between two pixels
                aq_sl = min(aq_mi + soil, aq_ne + soils[i+1, j]) - max(soil, soils[i+1, j])
                if aq_sl >= z_res and not np.isclose(aq_mi, 0.) and not np.isclose(aq_ne, 0.):
                    # diffusion can only happen when the water are in contact

                    ek_ne = ekins[i+1, j]
                    # asking the density of ekin to be similar over time
                    ek_diff = (aq_ne*ek_mi - aq_mi*ek_ne) / (aq_mi + aq_ne) * diffuse_eff
                    ek_diff = (
                        min(ek_diff,  ek_mi*aq_sl/aq_mi) if ek_diff > 0.0 else
                        max(ek_diff, -ek_ne*aq_sl/aq_ne)
                    )
                    ekins_dnew[i,   j] -= ek_diff
                    ekins_dnew[i+1, j] += ek_diff
                    
                    se_ne = sedis[i+1, j]
                    se_diff = (aq_ne*se_mi - aq_mi*se_ne) / (aq_mi + aq_ne) * diffuse_eff
                    se_diff = (
                        min(se_diff,  se_mi*aq_sl/aq_mi) if se_diff > 0.0 else
                        max(se_diff, -se_ne*aq_sl/aq_ne)
                    )
                    sedis_dnew[i,   j] -= se_diff
                    sedis_dnew[i+1, j] += se_diff

                aq_ne, aq_mi = aquas[i, j+1], aq - wrc
                aq_sl = min(aq_mi, aq_ne) - max(soil, soils[i, j+1])
                if aq_sl > z_res and not np.isclose(aq_mi, 0.) and not np.isclose(aq_ne, 0.):
                    # diffusion can only happen when the water are in contact
                    
                    ek_ne = ekins[i, j+1]
                    ek_diff = (aq_ne*ek_mi - aq_mi*ek_ne) / (aq_mi + aq_ne) * diffuse_eff
                    ek_diff = (
                        min(ek_diff,  ek_mi*aq_sl/aq_mi) if ek_diff > 0.0 else
                        max(ek_diff, -ek_ne*aq_sl/aq_ne)
                    )
                    ekins_dnew[i, j  ] -= ek_diff
                    ekins_dnew[i, j+1] += ek_diff
                    
                    se_ne = sedis[i, j+1]
                    se_diff = (aq_ne*se_mi - aq_mi*se_ne) / (aq_mi + aq_ne) * diffuse_eff
                    se_diff = (
                        min(se_diff,  se_mi*aq_sl/aq_mi) if se_diff > 0.0 else
                        max(se_diff, -se_ne*aq_sl/aq_ne)
                    )
                    sedis_dnew[i, j  ] -= se_diff
                    sedis_dnew[i, j+1] += se_diff

    
    # - do erosion -
    for i in prange(1, npix_x+1):
        for j in prange(1, npix_y+1):
            aq = aquas[i, j] + aquas_dnew[i, j]
            if aq:
                se = sedis[i, j] + sedis_dnew[i, j]
                ca = capas[i, j]
                dzs = np.empty(N_ADJ)    # altitude change
                dzs[0] = zs[i-1, j]
                dzs[1] = zs[i+1, j]
                dzs[2] = zs[i, j-1]
                dzs[3] = zs[i, j+1]
                dzs -= z
                # d_se: extra sediments to be absorbed by water
                d_se = (ca - se) * erosion_eff
                if d_se > 0.:
                    # cannot dig under the bedrock
                    d_se = min(
                        d_se,
                        soils[i, j],
                        max(-np.min(dzs)+hole_depth, 0.),
                    )
                else:
                    # cannot give more than have
                    d_se = max(d_se, -aq, -se)
                sedis_dnew[i, j] += d_se
                aquas_dnew[i, j] += d_se
                soils_dnew[i, j] -= d_se

    # - update database -
    soils += soils_dnew
    aquas += aquas_dnew
    ekins += ekins_dnew
    sedis += sedis_dnew
    # reset boundary
    for ik in prange(edges_n):
        i, j = edges_inds_x[ik], edges_inds_y[ik]
        # soils edge should not have changed
        aquas[i, j] = edges[i, j]
        ekins[i, j] = 0.
        sedis[i, j] = 0.
    
    return soils, aquas, ekins, sedis



_erode_rainfall_evolve_sub_default = (
    # _erode_rainfall_evolve_sub_cuda if CAN_CUDA else
    _erode_rainfall_evolve_sub_nb
)



def _erode_rainfall_evolve(
    soils: npt.NDArray[np.float32],
    aquas: npt.NDArray[np.float32],
    ekins: npt.NDArray[np.float32],
    sedis: npt.NDArray[np.float32],
    edges: npt.NDArray[np.float32],
    pix_widxy: tuple[np.float32, np.float32],
    z_config : tuple[np.float32, np.float32, np.float32, np.float32],
    rain_configs: npt.NDArray[np.float32] = np.array(
        [2.**(-7)], dtype=np.float32),
    flow_eff   : np.float32 = np.float32(0.25),
    visco_kin_range: tuple[np.float32, np.float32] = (
        np.float32(1e-6), np.float32(1.0)),
    sed_cap_fac : np.float32 = np.float32(1.0),
    erosion_eff : np.float32 = np.float32(0.125),
    diffuse_eff : np.float32 = np.float32(0.25),
    hole_depth  : np.float32 = np.float32(2.**(-8)),
    slope_facs : tuple[np.float32, np.float32] = (
        np.float32(1.0), np.float32(1.0)),
    v_cap: np.float32 = np.float32(16.),
    g_eff : np.float32 = np.float32(9.8),
    sub_func: Callable = _erode_rainfall_evolve_sub_default,
    **kwargs,
):
    """Erosion through simulating falling rains.

    Notes
    -----
    Assuming that for each pixel, velocity goes linearly
        from 0 at the bottom of the water, to vmax at the top of the water,
        i.e. v(z) = vmax * z / aq for z from 0 to aq, where aq = aquas[i, j].
        Therefore, ekins[i, j] = ek = $ 0.5 \\int v(z)^2 dz $ = aq*vmax**2/6
            is the kinetic energy (per pixel area divided by water density)
            stored at that pixel.

    Note: Single-precision floating-point np.float32 has only 23bits for
        storing numbers, so for z_max=4096 (2**12), this means only 11bits
        for the digits after the decimal point-
        so, many parameters, such as rain_configs[..., 0],
            is recommended to be >=2**(-10), i.e. >=0.001.
    
    ...
    Parameters
    ----------
    soils, aquas, ekins, sedis, edges: (npix+2, npix+2)-shaped numpy array
        soils: Ground level (excl. water)
        aquas: Water level. = pure water + sediment.
        ekins: Kinetic energy per water density per pixel area in m^3/s^2.
        sedis: Sediment volume per pixel area.
        edges: Constant level water spawner level
        Minimum value being zero for all (more digits for data precision).
        0 and -1 index in both x and y are edges.
        data are stored in [1:-1, 1:-1].
        Repeat- need to reset min level for soils to zero!

    z_config: tuple((z_min, z_sea, z_max, z_res))
        Minimum height allowed / Sea level / Maximum height allowed.
        *** Warning: z_sea = 0 will disable sea level mechanics ***

    rains_config: (n_step, 4)-shaped npt.NDArray[np.float32]
        configuration of rains at each step.
        for each line, the data should be like
            (strength, spread, loc_x, loc_y)

    flow_eff: float
        Flow efficiency. should be in 0. < flow_eff <= 1.
        Controls how well the water flows around.
        I do NOT recommend touching this.

    visco_kin_range: tuple((visco_kin_aqua, visco_kin_soil))
        Kinematic visocity of water and soils in SI units (m^2/s).
        Must have visco_kin_aqua <= visco_kin_soil.
        It is ~1e-6 for water and 1e-2 ~ 1e-1 for mud.
        
    erosion_eff: float
        Erosion/deposition efficiency. Should be 0. <= erosion_eff <= 1.
        Setting it to 0. will disable erosion and deposition.

    diffuse_eff: float
        Diffusion efficiency. Should be 0. <= diffuse_eff <= 1.
        Controls how fast sediments and kinetic energies spread in lakes etc.

    hole_depth: float
        Maximum depth of the hole allowed by the erosion process to dig
            per step. Should be >= 0.
        If > 0., the erosion process may dig lakes.
            (but may also dig single pixel holes)

    slope_facs: tuple((downhill_fac, uphill_fac))
        factor to used in the slope calculation. should be 0. < fac < 1.
        if downhill_fac > upfill_fac, will make more gently sloped hills;
        else, will make more cliffs
        
    v_cap: float
        Characteristic velocity for sediment capacity calculations, in m/s.
        Used to regulate the velocity in capas calc,
        So its influence flatten out when v is high.
        
    g_eff: float
        Gravitational constant in m/s2.
    ...
    
    ---------------------------------------------------------------------------
    """

    raise NotImplementedError('This CPU version no longer supported.')

    # - type cast -
    z_config = np.asarray(z_config, dtype=np.float32)
    rain_configs = np.asarray(rain_configs, dtype=np.float32)
    flow_eff = np.float32(flow_eff)
    visco_kin_range = tuple([
        np.float32(visco_kin) for visco_kin in visco_kin_range])
    sed_cap_fac = np.float32(sed_cap_fac)
    erosion_eff = np.float32(erosion_eff)
    diffuse_eff = np.float32(diffuse_eff)
    hole_depth  = np.float32(hole_depth)
    slope_facs = tuple([
        np.float32(slope_fac) for slope_fac in slope_facs])
    v_cap = np.float32(v_cap)
    g_eff = np.float32(g_eff)

    # - do things -
    for rain_config in rain_configs:

        # - let it rain -
        # amp for amplitude
        rain_amp = rain_config[0]
        aquas += rain_amp
        # *** Rain mechanics to be updated ***

        # - physics and erosion -
        soils, aquas, ekins, sedis = sub_func(
            soils, aquas, ekins, sedis, edges,
            pix_widxy, z_config,
            flow_eff = flow_eff,
            visco_kin_range = visco_kin_range,
            sed_cap_fac = sed_cap_fac,
            erosion_eff = erosion_eff,
            diffuse_eff = diffuse_eff,
            hole_depth  = hole_depth,
            slope_facs  = slope_facs,
            v_cap = v_cap,
            g_eff = g_eff,
            **kwargs,
        )
    
    capas = _erode_rainfall_get_capas(
        soils+aquas, aquas, ekins, sedis,
        pix_widxy, sed_cap_fac, slope_facs, v_cap)
    
    return soils, aquas, ekins, sedis, capas









#-----------------------------------------------------------------------------#
#    Erosion: Raindrop (ABANDONED)
#-----------------------------------------------------------------------------#

# Note: my implementation of the raindrop method for erosion below suffers
#    a multitude of problems, such as the eneryg conservation,
#    particles stuck at the bottom of the lake, more coding effort required
#    and generally me overthinking this way too much.
#    so let's try a different approach- see Rainfall method section below


@jit(nopython=True, fastmath=True)
def _raindrop_hop_v_redirect(
    v_x, v_y, v_z,
    dz_dx, dz_dy,
    E_conserv_fac   : float,
) -> tuple[float, float, float]:
    """Fixing the velocity direction to the tangent surface of the hmap.
    
    I.e., what to do if the drop hits a wall or slope
        that forces it to change its velocity direction:
    
    The velocity must be constrained on the HMap surface,
        which means that the velocity component alongside
        the normal vector of the surface at that point
            ( norm vec being (-dz_dx, -dz_dy, 1) )
        is either lost (conserve momentum)
        or redirected (conserve energy)
        or somewhere in between.
    """
    # unit normal vecs of hmap surface
    surf_x, surf_y, surf_z = _hat(-dz_dx, -dz_dy, 1.)
    # dp for dot product
    v_dp_surf = v_x * surf_x + v_y * surf_y + v_z * surf_z
    v_ec = _norm(v_x, v_y, v_z)
    # remove the part of v that directly hits surf
    v_x -= v_dp_surf * surf_x
    v_y -= v_dp_surf * surf_y
    v_z -= v_dp_surf * surf_z
    if E_conserv_fac:
        # adding back energy as directed
        # ec for energy-conserved
        # mc for momentum-conserved
        #v_ec = v
        v_mc = _norm(v_x, v_y, v_z)
        v_x, v_y, v_z = _hat(
            v_x, v_y, v_z,
            E_conserv_fac*v_ec + (1.- E_conserv_fac)*v_mc)
    return v_x, v_y, v_z




@jit(nopython=True, fastmath=True)
def _raindrop_hop(
    p_x, p_y, p_z,
    v_x, v_y, v_z,
    v, dz_dx, dz_dy,
    data            : npt.NDArray[np.float64],
    map_widxy       : tuple[float, float],
    map_wid_x_b     : float,
    map_wid_y_b     : float,
    ds_xy           : float,
    turning         : float,
    E_conserv_fac   : float,
    fric_coeff      : float,
    fric_static_fac : float,
    g_eff : float = 9.8,
):
    """Move the rain drop one step.
    
    Returns
    -------
        break_state,
        p_x, p_y, p_z,
        v_x, v_y, v_z,
        dz_dx, dz_dy
    -------
    break_state: int
        -1: can continue but step equals 0
        0 : continue
        1 : break due to droplet moved out of the map
        2 : break due to droplet got stuck (v_xy==0 and a_xy==0)
        3 : cannot conserve energy after deposition (not included here)
    """

    break_state : int = 0
    
    
    # gradient direction of z (b for nabla)
    # note that the surface norm vector is (-dz_dx, -dz_dy, 1.),
    # And the gradient should be in the same vertical plane of the surf norm,
    #    (i.e. it's perpendicular to (-dz_dy, dz_dx, 0.) )
    #    while being a linear combination of the 2 tangent vectors of the surf:
    #        (1./dz_dx, 0, 1.) and (0., 1./dz_dy, 1.)
    # i.e.,
    b_x, b_y, b_z = _hat(dz_dx, dz_dy, dz_dx**2+dz_dy**2)
    
    # - accelerations -
    # note that (g_x**2 + g_y**2 + g_z**2 + g_f**2)**0.5 == g_eff
    if np.isclose(_norm(b_x, b_y, b_z), 0.):
        # perfectly flat
        g_x, g_y, g_z, g_f = 0., 0., 0., g_eff
    else:
        g_x = -b_x * b_z * g_eff # / _norm(b_x, b_y, b_z)**2
        g_y = -b_y * b_z * g_eff # / _norm(b_x, b_y, b_z)**2
        g_z = -(b_x**2 + b_y**2) * g_eff # / _norm(b_x, b_y, b_z)**2
        g_f =  b_z * g_eff # for friction
    # reset velocity to gravity directions if turning
    if turning and not np.isclose(_norm(g_x, g_y), 0.):
        v_new_x, v_new_y, v_new_z = _hat(g_x, g_y, 0., v)
        v_x = (1. - turning) * v_x + turning * v_new_x
        v_y = (1. - turning) * v_y + turning * v_new_y
        v_z = (1. - turning) * v_z + turning * v_new_z
        v_x, v_y, v_z = _hat(v_x, v_y, v_z, v)
    # note: b_z/_norm(b_x, b_y, b_z) because
    #    the other two get cancelled out by ground's support force
    # friction (v**2/ds = v/dt)
    #    i.e. friction does not move things on its own
    #    also, approximate ds with ds_xy
    a_f  = fric_coeff * g_f
    if not np.isclose(v, 0.):
        # friction against velocity direction
        a_fx = -_minabs(a_f*v_x/v, (g_x + v_x**2/ds_xy))
        a_fy = -_minabs(a_f*v_y/v, (g_y + v_y**2/ds_xy))
        a_fz = -_minabs(a_f*v_z/v, (g_z + v_z**2/ds_xy))
    else:
        # static friction against gravity direction
        g_xyz= _norm(g_x, g_y, g_z)
        if not np.isclose(g_xyz, 0.):
            a_fx = -_minabs(a_f * g_x / g_xyz, g_x) * fric_static_fac
            a_fy = -_minabs(a_f * g_y / g_xyz, g_y) * fric_static_fac
            a_fz = -_minabs(a_f * g_z / g_xyz, g_z) * fric_static_fac
        else:
            a_fx, a_fy, a_fz = 0., 0., 0.
    a_x  = g_x + a_fx
    a_y  = g_y + a_fy
    a_z  = g_z + a_fz

    # - step -
    # get time step
    v_xy = _norm(v_x, v_y)
    a_xy = _norm(a_x, a_y)
    if not np.isclose(a_xy, 0.):
        dt  = ((v_xy**2 + 2 * a_xy * ds_xy)**0.5 - v_xy) / a_xy
    elif not np.isclose(v_xy, 0):
        dt  = ds_xy / v_xy
    else:
        # droplet stuck - terminate
        break_state = 2
        #break
        return (
            break_state,
            p_x, p_y, p_z,
            v_x, v_y, v_z,
            v, dz_dx, dz_dy)
    # getting step size
    #    normalize so that d_x**2 + d_y**2 == ds_xy
    d_x = v_x * dt + a_x * dt**2 / 2.
    d_y = v_y * dt + a_y * dt**2 / 2.
    #d_z = v_z * dt + a_z * dt**2 / 2
    norm_d_x_y = _norm(d_x, d_y)
    if np.isclose(norm_d_x_y, 0.):
        # break due to 0-sized step
        break_state = -1
        ##break
        #return (
        #    break_state,
        #    p_x, p_y, p_z,
        #    v_x, v_y, v_z,
        #    v, dz_dx, dz_dy)
    else:
        d_factor = ds_xy / norm_d_x_y
        d_x *= d_factor
        d_y *= d_factor
    #d_z *= d_factor    #d_z = dz_dx * d_x + dz_dy * d_y
    #ds  = (ds_xy**2 + d_z**2)**0.5

    # - update -
    # record original
    p_x_old, p_y_old, p_z_old = p_x, p_y, p_z
    dz_dx_old, dz_dy_old = dz_dx, dz_dy
    E_old = g_eff * p_z + v**2/2.    # specific energy
    # update position / direction
    p_x += d_x
    p_y += d_y
    p_z, dz_dx, dz_dy = _get_z_and_dz(p_x, p_y, data, map_widxy)
    d_z = p_z - p_z_old    # actual d_z that happened to the drop
    # Update velocity and Fix Energy
    # Obtaining E_new = E_old + _dot(a_f, d) (remove work from friction)
    # i.e. g_eff * p_z + v_new**2/2.
    #    = E_old + (a_fx*d_x + a_fy*d_y + a_fx*d_z)
    # v2 = v**2
    v2_new = 2 * (E_old + (a_fx*d_x + a_fy*d_y + a_fz*d_z) - g_eff * p_z)
    if v2_new < 0.:
        # the rain drop should be reflected back.
        # Cancel step
        p_x, p_y, p_z = p_x_old, p_y_old, p_z_old
        dz_dx, dz_dy  = dz_dx_old, dz_dy_old
        # Reflect velocities
        #    it should be perpendicular to surf norm vec,
        #        and maintain the same v, with v_z_new = -v_z.
        # So,
        #    -v_x_new = (dz_dy / dz_dx) * (v_y_new + v_y) + v_x
        # and
        #    v_x_new**2 + v_y_new**2 = v_x**2 + v_y**2
        # Solve above two equations and we find:
        if not np.isclose(dz_dx, 0.):
            tmp = dz_dy/dz_dx
            v_y_new = ((1 - tmp**2)*v_y - 2*tmp*v_x) / (tmp**2 + 1)
            v_x_new =  -tmp * (v_y_new + v_y) - v_x
        else:
            v_y_new = -v_y
            v_x_new = v_x
        v_x, v_y, v_z = v_x_new, v_y_new, -v_z
    else:
        v_new = v2_new**0.5
        # update velocity
        v_x += a_x * dt
        v_y += a_y * dt
        v_z += a_z * dt
        v    = _norm(v_x, v_y, v_z)
        # normalize specific energy to ensure energy is conserved
        if np.isclose(v, 0.):
            v_x, v_y, v_z = 0., 0., -v_new
        else:
            v_x, v_y, v_z = _hat(v_x, v_y, v_z, v_new)

    # - velocities direction -
    #    What to do if the drop hits a wall or slope
    #        that forces it to change its velocity direction
    v_x, v_y, v_z = _raindrop_hop_v_redirect(
        v_x, v_y, v_z, dz_dx, dz_dy, E_conserv_fac=E_conserv_fac)
    v = _norm(v_x, v_y, v_z)
    

    # check
    if (   p_x <= -map_wid_x_b
        or p_x >=  map_wid_x_b
        or p_y <= -map_wid_y_b
        or p_y >=  map_wid_y_b):
        # droplet out of bounds- terminate
        break_state = 1
        #break
    return (
        break_state,
        p_x, p_y, p_z,
        v_x, v_y, v_z,
        v, dz_dx, dz_dy)



@jit(nopython=True, fastmath=True)
def _erode_raindrop_test(
    data  : npt.NDArray[np.float64],
    map_widxy : tuple[float, float],
    z_min : float,
    z_sea : float,
    z_max : float,
    n_drop    : int = 1,
    n_step_max: int = 65536,
    initial_velocity  : float = 0.1,
    turning           : float = 0.1,
    E_conserv_fac     : float = 0.8,
    fric_coeff        : float = 0.01,
    fric_static_fac   : float = 1.0,
    g_eff : float = 9.8,
    do_erosion : bool = True,
    sed_cap_fac  : float = 1.0,
    sed_initial  : float = 0.0,
    erosion_eff  : float = 1.0,
):
    """Erosion.

    Best keep data in same resolution in both x and y.
    
    Parameters
    ----------
    ...
    n_drop : int
        How many raindrops we are simulating.
    
    n_step_max : int
        maximum steps (i.e. distance) the rain drop is allowed to travel.

    initial_velocity : float
        Initial velocity of the rain drop when spawned.
        Should be > 0.
        
    turning : float
        if we should immediately reset velocity direction
            to gravity direction (1.),
            or not (0.)
        
    E_conserv_fac : float
        Switch between energy-conserved mode and momentum-conserved mode
            when the raindrop changes its direction.
         0. for momentum conservation, 1. for energy conservation.
         i.e. when the rain drop hits a wall (or slope),
         Does the energy gets absorbed as the drop comes to an stop (0.),
         or does the velocity gets redirected (1.)
         
    fric_coeff : float
        Friction coefficient. Should be within 0. <= fric_coeff < 1.
            Physics says we cannot assign friction coeff to a liquid.
            Well let's pretend we can because it's all very approximate.

    fric_static_fac : float
        Static friction is gravity multiplied by this factor.
        Should be smaller but close to 1.
            
    g_eff : gravitational accceleration in m/s^2

    do_erosion: bool
        Switch to switch on erosion (i.e. actual change in data)

    sed_cap_fac : float
        Sediment capacity factor of the river:
            sediment capacity per speed per slope, in unit of s/m,
        with the sediment capacity being
            the sediment mass carried by water per water mass,
        and the slope being meters droped per meter moved horizontally

    sed_initial : float
        initial sediment content in the rain drop

    erosion_eff : float
        Erosion efficiency. Should be within [0., 1.].
    ...

    Returns
    ----------
    ...
    """

    print("*** Warning: Test code- currently broken.")

    # init
    drops = np.zeros_like(data, dtype=np.uint32)    # initial drops position
    paths = np.zeros_like(data, dtype=np.uint64)    # counts of passing drops
    lib_z = np.zeros((n_drop, n_step_max))
    lib_v = np.zeros((n_drop, n_step_max))
    # specific energy (i.e. energy per mass)
    lib_E = np.zeros((n_drop, n_step_max))
    lib_sed=np.zeros((n_drop, n_step_max))

    steps = np.zeros(n_drop, dtype=np.uint32)    # step count for each rain drop
    stats = np.zeros(n_drop, dtype=np.int8)    # break_states for each rain drop
    

    npix_x, npix_y = data.shape
    map_wid_x, map_wid_y = map_widxy
    # boundaries: [-map_wid_x_b, map_wid_x_b]
    map_wid_x_b = map_wid_x * (0.5 - 0.5/npix_x)
    map_wid_y_b = map_wid_y * (0.5 - 0.5/npix_y)
    #assert map_wid_x == map_wid_y
    
    # distance per step (regardless of velocity)
    ds_xy : float = (map_wid_x/npix_x + map_wid_y/npix_y)/2.    # fixed

    for i in prange(n_drop):

        # Step 1: Generate a raindrop at random locations
        
        # i for in index unit; no i for in physical unit

        # position
        p_x : float = random.uniform(-map_wid_x_b, map_wid_x_b)
        p_y : float = random.uniform(-map_wid_y_b, map_wid_y_b)
        p_z, dz_dx, dz_dy = _get_z_and_dz(p_x, p_y, data, map_widxy)
        x_i : int = _pos_to_ind_d(p_x, map_wid_x, npix_x)
        y_i : int = _pos_to_ind_d(p_y, map_wid_y, npix_y)
        drops[x_i, y_i] += 1
        paths[x_i, y_i] += 1
        # velocity
        v_theta = random.uniform(0., 2*np.pi)
        v   : float = initial_velocity
        v_x : float = v * np.sin(v_theta)
        v_y : float = v * np.cos(v_theta)
        v_z : float = 0.
        v_x, v_y, v_z = _raindrop_hop_v_redirect(
            v_x, v_y, v_z, dz_dx, dz_dy, E_conserv_fac=E_conserv_fac)
        v = _norm(v_x, v_y, v_z)
        # sediments
        slope: float = 0.    # height droped: + for downhill, - for uphill
        sed_c: float = 0.    # sediment capacity
        sed_m: float = 0.    # sediment mass
    
        for s in range(n_step_max):
        
            # Step 2: Simulate physics for the droplet to slide downhill

            # log
            x_i_old, y_i_old = x_i, y_i

            # evolve
            (
                break_state,
                p_x, p_y, p_z_new,
                v_x, v_y, v_z,
                v, dz_dx, dz_dy,
            ) = _raindrop_hop(
                p_x, p_y, p_z,
                v_x, v_y, v_z,
                v, dz_dx, dz_dy,
                data            = data,
                map_widxy       = map_widxy,
                map_wid_x_b     = map_wid_x_b,
                map_wid_y_b     = map_wid_y_b,
                ds_xy           = ds_xy,
                turning         = turning,
                E_conserv_fac   = E_conserv_fac,
                fric_coeff      = fric_coeff,
                fric_static_fac = fric_static_fac,
                g_eff = g_eff,
            )

            # info
            d_z   = p_z_new - p_z
            slope = -d_z / ds_xy
            p_z = p_z_new
            

            # log
            x_i = _pos_to_ind_d(p_x, map_wid_x, npix_x)
            y_i = _pos_to_ind_d(p_y, map_wid_y, npix_y)
            E_new =  g_eff * p_z + v**2/2.
            paths[x_i, y_i] += 1
            lib_z[i, s] = p_z
            lib_v[i, s] = v
            lib_E[i, s] = E_new


            # Step 3: Do erosion / deposition

            # *** Add code here ***

            # Recall Lane's Equation:
            #    In a stable river, [Sediment flow] * [Sediment median size]
            #    is proportional to [   Water flow] * [Slope]
            # Ignoring sediment size, sediment capacity can be
            sed_c = max(sed_cap_fac * v * abs(slope), 0.)

            # sediment mass to be absorbed into the water
            sed_d = (sed_c - sed_m) * erosion_eff
            # impose limit on sed_d
            if data[x_i_old, y_i_old] - sed_d < z_min:
                sed_d = data[x_i_old, y_i_old] - z_min
            elif data[x_i_old, y_i_old] - sed_d > z_max:
                sed_d = data[x_i_old, y_i_old] - z_max
            lib_sed[i, s] = sed_d

            # erode
            # *** to be improved ***
            if do_erosion:
                # erode the part behind us 
                #    so we don't dig a hole in front of a waterfall
                data[x_i_old, y_i_old] -= sed_d
                sed_m += sed_d
                # if sed_d > 0:
                #    data[x_i_old, x_i_old] -= sed_d
                #    sed_m += sed_d
                # elif sed_d < 0:
                #    data[x_i_old, x_i_old] -= sed_d
                #    sed_m += sed_d

                # update pos, v & regulate energy
                p_z, dz_dx, dz_dy = _get_z_and_dz(p_x, p_y, data, map_widxy)
                v2_ero = 2 * (E_new - g_eff * p_z)
                if v2_ero < 0:
                    # cannot conserve energy- break
                    break_state = 3
                    # give free energy so the code doesn't freak out
                    v2_ero = 0.
                v = v2_ero**0.5
                v_x, v_y, v_z = _raindrop_hop_v_redirect(
                    v_x, v_y, v_z, dz_dx, dz_dy, E_conserv_fac=0.0)
                v_x, v_y, v_z = _hat(v_x, v_y, v_z, v)
            

            # check
            if break_state > 0:
                break
                
        # finally
        steps[i] = s
        stats[i] = break_state
        
    
    return stats, steps, drops, paths, lib_z, lib_v, lib_E, lib_sed
    


#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
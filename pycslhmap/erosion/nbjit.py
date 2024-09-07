#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CPU version of .cuda codes. Incomplete. No longer supported.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""

from .cuda import (
    CAN_CUDA,
    _erode_rainfall_init_sub_cuda,
    _erode_rainfall_evolve_sub_cuda,
)

from typing import Self, Callable

from numba import jit, prange
import numpy as np
from numpy import typing as npt


#-----------------------------------------------------------------------------#
#    Erosion: Rainfall: Init
#-----------------------------------------------------------------------------#


@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_init_sub_nbjit(
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
#    Erosion: Rainfall: Evolve
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
    g : np.float32,
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
                            eks[kj] += wmfe*g*flow_eff*(
                                wms_w[kj] + (wrc_d + wmfe)*flow_eff/2.
                            )
                            #wms_w[kj] -= wms_w[k]
                        break
                    # otherwise
                    wrc += wrc_d
                    # optimizing below:
                    # wms[inds[ik:]] += wmfe
                    # eks[inds[ik:]] += wmfe*g*(wms_w[inds[ik:]] - wms_w[k]/2)
                    # wms_w[inds[ik:]] -= wms_w[k]
                    for jk in range(ik, N_ADJ):
                        kj = inds[jk]
                        wms[kj] += wmfe * flow_eff
                        eks[kj] += wmfe*g*flow_eff*(
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
    g : np.float32 = np.float32(9.8),
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
        
    g: float
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
    g = np.float32(g)

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
            g = g,
            **kwargs,
        )
    
    capas = _erode_rainfall_get_capas(
        soils+aquas, aquas, ekins, sedis,
        pix_widxy, sed_cap_fac, slope_facs, v_cap)
    
    return soils, aquas, ekins, sedis, capas



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
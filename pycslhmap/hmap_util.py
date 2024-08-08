#!/usr/bin/env python
# coding: utf-8

"""Functions for handling height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies

from typing import Self

from numba import jit, prange
import numpy as np
from numpy import typing as npt



#-----------------------------------------------------------------------------#
#    Vectors and HMaps
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
#    Erosion: Rainfall
#-----------------------------------------------------------------------------#


@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_init(
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

    z_sea: np.float32
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

    print("Test function - Not Yet finished.")
    

    npix_x, npix_y = data.shape
    z_min = np.float32(z_min)
    z_sea = np.float32(z_sea)
    z_max = np.float32(z_max)

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
    zs = np.full_like(soils, z_max) 
    zs = aquas + soils    # actual heights (water + ground)
    zs[1:-1, 1:-1] = z_max - z_min    # first fill, then drain
    # note: zs' edge elems are fixed
    n_cycles = 0    # debug
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

    aquas[1:-1, 1:-1] = (zs - soils)[1:-1, 1:-1]

    ekins = np.zeros_like(soils)
    sedis = np.zeros_like(soils) # is zero because speed is zero
    
    return soils, aquas, ekins, sedis, edges, n_cycles



@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_get_capas(
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
    


@jit(nopython=True, fastmath=True, parallel=True)
def _erode_rainfall_evolve(
    soils: npt.NDArray[np.float32],
    aquas: npt.NDArray[np.float32],
    ekins: npt.NDArray[np.float32],
    sedis: npt.NDArray[np.float32],
    edges: npt.NDArray[np.float32],
    pix_widxy: tuple[float, float],
    #z_min: float,
    z_sea: float,
    z_max: float,
    n_step: int = 1,
    rain_per_step:float = 2**(-4),
    #rains_config: npt.NDArray[np.float32],
    flow_eff   : float = 0.25,
    visco_kin_aqua: float = 1e-6,
    visco_kin_soil: float = 1.0,
    sed_cap_fac: float = 1.0,
    erosion_eff: float = 0.125,
    v_cap: float = 16.,
    g : float = 9.8,
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
        for the digits after the decimal point- so, some of the parameters
        (such as rain_per_step) is recommended to be >=2**(-10), i.e. >=0.001.
    
    ...
    Parameters
    ----------
    soils, aquas, ekins, sedis, edges: (npix+2, npix+2)-shaped numpy array
        soils: Ground level (excl. water)
        aquas: Water level. = pure water + sediment.
        ekins: Kinetic energy per water density per pixel area in m^3/s^2.
        sedis: Sediment volume per pixel area.
        edges: Constant level water spawner level
        Minimum value being zero for all.
        0 and -1 index in both x and y are edges.
        data are stored in [1:-1, 1:-1].
        Repeat- need to reset min level for soils to zero!

    z_sea: float
        Sea level.
        *** Warning: z_sea = 0 will disable sea level mechanics ***

    rain_per_step: float
        rain rate minus evaporation rate.

    rains_config: (n_step, 4)-shaped npt.NDArray[np.float32]
        configuration of rains at each step.
        for each line, the data should be like
            (strength, spread, loc_x, loc_y)
        *** To be updated ***

    flow_eff: float
        Flow efficiency. should be in 0. < flow_eff <= 1.
        Controls how well the water flows around.
        I do NOT recommend touching this.

    visco_kin_aqua, visco_kin_soil: float
        Kinematic visocity of water and soils in SI units (m^2/s).
        It is ~1e-6 for water and 1e-2 ~ 1e-1 for mud.
        
    erosion_eff : float
        Erosion efficiency. Should be 0. < erosion_eff <= 1.
        
    v_cap: float
        Characteristic velocity for sediment capacity calculations, in m/s.
        Used to regulate the velocity in capas calc,
        So its influence flatten out when v is high.
        
    g: float
        Gravitational constant in m/s2.
    ...
    
    ---------------------------------------------------------------------------
    """

    print("Test function - Not Yet finished.")

    # - init -
    # remember len(soils) is npix+2 because we added edges
    N_ADJ : int = 4    # number of adjacent cells
    npix_x, npix_y = soils.shape[0]-2, soils.shape[1]-2
    pix_wid_x, pix_wid_y = pix_widxy
    zs     = np.empty_like(soils)
    # the boundary will not be changed
    edges_inds_x, edges_inds_y = np.where(edges)
    edges_n = len(edges_inds_x)
    visco_kin = visco_kin_aqua
    
    for s in range(n_step):

        # - update height and gradient -
        zs = soils + aquas

        # - add rains and init -
        aquas += rain_per_step
        soils_dnew = np.zeros_like(soils)
        aquas_dnew = np.zeros_like(soils)
        ekins_dnew = np.zeros_like(soils)
        sedis_dnew = np.zeros_like(soils)

        capas = _erode_rainfall_get_capas(
            zs, aquas, ekins, pix_widxy, sed_cap_fac, v_cap)
        
        for i in range(1, npix_x+1):
            for j in range(1, npix_y+1):
                if aquas[i, j] > 0: # only do things if there is water
                    
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
                    if not np.isclose(wrc, 0.):
                        # move water
                        aquas_dnew[i,   j] -= wrc
                        aquas_dnew[i-1, j] += wms[0]
                        aquas_dnew[i+1, j] += wms[1]
                        aquas_dnew[i, j-1] += wms[2]
                        aquas_dnew[i, j+1] += wms[3]
                        # transfer kinetic energy
                        # will update ekins later
                        #ek_d = wrc / aq * ek
                        #ekins_dnew[i,   j] -= ek_d
                        ekins_dnew[i-1, j] += ek_d * (wms[0] / wrc) + eks[0]
                        ekins_dnew[i+1, j] += ek_d * (wms[1] / wrc) + eks[1]
                        ekins_dnew[i, j-1] += ek_d * (wms[2] / wrc) + eks[2]
                        ekins_dnew[i, j+1] += ek_d * (wms[3] / wrc) + eks[3]
                        # transfer sediments
                        se_d = wrc / aq * se
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
                    ek_d += visco_kin * (6*(ek-ek_d)/aq)**0.5
                    ek_d = min(ek_d, ek)    # make sure ekins don't go negative
                    ekins_dnew[i, j] -= ek_d


                    # diffusion
        
        
        # - do erosion -
        for i in prange(1, npix_x+1):
            for j in prange(1, npix_y+1):
                aq = aquas[i, j] + aquas_dnew[i, j]
                if aq:
                    se = sedis[i, j] + sedis_dnew[i, j]
                    ca = capas[i, j]
                    # d_se: extra sediments to be absorbed by water
                    d_se = (ca - se) * erosion_eff
                    if d_se > 0:
                        # cannot dig under the bedrock
                        # ****** Add code to prevet digging a hole ******
                        d_se = min(d_se, soils[i, j]) #, -np.min(dzs))
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
    
    return soils, aquas, ekins, sedis, capas
    


#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
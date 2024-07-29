#!/usr/bin/env python
# coding: utf-8

"""A class to handle height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies

from typing import Self
import random

from numba import jit, prange
import numpy as np
from numpy import pi
from numpy import typing as npt



#-----------------------------------------------------------------------------#
#    Functions: Vectors and HMaps
#-----------------------------------------------------------------------------#


@jit(nopython=True)
def _minabs(a, b):
    """Return the elem with minimum abs value."""
    return a if abs(a) < abs(b) else b



@jit(nopython=True)
def _norm(
    v_x: float, v_y: float, v_z: float,
) -> float:
    """Get the norm of a vector."""
    return (v_x**2 + v_y**2 + v_z**2)**0.5

    

@jit(nopython=True)
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
    


@jit(nopython=True)
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



@jit(nopython=True)
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



@jit(nopython=True)
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



@jit(nopython=True)
def _get_z_and_dz(
    pos_x: float,
    pos_y: float,
    data : npt.NDArray[np.float64],
    map_widxy: tuple[int, int],
) -> tuple[float, float, float]:
    """Get height and gradients at specified position in physical units.

    pos_xy in physical units, within range of [-map_widxy/2., map_widxy/2.]

    Returns: z, dz_dx, dz_dy
    Note that dz_dx is $ \\frac{\\partial z}{\\partial x} $
        i.e. partial derivative
    """

    # init
    map_wid_x, map_wid_y = map_widxy
    npix_x, npix_y = data.shape
    assert npix_x >= 2 and npix_y >= 2    # otherwise interpolation will break
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
#    Functions: Erosion
#-----------------------------------------------------------------------------#


@jit(nopython=True)
def _erode_raindrop_once(
    data: npt.NDArray[np.float64],
    map_widxy: tuple[int, int],
    z_seabed: float,
    z_sealvl: float,
    max_steps_per_drop: int   = 65536,
    turning           : float = 0.,
    friction_coeff    : float = 0.01,
    initial_velocity  : float = 0.1,
    g : float = 9.8,
):
    """Erosion.

    Best keep data in same resolution in both x and y.
    
    Parameters
    ----------
    ...
    max_steps_per_drop: int
        maximum steps (i.e. distance) the rain drop is allowed to travel.
        
    turning : float
        Switch between energy-conserved mode and momentum-conserved mode
            when the raindrop changes its direction.
         0. for momentum conservation, 1. for energy conservation.
         
    friction_coeff : float
        friction coefficient. should be within 0. <= friction_coeff < 1.
            Physics says we cannot assign friction coeff to a liquid.
            Well let's pretend we can because it's all very approximate.

    initial_velocity : float
        initial velocity of the rain drop when spawned.
        Should be > 0.
            
    g : gravitational accceleration in m/s^2
    ...
    """

    raise NotImplementedError

    # init
    paths = np.zeros_like(data, dtype=np.int64)
    lib_z = np.zeros(max_steps_per_drop)
    lib_v = np.zeros(max_steps_per_drop)
    lib_E = np.zeros(max_steps_per_drop)

    npix_x, npix_y = data.shape
    map_wid_x, map_wid_y = map_widxy
    # boundaries: [-map_wid_x_b, map_wid_x_b]
    map_wid_x_b = map_wid_x * (0.5 - 0.5/npix_x)
    map_wid_y_b = map_wid_y * (0.5 - 0.5/npix_y)
    #assert map_wid_x == map_wid_y

    if True:

        
        # step 1: Generate a raindrop at random locations
        
        # i for in index unit; no i for in physical unit
        # distance per step (regardless of velocity)
        ds_xy : float = (map_wid_x/npix_x + map_wid_y/npix_y)/2.    # fixed
        ds    : float = ds_xy
        dt    : float = 0.
        # position
        p_x : float = random.uniform(-map_wid_x_b, map_wid_x_b)
        p_y : float = random.uniform(-map_wid_y_b, map_wid_y_b)
        x_i : int = _pos_to_ind_d(p_x, map_wid_x, npix_x)
        y_i : int = _pos_to_ind_d(p_y, map_wid_y, npix_y)
        # direction (i.e. each step)
        d_x : float = random.uniform(-ds_xy, ds_xy)
        d_y : float = (ds_xy**2 - d_x**2)**0.5 * (random.randint(0, 1)*2 - 1)
        d_z : float = 0.
        # velocity
        v_x : float = initial_velocity * d_x / ds_xy
        v_y : float = initial_velocity * d_y / ds_xy
        v_z : float = 0.
        v   : float = _norm(v_x, v_y, v_z)
        # sediment content
        c   : float = 0.
    
        for s in range(max_steps_per_drop):
        
            # step 2: Simulate physics for the droplet to slide downhill
    
            # init
            x_i = _pos_to_ind_d(p_x, map_wid_x, npix_x)
            y_i = _pos_to_ind_d(p_y, map_wid_y, npix_y)
            paths[x_i, y_i] += 1
            p_z, dz_dx, dz_dy = _get_z_and_dz(p_x, p_y, data, map_widxy)

            # Fixing the velocity direction:
            #    What to do if the drop hits a wall or slope
            #        that forces it to change its velocity direction
            # The velocity must be constrained on the HMap surface,
            #     which means that the velocity component alongside
            #     the normal vector of the surface at that point
            #         ( norm vec being (-dz_dx, -dz_dy, 1) )
            #     is either lost (conserve momentum)
            #     or redirected (conserve energy) or somewhere in between

            # unit normal vecs of hmap surface
            surf_x, surf_y, surf_z = _hat(-dz_dx, -dz_dy, 1.)
            # dp for dot product
            v_dp_surf = v_x * surf_x + v_y * surf_y + v_z * surf_z
            #d_dp_surf = d_x * surf_x + d_y * surf_y + d_z * surf_z
            # remove the part of v that directly hits surf
            v_x -= v_dp_surf * surf_x
            v_y -= v_dp_surf * surf_y
            v_z -= v_dp_surf * surf_z
            #d_x -= d_dp_surf * surf_x
            #d_y -= d_dp_surf * surf_y
            #d_z -= d_dp_surf * surf_z
            if turning:
                # adding back energy as directed
                # ec for energy-conserved
                # mc for momentum-conserved
                v_ec = v
                v_mc = _norm(v_x, v_y, v_z)
                v_x, v_y, v_z = _hat(
                    v_x, v_y, v_z,
                    turning*v_ec + (1.- turning)*v_mc)
            
            # steps
            #    normalize so that d_x**2 + d_y**2 == ds_xy
            d_factor = ds_xy / (d_x**2 + d_y**2)
            d_x *= d_factor
            d_y *= d_factor
            #d_z *= d_factor
            d_z = dz_dx * d_x + dz_dy * d_y
            ds  = (ds_xy**2 + d_z**2)**0.5
            # gradient direction of z (b for nabla)
            b_x, b_y, b_z = _hat(dz_dx, dz_dy, 1.)
            # accelerations
            # note that (g_x**2 + g_y**2 + g_z**2 + g_f**2)**0.5 == g
            g_x = -b_x * b_z * g # / _norm(b_x, b_y, b_z)**2
            g_y = -b_y * b_z * g # / _norm(b_x, b_y, b_z)**2
            g_z = -(b_x**2 + b_y**2) * g # / _norm(b_x, b_y, b_z)**2
            g_f =  b_z * g # for friction
            # note: b_z/_norm(b_x, b_y, b_z) because
            #    the other two get cancelled out by ground's support force
            # friction (v**2/ds = v/dt)
            #    i.e. friction does not move things on its own
            a_f  = friction_coeff * g_f
            a_fx = _minabs(a_f * d_x / ds, g_x + v_x**2/ds)
            a_fy = _minabs(a_f * d_y / ds, g_y + v_y**2/ds)
            a_fz = _minabs(a_f * d_z / ds, g_z + v_z**2/ds)
            a_x  = g_x - a_fx
            a_y  = g_y - a_fy
            a_z  = g_z - a_fz
            v_xy = _norm(v_x, v_y, 0.)
            a_xy = _norm(a_x, a_y, 0.)
            # get time step
            if not np.isclose(a_xy, 0.):
                dt  = ((v_xy**2 + 2 * a_xy * ds_xy)**0.5 - v_xy) / a_xy
            elif not np.isclose(v_xy, 0):
                dt  = ds_xy / v_xy
            else:
                # droplet stuck - terminate
                break
            # update position / direction
            p_x += d_x
            p_y += d_y
            d_x = v_x * dt + a_x * dt**2 / 2
            d_y = v_y * dt + a_y * dt**2 / 2
            d_z = v_z * dt + a_z * dt**2 / 2
            # update velocity
            v_x += a_x * dt
            v_y += a_y * dt
            v_z += a_z * dt
            ## pending optimization
            #p_new_z, _, _ = _get_z_and_dz(p_x, p_y, data, map_widxy)
            #v_z  = (p_z_new - p_z) / dt if dt else 0.
            v    = _norm(v_x, v_y, v_z)
                

            # log
            lib_z[s] = p_z
            lib_v[s] = v
            lib_E[s] = g * p_z + v**2/2.
            if abs(d_x) > 2*ds_xy or abs(d_y) > 2*ds_xy:
                # something has gone horribly wrong
                print("error: at s=", s, "d_x=", d_x, "d_y=", d_y)
                break

            # check
            if (   p_x <= -map_wid_x_b
                or p_x >=  map_wid_x_b
                or p_y <= -map_wid_y_b
                or p_y >=  map_wid_y_b):
                # droplet out of bounds- terminate
                break


    
    return paths, lib_z, lib_v, lib_E, s, x_i, y_i, v, ds, dt, d_x, d_y



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
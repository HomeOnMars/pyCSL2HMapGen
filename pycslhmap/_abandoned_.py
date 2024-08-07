#!/usr/bin/env python
# coding: utf-8

"""Abandoned codes- for archive purposes only.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies
from .hmap_util import (
    _minabs, _norm, _hat,
    _ind_to_pos, _pos_to_ind_f, _pos_to_ind_d,
    _get_z_and_dz,
)

from typing import Self
import random

from numba import jit, prange
import numpy as np
from numpy import typing as npt






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
    g : float = 9.8,
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
    # note that (g_x**2 + g_y**2 + g_z**2 + g_f**2)**0.5 == g
    if np.isclose(_norm(b_x, b_y, b_z), 0.):
        # perfectly flat
        g_x, g_y, g_z, g_f = 0., 0., 0., g
    else:
        g_x = -b_x * b_z * g # / _norm(b_x, b_y, b_z)**2
        g_y = -b_y * b_z * g # / _norm(b_x, b_y, b_z)**2
        g_z = -(b_x**2 + b_y**2) * g # / _norm(b_x, b_y, b_z)**2
        g_f =  b_z * g # for friction
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
    E_old = g * p_z + v**2/2.    # specific energy
    # update position / direction
    p_x += d_x
    p_y += d_y
    p_z, dz_dx, dz_dy = _get_z_and_dz(p_x, p_y, data, map_widxy)
    d_z = p_z - p_z_old    # actual d_z that happened to the drop
    # Update velocity and Fix Energy
    # Obtaining E_new = E_old + _dot(a_f, d) (remove work from friction)
    # i.e. g * p_z + v_new**2/2.
    #    = E_old + (a_fx*d_x + a_fy*d_y + a_fx*d_z)
    # v2 = v**2
    v2_new = 2 * (E_old + (a_fx*d_x + a_fy*d_y + a_fz*d_z) - g * p_z)
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
    g : float = 9.8,
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
            
    g : gravitational accceleration in m/s^2

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
                g = g,
            )

            # info
            d_z   = p_z_new - p_z
            slope = -d_z / ds_xy
            p_z = p_z_new
            

            # log
            x_i = _pos_to_ind_d(p_x, map_wid_x, npix_x)
            y_i = _pos_to_ind_d(p_y, map_wid_y, npix_y)
            E_new =  g * p_z + v**2/2.
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
                v2_ero = 2 * (E_new - g * p_z)
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
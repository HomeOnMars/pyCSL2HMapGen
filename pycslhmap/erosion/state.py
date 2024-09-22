#!/usr/bin/env python
# coding: utf-8

"""Height map erosion with GPU-accelerations.

Require Cuda. CPU version no longer supported.


-------------------------------------------------------------------------------

Inspired by Sebastian Lague's Erosion code (
    See https://github.com/SebLague/Hydraulic-Erosion
) who in turn was inspired by Hans Theobald Beyer's Bachelor's Thesis
    'Implementation of a method for hydraulic erosion'.
However, my implementation ended up very different from theirs.

-------------------------------------------------------------------------------


Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (built-in)
from typing import Self, Callable
from copy import deepcopy

# imports (3rd party)
from numba import jit
import numpy as np
from numpy import typing as npt

# imports (my libs)
from ..util import (
    now, comment_docstring, _not_implemented_func,
    VerboseType,
)
from .defaults import (
    DEFAULT_PARS,
    _ErosionStateDataDtype, ErosionStateDataType,
    _ErosionStateDataExtendedDtype, ErosionStateDataExtendedType,
    ParsValueType, ParsType,
)
from .cuda import (
    CAN_CUDA,
    _erode_rainfall_init_sub_cuda,
    _erode_rainfall_evolve_cuda,
)
from .nbjit import (
    _erode_rainfall_init_sub_nbjit,
)
from ..hmap import HMap
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)


#-----------------------------------------------------------------------------#
#    Switches
#-----------------------------------------------------------------------------#


_erode_rainfall_init_sub_default = (
    _erode_rainfall_init_sub_cuda if CAN_CUDA else
    _erode_rainfall_init_sub_nbjit
)

_erode_rainfall_evolve_default = (
    _erode_rainfall_evolve_cuda if CAN_CUDA else
    _not_implemented_func
)



#-----------------------------------------------------------------------------#
#    Classes
#-----------------------------------------------------------------------------#


class ErosionState(HMap):
    """Erosion state snapshot.

    Rainfall erosion.
    
    Contains all informations necessary for resuming erosion operation.
    ---------------------------------------------------------------------------
    """
    
    #-------------------------------------------------------------------------#
    #    Meta: Initialize
    #-------------------------------------------------------------------------#
    
    def __init__(
        self,
        hmap: None|HMap = None,
        pars: ParsType = DEFAULT_PARS,
        do_init: None|bool = None,
        copy : bool = False,
        verbose: VerboseType = True,
    ):
        # init
        if do_init is None:
            do_init = hmap is not None
        if hmap is None:
            hmap = HMap()

        
        # variables

        self.__done__init: bool = False
        # actual values
        self.stats: ErosionStateDataType = np.zeros(
            hmap.npix_xy, dtype=_ErosionStateDataDtype)
        # boundary conditions (will stay constant)
        self.edges: ErosionStateDataType = np.zeros(
            hmap.npix_xy, dtype=_ErosionStateDataDtype)
        # parameters
        self.__pars = deepcopy(pars)
        # extended info on stats
        self.__stats_ext: None|ErosionStateDataExtendedType = None


        # do things
        
        super().__init__(
            hmap,
            copy = copy,
            use_data_meta = True,
            verbose = verbose,
        )
        
        if do_init:
            self.init(init_edges=True, verbose=verbose)

        self.__done__init = True    # do NOT change this flag afterwards
        self.normalize(verbose=verbose)

    

    def get_par(self, name: str) -> ParsValueType:
        """Return parameter of the name."""
        return self.__pars[name]['value']

    def set_par(self, name: str, value: ParsValueType):
        """Set the parameter of the name to value.

        Performs safety checks.
        """
        if name in self.__pars.keys():
            self.__pars[name]['value'] = value
        else:
            raise ValueError(
                f"Unrecognized parameter '{name}'. " +
                f"Possible parameters are: {self.__pars.keys()}"
            )
        return

    @property
    def pars(self) -> ParsType:
        return self.__pars

    @property
    def _pars_kwargs(self) -> dict[str, ParsValueType]:
        return {k: v['value'] for k, v in self.__pars.items()}

    @property
    def stats_data(self) -> ErosionStateDataExtendedType:
        """Return the center part of the self.stats"""
        return self.stats[1:-1, 1:-1]

    @property
    def stats_ext(self) -> ErosionStateDataExtendedType:
        """Calculate the extended data of self.stats"""
        # init
        if (
            self.__stats_ext is None
            or self.__stats_ext.shape != self.stats.shape
        ):
            self.__stats_ext = np.empty(
                self.stats.shape, dtype=_ErosionStateDataExtendedDtype)
        # re-calc
        mask_has_water = self.stats['aqua'] > 0
        self.__stats_ext['h'] = self.stats['sedi'] + self.stats['aqua']
        self.__stats_ext['z'] = self.__stats_ext['h'] + self.stats['soil']
        self.__stats_ext['m'] = (
            self.get_par('rho_soil_div_aqua') * self.stats['sedi']
            + self.stats['aqua'])
        self.__stats_ext['v'][~mask_has_water] = 0
        
        # # get v from energy ekin
        # self.__stats_ext['v'][mask_has_water] = (
        #     2 * self.stats['ekin'][mask_has_water]
        #     / self.__stats_ext['m'][mask_has_water])**0.5
        
        # get v form momentum p
        self.__stats_ext['v'][mask_has_water] = (
            self.stats['p_x'][mask_has_water]**2
            + self.stats['p_y'][mask_has_water]**2
        )**0.5 / self.__stats_ext['m'][mask_has_water]
        self.__stats_ext['v']
        return self.__stats_ext
        
    
    
    def normalize(
        self, overwrite: bool = False, verbose: VerboseType = True,
    ) -> Self:
        """Resetting and safety checks."""
        
        super().normalize(overwrite=overwrite, verbose=verbose)

        # self.stats['z'] = (
        #     self.stats['sedi'] + self.stats['soil'] + self.stats['aqua']
        # )
        # self.edges['z'] = (
        #     self.edges['sedi'] + self.edges['soil'] + self.edges['aqua']
        # )
        
        if self.__done__init:
            # safety checks
            assert self.stats.shape == self.npix_xy
            assert self.edges.shape == self.npix_xy

        return self



    #-------------------------------------------------------------------------#
    #    Meta: magic methods
    #-------------------------------------------------------------------------#

    
    def __repr__(self):
        txt = f"""Erosion state object:

# Meta data
    Pixels shape  : {self.data.shape = } | {self.npix_xy = }
    Map Widths    : NS/y {self._map_widxy[0]:.2f},\
    WE/x {self._map_widxy[1]:.2f},\
    with {len(self._map_widxy) = }

# Height data insight
    Average height: {np.average(self.data):.2f} +/- {np.std(self.data):.2f}
    Height  range : [{np.min(self.data):.2f}, {np.max(self.data):.2f}]
    Height data storage config :
    |  Minimum  | Sea level |  Maximum  | Resolution |
    | {self.z_min:9.2f} | {self.z_sea:9.2f} | {self.z_max:9.2f} \
| {self.z_res:10.7f} |

# Parameters

"""
        for k, v in self.__pars.items():
            txt += f"{comment_docstring(v['_DOC_'], '    # ')}\n"
            txt += f"    {k:16}: {v['_TYPE_STR']:16} = {v['value']}\n"
            txt += '\n'

        return txt



    def __str__(self):
        return self.__repr__()



    #-------------------------------------------------------------------------#
    #    Do erosion
    #-------------------------------------------------------------------------#


    def init(
        self,
        init_edges: bool = True,
        sub_func: Callable = _erode_rainfall_init_sub_default,
        verbose: VerboseType = True,
    ) -> Self:
        """Initialization for Rainfall erosion.
    
        Parameters
        ----------
        init_edges: bool
            whether boundary conditions (self.edges) should be initialized too.
            
        sub_func: function
            Provide the function for the sub process.
            Choose between _erode_rainfall_init_sub_nbjit()   (CPU)
                and        _erode_rainfall_init_sub_cuda()    (GPU)
    
        Returns
        -------
        self
    
        -----------------------------------------------------------------------
        """

        # - time it -
        runtime_t0 = now()
        if verbose:
            print(f"Time: ErosionState.init() Starting: {runtime_t0}")

        # - init -
        npix_x, npix_y = self.npix_xy
        z_min, z_sea, z_max, z_res = self.z_config
        
        # self.stats[:] = 0
    
        # init soils
        self.stats['soil'] = np.where(
            self.data <= z_min,
            0.,
            self.data - z_min,
        )
        soils = self.stats['soil']
        
        # init edges (i.e. const lvl water spawners)
        if init_edges:
            # could use optimization
            # self.edges['sedi'] and self.edges['ekin']
            #    are always 0 by default
            self.edges[:] = 0
            self.edges['soil'] = self.stats['soil']
            self.edges['soil'][1:-1, 1:-1] = -1.
            self.edges['aqua'] = np.where(
                self.edges['soil'] < 0.,
                -1.,
                np.where(
                    self.edges['soil'] < z_sea,
                    z_sea - self.edges['soil'],
                    0.,
                ),
            )
        
        # - fill basins -
        # (lakes / sea / whatev)
        zs, n_cycles = sub_func(
            self.stats['soil'], self.edges['soil'] + self.edges['aqua'],
            z_range=z_max-z_min)

        self.stats['aqua'] = zs - self.stats['soil']
        self.stats['sedi'] = 0.
        # speed is zero, so
        self.stats['p_x' ] = 0.
        self.stats['p_y' ] = 0.
        self.stats['ekin'] = 0. # is zero because speed is zero
        # self.stats['z'] = self.stats['soil'] + self.stats['aqua']

        # - time it -
        runtime_t1 = now()
        if verbose:
            print(
                f"Time: ErosionState.init() Ending: {runtime_t1}\n" +
                f"    Total used time: {runtime_t1 - runtime_t0}\n",
            )
            print(f"    Debug: {n_cycles} cycles used for initialization.")

        return self



    def evolve(
        self,
        n_step: int = 4,
        sub_func: Callable = _erode_rainfall_evolve_default,
        verbose: VerboseType = True,
    ) -> Self:
        """Wrapper func for rainfall erosion.
        
        Parameters used see self.pars
        Use self.set_par() to change them.
        """
        
        # - time it -
        runtime_t0 = now()
        if verbose:
            print(f"Time: ErosionState.evolve() Starting: {runtime_t0}")
        
        # - run -
        self.stats = sub_func(
            n_step, self.stats, self.edges, self.npix_xy,
            z_max=self.z_range, z_res=self.z_res,
            verbose=verbose, **self._pars_kwargs)

        # - time it -
        runtime_t1 = now()
        if verbose:
            print(
                f"Time: ErosionState.evolve() Ending: {runtime_t1}\n" +
                f"    Total used time: {runtime_t1 - runtime_t0}\n",
            )
        
        return self


#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
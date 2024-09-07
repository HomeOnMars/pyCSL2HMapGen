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


# Dependencies
from ..util import (
    now, comment_docstring, _not_implemented_func,
    VerboseType,
)
from .defaults import (
    DEFAULT_PARS,
    _ErosionStateDataDtype, ErosionStateDataType,
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

from typing import Self, Callable
from copy import deepcopy

from numba import jit
import numpy as np
from numpy import typing as npt



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


def _shape_add_two(shape: tuple[int, int]) -> tuple[int, int]:
    return tuple([x+2 for x in shape])



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
            _shape_add_two(hmap.npix_xy), dtype=_ErosionStateDataDtype)
        # boundary conditions (will stay constant)
        self.edges: ErosionStateDataType = np.zeros(
            _shape_add_two(hmap.npix_xy), dtype=_ErosionStateDataDtype)
        # parameters
        self.__pars = deepcopy(pars)


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
    def shape_stats(self) -> tuple[int, int]:
        return self.stats.shape
        
    @property
    def _shape_stats_calc(self) -> tuple[int, int]:
        return _shape_add_two(self.npix_xy)
        

    def normalize(
        self, overwrite: bool = False, verbose: VerboseType = True,
    ) -> Self:
        """Resetting and safety checks."""
        
        super().normalize(overwrite=overwrite, verbose=verbose)

        self.stats['z'] = (
            self.stats['sedi'] + self.stats['soil'] + self.stats['aqua']
        )
        self.edges['z'] = (
            self.edges['sedi'] + self.edges['soil'] + self.edges['aqua']
        )
        
        if self.__done__init:
            # safety checks
            assert self.shape_stats == self._shape_stats_calc

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
        data  = self.data
        soils = self.stats['soil']
    
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
        self.stats['soil'] = np.where(
            soils <= z_min,
            0.,
            soils - z_min,
        )
        soils = self.stats['soil']
        
        # init edges (i.e. const lvl water spawners)
        if init_edges:
            # could use optimization
            # self.edges['sedi'] and self.edges['ekin']
            #    are always 0 by default
            self.edges[:] = 0.
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
            self.edges['z'] = self.edges['soil'] + self.edges['aqua']
        
        # - fill basins -
        # (lakes / sea / whatev)
        zs, n_cycles = sub_func(
            self.stats['soil'], self.edges['z'], z_range=z_max-z_min)

        self.stats['aqua'] = self.edges['aqua']
        self.stats['aqua'][1:-1, 1:-1] = (zs - soils)[1:-1, 1:-1]
        self.stats['sedi'] = 0.
        self.stats['ekin'] = 0. # is zero because speed is zero
        self.stats['z'] = self.stats['soil'] + self.stats['aqua']

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
            n_step, self.stats, self.edges, verbose=verbose,
            **self._pars_kwargs)

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
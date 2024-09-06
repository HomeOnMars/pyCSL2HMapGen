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
    comment_docstring,
    VerboseType,
)
from .defaults import (
    DEFAULT_PARS,
    ParsValueType, ParsType,
)
from .cuda import (
    CAN_CUDA,
    _erode_rainfall_init_sub_cuda,
)
from ..hmap import HMap

from typing import Self, Callable
from copy import deepcopy

from numba import jit
import numpy as np
from numpy import typing as npt



#-----------------------------------------------------------------------------#
#    Classes
#-----------------------------------------------------------------------------#


_ErosionStateDataDtype : np.dtype = np.dtype([
    ('sedi', np.float32),    # sediment (in water) height
    ('soil', np.float32),    # soil (solid) height
    ('aqua', np.float32),    # water (excluding sediment) height
    ('ekin', np.float32),    # kinetic energy of water+sediment
    ('z'   , np.float32),    # total height (z = soil + sedi + aqua)
])


def _shape_add_two(shape: tuple[int, int]) -> tuple[int, int]:
    return tuple([x+2 for x in shape])



class ErosionState(HMap):
    """Erosion state snapshot.
    
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
        self.stats: npt.NDArray[_ErosionStateDataDtype] = np.zeros(
            _shape_add_two(hmap.npix_xy), dtype=_ErosionStateDataDtype)
        # boundary conditions (will stay constant)
        self.edges: npt.NDArray[_ErosionStateDataDtype] = np.zeros(
            _shape_add_two(hmap.npix_xy), dtype=_ErosionStateDataDtype)
        # parameters
        self.__pars = deepcopy(pars)


        # do things
        
        super().__init__(
            hmap,
            use_data_meta = True,
            verbose = verbose,
        )
        
        if do_init:
            raise NotImplementedError("Erosion init func wrapper to be added")

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
            pass

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



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
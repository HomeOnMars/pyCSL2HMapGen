#!/usr/bin/env python
# coding: utf-8

"""Height map erosion with GPU-accelerations.

Require Cuda. CPU version no longer supported.

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
    ParsValueType,
)
from .cuda import (
    CAN_CUDA,
    _erode_rainfall_init_sub_cuda,
)
from ..hmap import HMap

from typing import Self
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
])



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
        pars: dict[
            str, dict[str, type|str|float|np.float32|npt.NDArray]
        ] = DEFAULT_PARS,
        do_init: None|bool = None,
        verbose: VerboseType = True,
    ):
        # init
        if hmap is None:
            hmap = HMap()
            do_init = False
        if do_init is None:
            do_init = True
        super().__init__(
            hmap,
            use_data_meta = True,
            verbose = verbose,
        )

        
        # variables

        # actual values
        self.stats: npt.NDArray[_ErosionStateDataDtype] = np.empty(
            self.npix_xy, dtype=_ErosionStateDataDtype)
        # boundary conditions (will stay constant)
        self.edges: npt.NDArray[_ErosionStateDataDtype] = np.empty(
            self.npix_xy, dtype=_ErosionStateDataDtype)
        # parameters
        self.__pars = deepcopy(pars)
        
        if do_init:
            raise NotImplementedError("Erosion init func wrapper to be added")

    

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
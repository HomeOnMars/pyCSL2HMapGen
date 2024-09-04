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
        hmap: HMap = HMap(),
        pars:dict[
        str, dict[str, type|str|float|np.float32|npt.NDArray]] = DEFAULT_PARS,
        verbose: VerboseType = True,
    ):
        self.soils: npt.NDArray[np.float32]    # oi
        self.aquas: npt.NDArray[np.float32]    # aq
        self.sedis: npt.NDArray[np.float32]    # se
        self.ekins: npt.NDArray[np.float32]    # ek
        self.edges: npt.NDArray[np.float32]    # dg

        self.__pars = deepcopy(pars)
        super().__init__(
            hmap,
            use_data_meta = True,
            verbose = verbose,
        )

        # *** Add code to save parameters here ***
        

    def get_zs(self) -> npt.NDArray[np.float32]:
        """Return total height."""
        return self.soils + self.aquas + self.ekins

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
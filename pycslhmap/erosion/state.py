#!/usr/bin/env python
# coding: utf-8

"""Height map erosion with GPU-accelerations.

Require Cuda. CPU version no longer supported.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies
from ..util import VerboseType, comment_docstring
from .cuda import (
    CAN_CUDA, _erode_rainfall_init_sub_cuda,
)
from ..hmap import HMap

from typing import Self

from numba import jit
import numpy as np
from numpy import typing as npt



#-----------------------------------------------------------------------------#
#    Constants
#-----------------------------------------------------------------------------#


ParsValueType = float|np.float32|npt.NDArray[np.float32]

DEFAULT_PARS : dict[str, dict[str, type|ParsValueType|str]] = {
    # Note: Do NOT edit _TYPE and _DOC_ in real-time - they will not be used.
    'rain_configs': {
        '_TYPE': npt.NDArray[np.float32],
        'value': np.array([2.**(-7)], dtype=np.float32),
        '_DOC_': """Kinematic visocity of water and soils in SI units (m^2/s).
        .
            tuple((visco_kin_aqua, visco_kin_soil))
            Must have visco_kin_aqua <= visco_kin_soil.
            It is ~1e-6 for water and 1e-2 ~ 1e-1 for mud.
        .""",
    },
    'flow_eff': {
        '_TYPE': np.float32,
        'value': np.float32(0.25),
        '_DOC_': """Flow efficiency.
        .
            Should be in 0. < flow_eff <= 1.
            Controls how well the water flows around.
            Do NOT touch this unless you know what you are doing.
        .""",
    },
    'visco_kin_range': {
        '_TYPE': tuple[np.float32, np.float32],
        'value': (np.float32(1e-6), np.float32(1.0)),
        '_DOC_': """Kinematic visocity of water and soils in SI units (m^2/s).
        .
            Must have visco_kin_aqua <= visco_kin_soil.
            It is ~1e-6 for water and 1e-2 ~ 1e-1 for mud.
        .""",
    },
    'sed_cap_fac': {
        '_TYPE': np.float32,
        'value': np.float32(1.0),
        '_DOC_': """Sediment capacity factor of the river.
        .
            Limits the maximum of the sediemnt capacity.
        .""",
    },
    'erosion_eff': {
        '_TYPE': np.float32,
        'value': np.float32(0.125),
        '_DOC_': """Erosion/deposition efficiency.
        .
            Should be 0. <= erosion_eff <= 1.
            Setting it to 0. will disable erosion and deposition.
        .""",
    },
    'diffuse_eff': {
        '_TYPE': np.float32,
        'value': np.float32(0.25),
        '_DOC_': """Diffusion efficiency.
        .
            Should be 0. <= diffuse_eff <= 1.
            Controls how fast sediments and kinetic energies spread
                in lakes etc.
        .""",
    },
    'hole_depth': {
        '_TYPE': np.float32,
        'value': np.float32(2.**(-8)),
        '_DOC_': """Maximum depth of the hole allowed
        by the erosion process to dig per step. Should be >= 0.
        .
            If > 0., the erosion process may dig lakes.
                (but may also dig single pixel holes)
        .""",
    },
    'slope_facs': {
        '_TYPE': tuple[np.float32, np.float32],
        'value': (np.float32(1.0), np.float32(1.0)),
        '_DOC_': """Factor to used in the slope calculation.
        .
            Should be 0. < fac < 1.
            if downhill_fac > upfill_fac, will make more gently sloped hills;
            else, will make more cliffs
        .""",
    },
    'v_cap': {
        '_TYPE': np.float32,
        'value': np.float32(16.),
        '_DOC_': """Characteristic velocity for sediment capacity calculations,
        in m/s.
        .
            Used to regulate the velocity in capas calc,
            So its influence flatten out when v is high.
        .""",
    },
    'g': {
        '_TYPE': np.float32,
        'value': np.float32(9.8),
        '_DOC_': """Gravitational constant in m/s2.""",
    },
}

# finish init DEFAULT_PARS
for k, v in DEFAULT_PARS.items():
    if '_TYPE_STR' not in v.keys():
        if   v['_TYPE'] == np.float32:
            _type_str = 'float32'
        elif v['_TYPE'] == npt.NDArray[np.float32]:
            _type_str = 'NDArray[float32]'
        elif v['_TYPE'] == tuple[np.float32, np.float32]:
            _type_str = 'tuple[float32, float32]'
        else:
            raise NotImplementedError(
                'Unrecognized type in DEFAULT_PARS. Fix me!')
        v['_TYPE_STR'] = _type_str



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

        self.__pars = pars
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
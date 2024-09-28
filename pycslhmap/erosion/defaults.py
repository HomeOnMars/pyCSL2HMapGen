#!/usr/bin/env python
# coding: utf-8

"""Height map erosion with GPU-accelerations.

Require Cuda. CPU version no longer supported.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (3rd party)
import numpy as np
from numpy import typing as npt

# imports (my libs)
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)



#-----------------------------------------------------------------------------#
#    Constants and types
#-----------------------------------------------------------------------------#


# per rhoS means [per water density (rho) per pixel area (S)]
#    i.e. translates mass into mass-equivalent water height
# fluid is water + sediment

_ErosionStateDataDtype : np.dtype = np.dtype([
    # note: nan might be noted as -1 in positive fields
    #       positive means not negative (i.e. incl. zero)
    ('soil', np.float32),   # [positive][m] soil (solid) height
    ('sedi', np.float32),   # [ signed ][m] sediment (in water) height
                            #     Warning: could be negative from leftover unfulfilled sedi deposition requests
    ('aqua', np.float32),   # [positive][m] water (excluding sediment) height
    ('p_x' , np.float32),   # [ signed ][m^2/s] momentum per rhoS in x direction (NS)
    ('p_y' , np.float32),   # [ signed ][m^2/s] momentum per rhoS in y direction (WE)
    ('ekin', np.float32),   # [ signed ][m^3/s^2] LEFTOVER kinetic energy budget of fluid per rhoS
                            #     Could be either positive or negative
])
ErosionStateDataType = npt.NDArray[_ErosionStateDataDtype]

_ErosionStateDataExtendedDtype : np.dtype = np.dtype([
    # note: nan might be noted as -1 in positive fields
    ('z'   , np.float32),   # [positive][m] total height (z = soil+sedi+aqua)
    ('h'   , np.float32),   # [positive][m] fluid height (h =      sedi+aqua)
    ('d'   , np.float32),   # [positive][m] dirt  height (d = soil+sedi     )
    ('m'   , np.float32),   # [positive][m] fluid mass per rhoS
                            #        (m = rho_soil_div_aqua * sedi + aqua)
    ('v'   , np.float32),   # [positive][m/s] fluid speed
])
ErosionStateDataExtendedType = npt.NDArray[_ErosionStateDataExtendedDtype]

ParsValueType = (
    float|np.float32|npt.NDArray[np.float32]|tuple[np.float32, np.float32]
)

ParsType = dict[
    str,
    dict[str, type|str|float|np.float32|npt.NDArray]
]


DEFAULT_PARS : dict[str, dict[str, type|ParsValueType|str]] = {
    # Note: Do NOT edit _TYPE / _DOC_ / etc. in real-time -
    #    they will not be used.
    
    # Flow
    #--------------------------------------------------------------------------
    'evapor_rate': {
        '_TYPE': np.float32,
        'value': np.float32(-2.**(-10)),
        '_DOC_': """Evaporation rate per step.
        .
            float32 with abs(evapor_rate) >= z_res or == 0
            If negative: will add water per step instead of taking them away,
                i.e. assuming uniform raining.
        .""",
    },
    'flow_eff': {
        '_TYPE': np.float32,
        'value': np.float32(1.0),
        '_DOC_': """Flow efficiency.
        .
            Should be in 0. < flow_eff <= 1.
            Controls how well the water flows around.
            Do NOT touch this unless you know what you are doing.
        .""",
    },
    'rho_soil_div_aqua': {
        '_TYPE': np.float32,
        'value': np.float32(1.5),
        '_DOC_': """Soil/Sediment density vs water density ratio.
        .
            Assuming sediment density are the same as soil,
            Even if it mixes with the water.
        .""",
    },
    'v_cap': {
        '_TYPE': np.float32,
        'value': np.float32(16.),
        '_DOC_': """Characteristic velocity for sediment capacity calculations,
        in m/s.
        .
            Must: v_cap > 0.
            Fluid speed shall not exceed this;
            any extra kinetic energy will go to stats['ekin'].
            This also determines the time step dt.
        .""",
    },
    'g': {
        '_TYPE': np.float32,
        'value': np.float32(9.8),
        '_DOC_': """Gravitational constant in m/s2.""",
    },
    # Erosion
    #--------------------------------------------------------------------------
    'erosion_eff': {
        '_TYPE': np.float32,
        'value': np.float32(0.25),
        '_DOC_': """Overall Erosion/deposition efficiency.
        .
            Should be 0. <= erosion_eff <= 1.
            Setting it to 0. will disable erosion and deposition.
        .""",
    },
    'erosion_brush': {
        '_TYPE': npt.NDArray[np.float32],
        'value': np.array([0.5, 0.125, 0.125, 0.125, 0.125], dtype=np.float32),
        '_DOC_': """How much to be eroded for the cell and its adjacent cells.
        .
            Should have exactly 5 elements. Each 0. <= erosion_brush[i] <= 1.
            First element must not be zero.
            Sum of which should be 1 for equal speed of erosion and deposition.
            First one is for the cell itself, the rest for the adjacent cells.
        .""",
    },
    'sedi_capa_fac': {
        '_TYPE': np.float32,
        'value': np.float32(1.0),
        '_DOC_': """Sediment capacity factor of the river.
        .
            Must: sedi_capa_fac >= 0
            Ratio of max sediment vs existing water.
        .""",
    },
    'sedi_capa_fac_base': {
        '_TYPE': np.float32,
        'value': np.float32(0.5),
        '_DOC_': """Another Sediment capacity factor, this time for still water only.
        .
            Must: sedi_capa_fac_base >= 0
            Dimensionless.
                1   means the fastest water can carry twice as much sedi as still water;
                10  means the fastest water can carry 11/10 as much sedi as still water.
                0.5 means the fastest water can carry 1.5/0.5 = 3x as much sedi as still water.
        .""",
    },

    # Old / In Reserve / Abandoned
    #--------------------------------------------------------------------------
    'rain_configs': {
        '_TYPE': npt.NDArray[np.float32],
        'value': np.array([2.**(-6)], dtype=np.float32),
        '_DOC_': """Rain configuration.
        .
            ***type***
            ***Add doc here***
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
#    End
#-----------------------------------------------------------------------------#
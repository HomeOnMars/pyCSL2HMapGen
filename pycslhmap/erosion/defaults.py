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

ErosionStateDataDtype : np.dtype = np.dtype([
    # note: nan might be noted as -1 in positive fields
    #       positive means not negative (i.e. incl. zero)
    ('soil', np.float32),   # [positive][m] soil (solid) height
    ('sedi', np.float32),   # [ signed ][m] sediment (in water) height
                            #     Warning: could be negative from leftover unfulfilled sedi deposition requests
    ('aqua', np.float32),   # [positive][m] water (excluding sediment) height
    ('p_x' , np.float32),   # [ signed ][m^2/s] momentum per rhoS in x direction (NS)
    ('p_y' , np.float32),   # [ signed ][m^2/s] momentum per rhoS in y direction (WE)
    ('ekin', np.float32),   # [negative][m^3/s^2] LEFTOVER kinetic energy debt of fluid per rhoS
                            #     Could be either positive or negative
])
ErosionStateDataType = npt.NDArray[ErosionStateDataDtype]

# ErosionStateDataExtendedDtype
ErosionStateDataExtDtype : np.dtype = np.dtype([
    # note: nan might be noted as -1 in positive fields
    ('z'   , np.float32),   # [positive][m] total height (z = soil+sedi+aqua)
    ('h'   , np.float32),   # [positive][m] fluid height (h =      sedi+aqua)
    ('d'   , np.float32),   # [positive][m] dirt  height (d = soil+sedi     )
    ('m'   , np.float32),   # [positive][m] fluid mass per rhoS
                            #        (m = rho_sedi * sedi + aqua)
    ('v'   , np.float32),   # [positive][m/s] fluid speed
])
ErosionStateDataExtType = npt.NDArray[ErosionStateDataExtDtype]

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
    'n_step': {
        '_TYPE': int,
        'value': None,
        '_DOC_': """Number of steps for this part of erosion cycle.
        .
            None or int.
            If None, will be selected as 1/8 of the average pixel number per axis.
        .""",
    },
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
        'value': np.float32(0.625),  # 0.75
        '_DOC_': """Flow factor.
        .
            Should be in 0. <= flow_eff <= 1.
            Controls how the water flows around,
                i.e. momentum-based movements vs gravity-based movements.
                1 is all gravity,
                and 0 is all momentum (hence nothing will flow because no gravity is there to initiate movements)
            Changing this may drastically affect simulation results.
        .""",
    },
    # 'turning': {
    #     '_TYPE': np.float32,
    #     'value': np.float32(1.0),
    #     '_DOC_': """Turning efficiency.
    #     .
    #         Should be turning >= 0.
    #         Controls how fast the water velocity conforms to local gradient.
    #         1.0 means halfway, 0.0 means no turning effect, ~+inf means fullly confrom to local gradient.
    #         Warning: dependent on pixel resolution.
    #     .""",
    # },
    'turning_gradref': {
        '_TYPE': np.float32,
        'value': np.float32(0.1),
        '_DOC_': """Reference gradient for turning strength.
        .
            Should be turning_gradref >= 0.
            i.e. typical gradient that results in a typical turning.
            if the local gradient is the same as this,
                the velocity will be turned halfway in-between the original velocity and local gradient.
            if 0, will reset the velocity direction to the local gradient direction.
        .""",
    },
    'rho_sedi': {
        '_TYPE': np.float32,
        'value': np.float32(1.5),
        '_DOC_': """Sediment density in unit of water density.
        .
            Assuming sediment density are the same as soil,
            Even if it mixes with the water.
        .""",
    },
    'v_cap': {
        '_TYPE': np.float32,
        'value': np.float32(16.),
        '_DOC_': """Characteristic velocity for sediment capacity calculations, in m/s.
        .
            Must: v_cap > 0.
            Fluid speed shall not exceed this;
            any extra kinetic energy will go to stats['ekin'].
            This also determines the time step dt.
        .""",
    },
    'v_damping': {
        '_TYPE': np.float32,
        'value': np.float32(2**(-10)),
        '_DOC_': """Damping factor for momentums per step.
        .
            Must: 0. <= v_damping <= 1.
            0 means no damping,
            while 1 means all velocity immediately reset to 0 after each step.
        .""",
    },
    'g_eff': {
        '_TYPE': np.float32,
        'value': np.float32(8),
        '_DOC_': """Effective gravitational constant, in m/s2.
        .
            Must: g_eff >= 0
            Gravitational constant multiplied with how efficient the gravitatioal energy is being transformed into momentum.
            0 means no gravitational energy gain (so the water will not move)
        .""",
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
    'capa_fac': {
        '_TYPE': np.float32,
        'value': np.float32(1.0),
        '_DOC_': """Sediment capacity factor of the river.
        .
            Must: capa_fac >= 0
            Ratio of max sediment vs existing water.
        .""",
    },
    'capa_fac_v': {
        '_TYPE': np.float32,
        'value': np.float32(1.0),
        '_DOC_': """Another sediment capacity factor, this time for water velocity.
        .
            Must: capa_fac_v > -1
            Dimensionless.
                The fastest water can carry (1+capa_fac_v) as much sediments as still water.
                e.g.:
                -0.5 means the fastest water can carry half as much sedi as still water;
                0    means water velocity have no effect on sediment capacity;
                1    means the fastest water can carry twice as much sedi as still water.
        .""",
    },
    'capa_fac_slope': {
        '_TYPE': np.float32,
        'value': np.float32(1.0),
        '_DOC_': """Another sediment capacity factor, this time for slope.
        .
            Must: capa_fac_slope > -1
            Dimensionless.
                The steepest water can carry (1+capa_fac_slope) as much sediments as water with leveled water surface.
                e.g.:
                e.g.:
                -0.5 means the steepest water can carry half as much sedi as leveled water;
                0    means the slope have no effect on sediment capacity;
                1    means the steepest water can carry twice as much sedi as leveled water.
        .""",
    },
    'capa_fac_slope_cap': {
        '_TYPE': np.float32,
        'value': np.float32(1.0),
        '_DOC_': """Maximum slope for sediment capacity factor calculation purposes.
        .
            Dimensionless.
        .""",
    },

    # Old / In Reserve / Abandoned
    #--------------------------------------------------------------------------
    # 'rain_configs': {
    #     '_TYPE': npt.NDArray[np.float32],
    #     'value': np.array([2.**(-6)], dtype=np.float32),
    #     '_DOC_': """Rain configuration.
    #     .
    #         ***type***
    #         ***Add doc here***
    #     .""",
    # },
    # 'visco_kin_range': {
    #     '_TYPE': tuple[np.float32, np.float32],
    #     'value': (np.float32(1e-6), np.float32(1.0)),
    #     '_DOC_': """Kinematic visocity of water and soils in SI units (m^2/s).
    #     .
    #         Must have visco_kin_aqua <= visco_kin_soil.
    #         It is ~1e-6 for water and 1e-2 ~ 1e-1 for mud.
    #     .""",
    # },
    # 'diffuse_eff': {
    #     '_TYPE': np.float32,
    #     'value': np.float32(0.25),
    #     '_DOC_': """Diffusion efficiency.
    #     .
    #         Should be 0. <= diffuse_eff <= 1.
    #         Controls how fast sediments and kinetic energies spread
    #             in lakes etc.
    #     .""",
    # },
    # 'hole_depth': {
    #     '_TYPE': np.float32,
    #     'value': np.float32(2.**(-8)),
    #     '_DOC_': """Maximum depth of the hole allowed
    #     by the erosion process to dig per step. Should be >= 0.
    #     .
    #         If > 0., the erosion process may dig lakes.
    #             (but may also dig single pixel holes)
    #     .""",
    # },
    # 'slope_facs': {
    #     '_TYPE': tuple[np.float32, np.float32],
    #     'value': (np.float32(1.0), np.float32(1.0)),
    #     '_DOC_': """Factor to used in the slope calculation.
    #     .
    #         Should be 0. < fac < 1.
    #         if downhill_fac > upfill_fac, will make more gently sloped hills;
    #         else, will make more cliffs
    #     .""",
    # },
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
        elif v['_TYPE'] == int:
            _type_str = 'int'
        else:
            raise NotImplementedError(
                'Unrecognized type in DEFAULT_PARS. Fix me!')
        v['_TYPE_STR'] = _type_str



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
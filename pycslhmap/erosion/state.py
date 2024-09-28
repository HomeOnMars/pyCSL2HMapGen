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
    CAN_CUDA, CUDA_TPB_X, CUDA_TPB_Y,
    erode_rainfall_init_sub_cuda,
    erode_rainfall_evolve_cuda,
)
from .nbjit import (
    erode_rainfall_init_sub_nbjit,
)
from ..hmap import HMap
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)


#-----------------------------------------------------------------------------#
#    Switches
#-----------------------------------------------------------------------------#


_erode_rainfall_init_sub_default = (
    erode_rainfall_init_sub_cuda if CAN_CUDA else
    erode_rainfall_init_sub_nbjit
)

_erode_rainfall_evolve_default = (
    erode_rainfall_evolve_cuda if CAN_CUDA else
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
        do_trim: None|bool = False,
        copy : bool = True,
        verbose: VerboseType = True,
    ):
        """Init.

        do_trim:
            If we should throw away some cells at edges of the hmap.data
            so nx-2, ny-2 are multiples of 14
            This will optimize the process a little bit.
        """
        # init
        if do_init is None:
            do_init = hmap is not None
        if do_trim is None:
            do_trim = hmap is not None
        if hmap is None:
            hmap = HMap()
        if do_trim:
            hmap = HMap(hmap)
            rm_nx = ((hmap.npix_xy[0]-2)%(CUDA_TPB_X-2)+1)
            rm_nx0 = rm_nx // 2; rm_nx1 = rm_nx - rm_nx0
            rm_ny = ((hmap.npix_xy[1]-2)%(CUDA_TPB_Y-2)+1)
            rm_ny0 = rm_ny // 2; rm_ny1 = rm_ny - rm_ny0
            hmap.data = hmap.data[rm_nx0:-rm_nx1, rm_ny0:-rm_ny1]

        
        # variables

        self.__done__init: bool = False
        # actual values
        self.stats: ErosionStateDataType = np.zeros(
            hmap.npix_xy, dtype=_ErosionStateDataDtype)
        # boundary conditions (will stay constant)
        #    unset pixels will be denoted as np.nan
        self.edges: ErosionStateDataType = np.zeros(
            hmap.npix_xy, dtype=_ErosionStateDataDtype)
        # parameters
        self.__pars = deepcopy(pars)
        # extended info on stats
        self.__stats_ext: None|ErosionStateDataExtendedType = None
        # cycles of erosions ran after initialization
        self.__i_cycle: int = 0
        # log keeping
        self.__log_txts: list[str] = []


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
    def i_cycle(self) -> int:
        return self.__i_cycle
        
    @property
    def log_txt(self) -> str:
        return '\n'.join(self.__log_txts)

    @property
    def log_txts(self) -> tuple[str]:
        return tuple(self.__log_txts)

    @property
    def log_last(self) -> str:
        return self.__log_txts[-1] if self.__log_txts else ''

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
        self.__stats_ext['d'] = self.stats['soil'] + self.stats['sedi']
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



    @property
    def delta_height(self) -> npt.NDArray[np.float32]:
        """Return the change in height after erosion.

        I.e., getting the results.
        """
        return self.z_min + self.stats['soil'] + self.stats['sedi'] - self.data
        
    
    
    def normalize(
        self, overwrite: bool = False, verbose: VerboseType = True,
    ) -> Self:
        """Resetting and safety checks."""
        
        super().normalize(overwrite=overwrite, verbose=verbose)
        
        if self.__done__init:
            # safety checks
            assert self.stats.shape == self.npix_xy
            assert self.edges.shape == self.npix_xy

        return self



    #-------------------------------------------------------------------------#
    #    Log keeping
    #-------------------------------------------------------------------------#


    def get_info(self) -> str:
        stats_ext = self.stats_ext
        aq = self.stats['aqua']
        zs = stats_ext['z']
        hs = stats_ext['h']
        ds = stats_ext['d']
        return f"""
{self.i_cycle}
    water lvl: {np.average(aq):9.4f} +/- {np.std(aq):9.4f}   in [{np.min(aq):9.4f}, {np.max(aq):9.4f}]
    fluid lvl: {np.average(hs):9.4f} +/- {np.std(hs):9.4f}   in [{np.min(hs):9.4f}, {np.max(hs):9.4f}]
    dirt  lvl: {np.average(ds):9.4f} +/- {np.std(ds):9.4f}   in [{np.min(ds):9.4f}, {np.max(ds):9.4f}]
    total z  : {np.average(zs):9.4f} +/- {np.std(zs):9.4f}   in [{np.min(zs):9.4f}, {np.max(zs):9.4f}]
"""



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

# Erosion state insight
{self.__log_txts[-1] if self.__log_txts else 'N/A'}


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
            Choose between erode_rainfall_init_sub_nbjit()   (CPU)
                and        erode_rainfall_init_sub_cuda()    (GPU)
    
        Returns
        -------
        self
    
        -----------------------------------------------------------------------
        """

        # - time it -
        runtime_t0 = now()
        if verbose:
            print(f"Time: ErosionState.init() Starting: {runtime_t0}")
        self.__i_cycle = 0
        self.__log_txts = []
        

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

            # mask: at map border
            mask = np.full(self.stats.shape, False, dtype=np.bool_)
            mask[ :2] = True
            mask[-2:] = True
            mask[:,  :2] = True
            mask[:, -2:] = True

            # set edges at border
            self.edges[:] = np.nan
            self.edges[mask] = 0
            self.edges['soil'][mask] = self.stats['soil'][mask]
            self.edges['aqua'][mask] = (z_sea-z_min) - self.edges['soil'][mask]
            self.edges['aqua'][self.edges['aqua'] <= 0.] = 0.

        edges_zs = np.zeros_like(self.edges['soil'])
        mask = ~np.isnan(self.edges['soil']); mask_any = mask
        edges_zs[mask] += self.edges['soil'][mask]
        mask = ~np.isnan(self.edges['sedi']); mask_any = mask_any|mask
        edges_zs[mask] += self.edges['sedi'][mask]
        mask = ~np.isnan(self.edges['aqua']); mask_any = mask_any|mask
        edges_zs[mask] += self.edges['aqua'][mask]
        edges_zs[~mask_any] = np.nan
        
        # - fill basins -
        # (lakes / sea / whatev)
        zs, n_cycles = sub_func(
            self.stats['soil'], edges_zs, z_range=z_max-z_min)

        # *** Note: update below when updating _get_capa_cudev() ***
        #     Getting the initial sediments when speed v=0
        hs = zs - self.stats['soil']
        sedi_capa_fac = self.get_par('sedi_capa_fac')
        self.stats['aqua'] = hs / (1.+sedi_capa_fac)
        self.stats['sedi'] = hs * sedi_capa_fac/(1.+sedi_capa_fac)
        # speed is zero, so
        self.stats['p_x' ] = 0.
        self.stats['p_y' ] = 0.
        self.stats['ekin'] = 0. # is zero because speed is zero
        # self.stats['z'] = self.stats['soil'] + self.stats['aqua']

        # - time it -
        self.__log_txts.append(self.get_info())
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
        self.__i_cycle += 1
        
        # - run -
        self.stats = sub_func(
            n_step, self.stats, self.edges,
            z_max=self.z_range, z_res=self.z_res, pix_widxy = self.pix_widxy,
            verbose=verbose, **self._pars_kwargs)

        # - time it -
        self.__log_txts.append(self.get_info())
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
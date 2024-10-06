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
import matplotlib as mpl
from matplotlib import pyplot as plt

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
        self.__i_step : int = 0
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
    def i_step(self) -> int:
        return self.__i_step
        
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
        self.__stats_ext['h'] = self.stats['sedi'] + self.stats['aqua']
        self.__stats_ext['d'] = self.stats['soil'] + self.stats['sedi']
        self.__stats_ext['z'] = self.__stats_ext['h'] + self.stats['soil']
        self.__stats_ext['m'] = (
            self.get_par('rho_sedi') * self.stats['sedi']
            + self.stats['aqua'])
        mask_m = self.__stats_ext['m'] > 0
        self.__stats_ext['v'][~mask_m] = 0
        
        # # get v from energy ekin
        # self.__stats_ext['v'][mask_m] = (
        #     2 * self.stats['ekin'][mask_m]
        #     / self.__stats_ext['m'][mask_m])**0.5
        
        # get v form momentum p
        self.__stats_ext['v'][mask_m] = (
            self.stats['p_x'][mask_m]**2
            + self.stats['p_y'][mask_m]**2
            + (
                2*self.__stats_ext['m'][mask_m]
                * self.stats[   'ekin'][mask_m]
        ))    # m^2 v^2
        self.__stats_ext['v'][mask_m] = np.where(
            self.__stats_ext['v'][mask_m] > 0,
            self.__stats_ext['v'][mask_m],
            0)
        self.__stats_ext['v'][mask_m] = (
            self.__stats_ext['v'][mask_m]**0.5
            / self.__stats_ext['m'][mask_m])
        return self.__stats_ext



    @property
    def delta_height(self) -> npt.NDArray[np.float32]:
        """Return the change in height after erosion.

        I.e., getting the results.
        -----------------------------------------------------------------------
        """
        return self.z_min + self.stats['soil'] - self.data
        
    
    
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
{self.i_cycle:5d}:    Step #{self.i_step}
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
    #    Plotting
    #-------------------------------------------------------------------------#


    def plot_xsec(
        self,
        kj: None|int = None,
        figsize : tuple[int, int] = (13, 6),
        ylim: tuple[None|int, None|int] = (0, None),
    ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Plot a cross-section of the soil state for debug purposes."""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        kj = self.npix_xy[1]//2 if kj is None else kj
    
        xs = np.linspace(-self.map_widxy[0]/2, self.map_widxy[0]/2, self.npix_xy[0])
        
        ax.plot(xs, (self.data - self.z_min)[:, kj], '--', color='C8', label='original')
        
        ax.plot(xs, self.stats_ext['z'][:, kj], color='C0', label='aqua')
        ax.plot(xs, self.stats_ext['d'][:, kj], color='C2', label='sedi')
        ax.plot(xs, self.stats[ 'soil'][:, kj], color='C1', label='soil')
        
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ylim)

        ax.set_xlabel("$x$ / m")
        ax.set_ylabel("Height / m")
        ax.set_title(f"Cross section at y={self.ind_to_pos((0, kj))[1]:.1f}m")
        
        ax.legend()
        
        return fig, ax



    def plot_delta_height(
        self,
        fig : None|mpl.figure.Figure = None,
        ax  : None|mpl.axes.Axes     = None,
        figsize : tuple[int, int] = (8, 6),
        norm    : None|str|mpl.colors.Normalize = 'default',
        cmap    : str|mpl.colors.Colormap ='seismic',
        add_cbar: bool = True,
        **kwargs,
    ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Plot changes from erosion.

        Parameters
        ----------
        fig, ax:
            if either are None, will generate new figure.
        ...

        -----------------------------------------------------------------------
        """

        # init
        delta_height = self.delta_height
        if norm == 'default':
            scale = np.percentile(np.abs(delta_height), (1.-2**(-12))*100.)
            norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)

        # plot things
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cax  = ax.imshow(self.delta_height, norm=norm, cmap=cmap, **kwargs)
        if add_cbar:
            cbar = fig.colorbar(cax)
            cbar.set_label("$\\Delta h$")
        ax.set_title("Height Map changes from erosion / deposition")

        # update tick labels
        tick_locs = tuple([
            np.linspace(0, self.npix_xy[i], 9, dtype=np.int64)
            for i in range(2)
        ])
        tick_vals = tuple([
            (0.5 - tick_locs[0] / self.npix_xy[0]      ) * self._map_widxy[0],
            (      tick_locs[1] / self.npix_xy[1] - 0.5) * self._map_widxy[1],
        ])
        tick_labels = tuple([
            [f'{tick_val:.0f}' for tick_val in tick_vals[i]]
            for i in range(2)
        ])
        tick_labels[0][ 0] = f"NW\n{tick_labels[0][ 0]}"
        tick_labels[0][-1] = f"{    tick_labels[0][-1]}\n\n\nSW     "
        tick_labels[1][-1] = f"{    tick_labels[1][-1]}\nSE"
        ax.set_yticks(tick_locs[0])
        ax.set_yticklabels(tick_labels[0])
        ax.set_xticks(tick_locs[1])
        ax.set_xticklabels(tick_labels[1])

        return fig, ax



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
        self.__i_step  = 0
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
        capa_fac = self.get_par('capa_fac')
        capa_fac_v = self.get_par('capa_fac_v')
        capa_fac_slope = self.get_par('capa_fac_slope')
        capa_fac = capa_fac * (
            capa_fac_v / (1+capa_fac_v)
        ) * (
            capa_fac_slope / (1+capa_fac_slope)
        )
        self.stats['aqua'] = hs / (1.+capa_fac)
        self.stats['sedi'] = hs * capa_fac/(1.+capa_fac)
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
        steps_config: None|int|list[dict|int] = None,
        sub_func: Callable = _erode_rainfall_evolve_default,
        verbose: VerboseType = True,
    ) -> Self:
        """Wrapper func for rainfall erosion.
        
        Parameters used see self.pars
        Use self.set_par() to change them.

        Parameters
        ----------
        steps_config:
            If None: will auto-run n_step as per self.pars settings;
            If int : will run this number of steps;
            If list of dict: will replace self.pars with the settings in the
                given dict and run

        Returns
        -------
        self

        use self.delta_height to get the results for updating hmaps
        
        -----------------------------------------------------------------------
        """
        
        # - time it -
        runtime_t0 = now()
        if verbose:
            print(f"Time: ErosionState.evolve() Starting: {runtime_t0}")
        
        # - run -
        self.stats, d_n_step = sub_func(
            steps_config, self.stats, self.edges,
            z_max=self.z_range, z_res=self.z_res, pix_widxy = self.pix_widxy,
            pars = self.__pars, verbose=verbose)
        self.__i_cycle += 1
        self.__i_step += d_n_step

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
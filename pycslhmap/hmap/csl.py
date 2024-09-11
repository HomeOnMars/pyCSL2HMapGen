#!/usr/bin/env python
# coding: utf-8

"""A class to handle Cities: Skylines 2 height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (built-in)
from typing import Self

# imports (3rd party)
import numpy as np
from numpy import typing as npt
import matplotlib as mpl

# imports (my libs)
from ..util import VerboseType
from .base import HMap
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)


# constants
NPIX_CSL2   : int = 4096



#-----------------------------------------------------------------------------#
#    Functions
#-----------------------------------------------------------------------------#


def get_csl2_map_width(map_type: str) -> float:
    """Return map width in meters."""
    if   map_type == 'worldmap':
        return 57344.
    elif map_type == 'playable':
        return 14336.
    else:
        raise ValueError(
            f"Unrecognized {map_type=}:"
            + "Please set it to either 'worldmap' or 'playable'.")



#-----------------------------------------------------------------------------#
#    Classes
#-----------------------------------------------------------------------------#


class CSL2HMap(HMap):
    """CitiesSkylines2-specific Height Map object.

    Assuming square-shaped data.


    Extra Instance Variables
    ------------------------
    
    Public:

    map_type : {'worldmap', 'playable'}
        type of the map.
        
    map_name : str
        name of the map. For saving purposes.

    npix : int
        no of pixel in x and y dimensions.

    npix_4 : int
        == npix / 4
        
    npix_8 : int
        == npix / 8


    Private:

    _map_type : str
    
    ---------------------------------------------------------------------------
    """

    

    #-------------------------------------------------------------------------#
    #    Meta: Initialize
    #-------------------------------------------------------------------------#


    def __init__(
        self,
        data : Self|HMap|npt.ArrayLike = np.zeros(
            (NPIX_CSL2, NPIX_CSL2), dtype=np.float32),
        map_type: str = 'playable', # 'worldmap' or 'playable'
        map_name: str = '',
        z_config: tuple[float, float, float, float] = np.array(
            [64., 128., 4096., 2**(np.log2(4096)-23)], dtype=np.float32),
        copy : bool = True,
        use_data_meta: bool  = True,
        verbose: VerboseType = False,
    ):
        """Init.

        ...
        use_data_meta : bool
            If true and data is of type Self or HMap,
                will copy the metadata in it
                instead of the supplied parameters.
        ...
        -----------------------------------------------------------------------
        """

        # init
        
        if isinstance(data, HMap):
            if use_data_meta:
                z_config = data.z_config
                if isinstance(data, CSL2HMap):
                    map_type = data._map_type
                    map_name = data.map_name
            data = data.data
        
        
        # variables

        self._map_type: str   = map_type
        self.map_name : str   = map_name
        
        
        # do things
        
        map_width = get_csl2_map_width(map_type)
        
        super().__init__(
            data,
            map_width = map_width,
            z_config = z_config,
            copy = copy,
            verbose = False,
        )
        
        self.normalize(overwrite=False, verbose=verbose)


    
    @property
    def npix(self) -> int:
        """Number of pixels in x and y dimensions. Should be 4096."""
        return self.npix_xy[0]

    @property
    def npix_4(self) -> int:
        """1/4th of the number of pixels per dim. Should be 1024."""
        return int(self.npix / 4)

    @property
    def npix_8(self) -> int:
        """1/8th of the number of pixels per dim. Should be 512."""
        return int(self.npix / 8)

    @property
    def map_type(self) -> str:
        """type of the map. In {'worldmap', 'playable'}."""
        return self._map_type


    
    def normalize(
        self, overwrite: bool = False, verbose: VerboseType = True,
    ) -> Self:
        """Resetting parameters and do safety checks."""

        super().normalize(overwrite=overwrite, verbose=verbose)
            
        # safety checks
        assert self.ndim == 2
        assert self.npix_xy[0] == self.npix_xy[1]
        assert self.npix == self.npix_xy[0]
        assert self.npix == self.npix_8 * 8
        assert self.npix == NPIX_CSL2
        
        return self



    #-------------------------------------------------------------------------#
    #    Meta: magic methods
    #-------------------------------------------------------------------------#


    def __repr__(self):
        return f"""CSL2 {self._map_type} {self.map_name} {super().__repr__()}
# CSL2-specific data
    Map type      : {self._map_type = }
    Map name      : {self.map_name  = }
    Height scale  : {self.z_max     = :.2f}
        """


    def __str__(self):
        return self.__repr__()

    

    #-------------------------------------------------------------------------#
    #    Meta
    #-------------------------------------------------------------------------#

    
    def copy(self) -> Self:
        return CSL2HMap(self)
        
    

    #-------------------------------------------------------------------------#
    #    I/O
    #-------------------------------------------------------------------------#


    def load(
        self,
        dir_path: str = '',
        map_name: str = '',
        map_type: str = 'playable',
        z_min : float = 64.,
        z_sea : float = 128.,
        z_max : float = 4096.,
        verbose: VerboseType = True,
        **kwargs,
    ) -> Self:
        """Loading a Cities Skylines 2 Height Map.
        
        Parameters
        ----------
        map_name : str
            hmap file has the name of
            f"{dir_path}{map_type}_{map_name}.png"

        map_type  : 'worldmap' or 'playable'
        
        dir_path  : str
            Input directory path (i.e. filepath prefix)
            e.g. './out/'

        ...
        """

        self._map_type = map_type
        self.map_name  = map_name
        self.z_max     = z_max

        map_width = get_csl2_map_width(map_type)
        filename = f"{dir_path}{map_type}_{map_name}.png"

        return self.load_png(
            filename  = filename,
            map_width = map_width,
            z_min = z_min,
            z_sea = z_sea,
            z_max = z_max,
            verbose = verbose,
            **kwargs,
        )



    def save(
        self,
        dir_path: str,
        map_name: None|str = None,
        map_type: None|str = None,
        z_max   : None|float = None,
        compression: int = 9,    # maximum compression
        verbose : VerboseType = True,
        **kwargs,
    ) -> Self:
        """Save to Cities Skylines 2 compatible Height Map.

        
        Parameters
        ----------
        map_name : None|str
            hmap file has the name of
            f"{dir_path}{map_type}_{map_name}.png"
            
        dir_path  : str
            Input directory path (i.e. filepath prefix)

        map_type  : None|{'worldmap', 'playable'}
        ...
        """

        if map_name is None: map_name = self.map_name
        if map_type is None: map_type = self._map_type
        if z_max    is None: z_max = self.z_max
        
        filename = f"{dir_path}{map_type}_{map_name}.png"
        
        return self.save_png(
            filename = filename,
            bit_depth = 16,
            z_max = z_max,
            compression  = compression,
            verbose = verbose,
            **kwargs,
        )


    
    def plot(
        self,
        **kwargs,
    ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Return a plot of the data.

        See HMap.plot() docs for details.
        """
        fig, ax = super().plot(**kwargs)
        ax.set_title(f"{self._map_type} {self.map_name} {ax.get_title()}")
        return fig, ax
        
    

    #-------------------------------------------------------------------------#
    #    Resampling
    #-------------------------------------------------------------------------#

    
    def extract_playable(self, **kwargs) -> Self:
        """Extract playable area from world map."""

        # safety check
        assert self._map_type == 'worldmap'

        ans = self.resample(
            nslim_in_ind=(3*self.npix_8, 5*self.npix_8),
            welim_in_ind=(3*self.npix_8, 5*self.npix_8),
            new_npix_xy=(NPIX_CSL2, NPIX_CSL2),  **kwargs,
        )
        
        return CSL2HMap(
            ans,
            map_type ='playable',
            map_name = self.map_name,
            z_config = self.z_config,
            use_data_meta= True,
        )



    def insert_playable(self, playable_hmap: Self, **kwargs) -> Self:
        """Insert playable area back into the worldmap.

        A copy will be made.
        """
        
        # safety check
        assert self._map_type == 'worldmap'

        res = playable_hmap.resample(
            nslim_in_ind=(0, self.npix),
            welim_in_ind=(0, self.npix),
            new_npix_xy=(self.npix_4, self.npix_4), **kwargs,
        )

        ans = self.copy()
        ans.data[
            3*self.npix_8 : 5*self.npix_8,
            3*self.npix_8 : 5*self.npix_8,
        ] = res.data
        
        return ans
        
    

#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
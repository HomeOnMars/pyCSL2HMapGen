#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HomeOnMars' python module for handling height maps.

Author: HomeOnMars

-------------------------------------------------------------------------------
"""


from .hmap import HMap, CSL2HMap
from .io import get_CSL_height_maps
from .util import now
from .util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)


# - Load Order -
#
# pycslhmap.util                  # Utilities and odds and ends.
# pycslhmap.hmap.util             # Utilities and odds and ends.
# pycslhmap.hmap.base             # A class to handle height maps.
# pycslhmap.hmap.csl              # A class to handle Cities: Skylines 2 height maps.
# pycslhmap.hmap                  # HomeOnMars' python class for height maps.
# pycslhmap.io.tiff               # A simple script to extract CSL2 height map from JAXA:AW3D30 data files.
# pycslhmap.io                    # HomeOnMars' python module for importing height maps.
# pycslhmap                       # HomeOnMars' python module for handling height maps.
#
# pycslhmap.erosion.defaults      # Height map erosion with GPU-accelerations.
# pycslhmap.erosion.cuda          # GPU-accelerated functions.
# pycslhmap.erosion.nbjit         # CPU version of .cuda codes. Incomplete. No longer supported.
# pycslhmap.erosion.state         # Height map erosion with GPU-accelerations.
# pycslhmap.erosion               # HomeOnMars' python module for doing erosion with height maps.

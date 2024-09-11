#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HomeOnMars' python module for importing height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""

from .tiff import get_CSL_height_maps
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)
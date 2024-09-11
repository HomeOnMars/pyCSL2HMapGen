#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HomeOnMars' python module for doing erosion with height maps.

Author: HomeOnMars
-------------------------------------------------------------------------------
"""

from .state import ErosionState
from .defaults import DEFAULT_PARS
from ..util import _LOAD_ORDER; _LOAD_ORDER._add(__spec__, __doc__)
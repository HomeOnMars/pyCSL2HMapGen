#!/usr/bin/env python
# coding: utf-8

"""Utilities and odds and ends.

Should be loaded before all other modules in this folder.


Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# imports (built-in)
from typing import Self
from datetime import datetime, UTC

# imports (3rd party)
import numpy as np
from numpy import typing as npt

# imports (my libs)



#-----------------------------------------------------------------------------#
#    Types and documentation
#-----------------------------------------------------------------------------#


# global variables
_LOAD_ORDER_LIST : list[dict[str, str]] = []


# - types -
VerboseType : type = bool


# - functions -
now = lambda: datetime.now(UTC)


def comment_docstring(docstring: str, leading_txt: str = '# '):
    """Trim and add leading_txt before each line of the doc string."""

    # Modified from the example doc string trimming function
    #    from https://peps.python.org/pep-0257/    (public domain)
    
    INDENT_MAX_SIZE : int = 255
    
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = INDENT_MAX_SIZE
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < INDENT_MAX_SIZE:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # add leading texts
    trimmed = [leading_txt + line_new for line_new in trimmed]
    # Return a single string:
    return '\n'.join(trimmed)



def _not_implemented_func(*args, **kwargs):
    """A placeholder function without implementation."""
    raise NotImplementedError



class _LoadOrder:
    """A class for recording load order of python scripts.

    For internal uses only.
    """
    def __init__(self):
        self.__data : list[dict[str, str]] = []

    @property
    def data(self):
        return self.__data

    def _add(self, spec: type(__spec__), doc: str) -> Self:
        """Add the current module to the list.

        Run this once after the imports in each modules.
        """
        try:
            self.__data.append({
                'name': spec.name,
                'doc' : comment_docstring(doc),
            })
        except AttributeError as e:
            print(f"*   Error when determining metadata:\n\t{e}")
            
        return self

    def __repr__(self):
        return f"{self.__data}"

    def __str__(self):
        lines = [f"{it['name']:30}  {it['doc'].splitlines()[0]}" for it in self.__data]
        return '\n'.join(lines)


_LOAD_ORDER = _LoadOrder()._add(__spec__, __doc__)



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
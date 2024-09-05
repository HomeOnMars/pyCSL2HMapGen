#!/usr/bin/env python
# coding: utf-8

"""Utilities and odds and ends.

Should be loaded before all other modules in this folder.


Author: HomeOnMars
-------------------------------------------------------------------------------
"""


# Dependencies
import numpy as np
from numpy import typing as npt
from datetime import datetime, UTC



#-----------------------------------------------------------------------------#
#    Types and documentation
#-----------------------------------------------------------------------------#


now = lambda: datetime.now(UTC)


VerboseType = bool



def comment_docstring(docstring: str, leading_txt: str = '# '):
    """Trim and add leading_txt before each line of the doc string."""

    # Modified from the example doc string trimming function
    #    from https://peps.python.org/pep-0257/
    
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
    trimmed = [leading_txt + lines[0].strip()]
    if indent < INDENT_MAX_SIZE:
        for line in lines[1:]:
            trimmed.append(leading_txt + line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)



#-----------------------------------------------------------------------------#
#    End
#-----------------------------------------------------------------------------#
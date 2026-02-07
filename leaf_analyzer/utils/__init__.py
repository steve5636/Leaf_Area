#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility modules for Advanced Leaf Analyzer
"""

# Color analysis utilities (optional)
try:
    from .color_analysis_utils import DeltaE2000, CircularStatistics, RobustStatistics
    COLOR_UTILS_AVAILABLE = True
except ImportError:
    COLOR_UTILS_AVAILABLE = False
    DeltaE2000 = None
    CircularStatistics = None
    RobustStatistics = None

__all__ = [
    'DeltaE2000', 'CircularStatistics', 'RobustStatistics',
    'COLOR_UTILS_AVAILABLE'
]

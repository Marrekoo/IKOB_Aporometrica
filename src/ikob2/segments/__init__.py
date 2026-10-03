"""
Household-type x income-decile segment shares per CBS buurt.

A GSPREE-style method: a Poisson structure model fitted on municipality-level cross-tabs
(CBS 86161NED) supplies the household-type x income *association*;
each buurt's predicted table is then raked (IPF) to that buurt's own
household-type and income marginals derived from Kerncijfers wijken
en buurten (KWB). Output: 4 household types x 11 income classes
(D1..D10 + 'onbekend') = 44 segment columns per buurt.
"""

from ikob2.segments.config import SegmentConfig
from ikob2.segments.pipeline import SegmentResult, run_pipeline

__all__ = ["SegmentConfig", "SegmentResult", "run_pipeline"]

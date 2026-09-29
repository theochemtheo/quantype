"""Portable static contract for an application-generated catalogue."""

from typing import assert_type

from labquantities import Length, Pressure, SurfaceTension, u

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)

assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])

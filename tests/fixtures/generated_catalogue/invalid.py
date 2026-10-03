"""Every marked application-catalogue expression must be rejected."""

from labquantities import Length, Pressure, u

from quantype.systems import SI

point = 300 * u.K
total = point.sum()  # error
doubled = point + point  # error
mixed_systems = Length[float, SI](1, u.nm) * Pressure[float](1, u.pascal)  # error
wrong_named_unit = Length[float](1, Pressure.unit_named("pascal"))  # error

"""Every marked application-catalogue expression must be rejected."""

from labquantities import Length, Pressure, u

from quantype.systems import SI, Atomistic

point = 300 * u.K
total = point.sum()  # error
doubled = point + point  # error
pressure = Pressure[float, Atomistic](1, u.pascal)
mixed_systems = Length[float, SI](1, u.nm) * pressure  # error
wrong_named_unit = Length[float](1, Pressure.unit_named("pascal"))  # error

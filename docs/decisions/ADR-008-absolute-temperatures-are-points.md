# Absolute temperatures are points on a kelvin scale

## Context and Problem Statement

Celsius has an offset: 20 °C plus 20 °C has no physical meaning, while 30 °C
minus 20 °C is a difference of 10 K. Formulas still need `k_B * T`, `T / T0`,
and the mean of a set of temperatures. Which operations are allowed on an
absolute `Temperature`, and how do the static types and the runtime agree on
them?

## Considered Options

* Treat `Temperature` as any other kind
* Allow only differences of absolute temperatures
* Points and differences: a temperature minus a temperature is a
  `TemperatureDifference`; a difference shifts a temperature; two temperatures
  can't be added or summed; means, products, ratios, powers, and scaling are
  allowed

## Decision Outcome

Chosen option: "points and differences", because it rejects the additions that
depend on the unit's offset and keeps every formula that is valid on a kelvin
scale. Every unit system stores temperature in kelvin, so products and scaling
mean the same in all of them.

### Consequences

* Good, because `T + T` and `sum()` of temperatures are type errors and raise
  `TypeError`, while `k_B * T` is an `Energy`.
* Good, because a spread or difference of temperatures (`std`, `diff`, `var`) is
  a `TemperatureDifference` or its square.
* Bad, because scaling drops an offset display unit: twice 20 °C shows 586.3 K.
* Bad, because rejecting `sum()` and `0 - T` statically types
  `Temperature.__radd__` and `__rsub__` with `Never`, which Pyrefly and ty
  report as a narrowed override and the stubs suppress inline.

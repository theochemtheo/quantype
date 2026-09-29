"""Investigation fixture, not a passing contract: left fallback hides __rmul__."""

from typing import assert_type

from quantype import Force, Length, Quantity


class CustomKind:
    pass


class Custom(Quantity[CustomKind, float]):
    def __mul__(self, other: Length[float]) -> Force[float]:
        raise NotImplementedError("Typing-only probe")

    def __rmul__(self, other: Length[float]) -> Force[float]:
        raise NotImplementedError("Typing-only probe")


def probe(custom: Custom, length: Length[float]) -> None:
    assert_type(custom * length, Force[float])
    assert_type(length * custom, Force[float])

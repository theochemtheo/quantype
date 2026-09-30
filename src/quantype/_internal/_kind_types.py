"""Shared structural markers and the non-affine bound used by public stubs.

Application catalogues keep their named kinds nominally distinct, but use the
same structural markers so the handwritten Quantity API applies to their trees.
"""


class NonAffineKind:
    """A kind that may participate in products, ratios, and scalar scaling."""


class Mul[A, B](NonAffineKind):
    pass


class Div[A, B](NonAffineKind):
    pass


class Pow[A, N](NonAffineKind):
    pass

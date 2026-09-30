"""Shared structural markers used by public stubs.

Application catalogues keep their named kinds nominally distinct, but use the
same structural markers so the handwritten Quantity API applies to their trees.
"""


class Mul[A, B]:
    pass


class Div[A, B]:
    pass


class Pow[A, N]:
    pass

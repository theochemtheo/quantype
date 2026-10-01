"""quantype.testing accepts quantities of one kind, product classes included."""

import numpy as np

from quantype import Length, u
from quantype.testing import assert_allclose

assert_allclose(2 * u.nm, 20 * u.angstrom)
assert_allclose((2 * u.nm) * (3 * u.fs), (3 * u.fs) * (2 * u.nm))
assert_allclose(300 * u.K, 300.05 * u.K, atol=0.1 * u.delta_K)
assert_allclose(Length[float](2, u.nm), u.nm(np.float64(2)), rtol=1e-6, atol=0)

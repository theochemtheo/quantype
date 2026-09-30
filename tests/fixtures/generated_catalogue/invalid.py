"""Every marked application-catalogue expression must be rejected."""

from labquantities import u

point = 300 * u.K
scaled = point * 2  # error
reverse_scaled = 2 * point  # error
ratio = point / point  # error
inverse = 1.0 / point  # error
power = point**2  # error
total = point.sum()  # error
negative = -point  # error
absolute = abs(point)  # error
right_product = (2 * u.angstrom) * point  # error
right_ratio = (2 * u.angstrom) / point  # error
structural = (2 * u.angstrom) * (3 * u.fs)
structural_product = structural * point  # error

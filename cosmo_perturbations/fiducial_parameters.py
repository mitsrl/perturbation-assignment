"""Fiducial cosmological parameters."""

import numpy as np


#### Fundamental Parameters ####

# From D&S D.3
omega_b_h2 = 0.022447
omega_c_h2 = 0.11928
tau_rei = 0.0568
h = 0.6770
ns = 0.9682
logAs = 3.0482 - np.log(10**10)

# D&S 2.76
omega_gamma_h2 = 2.47e-5
# D&S 2.85
omega_rad_h2 = 4.15e-5


#### Derived Parameters ####
omega_m_h2 = omega_b_h2 + omega_c_h2
omega_nu_h2_massless = omega_rad_h2 - omega_gamma_h2
omega_lambda_h2 = h**2 - omega_m_h2 - omega_rad_h2



#### unit conversions (constants, not parameters) ####
hc_over_H0 = 2997.92458    # Hubble distace in Mpc/h or hc/H0 in Mpc.

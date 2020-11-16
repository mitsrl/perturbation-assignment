"""Linear cosmological perturbation theory equation solvers."""

import numpy as np
from scipy import optimize, linalg

from cosmo_perturbations import fiducial_parameters


class MatterRadiationSolver:
    """Solve for the evolution of the cosmological perturbation fields, including only matter,
    radiation, and gravity.

    Fields are packed into arrays in the following order:
    [0] Phi - the metric perturbation
    [1] delta - the dark matter perturbation
    [2] uj - the dark matter velocity time j (unit imaginary)
    [3] Theta_0 - the photon monopole
    [4] Theta_1 - the photon dipole

    """

    def __init__(self,
            h=fiducial_parameters.h,
            omega_m_h2=fiducial_parameters.omega_m_h2,
            omega_rad_h2=fiducial_parameters.omega_rad_h2,
            num_k = 200,
            ):

        self._h = h
        self._omega_rad_h2 = omega_rad_h2
        self._omega_m_h2 = omega_m_h2
        self._omega_lam_h2 = h**2 - omega_m_h2 - omega_rad_h2

        self._lna = np.log(np.logspace(-7, 0, 10000))
        k = np.logspace(-5, 0, num_k) * h
        self._k_tilde = k * fiducial_parameters.hc_over_H0

    @property
    def k_tilde(self):
        return self._k_tilde.copy()

    @property
    def k(self):
        """Wave number k in units 1/Mpc."""
        return self.k_tilde / fiducial_parameters.hc_over_H0

    @property
    def k_over_h(self):
        """Wave number/h in 1/Mpc. Same as wavenumber in units of h/Mpc."""
        return self.k / self._h

    @property
    def a(self):
        """Scale factors at which solution is evaluated and stored."""
        return np.exp(self._lna)

    def H_tilde(self, lna):
        a = np.exp(lna)
        return a * np.sqrt(self._omega_rad_h2 / a**4
                + self._omega_m_h2 / a**3 + self._omega_lam_h2)

    def H_over_c(self, lna):
        """Hubble parameter over c. In 1/Mpc."""
        a = np.exp(lna)
        return self.H_tilde(lna) / a / fiducial_parameters.hc_over_H0

    def field_ICs(self):
        raise NotImplementedError()
        return np.empty(5)

    def field_derivatives(self, field_values, lna, k_tilde):
        raise NotImplementedError()
        return np.empty(5)


    def solve(self):
        # Finds the solution at each k and stores it.
        raise NotImplementedError()
        self._solution = np.empty((len(self.k)), len(self.field_ICs), len(self.a))


def radiation_turnoff(lna):
    """Function used to smoothly "turn off" radiation evolution at z=500.

    Returns 1 for z >> 500, 0 for z << 500.

    """

    LN_A_STOP_RAD = np.log(1. / 500)
    LN_A_STOP_WID = 0.1
    return 1. - 1. / (1 + np.exp(-(lna - LN_A_STOP_RAD) / LN_A_STOP_WID))


def electron_decoupling(lna):
    """Function used to decouple electrons from photons at recombination.

    Returns 1 for z >> 1100, 0 for z << 1100.

    """
    a = np.exp(lna)
    A_RECOMB = 1. / 1100
    A_RECOMB_WID = 75. * A_RECOMB**2
    return 1. - 1. / (1 + np.exp(-(a - A_RECOMB) / A_RECOMB_WID))


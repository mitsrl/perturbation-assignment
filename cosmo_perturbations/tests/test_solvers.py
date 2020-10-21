"""Unit tests for solvers.py."""

import pytest
import numpy as np

from cosmo_perturbations import solvers, fiducial_parameters

def test_H0():
    solver = solvers.MatterRadiationSolver()
    assert solver.H_over_c(0.) == pytest.approx(fiducial_parameters.h
                                                / fiducial_parameters.hc_over_H0)
    new_h = 0.8
    solver = solvers.MatterRadiationSolver(h=new_h)
    assert solver.H_over_c(0.) == pytest.approx(new_h
                                                / fiducial_parameters.hc_over_H0)

def test_k_tilde():
    h = 0.5
    solver = solvers.MatterRadiationSolver(num_k=1000, h=h)
    # Make sure we get today's horizon scale correct.
    ktilde1_ind = np.argmin(np.abs(solver.k_tilde - h))
    k_over_h = solver.k_over_h[ktilde1_ind]
    assert k_over_h == pytest.approx(1 / fiducial_parameters.hc_over_H0, rel=0.01)

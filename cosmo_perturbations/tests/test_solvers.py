"""Unit tests for solvers.py."""

import pytest

from cosmo_perturbations import solvers, fiducial_parameters

def test_H0():
    solver = solvers.MatterRadiationSolver()
    assert solver.H_over_c(0.) == pytest.approx(fiducial_parameters.h
                                                / fiducial_parameters.hc_over_H0)
    new_h = 0.8
    solver = solvers.MatterRadiationSolver(h=new_h)
    assert solver.H_over_c(0.) == pytest.approx(new_h
                                                / fiducial_parameters.hc_over_H0)

import pytest
from drl.utilities import rk_step, function_derivatives, finite_difference

def test_rk_step():
    x = lambda t, x, u: x^2 * t + x + u * t
    xdot = lambda x, u: x^2 + u

    x0 = x()
    u = 1.0




    pytest.fail()

def test_function_derivatives():
    pytest.fail()

def test_finite_difference():
    pytest.fail()
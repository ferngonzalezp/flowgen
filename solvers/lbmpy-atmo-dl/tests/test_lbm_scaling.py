import pytest
from atmo import StableScaling


def test_stable_scaling():
    sc = StableScaling(dx=0.7, u_max=5, u_bulk_target=2)
    assert pytest.approx(sc.dt) == 0.007  # Timestep
    assert pytest.approx(sc.ν_from_dx()) == 0.0584795321637414  # Lowest stable ν
    assert pytest.approx(sc.Mach) == 0.014705882352941176  # Max Mach Number
    assert pytest.approx(sc.Reynolds(20)) == 684.0  # Re number of a 20 m object

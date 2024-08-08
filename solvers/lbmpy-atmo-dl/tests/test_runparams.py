import lbmpy as lp
from atmo import RunParams, StableScaling


def basic_runparams():
    dx = 1
    domain = (400.0, 150.0, 160.0)
    params = RunParams(
        ## - General physical parameters - ##
        scaling=StableScaling(
            dx=dx,  # [m]
            u_max=5,  # Expected max velocity [m/s]
            u_bulk_target=2,  # Target bulk velocity [m/s]
            ν_target=1.48e-5,  # Laminar viscosity [m^2/s]
        ),
        domain_size=domain,  # TODO: domain_size_x_y_z
        method=lp.Method.CUMULANT,
        compressible=True,
        smagorinsky=False,
        output_freq=10000,  # Frequency of output solutions [iterations]
        monitor_freq=50,  # Frequency of monitoring [iterations]
        # restart=Path(""),
    )
    return params


def test_create_runparams():
    basic_runparams()


def test_runparams_serialize():
    params = basic_runparams()
    params.serialize()


def test_runparams_deserialize():
    params = basic_runparams()
    bytes = params.serialize()
    deser = RunParams.deserialize(bytes)
    assert deser.scaling.dx == 1


def test_runparams_save(tmp_path):
    params = basic_runparams()
    params.save(tmp_path / "test_save.pkl")


def test_runparams_load(tmp_path):
    params = basic_runparams()
    params.save(tmp_path / "test_load.pkl")
    load = RunParams.load(tmp_path / "test_load.pkl")
    assert load.scaling.dx == 1

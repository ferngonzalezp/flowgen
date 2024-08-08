"""urban_surface.py

Example script to use lbmpy-atmo for a urban setting with a surface file
decribing arbitrarily complex surfaces.
"""
import lbmpy as lp
import numpy as np
import scipy as sc
from lbmpy.boundaries import NoSlip

from atmo import RunParams, StableScaling
from atmo.runners import AtmosphericPeriodic

# -----------------------------------------------------------------------------
# - Basic parameters - #
domain = (400.0, 150.0, 160.0)
# _domain_g = [d + 2 for d in domain]
direction = (1.0, 0.0, 0.0)
# angle = 5 * np.pi / 180  # Radians
# direction = (np.cos(angle), 0, np.sin(angle))
dx = 0.7


# -----------------------------------------------------------------------------
# - Parameter file - #
params = RunParams(
    ## - General physical parameters - ##
    scaling=StableScaling(
        dx=dx,  # [m]
        u_max=5,  # Expected max velocity [m/s]
        u_bulk_target=2,  # Target bulk velocity [m/s]
        ν_target=1.48e-5,  # Laminar viscosity [m^2/s]
    ),
    domain_size=domain,  # TODO: domain_size_x_y_z
    method="CUMULANT",
    compressible=True,
    smagorinsky=False,
    output_freq=10000,  # Frequency of output solutions [iterations]
    monitor_freq=50,  # Frequency of monitoring [iterations]
    # continue_previous=True,
)


# -----------------------------------------------------------------------------
# - Runtime - #
params.final_iteration = params.scaling.iterations_from_t(
    5000.0
)  # Final iteration of simulation [iterations]


# -----------------------------------------------------------------------------
# - Probes - #
domx, domy, domz = domain
probes = [[domx / 2, domy, domz / 2]]
centerline = [[domx / 8 * i, domy / 2, domz / 2] for i in range(8)]
probes += centerline

params.probes = probes


# -----------------------------------------------------------------------------
# - Initial field - #
# Add a bit of noise to trigger turbulence
# params.init_ul = 0.001 * np.random.random_sample(params.domain_cells + (3,))
# params.write_init_file = True
# TODO init_u ?


# -----------------------------------------------------------------------------
# - Forcing - #
def force(u_bulk):
    error = params.scaling.u_bulk_target - u_bulk
    return dict(zip("fx fy fz".split(), 1e-4 * error * np.array(direction)))


# Constant force:
# def force(*args):
#     return {'fx': 0, 'fy': 0, 'fz': 0}

# Add force to the exiting params instance
params.force = force


# -----------------------------------------------------------------------------
# - Boundaries - #
# Boundary built from the data relesed in M. Aghaei's repo on
# [github](https://github.com/MostafaAghaei/Prediction-of-the-roughness-equivalent-sandgrain-height/tree/master/3)
delta = 400 / 3
bnd_data = (
    np.loadtxt("./AGHAEI/chn_r6_Re1000_rnd_inc3/Surface.txt")[:, -1].reshape(400, 160)
) * delta
xrng = np.arange(400)
zrng = np.arange(160)
bnd_func = sc.interpolate.RegularGridInterpolator(
    (xrng, zrng), bnd_data, bounds_error=False
)


def surface(x, y, z):
    return y <= bnd_func((x * dx, z * dx))


def obstacles(bh):
    bh.set_boundary(
        NoSlip("Surface"),
        mask_callback=surface,
    )


params.obstacles = obstacles


# -----------------------------------------------------------------------------
# - Run - #
params.info()
ap = AtmosphericPeriodic(params)
ap.run()

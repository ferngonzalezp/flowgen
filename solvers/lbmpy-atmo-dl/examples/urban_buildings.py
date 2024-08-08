"""urban_buildings.py

Example script to use lbmpy-atmo for a urban setting with buildings
prescribed with simple parameters (see `create_building` function).
"""
import lbmpy as lp
import numpy as np
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
def create_building(x_center, z_center, dx, dy, dz):
    """Create callback function to mask a brick-shaped structure"""
    return (
        lambda x, y, z: (abs(x - x_center) < dx // 2)
        & (y < dy)
        & (abs(z - z_center) < dz // 2)
    )


def obstacles(bh):
    nx, _, nz = params.domain_cells
    bh.set_boundary(
        NoSlip("obstacle"),
        mask_callback=create_building(nx // 3, nz // 3, 6, 40, 40),
    )
    bh.set_boundary(
        NoSlip("obstacle"),
        mask_callback=create_building(2 * nx // 3, nz // 3, 4, 20, 40),
    )
    bh.set_boundary(
        NoSlip("obstacle"),
        mask_callback=create_building(nx // 3, 2 * nz // 3, 6, 40, 40),
    )
    bh.set_boundary(
        NoSlip("obstacle"),
        mask_callback=create_building(2 * nx // 3, 2 * nz // 3, 4, 40, 40),
    )


params.obstacles = obstacles


# -----------------------------------------------------------------------------
# - Run - #
params.info()
ap = AtmosphericPeriodic(params)
ap.run()

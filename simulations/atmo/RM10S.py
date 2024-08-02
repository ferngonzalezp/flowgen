"""RM10S.py

Script to simulation the RM10S configuration from (Cheng & Castro 2002)
using LBMpy-atmo.
"""
import lbmpy as lp
import numpy as np
import scipy.stats as stats
from lbmpy.boundaries import NoSlip

from atmo import RunParams, StableScaling
from atmo.runners import AtmosphericPeriodic

# -----------------------------------------------------------------------------
# - Domain and forcing related parameters - #
H = 10.0e-3 # mean heigth of the buildings (m)
continue_previous = True

x_factor, y_factor, z_factor  = 8, 8, 8 # domain size factors
domain = (x_factor*H, y_factor*H, z_factor*H)

direction = (1.0, 0.0, 0.0) # direction of the forcing 
# angle = 5 * np.pi / 180  # Radians
# direction = (np.cos(angle), 0, np.sin(angle))

Forcing_phy = 8.72 # Pressure gradient value (Re_tau = 500, u_tau = 0.835)

# -----------------------------------------------------------------------------
# - Parameter file - #
params = RunParams(
    ## - General physical parameters - ##
    scaling=StableScaling(
        dx=H/16,  # spatial step [m]
        u_max=6,  # Expected max velocity [m/s]
        u_bulk_target=5.95,  # Target bulk velocity [m/s]
        ν_target=1.48e-5,  # Laminar viscosity [m^2/s]
    ),
    domain_size=domain,
    method="CUMULANT",
    compressible=True,
    smagorinsky=False,
    write_on_disk=True,
    output_freq=500,  # Frequency of output solutions [iterations]
    monitor_freq=500,  # Frequency of monitoring [iterations]
    mean = True,
    rms = True,
    avg_every_sol = True,
    continue_previous=True,
    overwrite=False,
    stream = False,
    adios2_cfg = "adios2.xml",
)


# -----------------------------------------------------------------------------
# - Runtime - #
tc = 0.3 # Characteristic time (sqrt(Lz/u_tau))
params.final_iteration = params.scaling.iterations_from_t(
    2 * tc
)  # Final iteration of simulation [iterations], minimum 200*tc to be converged

# -----------------------------------------------------------------------------
# - Probes - #
#domx, domy, domz = domain
#probes = [[domx / 2, domy, domz / 2]]
#centerline = [[domx / 8 * i, domy / 2, domz / 2] for i in range(8)]
#probes += centerline

#params.probes = probes


# -----------------------------------------------------------------------------
# # - Initial field - #
# # Add a bit of noise to trigger turbulence
# params.init_ul = 0.08 * np.random.random_sample(params.domain_cells + (3,))
# #params.write_init_file = True
# # TODO init_u ?


# -----------------------------------------------------------------------------
# - Forcing - #
Cl, Crho, Ct = params.scaling.dx, 1.205, params.scaling.dt #LBM conversion factors (space, density and time)
Cf = Cl*Crho*Ct**(-2) # Forcing conversion factor
Forcing_lat = Forcing_phy/Cf


def force(u_bulk):
    error = params.scaling.u_bulk_target - u_bulk
    return dict(zip("fx fy fz".split(), Forcing_lat * np.array(direction)))

# Add force to the exiting params instance
params.force = force

# -----------------------------------------------------------------------------
# - Boundaries - #
def create_building(x_center, z_center,  dx, dy, dz):
    """Create callback function to mask a brick-shaped structure"""
    return (
        lambda x, y, z: (abs(x - x_center) <  dx // 2)
        & (y < dy)
        & (abs(z - z_center) < dz // 2)
    )

# Buildings definition
def obstacles(bh, Cubes_param=None):
        nx, _, nz = params.domain_cells
        Hlat = nx/x_factor
        dist = stats.truncnorm(H/5, H*3, loc = H, scale = H*7,)
        # x_center, z_center,  dx, dy, dz
        if not Cubes_param:
            Cubes_param = [[0,      Hlat,   Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [0,      3*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat], 
                        [0,      5*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [0,      7*Hlat, Hlat, dist.rvs()/ params.scaling.dx,  Hlat], # row 2
                        [2*Hlat, 0,      Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [2*Hlat, 2*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat], 
                        [2*Hlat, 4*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [2*Hlat, 6*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [2*Hlat, 8*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat], # row 1
                        [4*Hlat, Hlat,   Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [4*Hlat, 3*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat], 
                        [4*Hlat, 5*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [4*Hlat, 7*Hlat, Hlat, dist.rvs()/ params.scaling.dx,  Hlat], # row 2
                        [6*Hlat, 0,      Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [6*Hlat, 2*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [6*Hlat, 4*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [6*Hlat, 6*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat],
                        [6*Hlat, 8*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat], # row 3
                        [8*Hlat, Hlat,   Hlat, dist.rvs()/ params.scaling.dx, Hlat], 
                        [8*Hlat, 3*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat], 
                        [8*Hlat, 5*Hlat, Hlat, dist.rvs()/ params.scaling.dx, Hlat], 
                        [8*Hlat, 7*Hlat, Hlat, dist.rvs()/ params.scaling.dx,  Hlat], # row 4
                        ]

        for cubes in Cubes_param:
            bh.set_boundary(
                NoSlip("obstacle"),
            mask_callback=create_building(cubes[0], cubes[1], cubes[2], cubes[3], cubes[4]),
            )
        
        return Cubes_param

params.obstacles = obstacles


# -----------------------------------------------------------------------------
# - Run - #
params.info()
ap = AtmosphericPeriodic(params)
ap.run()

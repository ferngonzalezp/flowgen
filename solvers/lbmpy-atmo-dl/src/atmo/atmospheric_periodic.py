"""atmospheric_periodic.py

Created 2023-06-15 by C. Lapeyre (lapeyre@cerfacs.fr)

Wrapper for lbmpy in the specific case of atmospheric bi-periodic flows.
"""
# coding: utf-8

# import time
import os
import sys
from pathlib import Path
from time import time

import __main__
import lbmpy as lp
import numpy as np
import pystencils as ps
import sympy as sp
from lbmpy.boundaries import FreeSlip, LatticeBoltzmannBoundaryHandling, NoSlip
from loguru import logger
from pystencils.slicing import slice_from_direction
from tqdm import trange
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, ANIMALS, COLORS

from .run_params import RunParams
from .cupy_tools import setup_gpu, gpu_sync, gpu_memory_stats, gpu_memory_free
import cupy
from .atmo_solution import AtmoSolution
from .run_dir_tools import RUN_LOG_FILE, JOB_LOG_FILE, find_latest_job_directory
import adios2.bindings as adios2
from mpi4py import MPI
import pickle

# Phydll library
import sys, os
#sys.path.append("../phydll/src/python/")
#from pyphydll.pyphydll import PhyDLL


class AtmosphericPeriodic:
    SOURCE = "src"
    DESTIN = "dst"
    VELOCITY = "velField"

    def __init__(self, run_params: RunParams):
        logger.info("New AtmosphericPeriodic run")
        self.prm = run_params
        self.scl = run_params.scaling
        self.previous = None

        logger.add(
            os.environ("LOG_FILE") if "LOG_FILE" in os.environ else RUN_LOG_FILE,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level="DEBUG",
            backtrace=True,
            diagnose=True,
        )

        setup_gpu()
        logger.debug(gpu_memory_stats())

    @property
    def velocity(self):
        return self.dh.gpu_arrays[self.VELOCITY]

    @property
    def geometry(self):
        return self.dh.cpu_arrays["bhFlags"]

    @logger.catch
    def _run_init(self):
        """Initialize the simulation using self.prm RunParameters instance"""
        self.run_folder = Path.cwd()
        self.run_name = get_random_name(
            separator="_", style="lowercase", combo=[ADJECTIVES, COLORS, ANIMALS]
        )
        logger.info(
            f"New run will be called {self.run_name} and stored in {self.run_folder}"
        )
        logger.debug(gpu_memory_stats())

        self._init_data_handling()
        self._init_lbm()
        self._init_boundary()

        #if self.prm.use_phydll:
        #    self._init_phydll()

        self._setup_outputs()
        if self.prm.stream:
            self._init_adios2()
        
        self._init_fields()        
        if self.prm.write_init_file:
            self.write_solution()

    def _init_data_handling(self):
        """Initialize stencil, DataHandling, fields"""
        t = time()
        logger.info("Initializing stencil, DataHandling, fields")
        self.stencil = lp.LBStencil(getattr(lp.Stencil, self.prm.stencil))

        dh = ps.create_data_handling(
            domain_size=self.prm.domain_cells,
            periodicity=self.prm.periodicity,
            default_target=ps.Target.GPU,
        )
        self._src = dh.add_array(self.SOURCE, values_per_cell=len(self.stencil))
        dh.fill(self.SOURCE, 0.0, ghost_layers=True)
        self._dst = dh.add_array(self.DESTIN, values_per_cell=len(self.stencil))
        dh.fill(self.DESTIN, 0.0, ghost_layers=True)
        self._velField = dh.add_array(self.VELOCITY, values_per_cell=dh.dim)
        dh.fill(self.VELOCITY, 0.0, ghost_layers=True)
        dh.all_to_gpu()
        self.dh = dh

        gpu_sync()
        logger.debug(f"Data Handling init time: {time() - t} s")
        logger.debug(gpu_memory_stats())

    def _init_lbm(self):
        """Initialize lbm kernels"""
        t = time()
        logger.info("Initializing lbm kernels")
        symbolic_force = (sp.Symbol("fx"), sp.Symbol("fy"), sp.Symbol("fz"))
        symbolic_omega = sp.Symbol("omega")

        lbm_config = lp.LBMConfig(
            stencil=self.stencil,
            method=getattr(lp.Method, self.prm.method),
            compressible=self.prm.compressible,
            relaxation_rate=symbolic_omega,
            force=symbolic_force,
            smagorinsky=self.prm.smagorinsky,
        )

        self._method = lp.create_lb_method(lbm_config=lbm_config)

        macro_getter = lp.macroscopic_values_getter(
            self._method, density=None, velocity=self._velField, pdfs=self._src
        )
        self._getter_kernel = ps.create_kernel(
            macro_getter, target=self.dh.default_target
        ).compile()

        lbm_optimisation = lp.LBMOptimisation(
            symbolic_field=self._src, symbolic_temporary_field=self._dst
        )
        update = lp.create_lb_update_rule(
            lb_method=self._method,
            lbm_config=lbm_config,
            lbm_optimisation=lbm_optimisation,
        )

        ast_kernel = ps.create_kernel(
            update, target=self.dh.default_target, cpu_openmp=True
        )
        self._kernel = ast_kernel.compile()

        gpu_sync()
        logger.debug(f"LBM kernels init time: {time() - t} s")
        logger.debug(gpu_memory_stats())
    
    #def _init_phydll(self):
    #    # Init PhyDLL
    #    phyl = PhyDLL()
    #    phyl.init(instance="physical")

    #    self.phyl = phyl

    #    self.phyl.opt_enable_cpl_loop()

        # Get MPI local communicator
    #    self.comm = self.phyl.get_local_mpi_comm()
    
    def _init_adios2(self):
        comm = MPI.COMM_WORLD
        self.adios = adios2.ADIOS(self.prm.adios2_cfg, comm)
        self.io     = self.adios.DeclareIO("writerIO")
        self.engine = self.io.Open(str(self.run_name)+"/atmo_sol", adios2.Mode.Write) 

    def _init_boundary(self):
        """Setup boundary conditions"""
        t = time()
        logger.info("Setting up boundary conditions")
        if self.prm.continue_previous:
            latest_jobdir = find_latest_job_directory(self.run_folder, restartable=True)
            if latest_jobdir is None:
                logger.warning(
                    "No previous run found. Starting from scratch even though "
                    "continue_previous = True"
                )
                self.prm.continue_previous = False

        if self.prm.continue_previous:
            start_it, restart_path = latest_jobdir.restarts[-1]
            with open(restart_path.parents[0] / f"Cubes_param.obs", 'rb') as pickle_file:
                self.Cubes_param = pickle.load(pickle_file)
                pickle_file.close()
        else:
            self.Cubes_param =  None
            

        self.bh = LatticeBoltzmannBoundaryHandling(
                self._method,
                self.dh,
                self.SOURCE,
                name="bh",
                target=self.dh.default_target,
            )
        self._periodic = self.dh.synchronization_function(self.SOURCE)
        wall = NoSlip("wall")
        free_slip = FreeSlip(self.stencil, normal_direction=(0, -1, 0), name="freeSlip")

        self.bh.set_boundary(wall, slice_from_direction("S", self.prm.dim))
        self.bh.set_boundary(free_slip, slice_from_direction("N", self.prm.dim))
        self.Cubes_param = self.prm.obstacles(self.bh, self.Cubes_param)

        gpu_sync()
        logger.debug(f"Boundary init time: {time() - t} s")
        logger.debug(gpu_memory_stats())

    def _init_fields(self):
        """Initialize physical fields"""
        t = time()
        if self.prm.continue_previous:
            latest_jobdir = find_latest_job_directory(self.run_folder, restartable=True)
            if latest_jobdir is None:
                logger.warning(
                    "No previous run found. Starting from scratch even though "
                    "continue_previous = True"
                )
                self.prm.continue_previous = False

        if self.prm.continue_previous:
            start_it, restart_path = latest_jobdir.restarts[-1]
            logger.info(
                f"Restartable solution found: {restart_path}. "
                f"Starting from iteration {start_it}"
            )
            sol = AtmoSolution.from_h5file(restart_path)
            self.dh.cpu_arrays[self.SOURCE] = sol.pdf
            self.dh.to_gpu(self.SOURCE)
            self.ul_bulk = sol.misc["ul_bulk"]
            self.dh.run_kernel(
                self._getter_kernel,
                **self.prm.force(self.scl.u_from_ul(self.ul_bulk)),
                **self.prm.ω_dynamic(self.scl.t_from_iterations(start_it)),
            )
            self.dh.to_cpu(self.VELOCITY)
            self.iteration = start_it
            self.previous = sol.misc.get("run_name")
            logger.info(f"Last run was called {sol.misc.get('run_name', '<Unkown>')}")
        else:
            logger.info("Initializing physical fields from inputs")
            if self.prm.init_ul is not None:
                logger.info("Defining initial field according to init_ul")
                with cupy.cuda.Device(ps.GPU_DEVICE):
                    self.velocity[1:-1, 1:-1, 1:-1] = cupy.asarray(self.prm.init_ul)
                gpu_memory_free()
            self.bh()
            init = lp.pdf_initialization_assignments(
                self._method,
                self.prm.init_ρl,
                self._velField,
                self._src.center_vector,
            )
            kernel_init = ps.create_kernel(
                init, target=self.dh.default_target
            ).compile()
            self.dh.run_kernel(
                kernel_init, **self.prm.force(0), **self.prm.ω_dynamic(0)
            )
            self.dh.all_to_cpu()
            self.iteration = 0
            self.ul_bulk = 0


        if self.prm.stream:
            lx, ly, lz, nf = self.velocity.shape
            vel_id = np.zeros((nf, lx, ly, lz, self.prm.steps_to_save))
            print(vel_id.shape)
            self.vel_id = self.io.DefineVariable("velocity", vel_id, [*vel_id.shape], [0]*len(vel_id.shape), [*vel_id.shape],
                                                  adios2.ConstantDims)
        #if self.prm.use_phydll:
        #    self.phyl.define_phy(count = self.prm.n_fields_send, size = self.velocity.size)
        gpu_sync()
        logger.debug(f"Fields init time: {time() - t} s")
        logger.debug(gpu_memory_stats())

    def _setup_outputs(self):
        """Setup all outputs

        Create unique folder, temporal file, probe files, copy run file,
        allocate arrays to compute average quantities.
        """

        # Created dedicated job folder with unique name
        self.job_folder = self.run_folder / self.run_name
        self.job_folder.mkdir()
        self.tmp_file = self.job_folder / self.prm.temporal_file
        logger.info(f"Temporals stored in {self.tmp_file}")

        # log for the run in job folder
        logger.add(
            self.job_folder / JOB_LOG_FILE,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level="DEBUG",
            backtrace=True,
            diagnose=True,
            serialize=True,
        )

        # Temporal file
        with open(self.tmp_file, "w") as fh:
            fh.write(f"# Temporals for run {self.run_name}\n")
            fh.write(
                "# iteration time fx fy fz omega ul_max u_min u_max Mach_max ubulk Ec\n"
            )

        # Probes files
        self._probes = []
        if self.prm.probes is not None:
            probe_nodes = [
                np.minimum(
                    np.around(np.array(probe) / self.scl.dx).astype(int),
                    np.array(self.prm.domain_cells) - 1,
                ).tolist()
                for probe in self.prm.probes
            ]  # Avoid points in ghost layer due to wonky rounding errors
            probe_files = [
                self.job_folder / f"probe_{nx}_{ny}_{nz}.dat"
                for nx, ny, nz in probe_nodes
            ]
            self._probes = list(zip(probe_nodes, probe_files))
            for (nx, ny, nz), file in self._probes:
                if not file.is_file():
                    with open(file, "w") as fh:
                        fh.write(
                            f"# Probe file for node {(nx, ny, nz)} "
                            f"at location {tuple(n*self.scl.dx for n in [nx, ny, nz])} "
                            "\n"
                            f"# iteration time u v w"
                            "\n"
                        )

        # Allocate arrays to compute averages, rms as run advances for final output
        if self.prm.mean:
            with cupy.cuda.Device(ps.GPU_DEVICE):
                logger.debug("Allocating mean velocity array on GPU")
                self.vel_mean = cupy.zeros_like(self.dh.gpu_arrays[self.VELOCITY])
                if self.prm.rms:
                    logger.debug("Allocating mean squared velocity array on GPU")
                    self.vel_squared = cupy.zeros_like(self.vel_mean)
                self.nb_avg = 0
            logger.debug(gpu_memory_stats())

        # run file (called with `python run_file.py`)
        run_file = Path(__main__.__file__)
        (self.job_folder / (run_file.name + ".bck")).write_bytes(run_file.read_bytes())

    def write_solution(self, avg=False):
        """Write an hdf and matching xmf file of the current velocity field"""
        vel = self.scl.u_from_ul(self.velocity.get())
        sol = atmosol_from_ap(self)
        sol.misc = {"run_name": self.run_name}
        if self.previous is not None:
            sol.misc["previous"] = self.previous
        sol.velocity = vel
        sol.geometry = self.geometry
        if avg and self.prm.mean:
            if self.nb_avg == 0:
                logger.warning("Can't compute averages on 0 fields. Skipping")
            else:
                logger.info(f"Saving mean{', rms' if self.prm.rms else ''} in solution")
                sol.mean = self.scl.u_from_ul(self.vel_mean.get()) / self.nb_avg
                if self.prm.rms:
                    vsq = (
                        self.scl.u_from_ul(self.scl.u_from_ul(self.vel_squared.get()))
                        / self.nb_avg
                    )
                    sol.rms = np.sqrt(vsq - sol.mean**2)
        sol.write(
            self.job_folder / f"sol_{self.iteration:08}.h5",
            overwrite=self.prm.overwrite,
            compression=self.prm.compression,
            compression_opts=self.prm.compression_opts,
        )

    def write_restart(self):
        """Write an hdf file that is restartable"""
        sol = atmosol_from_ap(self)
        pdf = self.dh.gpu_arrays[self.SOURCE]
        logger.debug("Before restart .get(): " + gpu_memory_stats())
        # logger.warning("Experimental")
        # logger.debug(f"{sys.getrefcount(self._dst)}")
        # logger.debug(f"{sys.getrefcount(self.dh.gpu_arrays[self.DESTIN])}")
        # del self.dh.gpu_arrays[self.DESTIN]
        # logger.debug(f"{sys.getrefcount(self._dst)}")
        # logger.debug("After del DESTIN: " + gpu_memory_stats())
        # cupy.get_default_memory_pool().free_all_blocks()
        # logger.debug("After free blocks: " + gpu_memory_stats())
        sol.pdf = pdf.get()
        logger.debug("After restart .get(): " + gpu_memory_stats())
        sol.misc = {
            "ul_bulk": self.ul_bulk,
            "run_name": self.run_name,
        }
        if self.previous is not None:
            sol.misc["previous"] = self.previous
        sol.write(
            self.job_folder / f"rst_{self.iteration:08}.h5",
            xmf=False,
            overwrite=self.prm.overwrite,
        )
        with open(self.job_folder / f"Cubes_param.obs", 'wb') as pickle_file:
            pickle.dump(self.Cubes_param, pickle_file)
            pickle_file.close()

    def _monitor(self):
        """Monitoring: temporal files, compute averages, integrals"""
        self.dh.run_kernel(
            self._getter_kernel,
            **self.prm.force(self.scl.u_from_ul(self.ul_bulk)),
            **self.prm.ω_dynamic(self.scl.t_from_iterations(self.iteration)),
        )

        if cupy.isnan(self.velocity).any().get():
            logger.error(" >> NaN detected. Stopping ")
            return

        mach_max = (
            cupy.linalg.norm(self.velocity, axis=-1).max().get() / self.scl.csoundl
        )
        if mach_max > 0.2:
            logger.warning(
                f"Maximum Mach is above 0.2 ({mach_max}). "
                "Your computation might crash!"
            )

        if self.prm.mean:
            self.vel_mean += self.velocity
            if self.prm.rms:
                vsq = self.velocity**2
                self.vel_squared += vsq
            self.nb_avg += 1

        self.ul_bulk = self.velocity[1:-1, 1:-1, 1:-1, 0].sum().get() / self.prm.ncells
        force = self.prm.force(self.scl.u_from_ul(self.ul_bulk))
        ul_min, ul_max = (
            self.velocity.min().get(),
            self.velocity.max().get(),
        )
        it = self.iteration
        with open(self.tmp_file, "+a", buffering=1) as fh:
            fh.write(
                f"{it:08} {it*self.scl.dt:.9e} "
                f"{force['fx']:.9e} {force['fy']:.9e} {force['fz']:.9e} "
                f"{self.prm.ω_dynamic(self.scl.t_from_iterations(it))['omega']} "
                f"{ul_max:.9e} "
                f"{self.scl.u_from_ul(ul_min):.9e} "
                f"{self.scl.u_from_ul(ul_max):.9e} "
                f"{mach_max:.9e} {self.scl.u_from_ul(self.ul_bulk):.9e} "
                f"{vsq.sum().get()* self.scl.dx**3:.9e} "
                "\n"
            )
        for (nx, ny, nz), file in self._probes:
            with open(file, "+a", buffering=1) as fh:
                u, v, w = self.prm.scaling.u_from_ul(
                    self.velocity[nx + 1, ny + 1, nz + 1].get()
                )
                fh.write(
                    f"{it:08} {it*self.scl.dt:.9e} " f"{u:.9e} {v:.9e} {w:.9e} " "\n"
                )

    @logger.catch
    def run(self):
        self._run_init()
        logger.info("Start computation ")
        start_time = time()
        start_it = self.iteration
        count = 0
        saved_vel= []
        with cupy.cuda.Device(ps.GPU_DEVICE):
            for it in trange(
                start_it + 1,
                self.prm.final_iteration + 1,
                miniters=self.prm.monitor_freq,
            ):
                if (self.run_folder / "stop").is_file():
                    logger.warning("'stop' file detected, stopping cleanly")
                    break

                self.iteration = it

                if (it - start_it) % self.prm.monitor_freq == 0 or it == start_it:
                    self._monitor()
                
                # Regular interval solution outputs
                if (it - start_it) % self.prm.output_freq == 0:
                    if self.prm.write_on_disk:
                        self.write_solution(avg=self.prm.avg_every_sol)
                    logger.debug(gpu_memory_stats())
                    
                    #if self.prm.use_phydll:

                    #    self.phyl.set_field(field = atmosol_from_ap(self).velocity, label = "field_" + str(count))
                    #    count += 1 
                    #    if count == self.prm.n_fields_send:
                    #        self.phyl.send()
                    #        #dummy_field = self.phyl.wait_irecv(only=False)
                    #        #self.phyl.irecv()
                            
                    #        count = 0

                    if self.prm.stream:
                        vel = self.scl.u_from_ul(self.velocity.get())
                        saved_vel.append(np.stack([vel[...,0], vel[...,1], vel[...,2]]))
                        count += 1
                        if count == self.prm.steps_to_save:
                            saved_vel = np.stack(saved_vel,-1)
                            self.engine.BeginStep()
                            self.engine.Put(self.vel_id, saved_vel)
                            self.engine.EndStep()  
                            saved_vel = []
                            count = 0
                            os.system("rm -r "+str(self.job_folder)+"/sol*")
                            os.system("rm -r "+str(self.job_folder)+"/rst*")
                            self.write_solution(avg=self.prm.avg_every_sol)
                            self.write_restart()

                # Progress 1 iteration
                self._periodic()
                self.bh()
                self.dh.run_kernel(
                    self._kernel,
                    **self.prm.force(self.scl.u_from_ul(self.ul_bulk)),
                    **self.prm.ω_dynamic(self.scl.t_from_iterations(it)),
                )
                self.dh.swap(self.SOURCE, self.DESTIN)

        self._monitor()
        gpu_sync()
        logger.success("Simulation Done")
        sim_time = time() - start_time
        logger.info(f"Time needed for the calculation: {sim_time} seconds")
        mlups = self.prm.ncells * (self.iteration - start_it) / sim_time * 1e-6
        logger.info(f"MLUPS: {mlups}")
        logger.debug(gpu_memory_stats())

        self.write_solution(avg=True)
        if self.prm.save_restart:
            self.write_restart()
        #self.phyl.finalize()
        if self.prm.stream:
            self.engine.Close()

def atmosol_from_ap(ap: AtmosphericPeriodic) -> AtmoSolution:
    """Helper function to initalize solution from a run"""
    return AtmoSolution(ap.iteration, ap.prm)

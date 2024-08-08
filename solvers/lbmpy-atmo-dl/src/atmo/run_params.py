from __future__ import annotations
from typing import Union, Optional
from pathlib import Path
from dataclasses import dataclass

import dill
import numpy as np
#import lbmpy as lp

from loguru import logger

from atmo.lbm_scaling import LBMScaling
from atmo.run_dir_tools import TEMPORAL_FILE


@dataclass
class RunParams:
    """A class to hold the run parameters of an LBM run"""

    scaling: LBMScaling  # Scaling parameters
    domain_size: tuple[float] = (10, 10, 10)  # [m, m, m]
    method: str = "CUMULANT"
    stencil: str = "D3Q19"
    compressible: bool = True
    smagorinsky: bool = False
    dim: int = 3

    # - Initialization - #
    # Continue previous simulation
    continue_previous: bool = False
    # Or using constant values
    init_ρl: float = 1  # intial lattice density
    init_ul: Optional[np.typing.ArrayLike] = None  # initial lattice velocity
    write_init_file: bool = False
    write_on_disk: bool = True

    # - Run - #
    final_iteration: float = 10  # Final iteration of simulation [iterations]
    output_freq: int = 10000  # Frequency of output solutions [iterations]
    monitor_freq: int = 50  # Frequency of monitoring [iterations]
    overwrite: bool = False  # Flag to overwrite existing solution files
    save_restart: bool = True  # Save file necessary for restart at the end

    # HDF5 verbatim compression options (see https://docs.h5py.org/en/stable/high/dataset.html#lossless-compression-filters)
    compression: Optional[str] = None  # One of gzip, lzf, szip
    compression_opts: Optional[int] = None  # Between 0 and 9

    # - Boundary - #
    periodicity: tuple[bool] = (True, False, True)
    direction: tuple[float] = (1, 0, 0)

    temporal_file: str = TEMPORAL_FILE
    # temporals: str = "ubulk"
    mean: bool = True
    rms: bool = True
    avg_every_sol: bool = False  # Add mean, rms to every solution

    probes: list[list[float]] = None

    # PhyDLL#
    use_phydll: bool = False #activate PhyDLL in atmo
    n_fields_send: int = 10 # Number of fields to send to DL training run

    #Stream data using adios2
    stream: bool = True
    adios2_cfg: str = None
    steps_to_save: int = 5

    def __post_init(self):
        if self.rms and not self.mean:
            logger.error("Can't compute RMS without averages. Setting averages to True")
            self.mean = True

    def obstacles(*args):
        """Obstacles in the volume. By default, no obstacles"""
        pass

    def ω_dynamic(self, *args):
        """Schedule of values for ω"""
        return {"omega": self.scaling.ω}

    def force(*args) -> dict:
        """Volume forcing term. Default is no force"""
        return {"fx": 0, "fy": 0, "fz": 0}

    @property
    def domain_cells(self):
        return tuple([int(ds / self.scaling.dx) for ds in self.domain_size])

    @property
    def ncells(self):
        return np.prod(self.domain_cells)

    @property
    def volume(self):
        return self.scaling.dx**3 * self.ncells

    @property
    def final_time(self):
        return self.final_iteration * self.scaling.dt

    @property
    def output_time(self):
        return self.output_freq * self.scaling.dt

    def info(self, log=True):
        """Log information about run"""
        msg = f"""Run info:
            Domain has {self.ncells} cells, for a volume of {self.volume} m^3
            Run scheduled until {self.final_time} seconds
                       which is {self.final_iteration} iterations
            Saving 3D field every {self.output_freq * self.scaling.dt} seconds
                         which is {self.output_freq} iterations
            With a timestep of {self.scaling.dt} seconds
            Bulk target velocity is {self.scaling.u_bulk_target} m/s
                           which is {self.scaling.ul_bulk_target} lattices / iteration
            Effective ν is {self.scaling.ν} m^2/s ({self.scaling.ν/self.scaling.ν_target} times target)
            Numerics: ul_max = {self.scaling.ul_max}, ω (except dynamic)= {self.scaling.ω}"""
        if log:
            logger.info(msg)
        return msg

    def serialize(self) -> bytes:
        """Serialize self for storage and future reference / reuse"""
        return dill.dumps(self)

    def serialize_h5(self) -> np.typing.ArrayLike:
        """Serialize self for an HDF5 file

        In an HDF5 file, a trick is needed to avoid an error:
            ValueError: VLEN strings do not support embedded NULLs
        """
        return np.fromstring(dill.dumps(self), dtype=np.uint8)

    # @classmethod
    # def deserialize(cls, dict_):
    @staticmethod
    def deserialize(bytes_: bytes) -> RunParams:
        """Deserialize self from previous serialization"""
        return dill.loads(bytes_)

    @staticmethod
    def deserialize_h5(array: np.typing.ArrayLike) -> RunParams:
        """Trick needed to deserialize from an HDF5 file"""
        return dill.loads(array.tobytes())

    def save(self, filename, overwrite=False):
        """Save to file"""
        filename = Path(filename)
        if filename.is_file() and not overwrite:
            logger.error(f"Cannot overwrite existing file {filename}")
            return
        with open(filename, "wb") as fh:
            fh.write(self.serialize())

    @classmethod
    def load(cls, filename: Union[str, Path]) -> RunParams:
        """Load from file"""
        with open(filename, "rb") as fh:
            load = cls.deserialize(fh.read())
            # load = cls.deserialize(json.load(fh))
        return load

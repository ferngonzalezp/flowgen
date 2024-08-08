from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional, ClassVar
from pathlib import Path
from time import time

import numpy as np
from h5py import File
from loguru import logger

import atmo
from .xmf import write_xmf

def is_restartable(file_: Union[str, Path]) -> bool:
    """Fast check whether a solution file is restartable"""
    with File(file_) as h5f:
        return AtmoSolution.PDF in h5f


@dataclass
class AtmoSolution:
    """Holder class for an Atmo solution"""

    iteration: int
    params: atmo.RunParams
    misc: Optional[dict] = None
    velocity: Optional[np.typing.ArrayLike] = None
    geometry: Optional[np.typing.ArrayLike] = None
    mean: Optional[np.typing.ArrayLike] = None
    rms: Optional[np.typing.ArrayLike] = None
    pdf: Optional[np.typing.ArrayLike] = None

    VERSION: ClassVar[str] = "version"
    MISC: ClassVar[str] = "miscellaneous"
    ITERATION: ClassVar[str] = "iteration"
    RUNPARAMS: ClassVar[str] = "params"
    VELOCITY: ClassVar[str] = ("u", "v", "w")
    GEOMETRY: ClassVar[str] = "geometry"
    MEAN: ClassVar[str] = ("u_mean", "v_mean", "w_mean")
    RMS: ClassVar[str] = ("u_rms", "v_rms", "w_rms")
    PDF: ClassVar[str] = "pdf"

    @property
    def is_restartable(self) -> bool:
        """Can a full restart be performed from this solution?"""
        return self.pdf is not None

    @classmethod
    def from_h5file(cls, filename: str) -> AtmoSolution:
        """Generate this object from an hdf5 file on disk"""
        with File(filename) as h5f:
            file_atmo_version = h5f[cls.VERSION].asstr()[()]
            if file_atmo_version != atmo.__version__:
                logger.warning(
                    f"Loading previous solution from version {file_atmo_version}, "
                    f"but running with version {atmo.__version__}."
                )
            out = cls(
                iteration=h5f[cls.ITERATION][()],
                params=atmo.RunParams.deserialize_h5(h5f[cls.RUNPARAMS][()]),
            )
            if cls.VELOCITY[0] in h5f:
                out.velocity = np.stack([h5f[key][()] for key in cls.VELOCITY], axis=-1)
            if cls.GEOMETRY in h5f:
                out.geometry = np.stack([h5f[key][()] for key in cls.VELOCITY], axis=-1)
            if cls.MEAN[0] in h5f:
                out.mean = np.stack([h5f[key][()] for key in cls.MEAN], axis=-1)
            if cls.RMS[0] in h5f:
                out.rms = np.stack([h5f[key][()] for key in cls.RMS], axis=-1)
            if cls.PDF in h5f:
                out.pdf = h5f[cls.PDF][()]
            if cls.MISC in h5f:
                out.misc = {key: h5f[cls.MISC][key][()] for key in h5f[cls.MISC]}
        return out

    def write(
        self,
        filename: Union[str, Path],
        overwrite: bool = False,
        xmf: bool = True,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
        stream: bool = False
    ):
        """Write HDF5 and (optionally) matching xmf for easy viewing"""
        start_time = time()
        filename = Path(filename)
        if filename.is_file() and not overwrite:
            logger.error(
                f"Cannot overwrite existing file {filename}. "
                "Use overwrite=True to force."
            )
            return

        logger.info(
            f"Saving solution file {filename}{' and matching xmf' if xmf else ''}"
        )
        with File(filename, "w") as dst:
            if compression is not None:
                logger.warning(
                    f"Using {compression} compression. "
                    "Tests suggest this is much slower for little benefit."
                )
                if compression_opts is None:
                    compression_opts = 4

            def create_ds(key, data):
                dst.create_dataset(
                    key,
                    data=data,
                    compression=compression,
                    compression_opts=compression_opts,
                )

            dst[self.VERSION] = atmo.__version__
            dst[self.ITERATION] = self.iteration
            dst[self.RUNPARAMS] = self.params.serialize_h5()

            xmf_scalars = []
            xmf_vects = []
            if self.velocity is not None:
                create_ds(self.VELOCITY[0], self.velocity[:, :, :, 0])
                create_ds(self.VELOCITY[1], self.velocity[:, :, :, 1])
                create_ds(self.VELOCITY[2], self.velocity[:, :, :, 2])
                xmf_scalars += list(self.VELOCITY)
            if self.geometry is not None:
                create_ds(self.GEOMETRY, self.geometry)
                xmf_scalars += [self.GEOMETRY]
            if self.mean is not None:
                create_ds(self.MEAN[0], self.mean[:, :, :, 0])
                create_ds(self.MEAN[1], self.mean[:, :, :, 1])
                create_ds(self.MEAN[2], self.mean[:, :, :, 2])
                xmf_scalars += list(self.MEAN)
            if self.rms is not None:
                create_ds(self.RMS[0], self.rms[:, :, :, 0])
                create_ds(self.RMS[1], self.rms[:, :, :, 1])
                create_ds(self.RMS[2], self.rms[:, :, :, 2])
                xmf_scalars += list(self.RMS)
            if self.pdf is not None:
                create_ds(self.PDF, self.pdf)
                xmf_vects = [self.PDF]
            if self.misc is not None:
                grp = dst.create_group(self.MISC)
                for key in self.misc:
                    grp[key] = self.misc[key]

        if xmf:
            write_xmf(
                filename,
                self.params.domain_cells,
                (self.params.scaling.dx,) * 3,
                xmf_scalars,
                xmf_vects,
            )
        logger.debug(f"Write time for {filename} and xmf: {time() - start_time}")
"""Run Directory tools

One *run* is defined as a folder under which a number of chained
and restarted *jobs* take place.

Run directories are organized as follows:
root/
  run.py              # Runnable python file from which runs are started
  RUN_LOG_FILE        # Log file for the run
  unique_job_name/
    JOB_LOG_FILE      # JSON serialized log for the job
    sol_00010000.h5   # HDF5 solution
    sol_00010000.xmf  # XMF to read the solution
    rst_00010000.h5   # restartable solution
    temporal.dat      # temporal data
    probe_X_Y_Z.dat   # probe temporal data
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from h5py import File
import numpy as np

# from loguru import logger

import atmo


RUN_LOG_FILE = "atmo.log"
JOB_LOG_FILE = "atmo_log.json"
TEMPORAL_FILE = "temporal.dat"


@dataclass
class JobDirectory:
    path: Path

    def __post_init__(self):
        self.path = Path(self.path)

        def iteration(file_):
            with File(file_) as h5f:
                return h5f[atmo.AtmoSolution.ITERATION][()]

        h5s = [(iteration(file_), file_) for file_ in self.path.glob("*.h5")]
        self.solutions = {
            it: file_ for it, file_ in h5s if not atmo.is_restartable(file_)
        }
        self.restarts = [
            (it, file_) for it, file_ in h5s if file_ not in self.solutions.values()
        ]
        self.probes = [p.name for p in self.path.glob("probe_*.dat")]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_restartable(self) -> bool:
        return bool(self.restarts)

    @property
    def iteration_interval(self) -> list[int, int]:
        temporal = np.loadtxt(self.path / TEMPORAL_FILE)
        return int(temporal[0, 0]), int(temporal[-1, 0])

    @property
    def sol_iterations(self) -> list:
        """Ordered list of iterations that have a solution"""
        return sorted(self.solutions.keys())

    @property
    def previous_job_name(self) -> str:
        """Previous job that this one was restarted from"""
        if self.solutions:
            with File(self.solutions[min(self.sol_iterations)]) as h5f:
                misc = h5f[atmo.AtmoSolution.MISC]
                if "previous" in misc:
                    return misc["previous"][()].decode()
                return None
        if self.restarts:
            with File(self.restarts[-1]) as h5f:
                misc = h5f[atmo.AtmoSolution.MISC]
                if "previous" in misc:
                    return misc["previous"][()].decode()
                return None

    @property
    def previous_job(self) -> JobDirectory:
        """JobDirectory of previous job"""
        previous = self.previous_job_name
        if previous is not None:
            previous = Path(previous)
            if previous.is_dir() and (previous / JOB_LOG_FILE).is_file():
                return JobDirectory(previous)


def find_job_directories(dir_=None):
    """Find directories under dir_ that are job directories"""
    if dir_ is None:
        dir_ = Path.cwd()
    return [
        JobDirectory(path)
        for path in Path(dir_).iterdir()
        if path.is_dir() and (path / JOB_LOG_FILE).is_file()
    ]


def find_job_chain(jobname):
    """List of jobs (last to first) that led to jobname"""
    chain = [JobDirectory(jobname)]
    while True:
        previous = chain[-1].previous_job
        if previous is None:
            return chain
        chain.append(previous)


def find_latest_job_directory(dir_=None, restartable=False):
    """Find the latest JobDirectory"""
    if dir_ is None:
        dir_ = Path.cwd()

    jobdirs = find_job_directories(dir_)
    if restartable:
        jobdirs = [job for job in jobdirs if job.is_restartable]
    if not jobdirs:
        return None
    return sorted(jobdirs, key=lambda jd: jd.iteration_interval[-1])[-1]


def join_temporals(last_jobname, temporal_file=None):
    """Join all binaries in subfolders"""

    if temporal_file is None:
        temporal_file = TEMPORAL_FILE

    chain = find_job_chain(last_jobname)

    # Establish a list of all temporal files in job chain
    temp_files = {probe for jd in chain for probe in jd.probes}
    temp_files.add(temporal_file)

    def join_temp_file(filename):
        out = [
            (jd.path / filename).read_text()
            for jd in chain
            if (jd.path / filename).is_file()
        ][::-1]

        # If there is more than one file, check that headers match
        # Oh but wait, that can't work! The name of the run is included in there
        #
        # def split_header(contents):
        #     for i in range(len(contents) - 1):
        #         if contents[i] == "\n":
        #             if contents[i + 1] != "#":
        #                 break
        #     return contents[:i], contents[i:]
        # headers, contents = zip(*(split_header(f) for f in out))
        # if len(headers) > 1:
        #     if not all(h == headers[0] for h in headers[1]):
        #         logger.warning(f"Headers are incoherent for {filename}. Skipping")
        #         return None
        # return "\n".join([headers[0]] + contents)

        return "\n".join(out)

    return {filename: join_temp_file(filename) for filename in temp_files}

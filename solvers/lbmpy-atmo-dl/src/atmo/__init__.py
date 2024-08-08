__version__ = "0.3.0"

import sys
from loguru import logger

from atmo.run_params import RunParams
from atmo.lbm_scaling import LBMScaling, StableScaling
from atmo.atmo_solution import AtmoSolution, is_restartable


logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    backtrace=True,
    diagnose=True,
)

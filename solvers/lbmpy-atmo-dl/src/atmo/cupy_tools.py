import os

from loguru import logger

import pystencils as ps


try:
    import cupy
except ImportError:
    logger.warning("CUDA environment failed to load. Pre/postproc mode only")
    cupy = None


def setup_gpu():
    ps.GPU_DEVICE = int(os.environ["CUDA_DEVICE"]) if "CUDA_DEVICE" in os.environ else 0
    # Test whether GPU is accessible:
    cupy.cuda.Device(ps.GPU_DEVICE).compute_capability
    logger.info(f"Running on GPU n°{ps.GPU_DEVICE}")


def gpu_sync():
    """Syncronize CUDA threads. Useful for timers"""
    with cupy.cuda.Device(ps.GPU_DEVICE):
        cupy.cuda.runtime.deviceSynchronize()


def gpu_memory_stats():
    """Useful statistics about GPU memory"""
    with cupy.cuda.Device(ps.GPU_DEVICE):
        mempool = cupy.get_default_memory_pool()
        return (
            f"GPU bytes: {mempool.used_bytes()} used / {mempool.total_bytes()} "
            f"allocated. {mempool.n_free_blocks()} block(s) could be freed."
        )


def gpu_memory_free():
    """Deallocate unused GPU memory"""
    with cupy.cuda.Device(ps.GPU_DEVICE):
        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

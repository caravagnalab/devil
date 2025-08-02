"""GPU utilities and detection for devil package."""

import warnings
from typing import Optional, Tuple, Union
import numpy as np

# Try to import CuPy for GPU support
try:
    import cupy as cp
    # Note: cupyx.scipy.linalg may fail due to CUDA library compatibility
    try:
        import cupyx.scipy.linalg as cp_linalg
    except ImportError:
        cp_linalg = None
    try:
        import cupyx.scipy.sparse as cp_sparse
    except ImportError:
        cp_sparse = None
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_linalg = None
    cp_sparse = None
    CUPY_AVAILABLE = False


def is_gpu_available(use_memory: bool = True, memory_cutoff_gb: float = 10.0) -> bool:
    """
    Check if GPU (CUDA) is available for computation.
    Args:
        use_memory: Whether to check if GPU memory is available.
        memory_cutoff_gb: Minimum amount of GPU memory required in GB.
    Returns:
        True if GPU is available and CuPy is installed, False otherwise.
    """
    if not CUPY_AVAILABLE:
        return False
    
    can_use_gpu = False
    try:
        # Try to create a simple array on GPU
        cp.cuda.Device(0).use()
        test_array = cp.array([1, 2, 3])
        del test_array
        can_use_gpu = True
    except Exception:
        pass
    
    if use_memory:
        free_mem, total_mem = get_gpu_memory_info()
        if free_mem / 1e9 < memory_cutoff_gb:
            can_use_gpu = False
    return can_use_gpu


def get_gpu_memory_info() -> Tuple[int, int]:
    """
    Get GPU memory information.
    
    Returns:
        Tuple of (free_memory, total_memory) in bytes.
        Returns (0, 0) if GPU not available.
    """
    if not is_gpu_available(use_memory=False):
        return 0, 0
    
    try:
        # Get actual GPU device memory information
        device = cp.cuda.Device()
        free_memory, total_memory = device.mem_info
        return free_memory, total_memory
    except Exception:
        return 0, 0


def estimate_batch_size(
    n_genes: int,
    n_samples: int,
    n_features: int,
    dtype: np.dtype = np.float32,
    memory_fraction: float = 0.8
) -> int:
    """
    Estimate optimal batch size for GPU processing based on available memory.
    
    Args:
        n_genes: Number of genes to process.
        n_samples: Number of samples.
        n_features: Number of features in design matrix.
        dtype: Data type for calculations.
        memory_fraction: Fraction of GPU memory to use.
        
    Returns:
        Recommended batch size for processing genes.
    """
    if not is_gpu_available():
        return min(n_genes, 1024)  # Default CPU batch size
    
    try:
        # Get available GPU memory
        free_memory, total_memory = get_gpu_memory_info()
        available_memory = int(total_memory * memory_fraction)
        
        # Estimate memory per gene (rough calculation)
        bytes_per_element = np.dtype(dtype).itemsize
        
        # Memory needed per gene:
        # - Count data: n_samples
        # - Beta coefficients: n_features  
        # - Working arrays: ~3 * n_samples (mu, weights, residuals)
        # - Design matrix (shared): n_samples * n_features
        # - Temporary arrays: ~2 * n_samples * n_features
        
        memory_per_gene = bytes_per_element * (
            n_samples +  # count data
            n_features +  # beta coefficients
            3 * n_samples +  # working arrays
            2 * n_samples * n_features  # temporary computation
        )
        
        # Add shared memory for design matrix
        shared_memory = bytes_per_element * n_samples * n_features
        
        # Calculate batch size
        batch_size = max(1, (available_memory - shared_memory) // memory_per_gene)
        batch_size = min(batch_size, n_genes)
        
        return batch_size
        
    except Exception:
        return min(n_genes, 1024)


def to_gpu(array: np.ndarray, dtype: Optional[np.dtype] = None) -> "cp.ndarray":
    """
    Transfer numpy array to GPU.
    
    Args:
        array: NumPy array to transfer.
        dtype: Optional target dtype.
        
    Returns:
        CuPy array on GPU.
        
    Raises:
        RuntimeError: If GPU not available.
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    if dtype is not None:
        array = array.astype(dtype)
    
    return cp.asarray(array)


def to_cpu(array: "cp.ndarray") -> np.ndarray:
    """
    Transfer CuPy array back to CPU.
    
    Args:
        array: CuPy array to transfer.
        
    Returns:
        NumPy array on CPU.
    """
    if not CUPY_AVAILABLE:
        return array  # Already numpy array
    
    return cp.asnumpy(array)


class GPUMemoryManager:
    """Context manager for GPU memory management."""
    
    def __init__(self, clear_cache: bool = True):
        """
        Initialize GPU memory manager.
        
        Args:
            clear_cache: Whether to clear GPU memory cache on exit.
        """
        self.clear_cache = clear_cache
        self.initial_memory = None
        
    def __enter__(self):
        if is_gpu_available():
            self.initial_memory = get_gpu_memory_info()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_gpu_available() and self.clear_cache:
            try:
                # Clear memory pool
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                
                # Clear cache
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()
                
            except Exception as e:
                warnings.warn(f"Error clearing GPU memory: {e}")


def check_gpu_requirements(
    n_genes: int,
    n_samples: int,
    n_features: int,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Check if GPU computation is feasible for given problem size.
    
    Args:
        n_genes: Number of genes.
        n_samples: Number of samples.
        n_features: Number of features.
        verbose: Whether to print detailed info.
        
    Returns:
        Tuple of (feasible, message).
    """
    if not CUPY_AVAILABLE:
        return False, "CuPy not installed. Install with: pip install cupy-cuda11x"
    
    if not is_gpu_available():
        return False, "No GPU detected or CUDA not available"
    
    # Check memory requirements
    batch_size = estimate_batch_size(n_genes, n_samples, n_features)
    
    if batch_size < 1:
        return False, "Insufficient GPU memory for computation"
    
    if batch_size < 10:
        message = (
            f"GPU batch size very small ({batch_size}). "
            "Consider using CPU or reducing problem size."
        )
        if verbose:
            warnings.warn(message)
        return True, message
    
    free_mem, total_mem = get_gpu_memory_info()
    message = (
        f"GPU computation feasible. Batch size: {batch_size}, "
        f"GPU memory: {free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total"
    )
    
    if verbose:
        print(message)
    
    return True, message

# main
if __name__ == "__main__":
    # python -m devil.gpu
    print(is_gpu_available())
    free_mem, total_mem = get_gpu_memory_info()
    print(f"GPU memory available: {free_mem/1e9:.1f}GB / {total_mem/1e9:.1f}GB")
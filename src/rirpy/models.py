# -*- coding: utf-8 -*-

from enum import StrEnum
from itertools import product
import logging
import math
from typing import Tuple, List, Optional

from numba import njit, prange, complex64, float32
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Models(StrEnum):
    FREQUENCY_DOMAIN = "freq"
    # FREQUENCY_DOMAIN_FREE_SPACE = "freq_free"
    # TIME_DOMAIN = "time"
    # TIME_DOMAIN_FREE_SPACE = "time_free"


@njit(parallel=True, fastmath=True, nopython=True)
def compute_image_contributions(
    l_vals: np.ndarray,
    m_vals: np.ndarray, 
    n_vals: np.ndarray,
    r_source: np.ndarray,
    r_receiver: np.ndarray,
    omega: np.ndarray,
    Lx: float,
    Ly: float,
    Lz: float,
    c: float,
    beta_wall: float,
    beta_surface: float
) -> np.ndarray:
    """
    Optimized computation of Green's function contributions.
    """
    # Initialize result array
    num_valid = len(l_vals)
    num_freqs = len(omega)
    result = np.zeros((num_freqs,), dtype=np.complex64)
    
    # Precompute k values and constants
    k_values = omega / c
    four_pi = 4.0 * math.pi
    
    # Pre-compute all 8 image configuration offsets (avoid recreating in inner loop)
    image_offsets = np.zeros((8, 3), dtype=np.float32)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                image_offsets[idx, 0] = 1 - 2 * i
                image_offsets[idx, 1] = 1 - 2 * j
                image_offsets[idx, 2] = 1 - 2 * k
                idx += 1
    
    # Process each valid grid point in parallel
    for idx in prange(num_valid):
        l, m, n = l_vals[idx], m_vals[idx], n_vals[idx]
        
        # Pre-compute translation vector (done once per grid point)
        r_trans_x = 2 * l * Lx
        r_trans_y = 2 * m * Ly
        r_trans_z = 2 * n * Lz
        
        # Process all 8 image configurations with a single loop
        for img_idx in range(8):
            # Extract the current image offset (memory locality)
            offset_x = image_offsets[img_idx, 0]
            offset_y = image_offsets[img_idx, 1]
            offset_z = image_offsets[img_idx, 2]
            
            # Binary values for i, j, k from offset calculation
            i = 0 if offset_x > 0 else 1
            j = 0 if offset_y > 0 else 1
            k = 0 if offset_z > 0 else 1
            
            # Compute image position (inline without creating temporary arrays)
            r_image_x = r_trans_x + offset_x * r_source[0]
            r_image_y = r_trans_y + offset_y * r_source[1]
            r_image_z = r_trans_z + offset_z * r_source[2]
            
            # Calculate distance to receiver (inline to avoid temporaries)
            r_diff_x = r_receiver[0] - r_image_x
            r_diff_y = r_receiver[1] - r_image_y
            r_diff_z = r_receiver[2] - r_image_z
            distance_squared = r_diff_x*r_diff_x + r_diff_y*r_diff_y + r_diff_z*r_diff_z
            distance = math.sqrt(distance_squared)
            
            # Calculate reflection coefficients
            wall_reflections = abs(l-i) + abs(l) + abs(m-j) + abs(m) + abs(n-k)
            surface_reflections = abs(n)
            
            # Use math.pow for better performance with integer exponents
            beta_factor = math.pow(beta_wall, wall_reflections) * math.pow(beta_surface, surface_reflections)
            
            # Skip if too close to avoid singularity
            if distance > 1e-10:
                # Precompute 1/(4πr) factor used for all frequencies
                amplitude_factor = beta_factor / (four_pi * distance)
                
                # Vectorized computation for all frequencies
                for f_idx in range(num_freqs):
                    # Complex phase calculation
                    phase_val = distance * k_values[f_idx]
                    # Use separate real and imaginary components for better performance
                    cos_val = math.cos(phase_val)
                    sin_val = math.sin(phase_val)
                    # Add contribution to the result using complex math
                    result[f_idx] += amplitude_factor * complex(cos_val, sin_val)
    
    return result


def impulse_response_freq_domain(
    r_source: List[float],
    r_receiver: List[float],
    omega: List[float],
    Lx: float,
    Ly: float,
    Lz: float,
    sound_speed: float,
    beta_wall: float,
    beta_surface: float,
    cutoff_time: float,
    num_threads: Optional[int] = None,
    batch_size: int = 1000,
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Compute the frequency-domain Green's function for a tank using Numba acceleration.

    Args:
        r_source: Source position [x, y, z]
        r_receiver: Receiver position [x, y, z]
        omega: Array of angular frequencies
        Lx, Ly, Lz: Tank dimensions
        c: Sound speed
        beta_wall: Wall reflection coefficient
        beta_surface: Surface reflection coefficient
        cutoff_time: Time cutoff for reflections
        num_threads: Number of threads for parallel execution (None = use all available)

    Returns:
        Complex array of Green's function values for each frequency
    """
    # Convert inputs to NumPy arrays
    r_source = np.array(r_source, dtype=np.float32)
    r_receiver = np.array(r_receiver, dtype=np.float32)
    omega = np.array(omega, dtype=np.float32)

    # Compute limits of sum from cutoff time
    cutoff_distance = cutoff_time * sound_speed
    l_max = np.ceil(cutoff_distance / (Lx * 2))
    m_max = np.ceil(cutoff_distance / (Ly * 2))
    n_max = np.ceil(cutoff_distance / (Lz * 2))
    max_diagonal_length = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Initialize Green's function
    g_tank = np.zeros(omega.shape, dtype=np.complex64)

    # Set thread count for Numba parallel execution if specified
    if num_threads is not None:
        import numba

        numba.set_num_threads(num_threads)
        print(f"Numba using {numba.get_num_threads()} threads")

    # Calculate total grid points
    n_size = int(2 * n_max + 1)
    m_size = int(2 * m_max + 1)
    l_size = int(2 * l_max + 1)
    total_grid_points = l_size * m_size * n_size

    # Process in chunks to manage memory
    # batch_size = min(5000, total_grid_points)  # Adjust based on available memory
    print(f"Processing {total_grid_points:,} grid points in batches of {batch_size}")

    # Process all grid points in chunks
    for chunk_start in tqdm(range(0, total_grid_points, batch_size), desc="Processing batches"):
        # Determine chunk size
        this_chunk_size = min(batch_size, total_grid_points - chunk_start)

        # Create array of indices
        indices = np.arange(chunk_start, chunk_start + this_chunk_size)

        # Calculate l, m, n indices from flattened index
        n_idx = indices % n_size
        m_idx = (indices // n_size) % m_size
        l_idx = indices // (n_size * m_size)

        # Convert indices to actual l, m, n values
        l_vals = l_idx - l_max
        m_vals = m_idx - m_max
        n_vals = n_idx - n_max

        # Create translation vectors
        r_translations = np.zeros((this_chunk_size, 3), dtype=np.float32)
        r_translations[:, 0] = 2 * l_vals * Lx
        r_translations[:, 1] = 2 * m_vals * Ly
        r_translations[:, 2] = 2 * n_vals * Lz

        # Check which translations are within cutoff distance
        translation_norms = np.sqrt(np.sum(r_translations**2, axis=1))
        valid_mask = translation_norms - 2 * max_diagonal_length <= cutoff_distance

        if not np.any(valid_mask):
            continue

        # Get valid translations and their l, m, n values
        valid_l = l_vals[valid_mask]
        valid_m = m_vals[valid_mask]
        valid_n = n_vals[valid_mask]

        if len(valid_l) > 0:
            # Process this chunk with Numba-accelerated function
            chunk_result = compute_image_contributions(
                valid_l,
                valid_m,
                valid_n,
                r_source,
                r_receiver,
                omega,
                Lx,
                Ly,
                Lz,
                sound_speed,
                beta_wall,
                beta_surface,
            )

            # Add contributions to total
            g_tank += chunk_result

    return g_tank

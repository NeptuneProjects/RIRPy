# -*- coding: utf-8 -*-

from enum import StrEnum
from itertools import product
import math
from typing import Tuple, List, Optional

from numba import njit, prange, complex64, float32
import numpy as np


class Models(StrEnum):
    FREQUENCY_DOMAIN = "freq"
    # FREQUENCY_DOMAIN_FREE_SPACE = "freq_free"
    # TIME_DOMAIN = "time"
    # TIME_DOMAIN_FREE_SPACE = "time_free"


@njit(parallel=True)
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
    beta_surface: float,
) -> np.ndarray:
    """
    Compute Green's function contributions for all valid image sources.
    This function is JIT-compiled with Numba for high performance.

    Args:
        l_vals, m_vals, n_vals: Arrays of grid indices for translations
        r_source, r_receiver: Source and receiver positions
        omega: Array of angular frequencies
        Lx, Ly, Lz: Tank dimensions
        c: Sound speed
        beta_wall, beta_surface: Reflection coefficients

    Returns:
        Contribution to the Green's function for this chunk of grid points
    """
    # Initialize result array
    num_valid = len(l_vals)
    num_freqs = len(omega)
    result = np.zeros((num_freqs,), dtype=np.complex64)

    # Precompute k values
    k_values = omega / c

    # Process each valid grid point
    for idx in prange(num_valid):
        l, m, n = l_vals[idx], m_vals[idx], n_vals[idx]

        # Create translation vector
        r_trans = np.array([2 * l * Lx, 2 * m * Ly, 2 * n * Lz], dtype=np.float32)

        # Process all 8 image configurations
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Compute image position
                    image_offset = np.array(
                        [1 - 2 * i, 1 - 2 * j, 1 - 2 * k], dtype=np.float32
                    )
                    r_image = r_trans + image_offset * r_source

                    # Calculate distance to receiver
                    r_diff = r_receiver - r_image
                    distance = np.sqrt(np.sum(r_diff**2))

                    # Calculate reflection coefficients
                    wall_reflections = (
                        abs(l - i) + abs(l) + abs(m - j) + abs(m) + abs(n - k)
                    )
                    surface_reflections = abs(n)
                    beta_factor = (
                        beta_wall**wall_reflections * beta_surface**surface_reflections
                    )

                    # Compute Green's function for all frequencies
                    if distance > 0:  # Avoid singularity
                        for f_idx in range(num_freqs):
                            phase = 1j * distance * k_values[f_idx]
                            g_free = np.exp(phase) / (4 * np.pi * distance)
                            result[f_idx] += g_free * beta_factor

    return result


def impulse_response_freq_domain(
    r_source: List[float],
    r_receiver: List[float],
    omega: List[float],
    Lx: float,
    Ly: float,
    Lz: float,
    c: float,
    beta_wall: float,
    beta_surface: float,
    cutoff_time: float,
    num_threads: Optional[int] = None,
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
    cutoff_distance = cutoff_time * c
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
    n_size = 2 * n_max + 1
    m_size = 2 * m_max + 1
    l_size = 2 * l_max + 1
    total_grid_points = l_size * m_size * n_size

    # Process in chunks to manage memory
    chunk_size = min(5000, total_grid_points)  # Adjust based on available memory
    print(f"Processing {total_grid_points} grid points in chunks of {chunk_size}")

    # Process all grid points in chunks
    for chunk_start in range(0, total_grid_points, chunk_size):
        # Determine chunk size
        this_chunk_size = min(chunk_size, total_grid_points - chunk_start)

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
                c,
                beta_wall,
                beta_surface,
            )

            # Add contributions to total
            g_tank += chunk_result

    return g_tank

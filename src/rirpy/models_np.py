# -*- coding: utf-8 -*-

from enum import StrEnum
from itertools import product
import logging

import numpy as np

logger = logging.getLogger(__name__)

class Models(StrEnum):
    FREQUENCY_DOMAIN = "freq"
    # FREQUENCY_DOMAIN_FREE_SPACE = "freq_free"
    # TIME_DOMAIN = "time"
    # TIME_DOMAIN_FREE_SPACE = "time_free"


def impulse_response_freq_domain(
    r_source: np.array,
    r_receiver: np.array,
    omega: np.array,
    Lx: float,
    Ly: float,
    Lz: float,
    c: float,
    beta_wall: float,
    beta_surface: float,
    cutoff_time: float,
    batch_size: int = 1000,
    device = None,
) -> np.array:

    # Compute limits of sum from cutoff time
    cutoff_distance = cutoff_time * c
    l_max = np.ceil(cutoff_distance / (Lx * 2))
    m_max = np.ceil(cutoff_distance / (Ly * 2))
    n_max = np.ceil(cutoff_distance / (Lz * 2))
    max_diagonal_length = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Initialize Green's function
    g_tank = np.zeros(omega.shape, dtype=np.complex128)

    # Create all translation vectors at once
    # This can be memory-intensive, so we process in chunks

    # Convert flat indices to l, m, n coordinates
    n_size = 2 * n_max + 1
    m_size = 2 * m_max + 1

    # Create a tensor with all grid points
    # We'll process this in chunks to avoid memory issues
    total_grid_points = int((2 * l_max + 1) * (2 * m_max + 1) * (2 * n_max + 1))
    logging.info(
        f"Total grid points to process: {total_grid_points} (l: {2 * l_max + 1}, m: {2 * m_max + 1}, n: {2 * n_max + 1})"
    )
    # Process all grid points in chunks
    for chunk_start in range(0, total_grid_points, batch_size):
        logging.info(
            f"Processing chunk starting at index {chunk_start} of size {batch_size}"
        )
        # Determine l, m, n for this chunk of grid points
        chunk_points = min(batch_size, total_grid_points - chunk_start)

        # Create array of indices
        indices = np.arange(chunk_start, chunk_start + chunk_points)

        # Calculate l, m, n indices from flattened index
        n_idx = indices % n_size
        m_idx = (indices // n_size) % m_size
        l_idx = indices // (n_size * m_size)

        # Convert indices to actual l, m, n values
        l_vals = l_idx - l_max
        m_vals = m_idx - m_max
        n_vals = n_idx - n_max

        # Create translation vectors
        r_translations = np.zeros((chunk_points, 3))
        r_translations[:, 0] = 2 * l_vals * Lx
        r_translations[:, 1] = 2 * m_vals * Ly
        r_translations[:, 2] = 2 * n_vals * Lz

        # Check which translations are within cutoff distance
        translation_norms = np.linalg.norm(r_translations, axis=1)
        valid_mask = translation_norms - 2 * max_diagonal_length <= cutoff_distance

        if not np.any(valid_mask):
            continue

        # Get valid translations and their l, m, n values
        valid_translations = r_translations[valid_mask]
        valid_l = l_vals[valid_mask]
        valid_m = m_vals[valid_mask]
        valid_n = n_vals[valid_mask]

        # Process all 8 image configurations for each valid translation at once
        for i, j, k in product([0, 1], repeat=3):
            # Compute all image positions
            image_offset = np.array(
                [1 - 2 * i, 1 - 2 * j, 1 - 2 * k]
            )
            r_images = valid_translations + image_offset * r_source

            # Calculate distances to receiver
            r_diff = np.expand_dims(r_receiver, 0) - r_images
            distances = np.expand_dims(np.linalg.norm(r_diff, axis=1), 1)  # [num_valid, 1]

            # Compute Green's function for all frequencies
            k_values = omega / c
            phase = 1j * distances * np.expand_dims(k_values, 0)
            g_free = np.exp(phase) / (4 * np.pi * distances)

            # Calculate reflection coefficients vectorized
            wall_reflections = (
                np.abs(valid_l - i)
                + np.abs(valid_l)
                + np.abs(valid_m - j)
                + np.abs(valid_m)
                + np.abs(valid_n - k)
            )
            surface_reflections = np.abs(valid_n)
            
            beta_factors = np.pow(beta_wall, wall_reflections) * np.pow(
                beta_surface, surface_reflections
            )

            # Apply reflection coefficients and sum
            contribution = g_free * np.expand_dims(beta_factors, 1)
            g_tank += np.sum(contribution, axis=0)

    return g_tank

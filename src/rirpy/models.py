# -*- coding: utf-8 -*-

from enum import StrEnum
from itertools import product
import logging
import math

import torch
from tqdm import tqdm

class Models(StrEnum):
    FREQUENCY_DOMAIN = "freq"
    # FREQUENCY_DOMAIN_FREE_SPACE = "freq_free"
    # TIME_DOMAIN = "time"
    # TIME_DOMAIN_FREE_SPACE = "time_free"


def impulse_response_freq_domain(
    r_source: torch.Tensor,
    r_receiver: torch.Tensor,
    omega: torch.Tensor,
    Lx: float,
    Ly: float,
    Lz: float,
    c: float,
    beta_wall: float,
    beta_surface: float,
    cutoff_time: float,
    batch_size: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Highly optimized version with only a single main loop and maximum vectorization.
    """

    # Compute limits of sum from cutoff time
    cutoff_distance = cutoff_time * c
    l_max = math.ceil(cutoff_distance / (Lx * 2))
    m_max = math.ceil(cutoff_distance / (Ly * 2))
    n_max = math.ceil(cutoff_distance / (Lz * 2))
    max_diagonal_length = math.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Initialize Green's function
    g_tank = torch.zeros(omega.shape, dtype=torch.complex64, device=device)

    # Create all translation vectors at once
    # This can be memory-intensive, so we process in chunks

    # Convert flat indices to l, m, n coordinates
    n_size = 2 * n_max + 1
    m_size = 2 * m_max + 1

    # Create a tensor with all grid points
    # We'll process this in chunks to avoid memory issues
    total_grid_points = (2 * l_max + 1) * (2 * m_max + 1) * (2 * n_max + 1)
    logging.info(
        f"Total grid points to process: {total_grid_points} (l: {2 * l_max + 1}, m: {2 * m_max + 1}, n: {2 * n_max + 1})"
    )
    # Process all grid points in chunks
    for chunk_start in tqdm(range(0, total_grid_points, batch_size), desc="Processing chunks"):
        # logging.info(
        #     f"Processing chunk starting at index {chunk_start} of size {batch_size}"
        # )
        # Determine l, m, n for this chunk of grid points
        chunk_points = min(batch_size, total_grid_points - chunk_start)

        # Create array of indices
        indices = torch.arange(chunk_start, chunk_start + chunk_points, device=device)

        # Calculate l, m, n indices from flattened index
        n_idx = indices % n_size
        m_idx = (indices // n_size) % m_size
        l_idx = indices // (n_size * m_size)

        # Convert indices to actual l, m, n values
        l_vals = l_idx - l_max
        m_vals = m_idx - m_max
        n_vals = n_idx - n_max

        # Create translation vectors
        r_translations = torch.zeros((chunk_points, 3), device=device)
        r_translations[:, 0] = 2 * l_vals * Lx
        r_translations[:, 1] = 2 * m_vals * Ly
        r_translations[:, 2] = 2 * n_vals * Lz

        # Check which translations are within cutoff distance
        translation_norms = torch.norm(r_translations, dim=1)
        valid_mask = translation_norms - 2 * max_diagonal_length <= cutoff_distance

        if not torch.any(valid_mask):
            continue

        # Get valid translations and their l, m, n values
        valid_translations = r_translations[valid_mask]
        valid_l = l_vals[valid_mask]
        valid_m = m_vals[valid_mask]
        valid_n = n_vals[valid_mask]

        # Process all 8 image configurations for each valid translation at once
        for i, j, k in product([0, 1], repeat=3):
            # Compute all image positions
            image_offset = torch.tensor(
                [1 - 2 * i, 1 - 2 * j, 1 - 2 * k], device=device
            )
            r_images = valid_translations + image_offset * r_source

            # Calculate distances to receiver
            r_diff = r_receiver.unsqueeze(0) - r_images
            distances = torch.norm(r_diff, dim=1).unsqueeze(1)  # [num_valid, 1]

            # Compute Green's function for all frequencies
            k_values = omega / c
            phase = 1j * distances * k_values.unsqueeze(0)
            g_free = torch.exp(phase) / (4 * math.pi * distances)

            # Calculate reflection coefficients vectorized
            wall_reflections = (
                torch.abs(valid_l - i)
                + torch.abs(valid_l)
                + torch.abs(valid_m - j)
                + torch.abs(valid_m)
                + torch.abs(valid_n - k)
            )
            surface_reflections = torch.abs(valid_n)

            beta_factors = torch.pow(beta_wall, wall_reflections) * torch.pow(
                beta_surface, surface_reflections
            )

            # Apply reflection coefficients and sum
            contribution = g_free * beta_factors.unsqueeze(1)
            g_tank += torch.sum(contribution, dim=0)

    return g_tank

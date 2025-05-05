# -*- coding: utf-8 -*-

import logging
import math

import numpy as np
import numpy.typing as npt
import numba

logger = logging.getLogger(__name__)


def propagate_signal(
    y_source: npt.NDArray[np.float64],
    fs: float,
    r_source: npt.NDArray[np.float64],
    r_receiver: npt.NDArray[np.float64],
    Lx: float,
    Ly: float,
    Lz: float,
    c: float,
    beta_wall: float,
    beta_surface: float,
    cutoff_time: float,
    num_threads: int = 4,
) -> npt.NDArray[np.float64]:
    """
    Function to compute received signal in a tank based on reflection
    from walls computed in the time domain.

    Parameters:
    -----------
    y_source : ndarray
        Source time series, can be multiple column vectors [NxM]
    t : ndarray
        Time vector (s) [Nx1]
    c : float
        Speed of sound (m/s)
    image_distances : ndarray
        Array specifying distances of images to be considered (m) [Lx1]
    image_coefficients : ndarray
        Array specifying the combined reflection coefficient for each image [Lx1]

    Returns:
    --------
    y_receiver : ndarray
        Signal at the receiver location [NxM]
    """

    numba.set_num_threads(num_threads)
    logging.info(f"Numba is using {numba.get_num_threads()} threads.")

    dt = 1 / fs

    image_distances, image_coefficients = (
        compute_source_image_distances_and_reflection_coefficients(
            r_source, r_receiver, Lx, Ly, Lz, c, beta_wall, beta_surface, cutoff_time
        )
    )

    # Initialize output with same shape as input
    y_receiver = np.zeros_like(y_source, dtype=np.float64)

    # Compute time step
    # dt = t[1] - t[0]

    # Compute time offset for each image
    t_offset = image_distances / c
    t_offset_ind = np.round(t_offset / dt).astype(np.int64)

    # Find images that have arrivals within the signal duration
    num_samples = y_source.shape[0]

    # Process each relevant image
    for i in range(len(image_distances)):
        if t_offset_ind[i] >= num_samples:
            break

        shift_amount = t_offset_ind[i]

        if len(y_source.shape) == 1:  # 1D case
            y_image = np.zeros_like(y_source)
            if shift_amount > 0:
                y_image[shift_amount:] = y_source[: num_samples - shift_amount]
            else:
                y_image[:] = y_source[:]

            # Apply coefficient and distance attenuation
            y_image = y_image * (image_coefficients[i] / image_distances[i])

            # Zero out values before the arrival time
            y_image[:shift_amount] = 0

            # Add to result
            y_receiver += y_image

        else:  # 2D case - multiple columns
            for col in range(y_source.shape[1]):
                y_image = np.zeros_like(y_source[:, col])
                if shift_amount > 0:
                    y_image[shift_amount:] = y_source[: num_samples - shift_amount, col]
                else:
                    y_image[:] = y_source[:, col]

                # Apply coefficient and distance attenuation
                y_image = y_image * (image_coefficients[i] / image_distances[i])

                # Zero out values before the arrival time
                y_image[:shift_amount] = 0

                # Add to result
                y_receiver[:, col] += y_image

    return y_receiver


@numba.njit
def process_block(
    l: int,
    m_max: int,
    n_max: int,
    r_source: npt.NDArray[np.float64],
    r_receiver: npt.NDArray[np.float64],
    Lx: float,
    Ly: float,
    Lz: float,
    beta_wall: float,
    beta_surface: float,
    cutoff_distance: float,
    max_diagonal_length: float,
) -> tuple[list[float], list[float]]:
    """Helper function to process one block of the calculation for parallelization"""
    distances_list = []
    coefficients_list = []

    for m in range(-m_max, m_max + 1):
        for n in range(-n_max, n_max + 1):
            # Compute lattice displacement vector
            r_translation = np.array([2.0 * l * Lx, 2.0 * m * Ly, 2.0 * n * Lz])

            if (
                np.sqrt(np.sum(r_translation**2)) - 2 * max_diagonal_length
                <= cutoff_distance
            ):
                # Iterate over the 8 source images within this block
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            # Compute the source image separation vector
                            image_factors = np.array(
                                [1.0 - 2.0 * i, 1.0 - 2.0 * j, 1.0 - 2.0 * k]
                            )
                            r_image = r_translation + image_factors * r_source

                            # Compute distance
                            R = r_image - r_receiver
                            distance = np.sqrt(np.sum(R**2))

                            # Compute reflection coefficient product
                            b = (
                                beta_wall
                                ** (
                                    abs(l - i)
                                    + abs(l)
                                    + abs(m - j)
                                    + abs(m)
                                    + abs(n - k)
                                )
                            ) * (beta_surface ** abs(n))

                            distances_list.append(distance)
                            coefficients_list.append(b)

    return distances_list, coefficients_list


@numba.njit(parallel=True, fastmath=True)
def compute_source_image_distances_and_reflection_coefficients(
    r_source: npt.NDArray[np.float64],
    r_receiver: npt.NDArray[np.float64],
    Lx: float,
    Ly: float,
    Lz: float,
    c: float,
    beta_wall: float,
    beta_surface: float,
    cutoff_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distances and reflection coefficients for source images in a tank.

    Parameters:
    -----------
    r_source : ndarray
        Vector position of the source (m) [3x1]
    r_receiver : ndarray
        Vector position of the receiver (m) [3x1]
    Lx, Ly, Lz : float
        Dimensions of the tank in each coordinate (m)
    c : float
        Sound speed (m/s)
    beta_wall : float
        Reflection coefficient for the 5 non-surface walls of the tank
    beta_surface : float
        Reflection coefficient for the water surface
    cutoff_time : float
        Time over which to sum reflected paths (s)

    Returns:
    --------
    distances : ndarray
        Vector containing the apparent distance of each source image from the receiver
    coefficients : ndarray
        Vector containing the product of reflection coefficients for each source image
    """
    # Compute limits of sum from cutoff time
    cutoff_distance = cutoff_time * c
    l_max = math.ceil(cutoff_distance / (Lx * 2))
    m_max = math.ceil(cutoff_distance / (Ly * 2))
    n_max = math.ceil(cutoff_distance / (Lz * 2))
    max_diagonal_length = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Since we can't dynamically grow arrays in parallel regions with Numba,
    # we'll collect results from each parallel task and combine them later
    all_distances = []
    all_coefficients = []

    # Process blocks in parallel
    for l in numba.prange(-l_max, l_max + 1):
        distances_local, coefficients_local = process_block(
            l,
            m_max,
            n_max,
            r_source,
            r_receiver,
            Lx,
            Ly,
            Lz,
            beta_wall,
            beta_surface,
            cutoff_distance,
            max_diagonal_length,
        )

        # Thread-local lists to be combined later
        all_distances.append(distances_local)
        all_coefficients.append(coefficients_local)

    # Combine results from all threads
    distances_unsorted = []
    coefficients_unsorted = []

    for thread_distances, thread_coefficients in zip(all_distances, all_coefficients):
        distances_unsorted.extend(thread_distances)
        coefficients_unsorted.extend(thread_coefficients)

    # # Convert to numpy arrays
    # if len(distances_unsorted) == 0:
    #     # Return empty arrays if no images found
    #     return np.nan, np.nan

    distances_unsorted = np.array(distances_unsorted)
    coefficients_unsorted = np.array(coefficients_unsorted)

    # Sort arrays by distance
    sort_indices = np.argsort(distances_unsorted)
    distances = distances_unsorted[sort_indices]
    coefficients = coefficients_unsorted[sort_indices]

    return distances, coefficients

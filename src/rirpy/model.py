# -*- coding: utf-8 -*-

import logging
import math

import numpy as np
import numpy.typing as npt
import numba

logger = logging.getLogger(__name__)


def propagate_signal(
    source_signal: npt.NDArray[np.float64],
    sampling_rate: float,
    source_position: npt.NDArray[np.float64],
    receiver_position: npt.NDArray[np.float64],
    length_x: float,
    length_y: float,
    length_z: float,
    sound_speed: float,
    refl_coeff_wall: float,
    refl_coeff_ceil: float,
    cutoff_time: float,
    num_threads: int = 4,
) -> npt.NDArray[np.float64]:
    """
    Computes the received signal in a tank based on reflections from walls
    in the time domain.

    Args:
        source_signal: Source time series, can be multiple column vectors [NxM].
        sampling_rate: Sampling frequency (Hz).
        source_position: Vector position of the source (m) [3x1].
        receiver_position: Vector position of the receiver (m) [3x1].
        length_x: Length of the tank in the x-dimension (m).
        length_y: Length of the tank in the y-dimension (m).
        length_z: Length of the tank in the z-dimension (m).
        sound_speed: Speed of sound (m/s).
        refl_coeff_wall: Reflection coefficient for the 5 non-surface walls of the tank.
        refl_coeff_ceil: Reflection coefficient for the water surface.
        cutoff_time: Time over which to sum reflected paths (s).
        num_threads: Number of threads for parallel processing. Defaults to 4.

    Returns:
        Signal at the receiver location [NxM].
    """
    validate_geometry(
        source_position,
        receiver_position,
        length_x,
        length_y,
        length_z,
    )

    numba.set_num_threads(num_threads)
    logging.info(f"Numba is using {numba.get_num_threads()} threads.")

    dt = 1 / sampling_rate

    image_distances, image_coefficients = (
        compute_source_image_distances_and_reflection_coefficients(
            source_position,
            receiver_position,
            length_x,
            length_y,
            length_z,
            sound_speed,
            refl_coeff_wall,
            refl_coeff_ceil,
            cutoff_time,
        )
    )
    logging.info(f"{len(image_distances):,} reflections computed.")

    # Initialize output with same shape as input
    y_receiver = np.zeros_like(source_signal, dtype=np.float64)

    # Compute time offset for each image
    t_offset = image_distances / sound_speed
    t_offset_ind = np.round(t_offset / dt).astype(np.int64)

    # Find images that have arrivals within the signal duration
    num_samples = source_signal.shape[0]

    # Process each relevant image
    for i in range(len(image_distances)):
        if t_offset_ind[i] >= num_samples:
            break

        # Calculate the shift amount based on the time offset
        shift_amount = t_offset_ind[i]
        # Apply distance attenuation (assumes cylindrical spreading)
        coefficients = image_coefficients[i] / image_distances[i]

        if len(source_signal.shape) == 1:  # 1D case
            source_subset = (
                source_signal[: num_samples - shift_amount]
                if shift_amount > 0
                else source_signal
            )
            y_image = np.zeros_like(source_signal)
            y_image[shift_amount : shift_amount + len(source_subset)] = source_subset
        else:  # 2D case - multiple columns
            y_image = np.zeros_like(source_signal)
            for col in range(source_signal.shape[1]):
                source_subset = (
                    source_signal[: num_samples - shift_amount, col]
                    if shift_amount > 0
                    else source_signal[:, col]
                )
                y_image[shift_amount : shift_amount + len(source_subset), col] = (
                    source_subset
                )

        # Apply coefficient and distance attenuation
        y_image *= coefficients

        # Zero out values before the arrival time
        y_image[:shift_amount] = 0

        # Add to result
        y_receiver += y_image

    return y_receiver


@numba.njit
def process_block(
    i_current: int,
    j_max: int,
    k_max: int,
    source_position: npt.NDArray[np.float64],
    receiver_position: npt.NDArray[np.float64],
    length_x: float,
    length_y: float,
    length_z: float,
    refl_coeff_wall: float,
    refl_coeff_ceil: float,
    cutoff_distance: float,
    max_diagonal_length: float,
) -> tuple[list[float], list[float]]:
    """
    Processes one block of the calculation for parallelization.

    Args:
        i_current: Current block index in the x-dimension.
        j_max: Maximum block index in the y-dimension.
        k_max: Maximum block index in the z-dimension.
        source_position: Vector position of the source (m) [3x1].
        receiver_position: Vector position of the receiver (m) [3x1].
        length_x: Length of the tank in the x-dimension (m).
        length_y: Length of the tank in the y-dimension (m).
        length_z: Length of the tank in the z-dimension (m).
        refl_coeff_wall: Reflection coefficient for the 5 non-surface walls of the tank.
        refl_coeff_ceil: Reflection coefficient for the water surface.
        cutoff_distance: Maximum distance to consider for reflections (m).
        max_diagonal_length: Maximum diagonal length of the tank (m).

    Returns:
        Distances and reflection coefficients for the block.
    """
    distances_list = []
    coefficients_list = []

    for j in range(-j_max, j_max + 1):
        for k in range(-k_max, k_max + 1):
            # Compute lattice displacement vector
            r_translation = np.array(
                [2.0 * i_current * length_x, 2.0 * j * length_y, 2.0 * k * length_z]
            )

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
                            r_image = r_translation + image_factors * source_position

                            # Compute distance
                            R = r_image - receiver_position
                            distance = np.sqrt(np.sum(R**2))

                            # Compute reflection coefficient product
                            b = (
                                refl_coeff_wall
                                ** (
                                    abs(i_current - i)
                                    + abs(i_current)
                                    + abs(j - j)
                                    + abs(j)
                                    + abs(k - k)
                                )
                            ) * (refl_coeff_ceil ** abs(k))

                            distances_list.append(distance)
                            coefficients_list.append(b)

    return distances_list, coefficients_list


@numba.njit(parallel=True)
def compute_source_image_distances_and_reflection_coefficients(
    source_location: npt.NDArray[np.float64],
    receiver_location: npt.NDArray[np.float64],
    length_x: float,
    length_y: float,
    length_z: float,
    sound_speed: float,
    refl_coeff_wall: float,
    refl_coeff_ceil: float,
    cutoff_time: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculates the distances and reflection coefficients for source images in a tank.

    Args:
        source_location: Vector position of the source (m) [3x1].
        receiver_location: Vector position of the receiver (m) [3x1].
        length_x: Length of the tank in the x-dimension (m).
        length_y: Length of the tank in the y-dimension (m).
        length_z: Length of the tank in the z-dimension (m).
        sound_speed: Speed of sound (m/s).
        refl_coeff_wall: Reflection coefficient for the 5 non-surface walls of the tank.
        refl_coeff_ceil: Reflection coefficient for the water surface.
        cutoff_time: Time over which to sum reflected paths (s).

    Returns:
        Distances and reflection coefficients for source images.
    """
    # Compute limits of sum from cutoff time
    cutoff_distance = cutoff_time * sound_speed
    i_max = math.ceil(cutoff_distance / (length_x * 2))
    j_max = math.ceil(cutoff_distance / (length_y * 2))
    k_max = math.ceil(cutoff_distance / (length_z * 2))
    max_diagonal_length = np.sqrt(length_x**2 + length_y**2 + length_z**2)

    # Since we can't dynamically grow arrays in parallel regions with Numba,
    # we'll collect results from each parallel task and combine them later
    all_distances = []
    all_coefficients = []

    # Process blocks in parallel
    for l in numba.prange(-i_max, i_max + 1):
        distances_local, coefficients_local = process_block(
            l,
            j_max,
            k_max,
            source_location,
            receiver_location,
            length_x,
            length_y,
            length_z,
            refl_coeff_wall,
            refl_coeff_ceil,
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

    distances_unsorted = np.array(distances_unsorted)
    coefficients_unsorted = np.array(coefficients_unsorted)

    # Sort arrays by distance
    sort_indices = np.argsort(distances_unsorted)
    distances = distances_unsorted[sort_indices]
    coefficients = coefficients_unsorted[sort_indices]

    return distances, coefficients


def validate_geometry(
    source_location: npt.NDArray[np.float64],
    receiver_location: npt.NDArray[np.float64],
    length_x: float,
    length_y: float,
    length_z: float,
) -> None:
    """Validate geometry of model paramterization to ensure source and receiver
    are within the bounded volume.

    Args:
        source_location: Vector position of the source (m) [3x1].
        receiver_location: Vector position of the receiver (m) [3x1].
        length_x: Length of the tank in the x-dimension (m).
        length_y: Length of the tank in the y-dimension (m).
        length_z: Length of the tank in the z-dimension (m).
    Raises:
        ValueError: If source or receiver positions are outside the tank dimensions.
    """
    # Ensure positions are valid
    if len(source_location) != 3 or len(receiver_location) != 3:
        raise ValueError("Source and receiver positions must be 3D vectors")

    # Ensure room dimensions are valid
    if length_x <= 0 or length_y <= 0 or length_z <= 0:
        raise ValueError("Tank dimensions must be positive")

    # Check if source and receiver are within the tank dimensions
    if (
        not (0 <= source_location[0] <= length_x)
        or not (0 <= source_location[1] <= length_y)
        or not (0 <= source_location[2] <= length_z)
    ):
        raise ValueError("Source position is outside the tank dimensions")
    if (
        not (0 <= receiver_location[0] <= length_x)
        or not (0 <= receiver_location[1] <= length_y)
        or not (0 <= receiver_location[2] <= length_z)
    ):
        raise ValueError("Receiver position is outside the tank dimensions")

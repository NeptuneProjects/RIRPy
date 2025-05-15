from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
import functools
import logging
import math
import time
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import numba

T = TypeVar("T")
T_GreensResult = npt.NDArray[np.complex128]
T_ImagesResult = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


class MethodChoice(StrEnum):
    BOTH = "both"
    GREENS = "greens"
    IMAGES = "images"


@dataclass
class Environment:
    """Class to hold the environment parameters for the model.

    Attributes:
        space_dimensions: Length of the room in the x, y, and z dimensions (m) [3x1].
        sound_speed: Speed of sound in the medium (m/s).
        refl_coeff_wall: Wall (& floor) reflection coefficient.
        refl_coeff_ceil: Ceiling (surface) reflection coefficient.
    """

    space_dimensions: Sequence[float]
    sound_speed: float
    refl_coeff_wall: float
    refl_coeff_ceil: float


def _log_execution_time(
    message: str = "Function",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that logs the execution time of a function.

    Args:
        message: Description of the function for logging

    Returns:
        Decorated function that logs timing information
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logging.info(message)
            time_start = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - time_start
            logging.info(f"Completed in {execution_time:.6f} seconds.")
            return result

        return wrapper

    return decorator


def convert_frequency_to_angular(
    frequency: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Convert frequency in Hz to angular frequency in radians.

    Args:
        frequency: Frequency in Hz (float or array).

    Returns:
        Angular frequency in radians.
    """
    return 2 * np.pi * frequency


@numba.njit
def count_reflections(
    i_max: int,
    j_max: int,
    k_max: int,
    length_x: float,
    length_y: float,
    length_z: float,
    cutoff_distance: float,
    max_diagonal_length: float,
) -> int:
    """Count the total number of valid reflections.

    Args:
        i_max: Maximum number of reflections in the x-dimension.
        j_max: Maximum number of reflections in the y-dimension.
        k_max: Maximum number of reflections in the z-dimension.
        length_x: Length of room in the x-dimension (m).
        length_y: Length of room in the y-dimension (m).
        length_z: Length of room in the z-dimension (m).
        cutoff_distance: Cutoff distance for reflections (m).
        max_diagonal_length: Maximum diagonal length of the room (m).

    Returns:
        Total number of valid reflections.
    """
    count = 0
    for i_current in range(-i_max, i_max + 1):
        for j in range(-j_max, j_max + 1):
            for k in range(-k_max, k_max + 1):
                r_translation = np.array(
                    [2.0 * i_current * length_x, 2.0 * j * length_y, 2.0 * k * length_z]
                )

                if (
                    np.sqrt(np.sum(r_translation**2)) - 2 * max_diagonal_length
                    <= cutoff_distance
                ):
                    # 8 images per valid block
                    count += 8
    return count


@_log_execution_time("Computing distances and reflection coefficients.")
@numba.njit
def distances_and_amplitudes(
    source_location: npt.NDArray[np.float64],
    receiver_location: npt.NDArray[np.float64],
    space_dimensions: npt.NDArray[np.float64],
    sound_speed: float,
    refl_coeff_wall: float,
    refl_coeff_ceil: float,
    cutoff_time: float,
) -> T_ImagesResult:
    """
    Calculates the distances and reflection coefficients for source images in a room.

    Args:
        source_location: Vector position of the source (m) [3x1].
        receiver_location: Vector position of the receiver (m) [3x1].
        space_dimensions: Length of the room in the x, y, and z dimensions (m) [3x1].
        sound_speed: Speed of sound (m/s).
        refl_coeff_wall: Reflection coefficient for the 5 non-surface walls of the room.
        refl_coeff_ceil: Reflection coefficient for the water surface.
        cutoff_time: Time over which to sum reflected paths (s).

    Returns:
        Distances and reflection coefficients for source images.
    """
    # Lengths must be extracted as follows rather than using tuple unpacking
    # to avoid issues with Numba's type inference
    length_x = space_dimensions[0]
    length_y = space_dimensions[1]
    length_z = space_dimensions[2]

    # Compute limits of sum from cutoff time
    cutoff_distance = cutoff_time * sound_speed
    i_max = math.ceil(cutoff_distance / (length_x * 2))
    j_max = math.ceil(cutoff_distance / (length_y * 2))
    k_max = math.ceil(cutoff_distance / (length_z * 2))
    max_diagonal_length = np.sqrt(length_x**2 + length_y**2 + length_z**2)

    # First, count the total number of valid reflections to allocate arrays
    total_reflections = count_reflections(
        i_max,
        j_max,
        k_max,
        length_x,
        length_y,
        length_z,
        cutoff_distance,
        max_diagonal_length,
    )

    # Pre-allocate arrays
    distances = np.zeros(total_reflections, dtype=np.float64)
    coefficients = np.zeros(total_reflections, dtype=np.float64)

    # Fill arrays
    idx = 0
    for i_current in numba.prange(-i_max, i_max + 1):
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
                    for l in range(2):
                        for m in range(2):
                            for n in range(2):
                                # Compute the source image separation vector
                                image_factors = np.array(
                                    [1.0 - 2.0 * l, 1.0 - 2.0 * m, 1.0 - 2.0 * n]
                                )
                                r_image = (
                                    r_translation + image_factors * source_location
                                )

                                # Compute distance
                                R = r_image - receiver_location
                                distance = np.sqrt(np.sum(R**2))

                                # Compute reflection coefficient product
                                b = (
                                    refl_coeff_wall
                                    ** (
                                        abs(i_current - l)
                                        + abs(i_current)
                                        + abs(j - m)
                                        + abs(j)
                                        + abs(k - n)
                                    )
                                ) * (refl_coeff_ceil ** abs(k))

                                distances[idx] = distance
                                coefficients[idx] = b
                                idx += 1

    # Sort arrays by distance
    sort_indices = np.argsort(distances)
    distances = distances[sort_indices]
    coefficients = coefficients[sort_indices]

    # Normalize coefficients by distance (cylindrical spreading)
    normalized_coefficients = coefficients / distances

    return distances, normalized_coefficients


@numba.jit
def free_field_greens_function(
    source_location: npt.NDArray[np.float64],
    receiver_location: npt.NDArray[np.float64],
    sound_speed: float,
    omega: npt.NDArray[np.float64],
) -> T_GreensResult:
    """Compute the free-field Green's function in frequency domain for a
    point source and receiver at positions r1 and r2.

    Args:
        source_location: Vector position of the source (m) [array of size 3]
        receiver_location: Vector position of the receiver (m) [array of size 3]
        sound_speed: Sound speed (m/s)
        omega: Angular frequencies at which to compute g (radians) [array of size N]

    Returns:
        greens_func_free: Green's function for propagation between the source
            and receiver positions as a function of omega.
    """
    N = omega.size
    greens_func_free = np.zeros(N, dtype=np.complex128)

    distance = np.linalg.norm(receiver_location - source_location)

    if N % 2 == 0:
        midpoint = N // 2
        greens_func_free[:midpoint] = (
            np.exp(-1j * distance * omega[:midpoint] / sound_speed) / distance
        )
        greens_func_free[midpoint:] = np.conj(greens_func_free[:midpoint][::-1])
    else:
        midpoint = N // 2 + 1
        greens_func_free[:midpoint] = (
            np.exp(-1j * distance * omega[:midpoint] / sound_speed) / distance
        )
        greens_func_free[midpoint:] = np.conj(greens_func_free[: midpoint - 1][::-1])

    return greens_func_free


@_log_execution_time("Computing Green's function.")
@numba.jit(parallel=True)
def greens_function(
    source_location: npt.NDArray[np.float64],
    receiver_location: npt.NDArray[np.float64],
    space_dimensions: npt.NDArray[np.float64],
    sound_speed: float,
    refl_coeff_wall: float,
    refl_coeff_ceil: float,
    cutoff_time: float,
    omega: npt.NDArray[np.float64],
) -> T_GreensResult:
    """Compute the Green's function for a rectilinear space using method of images.

    Args:
        source_location: Vector position of the source (m) [3x1].
        receiver_location: Vector position of the receiver (m) [3x1].
        space_dimensions: Length of the room in the x, y, and z dimensions (m) [3x1].
        sound_speed: Speed of sound in the
            medium (m/s).
        refl_coeff_wall: Wall (& floor) reflection coefficient.
        refl_coeff_ceil: Ceiling (surface) reflection coefficient.
        cutoff_time: Cutoff time for reflections (s).
        omega: Angular frequencies at which to compute g (radians) [array of size N].
    Returns:
        greens_func: Green's function for propagation between the source
            and receiver positions as a function of frequency.
    """
    # Lengths must be extracted as follows rather than using tuple unpacking
    # to avoid issues with Numba's type inference
    length_x = space_dimensions[0]
    length_y = space_dimensions[1]
    length_z = space_dimensions[2]

    # Compute limits of sum from cutoff time
    cutoff_distance = cutoff_time * sound_speed
    l_max = int(np.ceil(cutoff_distance / (length_x * 2)))
    m_max = int(np.ceil(cutoff_distance / (length_y * 2)))
    n_max = int(np.ceil(cutoff_distance / (length_z * 2)))
    max_diagonal_length = np.sqrt(length_x**2 + length_y**2 + length_z**2)

    greens_func = np.zeros(omega.size, dtype=np.complex128)

    # Iterate over lattice displacement vectors using prange for parallel loops
    for l in numba.prange(-l_max, l_max + 1):
        # for l in range(-l_max, l_max + 1):
        for m in range(-m_max, m_max + 1):
            for n in range(-n_max, n_max + 1):
                # Compute lattice displacement vector and check cutoff distance
                r_translation = 2 * np.array([l * length_x, m * length_y, n * length_z])
                if (
                    np.linalg.norm(r_translation) - 2 * max_diagonal_length
                    <= cutoff_distance
                ):
                    # Iterate over the 8 source images within this block
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                # Compute the source image separation vector
                                sign = np.array([1 - 2 * i, 1 - 2 * j, 1 - 2 * k])
                                r_image = r_translation + sign * source_location

                                # Compute free-field Green's function
                                g_free = free_field_greens_function(
                                    r_image, receiver_location, sound_speed, omega
                                )

                                # Multiplength_y by reflection coefficients and add to the sum
                                reflection_coefs = refl_coeff_wall ** (
                                    abs(l - i)
                                    + abs(l)
                                    + abs(m - j)
                                    + abs(m)
                                    + abs(n - k)
                                ) * refl_coeff_ceil ** abs(n)
                                greens_func += reflection_coefs * g_free

    return greens_func


def run(
    source_location: Sequence[float] | npt.NDArray[np.float64],
    receiver_location: Sequence[float] | npt.NDArray[np.float64],
    environment: Environment,
    cutoff_time: float,
    frequency: float | Sequence[float] | npt.NDArray[np.float64],
    method: MethodChoice | str = MethodChoice.BOTH,
    num_threads: int = 4,
) -> T_ImagesResult | T_GreensResult | tuple[T_ImagesResult, T_GreensResult]:
    """Compute the Green's function for a rectilinear space using method of images.

    Args:
        source_location: Vector position of the source (m) [3x1].
        receiver_location: Vector position of the receiver (m) [3x1].
        environment: Environment parameters.
        cutoff_time: Cutoff time for reflections (s).
        frequency: Frequency in Hz (float or array).
        method: Method to use for computation. Options are "image", "greens",
            or "both". "image" computes distances and coefficients, "greens"
            computes the Green's function, and "both" computes both.
        num_threads: Number of threads to use for parallel computation.

    Returns:
        greens_func: Green's function for propagation between the source
            and receiver positions as a function of frequency.
    """
    source_location = np.asarray(source_location, dtype=np.float64)
    receiver_location = np.asarray(receiver_location, dtype=np.float64)
    space_dimensions = np.asarray(environment.space_dimensions, dtype=np.float64)
    omega = convert_frequency_to_angular(np.asarray(frequency, dtype=np.float64))
    validate_geometry(
        source_location=source_location,
        receiver_location=receiver_location,
        space_dimensions=space_dimensions,
    )

    numba.set_num_threads(num_threads)
    logging.info(f"Numba is using {numba.get_num_threads()} threads.")

    kwargs = {
        "source_location": source_location,
        "receiver_location": receiver_location,
        "space_dimensions": space_dimensions,
        "sound_speed": environment.sound_speed,
        "refl_coeff_wall": environment.refl_coeff_wall,
        "refl_coeff_ceil": environment.refl_coeff_ceil,
        "cutoff_time": cutoff_time,
    }

    if method == MethodChoice.IMAGES:
        return distances_and_amplitudes(**kwargs)
    if method == MethodChoice.GREENS:
        return greens_function(**(kwargs | {"omega": omega}))
    if method == MethodChoice.BOTH:
        return distances_and_amplitudes(**kwargs), greens_function(
            **(kwargs | {"omega": omega})
        )
    raise ValueError(
        f"Invalid method choice: {method}. Must be one of `MethodChoice.IMAGES`."
    )


def validate_geometry(
    source_location: npt.NDArray[np.float64],
    receiver_location: npt.NDArray[np.float64],
    space_dimensions: npt.NDArray[np.float64],
) -> None:
    """Validate geometry of model paramterization to ensure source and receiver
    are within the bounded volume.

    Args:
        source_location: Vector position of the source (m) [3x1].
        receiver_location: Vector position of the receiver (m) [3x1].
        length_x: Length of the room in the x-dimension (m).
        length_y: Length of the room in the y-dimension (m).
        length_z: Length of the room in the z-dimension (m).
    Raises:
        ValueError: If source or receiver positions are outside the room dimensions.
    """
    length_x, length_y, length_z = tuple(space_dimensions)
    # Ensure positions are valid
    if len(source_location) != 3 or len(receiver_location) != 3:
        raise ValueError("Source and receiver positions must be 3D vectors")

    # Ensure room dimensions are valid
    if length_x <= 0 or length_y <= 0 or length_z <= 0:
        raise ValueError("Room dimensions must be positive")

    # Check if source and receiver are within the room dimensions
    if (
        not (0 <= source_location[0] <= length_x)
        or not (0 <= source_location[1] <= length_y)
        or not (0 <= source_location[2] <= length_z)
    ):
        raise ValueError("Source position is outside the room dimensions")
    if (
        not (0 <= receiver_location[0] <= length_x)
        or not (0 <= receiver_location[1] <= length_y)
        or not (0 <= receiver_location[2] <= length_z)
    ):
        raise ValueError("Receiver position is outside the room dimensions")

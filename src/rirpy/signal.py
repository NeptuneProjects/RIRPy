import numpy as np
import numpy.typing as npt
import scipy.fft as fft


def signal_impulse(
    duration: float, sampling_rate: float, delay: float = 0.0
) -> npt.NDArray[np.float64]:
    """
    Generates an impulse signal of specified duration and sampling rate.

    Args:
        duration: Duration of the impulse signal in seconds.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Impulse signal array.
    """
    num_samples = int(duration * sampling_rate)
    impulse_signal = np.zeros(num_samples)
    delay_samples = int(delay * sampling_rate)
    impulse_signal[delay_samples] = (
        1.0  # Set the first sample to 1 to create an impulse
    )
    return impulse_signal


def signal_lfm_chirp(
    chirp_duration: float,
    sampling_rate: float,
    start_freq: float,
    end_freq: float,
    front_padding_duration: float = 0.0,
    end_padding_duration: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Generates a linear frequency modulated (LFM) chirp signal.

    Args:
        chirp_duration: Duration of the chirp signal in seconds.
        sampling_rate: Sampling rate in Hz.
        start_freq: Start frequency of the chirp in Hz.
        end_freq: End frequency of the chirp in Hz.
        front_padding_duration: Duration of front padding in seconds.
        end_padding_duration: Duration of end padding in seconds.

    Returns:
        LFM chirp signal array.
    """
    num_samples = int(chirp_duration * sampling_rate)
    t = np.linspace(0, chirp_duration, num_samples, endpoint=False)
    k = (end_freq - start_freq) / chirp_duration  # Chirp rate
    chirp_signal = np.sin(2 * np.pi * (start_freq * t + 0.5 * k * t**2))

    # Add front and end padding if specified
    front_padding_samples = int(front_padding_duration * sampling_rate)
    end_padding_samples = int(end_padding_duration * sampling_rate)

    return np.pad(
        chirp_signal, (front_padding_samples, end_padding_samples), mode="constant"
    )


def signal_sine(
    duration: float,
    sampling_rate: float,
    frequency: float,
) -> npt.NDArray[np.float64]:
    """
    Generates a sine wave signal.

    Args:
        duration: Duration of the sine wave in seconds.
        sampling_rate: Sampling rate in Hz.
        frequency: Frequency of the sine wave in Hz.

    Returns:
        Sine wave signal array.
    """
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    return np.sin(2 * np.pi * frequency * t)


def simulate_propagation(
    source_signal: npt.NDArray[np.float64], greens_function: npt.NDArray[np.complex128]
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    npt.NDArray[np.complex128],
]:
    """Simulates the propagation of a source signal through a medium using
    a Green's function.

    Args:
        source_signal: The original signal (1D or 2D array).
        greens_function: The Green's function (1D or 2D array).
    Returns:
        FFT of source signal, received signal, and FFT of received signal.
    """
    num_samples = source_signal.size
    source_fft = fft.rfft(source_signal, n=num_samples)
    received_fft = greens_function * source_fft
    received_signal = fft.irfft(received_fft, n=num_samples)
    return source_fft, received_signal, received_fft


def validate_source_signal(source_signal: npt.NDArray[np.float64]) -> None:
    """Validate the source signal to ensure it is a 1D or 2D array and not empty.

    Args:
        source_signal: The original signal (1D or 2D array)

    Raises:
        ValueError: If the source signal is not a 1D or 2D array or is empty.
    """
    if source_signal.ndim > 2:
        raise ValueError(f"Expected 1D or 2D array, got {source_signal.ndim}D")
    if source_signal.size == 0:
        raise ValueError("Input signal cannot be empty")
    if source_signal.ndim == 0:
        raise ValueError("Input signal cannot be a scalar")

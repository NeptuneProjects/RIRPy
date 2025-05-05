#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import rirpy.model as model
from rirpy.plot import plot_ir_and_signals

logger = logging.getLogger(__name__)

NUM_THREADS = 12


def signal_impulse(duration: float, fs: float) -> npt.NDArray[np.float64]:
    """
    Generates an impulse signal of specified duration and sampling frequency.

    Args:
        duration: Duration of the impulse signal in seconds.
        fs: Sampling frequency in Hz.

    Returns:
        Impulse signal array.
    """
    num_samples = int(duration * fs)
    impulse_signal = np.zeros(num_samples)
    impulse_signal[0] = 1.0  # Set the first sample to 1 to create an impulse
    return impulse_signal


def signal_lfm_chirp(
    chirp_duration: float,
    fs: float,
    f0: float,
    f1: float,
    front_padding_duration: float = 0.0,
    end_padding_duration: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Generates a linear frequency modulated (LFM) chirp signal.

    Args:
        chirp_duration: Duration of the chirp signal in seconds.
        fs: Sampling frequency in Hz.
        f0: Start frequency of the chirp in Hz.
        f1: End frequency of the chirp in Hz.
        front_padding_duration: Duration of front padding in seconds.
        end_padding_duration: Duration of end padding in seconds.

    Returns:
        LFM chirp signal array.
    """
    num_samples = int(chirp_duration * fs)
    t = np.linspace(0, chirp_duration, num_samples, endpoint=False)
    k = (f1 - f0) / chirp_duration  # Chirp rate
    chirp_signal = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    # Add front and end padding if specified
    front_padding_samples = int(front_padding_duration * fs)
    end_padding_samples = int(end_padding_duration * fs)

    return np.pad(
        chirp_signal, (front_padding_samples, end_padding_samples), mode="constant"
    )


def main():
    print("-" * 60)
    print("🔊 RIRTorch: Room Impulse Response Modeling")
    print("-" * 60)

    fs = 20000.0  # Sampling rate (Hz)
    y_impulse = signal_impulse(1.0, 20000.0)  # 1 s impulse signal
    y_source = signal_lfm_chirp(
        chirp_duration=0.1,  # 100 ms chirp duration
        fs=fs,
        f0=1000.0,  # Start frequency of the chirp (Hz)
        f1=5000.0,  # End frequency of the chirp (Hz)
        front_padding_duration=0.45,  # 450 ms front padding
        end_padding_duration=0.45,  # 450 ms end padding
    )
    time = np.arange(0, len(y_source)) / fs  # Time vector (s)

    r_source = np.array([1.0, 1.0, 0.75])  # Source position (m)
    r_receiver = np.array([2.0, 2.0, 0.75])  # Receiver position (m)
    Lx = 3.0  # Room length in x direction
    Ly = 3.0  # Room length in y direction
    Lz = 1.5  # Room height
    sound_speed = 1500.0  # Speed of sound (m/s)
    beta_wall = 0.9  # Wall reflection coefficient
    beta_surface = -0.9  # Surface reflection coefficient
    cutoff_time = 0.01  # Cutoff time for reflections (s)

    # The model function can accept multiple signals of the same length.
    y_combined = np.vstack([y_impulse, y_source]).T
    y_receiver = model.propagate_signal(
        y_combined,
        fs,
        r_source,
        r_receiver,
        Lx,
        Ly,
        Lz,
        sound_speed,
        beta_wall,
        beta_surface,
        cutoff_time,
        num_threads=NUM_THREADS,
    )

    fig = plot_ir_and_signals(
        time,
        y_source,
        y_receiver[:, 0],
        y_receiver[:, 1],
        ir_window=2 * cutoff_time,
    )
    plt.show()


if __name__ == "__main__":
    main()

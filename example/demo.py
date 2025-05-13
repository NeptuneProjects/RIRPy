#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import numpy as np

import rirpy.model as model
import rirpy.plot as plot
import rirpy.signal as signal

NUM_THREADS = 8
SAMPLING_RATE = 44000.0

impulse_kwargs = {
    "duration": 1.0,  # Duration of the impulse signal (s)
    "sampling_rate": SAMPLING_RATE,  # Sampling rate (Hz)
    "delay": 0.2,  # Delay before the impulse (s)
}
lfm_chirp_kwargs = {
    "chirp_duration": 0.1,  # Chirp duration (s)
    "sampling_rate": SAMPLING_RATE,  # Sampling rate (Hz)
    "start_freq": 1000.0,  # Start frequency of the chirp (Hz)
    "end_freq": 5000.0,  # End frequency of the chirp (Hz)
    "front_padding_duration": 0.45,  # Front padding duration (s)
    "end_padding_duration": 0.45,  # End padding duration (s)
}
sine_kwargs = {
    "duration": 1.0,
    "sampling_rate": SAMPLING_RATE,  # Sampling rate (Hz)
    "frequency": 700.0,  # Frequency of the sine wave (Hz)
}
model_kwargs = {
    "source_location": np.array([1.0, 1.0, 0.75]),  # Source position (m)
    "receiver_location": np.array([2.0, 2.0, 0.75]),  # Receiver position (m)
    "length_x": 3.0,  # Room length in x direction
    "length_y": 3.0,  # Room length in y direction
    "length_z": 1.5,  # Room height
    "sound_speed": 343.0,  # Speed of sound (m/s) in air
    "refl_coeff_wall": 0.9,  # Wall (& floor) reflection coefficient
    "refl_coeff_ceil": 0.9,  # Ceiling (surface) reflection coefficient
    "cutoff_time": 0.1,  # Cutoff time for reflections (s)
}


def main() -> None:
    print("-" * 60, "\n🔊RIRPy: Room Impulse Response Modeling\n", "-" * 60, sep="")
    duration = 1.0
    num_samples = int(duration * SAMPLING_RATE)  # Number of samples
    time = np.arange(0, num_samples) / SAMPLING_RATE  # Time vector (s)
    frequency = np.fft.rfftfreq(num_samples, 1 / SAMPLING_RATE)  # Frequency vector (Hz)

    (distances, amplitudes), greens_function = model.run(
        frequency=frequency, num_threads=NUM_THREADS, method="both", **model_kwargs
    )

    simulated_signals = [
        signal.signal_impulse(
            **impulse_kwargs,
        ),
        signal.signal_lfm_chirp(**lfm_chirp_kwargs),
        signal.signal_sine(**sine_kwargs),
    ]
    titles = ["Impulse Signal", "LFM Chirp Signal", "Sine Wave Signal"]
    # Define a reflection time offset to align the signals with the reflections
    refl_ref_times = [
        impulse_kwargs["delay"],
        lfm_chirp_kwargs["front_padding_duration"],
        0.0,
    ]

    for title, ref_time, x in zip(titles, refl_ref_times, simulated_signals):
        X, y, Y = signal.simulate_propagation(
            source_signal=x,
            greens_function=greens_function,
        )

        fig = plot.plot_channel_response(
            frequency=frequency,
            freq_domain_data={
                "Green's function": {
                    "data": np.abs(greens_function),
                    "kwargs": {"color": "k"},
                },
                "Source signal": {
                    "data": np.abs(X),
                    "kwargs": {"color": "tab:blue"},
                },
                "Received signal": {
                    "data": np.abs(Y),
                    "kwargs": {"color": "tab:orange"},
                },
            },
            time=time,
            time_domain_data={
                "Source signal": {
                    "data": x,
                    "kwargs": {"color": "tab:blue"},
                },
                "Received signal": {
                    "data": y,
                    "kwargs": {"color": "tab:orange"},
                },
            },
            refl_times=distances / model_kwargs["sound_speed"],
            refl_amplitudes=amplitudes,
            refl_ref_time=ref_time,
        )
        fig.suptitle(title, fontsize=16, y=0.91)
        
        savepath = Path.cwd() / "example"
        savepath.mkdir(parents=True, exist_ok=True)
        figname = f"{title.lower().replace(' ', '_')}_response.png"
        fig.savefig(
            savepath / figname,
            dpi=300,
            bbox_inches="tight",
        )
        logging.info(f"Saved figure to {savepath / figname}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_ir_and_signals(
    time: npt.NDArray[np.float64],
    source_signal: npt.NDArray[np.float64],
    impulse_response: npt.NDArray[np.float64],
    receiver_signal: npt.NDArray[np.float64],
    ir_window: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"hspace": 0.2})

    ax = axes[0]
    impulse_response[impulse_response == 0.0] = np.nan
    markerline, _, _ = ax.stem(
        time, impulse_response, markerfmt="o", linefmt="gray", basefmt="k-"
    )
    markerline.set_markeredgecolor("r")
    markerline.set_markerfacecolor("none")
    ax.set_ylabel("Impulse Response\nAmplitude")
    ax.grid()
    if ir_window is not None:
        ax.set_xlim(0, ir_window)

    ax = axes[1]
    ax.plot(time, source_signal, label="Source Signal", color="tab:blue")
    ax.set_ylabel("Source Signal\nAmplitude")
    ax.legend()
    ax.grid()

    ax = axes[2]
    ax.plot(time, receiver_signal, label="Receiver Signal", color="tab:green")
    ax.set_ylabel("Received Signal\nAmplitude")
    ax.legend()
    ax.grid()
    plt.xlabel("Time (s)")

    fig.suptitle(title)

    return fig

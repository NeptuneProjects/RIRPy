from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_channel_response(
    frequency: npt.NDArray[np.float64],
    freq_domain_data: dict,
    time: npt.NDArray[np.float64],
    time_domain_data: dict,
    refl_times: npt.NDArray[np.float64],
    refl_ref_time: float,
    refl_amplitudes: npt.NDArray[np.float64],
    figsize: tuple[int, int] = (12, 12),
    tlim: tuple[float, float] | None = None,
) -> Figure:
    if tlim is None:
        tlim = (0, max(time))

    fig, axs = plt.subplots(nrows=3, figsize=figsize)
    ax = axs[0]
    for key, args in freq_domain_data.items():
        ax.plot(frequency, args["data"], label=key, **args.get("kwargs", None))
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Channel Response\nAmplitude")

    ax = axs[1]
    markerline, _, _ = ax.stem(
        refl_times + refl_ref_time,
        refl_amplitudes,
        markerfmt="o",
        linefmt="gray",
        basefmt="k-",
    )
    markerline.set_markeredgecolor("k")
    markerline.set_markerfacecolor("none")
    ax.set_xlim(tlim)
    ax.grid(True)
    ax.set_ylabel("Reflections\nAmplitude")

    ax = axs[2]
    ax.sharex(axs[1])
    for key, args in time_domain_data.items():
        ax.plot(time, args["data"], label=key, **args.get("kwargs", None))
    ax.set_xlim(tlim)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel Response\nAmplitude")

    return fig

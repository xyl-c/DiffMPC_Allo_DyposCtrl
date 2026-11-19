from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path("../sim_data")

FILES = {"DAMPC": BASE / "DANMPC" / "DANMPC_results_2.csv", "ARMPC": BASE / "ARNMPC" / "ARNMPC_results_2.csv", "MPC": BASE / "NMPC" / "NMPC_results_2.csv",}

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 85
HEIGHT_MM = 100
DPI = 800

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

STYLES = {"DAMPC": dict(color="#0072B2", linestyle="-"), "ARMPC": dict(color="#D55E00", linestyle="--"), "MPC": dict(color="#009E73", linestyle=":"),}

PANELS = [
    ("T1_N", r"$T_1^{\mathrm{cmd}}$ (N)", "(a)"),
    ("delta1_rad", r"$\delta_1^{\mathrm{cmd}}$ (rad)", "(b)"),
    ("T2_N", r"$T_2^{\mathrm{cmd}}$ (N)", "(c)"),
    ("delta2_rad", r"$\delta_2^{\mathrm{cmd}}$ (rad)", "(d)"),
]

required_columns = ["time_s"] + [panel[0] for panel in PANELS]
data = {}

for name, path in FILES.items():
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig", autostrip=True, ndmin=1,)
    available_columns = set(arr.dtype.names or ())
    missing = set(required_columns) - available_columns
    valid_time = np.isfinite(arr["time_s"])
    t = arr["time_s"][valid_time]
    values = {"time_s": t}
    for column, _, _ in PANELS:
        signal = arr[column][valid_time].copy()
        invalid = ~np.isfinite(signal)
        values[column] = signal
    data[name] = values

t_min = min(values["time_s"].min() for values in data.values())
t_max = max(values["time_s"].max() for values in data.values())

fig, axes = plt.subplots(4, 1, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=150, sharex=True,)

for ax, (column, ylabel, title) in zip(axes, PANELS):
    for name, values in data.items():
        ax.plot(values["time_s"], values[column], label=name, **STYLES[name],)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, loc="center", pad=3)
    ax.set_xlim(t_min, t_max)
    limit = 1200.0 if column in ("T1_N", "T2_N") else np.pi / 2
    for bound in (-limit, limit):
        ax.axhline(y=bound, color="0.35", linestyle="-.", linewidth=0.7, zorder=2,)
    finite_values = np.concatenate([values[column][np.isfinite(values[column])] for values in data.values()])
    y_low = min(float(finite_values.min()), -limit)
    y_high = max(float(finite_values.max()), limit)
    padding = 0.08 * (y_high - y_low)
    ax.set_ylim(y_low - padding, y_high + padding)
    ax.grid(True, linewidth=0.35, color="0.85", linestyle="--",)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=5))
axes[-1].set_xlabel("Time (s)", labelpad=3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.55, 0.995),
    ncol=3,
    frameon=False,
    handlelength=2.0,
    handletextpad=0.45,
    columnspacing=1.0,
    prop={"family": "Times New Roman", "size": 7},
)

fig.subplots_adjust(left=0.20, right=0.98, bottom=0.09, top=0.92, hspace=0.16,)

fig.align_ylabels(axes)

stem = OUT_DIR / "controllers_control_comparison_2"

fig.savefig(stem.with_suffix(".png"), dpi=DPI)

print(f"Saved：{stem.with_suffix('.png').resolve()}")
plt.show()

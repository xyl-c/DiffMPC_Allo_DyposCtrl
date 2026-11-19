from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path("../sim_data")

FILES = {"DAMPC": BASE / "DANMPC" / "DANMPC_results_1.csv", "ARMPC": BASE / "ARNMPC" / "ARNMPC_results_1.csv", "MPC": BASE / "NMPC" / "NMPC_results_1.csv",}

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 85
HEIGHT_MM = 85
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
    ("Xcmd_N", r"$\tau^{*,\mathrm{cmd}}_x$ (N)", "(a)", 400.0),
    ("Ycmd_N", r"$\tau^{*,\mathrm{cmd}}_y$ (N)", "(b)", 400.0),
    ("Ncmd_Nm", r"$\tau^{*,\mathrm{cmd}}_\psi$ (N$\cdot$m)", "(c)", 600.0),
]

required_columns = {"time_s"} | {panel[0] for panel in PANELS}
data = {}

for name, path in FILES.items():
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig", autostrip=True, ndmin=1,)
    missing = required_columns - set(arr.dtype.names or ())
    valid_time = np.isfinite(arr["time_s"])
    t = arr["time_s"][valid_time]
    values = {"time_s": t}
    for column, _, _, _ in PANELS:
        signal = arr[column][valid_time].copy()
        invalid = ~np.isfinite(signal)
        values[column] = signal
    data[name] = values

t_min = min(v["time_s"].min() for v in data.values())
t_max = max(v["time_s"].max() for v in data.values())

fig, axes = plt.subplots(3, 1, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=150, sharex=True,)

for ax, (column, ylabel, title, limit) in zip(axes, PANELS):
    for name, values in data.items():
        ax.plot(values["time_s"], values[column], label=name, **STYLES[name],)
    for bound in (-limit, limit):
        ax.axhline(y=bound, color="0.35", linestyle="-.", linewidth=0.7, zorder=2,)
    finite_values = np.concatenate([v[column][np.isfinite(v[column])] for v in data.values()])
    y_low = min(float(finite_values.min()), -limit)
    y_high = max(float(finite_values.max()), limit)
    padding = 0.08 * (y_high - y_low)
    ax.set_ylim(y_low - padding, y_high + padding)
    ax.set_xlim(t_min, t_max)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, loc="center", pad=3)
    if y_low == -limit and y_high == limit:
        ax.set_yticks([-limit, 0, limit])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, linewidth=0.35, color="0.85", linestyle="--",)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2)

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

fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.90, hspace=0.22,)

fig.align_ylabels(axes)

stem = OUT_DIR / "controllers_cmd_comparison"

fig.savefig(stem.with_suffix(".png"), dpi=DPI)

print(f"Saved：{stem.with_suffix('.png').resolve()}")
plt.show()


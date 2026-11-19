from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path("../sim_data")

FILES = {"DAMPC": BASE / "DANMPC" / "DANMPC_results_2.csv", "ARMPC": BASE / "ARNMPC" / "ARNMPC_results_2.csv", "MPC": BASE / "NMPC" / "NMPC_results_2.csv",}

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 85
HEIGHT_MM = 82
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

required_columns = ["time_s", "x_m", "y_m", "psi_rad"]
data = {}

for name, path in FILES.items():
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig", autostrip=True, ndmin=1,)
    available_columns = arr.dtype.names or ()
    missing = set(required_columns) - set(available_columns)
    values = {column: arr[column] for column in required_columns}
    data[name] = values

fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=150,)

ax_x, ax_y, ax_psi, ax_xy = axes.ravel()

for name, values in data.items():
    style = STYLES[name]
    t = values["time_s"]
    x = values["x_m"]
    y = values["y_m"]
    psi = values["psi_rad"]
    ax_x.plot(t, x, label=name, **style)
    ax_y.plot(t, y, label=name, **style)
    ax_psi.plot(t, psi, label=name, **style)
    ax_xy.plot(x, y, label=name, **style)

t_min = min(v["time_s"].min() for v in data.values())
t_max = max(v["time_s"].max() for v in data.values())

for ax, ylabel, title in [(ax_x, r"$x$ (m)", "(a)"), (ax_y, r"$y$ (m)", "(b)"), (ax_psi, r"$\psi$ (rad)", "(c)"),]:
    ax.set_xlabel("Time (s)", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, loc="center", pad=3)
    ax.set_xlim(t_min, t_max)

ax_xy.set_xlabel(r"$x$ (m)", labelpad=2)
ax_xy.set_ylabel(r"$y$ (m)", labelpad=2)
ax_xy.set_title("(d)", loc="center", pad=3)
ax_xy.set_aspect("equal", adjustable="box")

for ax in axes.ravel():
    ax.grid(True, linewidth=0.35, color="0.85", linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.margins(x=0.04, y=0.08)

handles, labels = ax_x.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.53, 0.995),
    ncol=3,
    frameon=False,
    handlelength=2.0,
    handletextpad=0.45,
    columnspacing=1.0,
    prop={"family": "Times New Roman", "size": 7},
)

fig.subplots_adjust(left=0.15, right=0.985, bottom=0.11, top=0.88, wspace=0.32, hspace=0.35,)

zoom_xlim = (-0.1, 0.1)
zoom_ylim = (-0.1, 0.1)

ax_zoom = ax_xy.inset_axes([0.08, 0.53, 0.44, 0.43], zorder=5)
ax_zoom.set_facecolor("white")

for name, values in data.items():
    ax_zoom.plot(values["x_m"], values["y_m"], linewidth=0.8, **STYLES[name],)

ax_zoom.set_xlim(*zoom_xlim)
ax_zoom.set_ylim(*zoom_ylim)
ax_zoom.set_aspect("equal", adjustable="box")
ax_zoom.set_xticks([-0.1, 0, 0.1])
ax_zoom.set_yticks([-0.1, 0, 0.1])
ax_zoom.tick_params(axis="both", labelsize=5, direction="in", length=1.5, width=0.4, pad=1,)
ax_zoom.grid(True, linewidth=0.3, color="0.85", linestyle="--",)
ax_zoom.set_axisbelow(True)

for spine in ax_zoom.spines.values():
    spine.set_linewidth(0.5)

ax_xy.indicate_inset_zoom(ax_zoom, edgecolor="0.35", linewidth=0.5, alpha=0.8,)
stem = OUT_DIR / "controllers_comparison_2"

fig.savefig(stem.with_suffix(".png"), dpi=DPI)
print(f"Saved：{OUT_DIR}")
plt.show()

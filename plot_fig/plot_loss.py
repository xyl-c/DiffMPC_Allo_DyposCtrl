from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path("../sim_data")

FILES = {"DAMPC": BASE / "DANMPC" / "DANMPC_results_2.csv", "ARMPC": BASE / "ARNMPC" / "ARNMPC_results_2.csv", "MPC": BASE / "NMPC" / "NMPC_results_2.csv",}

LOSS_COLUMNS = {"DAMPC": "loss_rl", "ARMPC": "loss", "MPC": "loss",}

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 85
HEIGHT_MM = 65
DPI = 800
ZOOM_START = 5.0

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

data = {}

for name, path in FILES.items():
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig", autostrip=True, ndmin=1,)
    loss_column = LOSS_COLUMNS[name]
    required_columns = {"time_s", loss_column}
    available_columns = set(arr.dtype.names or ())
    missing = required_columns - available_columns
    t = arr["time_s"]
    loss = arr[loss_column]
    valid_time = np.isfinite(t)
    t = t[valid_time]
    loss = loss[valid_time].copy()
    invalid_loss = ~np.isfinite(loss)
    data[name] = {"time": t, "loss": loss}

t_min = min(values["time"].min() for values in data.values())
t_max = max(values["time"].max() for values in data.values())

zoom_losses = []

for name, values in data.items():
    mask = values["time"] >= ZOOM_START
    losses = values["loss"][mask]
    finite_losses = losses[np.isfinite(losses)]
    zoom_losses.append(finite_losses)

all_zoom_losses = np.concatenate(zoom_losses)

fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=150,)

for name, values in data.items():
    ax.plot(values["time"], values["loss"], label=name, **STYLES[name],)

ax.set_xlabel("Time (s)", labelpad=2)
ax.set_ylabel("Loss", labelpad=2)
ax.set_xlim(t_min, t_max)
ax.margins(y=0.08)

ax.grid(True, linewidth=0.35, color="0.85", linestyle="--",)
ax.set_axisbelow(True)
ax.tick_params(pad=2)
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

handles, labels = ax.get_legend_handles_labels()

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

fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.88,)

ax_zoom = ax.inset_axes([0.42, 0.43, 0.54, 0.51], zorder=5,)
ax_zoom.set_facecolor("white")

for name, values in data.items():
    t = values["time"]
    loss = values["loss"]
    mask = t >= ZOOM_START
    ax_zoom.plot(t[mask], loss[mask], linewidth=0.8, **STYLES[name],)

y_min = min(0.0, float(all_zoom_losses.min()))
y_max = max(0.0, float(all_zoom_losses.max()))
y_span = y_max - y_min
y_pad = 0.08 * y_span if y_span > 0 else 1e-6

ax_zoom.set_xlim(ZOOM_START, t_max)
ax_zoom.set_ylim(0, 9)
ax_zoom.set_yticks([0, 3, 6, 9])

ax_zoom.xaxis.set_major_locator(MaxNLocator(nbins=4))


ax_zoom.tick_params(axis="both", labelsize=5.5, direction="in", length=1.5, width=0.4, pad=1,)

ax_zoom.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True, useOffset=False,)
ax_zoom.yaxis.get_offset_text().set_fontsize(5.5)

ax_zoom.grid(True, linewidth=0.3, color="0.85", linestyle="--",)
ax_zoom.set_axisbelow(True)

for spine in ax_zoom.spines.values():
    spine.set_linewidth(0.5)

ax.indicate_inset_zoom(ax_zoom, edgecolor="0.35", linewidth=0.5, alpha=0.8,)

stem = OUT_DIR / "controllers_loss_comparison_2"

fig.savefig(stem.with_suffix(".png"), dpi=DPI)

print(f"Saved：{stem.with_suffix('.png').resolve()}")
plt.show()


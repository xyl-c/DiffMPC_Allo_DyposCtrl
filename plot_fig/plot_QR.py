from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path("../sim_data")
FILE = BASE / "DANMPC" / "DANMPC_results_2.csv"

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 178
HEIGHT_MM = 105
DPI = 800

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
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

PANELS = [
    ("Q_1", r"$Q_1$"),
    ("Q_2", r"$Q_2$"),
    ("Q_3", r"$Q_3$"),
    ("R_1", r"$R_{\tau,1}$"),
    ("R_2", r"$R_{\tau,2}$"),
    ("R_3", r"$R_{\tau,3}$"),
    ("R_delta_1", r"$R_{\Delta\tau,1}$"),
    ("R_delta_2", r"$R_{\Delta\tau,2}$"),
    ("R_delta_3", r"$R_{\Delta\tau,3}$"),
]

arr = np.genfromtxt(FILE, delimiter=",", names=True, dtype=float, encoding="utf-8-sig", autostrip=True, ndmin=1,)

required_columns = {"time_s"} | {column for column, _ in PANELS}
missing = required_columns - set(arr.dtype.names or ())

valid_time = np.isfinite(arr["time_s"])

t = arr["time_s"][valid_time]

data = {}

for column, _ in PANELS:
    values = arr[column][valid_time].copy()
    invalid = ~np.isfinite(values)
    data[column] = values

fig, axes = plt.subplots(3, 3, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=150, sharex=True,)

for index, (ax, (column, label)) in enumerate(zip(axes.ravel(), PANELS)):
    values = data[column]
    ax.plot(t, values, color="#0072B2", linestyle="-", linewidth=1.0,)
    ax.set_title(f"({chr(ord('a') + index)})  {label}", loc="center", pad=4,)
    ax.set_xlim(t.min(), t.max())
    ax.margins(y=0.10)
    ax.grid(True, linewidth=0.35, color="0.85", linestyle="--",)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True, useOffset=False,)
    ax.yaxis.get_offset_text().set_fontsize(6)

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)", labelpad=2)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.93, wspace=0.10, hspace=0.28,)

stem = OUT_DIR / "DANMPC_weights_comparison_2"

fig.savefig(stem.with_suffix(".png"), dpi=DPI)

print(f"Saved：{stem.with_suffix('.png').resolve()}")
plt.show()

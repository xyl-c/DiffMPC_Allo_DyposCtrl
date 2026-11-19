from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path("../sim_data/ARNMPC")

RESULT_FILE = BASE / "ARNMPC_results_1.csv"
PRED_FILE = BASE / "pre_Data_20.csv"

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_START = 10.0
HISTORY_END = 20.0

PRED_START = 20.0
PRED_TS = 0.2

PRED_FIRST_STEP = 0

TIME_START = 0.0
TIME_END = 25.0

WIDTH_MM = 85
HEIGHT_MM = 90
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

PANELS = [
    ("Xd_N", "Xd_hat_N", r"$\hat{d}_{lp,x}$ (N)", "(a)"),
    ("Yd_N", "Yd_hat_N", r"$\hat{d}_{lp,y}$ (N)", "(b)"),
    ("Nd_Nm", "Nd_hat_Nm", r"$\hat{d}_{lp,\psi}$ (N$\cdot$m)", "(c)"),
]

arr = np.genfromtxt(RESULT_FILE, delimiter=",", names=True, dtype=float, encoding="utf-8-sig", autostrip=True, ndmin=1,)

required = {"time_s"}
for true_col, hat_col, _, _ in PANELS:
    required.update([true_col, hat_col])

missing = required - set(arr.dtype.names or ())
time_all = arr["time_s"]
mask = (np.isfinite(time_all) & (time_all >= TIME_START) & (time_all <= TIME_END))
t = time_all[mask]
data = {}
for column in required - {"time_s"}:
    values = arr[column][mask].copy()
    invalid = ~np.isfinite(values)
    data[column] = values

prediction = np.loadtxt(PRED_FILE, delimiter=",", encoding="utf-8-sig", ndmin=2,)

t_pred = PRED_START + (np.arange(prediction.shape[1]) + PRED_FIRST_STEP) * PRED_TS

fig, axes = plt.subplots(3, 1, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=150, sharex=True,)

for i, (ax, panel) in enumerate(zip(axes, PANELS)):
    true_col, hat_col, ylabel, title = panel
    ax.axvspan(HISTORY_START, HISTORY_END, color="0.94", linewidth=0, zorder=0,)
    ax.plot(t, data[true_col], color="#0072B2", linestyle="-", linewidth=1.0, label="True",)
    ax.plot(t, data[hat_col], color="#D55E00", linestyle="--", linewidth=1.0, label="Estimated",)
    ax.plot(
        t_pred,
        prediction[i],
        color="#009E73",
        linestyle="-.",
        linewidth=1.0,
        marker="o",
        markersize=2.0,
        markevery=3,
        markerfacecolor="white",
        markeredgewidth=0.5,
        label="Predicted",
        zorder=4,
    )
    ax.axvline(PRED_START, color="0.4", linestyle=":", linewidth=0.7,)
    ax.set_xlim(TIME_START, TIME_END)
    ax.margins(y=0.12)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, loc="center", pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, linewidth=0.35, color="0.85", linestyle="--",)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2)

axes[-1].set_xticks(np.arange(TIME_START, TIME_END + 1, 5))
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

fig.subplots_adjust(left=0.20, right=0.98, bottom=0.11, top=0.91, hspace=0.24,)
fig.align_ylabels(axes)

output_file = OUT_DIR / "ARNMPC_disturbance_prediction_20s.png"
fig.savefig(output_file, dpi=DPI)

print(f"Saved：{output_file.resolve()}")
plt.show()





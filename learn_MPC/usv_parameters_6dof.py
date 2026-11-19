import numpy as np
import math

# ============================================================
# 6-DOF effective inertia
# [surge, sway, heave, roll, pitch, yaw]
# ============================================================
M_11 = 550.0
M_22 = 900.0
M_66 = 750.0
M_33 = 700.0     # heave effective mass [kg]
M_44 = 650.0     # roll effective inertia [kg m^2]
M_55 = 900.0     # pitch effective inertia [kg m^2]

M_11_dis = M_11 + 0.05*M_11
M_22_dis = M_22 + 0.05*M_22
M_66_dis = M_66 + 0.05*M_66
M_33_dis = M_33 + 0.05*M_33
M_44_dis = M_44 + 0.05*M_44
M_55_dis = M_55 + 0.05*M_55

# ============================================================
# Linear hydrodynamic damping
# ============================================================
D_11 = 150.0
D_22 = 300.0
D_66 = 250.0
D_33 = 500.0     # heave [N/(m/s)]
D_44 = 350.0     # roll  [N m/(rad/s)]
D_55 = 450.0     # pitch [N m/(rad/s)]

D_11_dis = D_11 + 0.05*D_11
D_22_dis = D_22 + 0.05*D_22
D_66_dis = D_66 + 0.05*D_66
D_33_dis = D_33 + 0.05*D_33
D_44_dis = D_44 + 0.05*D_44
D_55_dis = D_55 + 0.05*D_55

# ============================================================
# Quadratic damping
# ============================================================

DQ_11 = 150.0
DQ_22 = 300.0
DQ_33 = 830.0
DQ_44 = 1750.0
DQ_55 = 2250.0
DQ_66 = 1250.0

# ============================================================
# Hydrostatic restoring stiffness
# ============================================================
K_HEAVE = 2500.0       # N/m
K_ROLL  = 3500.0       # N m/rad
K_PITCH = 4500.0       # N m/rad

# ============================================================
# JONSWAP sea
# ============================================================
WAVE_HS = 1.6
WAVE_HS_SMALL = 0.1
WAVE_TP = 8.0
WAVE_GAMMA = 3.3
WAVE_DIRECTION = np.deg2rad(30.0)

WAVE_N_FREQ = 200
WAVE_F_MIN = 0.02
WAVE_F_MAX = 1.0
WATER_DEPTH = None
WAVE_SEED = 2
WAVE_TRANSITION_DURATION = 2.0

# ============================================================
# Ocean current
# ============================================================

CURRENT_SPEED = 0.0                   # m/s 0.2
CURRENT_DIRECTION = np.deg2rad(30.0)  # world-frame

# ============================================================
# Wave-load transfer gains
# ============================================================
WAVE_FORCE_GAINS = np.array([
    350.0,   # surge X
    450.0,   # sway  Y
    700.0,   # heave Z
], dtype=np.float64)

WAVE_MOMENT_GAINS = np.array([
    500.0,   # roll  K
    650.0,   # pitch M
    400.0,   # yaw   N
], dtype=np.float64)

# ============================================================
# Two azimuth thrusters
# ============================================================
LX1 = -1.5
LY1 =  0.8
LX2 = -1.5
LY2 = -0.8

T_MIN = -1200.0
T_MAX =  1200.0
DELTA_MIN = -0.5*math.pi
DELTA_MAX = 0.5*math.pi

TAU_T = 0.20
TAU_DELTA = 0.15

T_RATE_MAX = 5000.0              # N/s
DELTA_RATE_MAX = np.deg2rad(90)  # rad/s

# ============================================================
# Optional planar generalized-force limits for the NMPC/allocator layer.
# ============================================================
FX_MIN = -400.0
FX_MAX =  400.0
FY_MIN = -400.0
FY_MAX =  400.0
M_MIN  = -600.0
M_MAX  =  600.0

delta_tau_max_X = 40.0
delta_tau_max_Y = 40.0
delta_tau_max_M = 60.0



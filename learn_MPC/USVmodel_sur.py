import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from usv_parameters_6dof import *

class Space(object):

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)

class surUSV(nn.Module):
    def __init__(self, Ts, sim_time):
        super().__init__()
        self.nx = 6
        self.np = 3
        self.nu = 4
        self.Ts = Ts
        self.sim_time = sim_time
        self.m_11 = M_11
        self.m_22 = M_22
        self.m_33 = M_66
        self.d_11 = D_11
        self.d_22 = D_22
        self.d_33 = D_66
        self.lx1 = LX1
        self.ly1 = LY1
        self.lx2 = LX2
        self.ly2 = LY2
        self.Fx_max = FX_MAX
        self.Fx_min = FX_MIN
        self.Fy_max = FY_MAX
        self.Fy_min = FY_MIN
        self.M_max = M_MAX
        self.M_min = M_MIN
        self.T_min = T_MIN
        self.T_max = T_MAX
        self.delta_min = DELTA_MIN
        self.delta_max = DELTA_MAX
        self.tau_T = TAU_T
        self.tau_delta = TAU_DELTA
        self.t = 0

    def f_c(self, state, action, disturbance):
        psi = state[:, 2:3]
        u = state[:, 3:4]
        v = state[:, 4:5]
        r = state[:, 5:6]
        T1 = state[:, 6:7]
        delta1 = state[:, 7:8]
        T2 = state[:, 8:9]
        delta2 = state[:, 9:10]
        T1_cmd = action[:, 0:1]
        delta1_cmd = action[:, 1:2]
        T2_cmd = action[:, 2:3]
        delta2_cmd = action[:, 3:4]
        Fx1 = T1 * torch.cos(delta1)
        Fy1 = T1 * torch.sin(delta1)
        Fx2 = T2 * torch.cos(delta2)
        Fy2 = T2 * torch.sin(delta2)
        X = Fx1 + Fx2
        Y = Fy1 + Fy2
        N = (self.lx1 * Fy1 - self.ly1 * Fx1 + self.lx2 * Fy2 - self.ly2 * Fx2)
        disturbance = torch.as_tensor( disturbance, dtype=state.dtype, device=state.device)
        if disturbance.ndim == 1:
            disturbance = disturbance.reshape(1, -1)
        X_w = disturbance[:, 0:1]
        Y_w = disturbance[:, 1:2]
        N_w = disturbance[:, 2:3]
        dx = (u * torch.cos(psi) - v * torch.sin(psi))
        dy = (u * torch.sin(psi) + v * torch.cos(psi))
        dpsi = r
        du = ((self.m_22 / self.m_11) * v * r - (self.d_11 / self.m_11) * u + (X + X_w) / self.m_11)
        dv = (-(self.m_11 / self.m_22) * u * r - (self.d_22 / self.m_22) * v + (Y + Y_w ) / self.m_22)
        dr = (((self.m_11 - self.m_22) / self.m_33) * u * v - (self.d_33 / self.m_33) * r + (N + N_w) / self.m_33)
        dT1 = (T1_cmd - T1) / self.tau_T
        dT2 = (T2_cmd - T2) / self.tau_T
        ddelta1 = angle_error_torch(delta1_cmd, delta1) / self.tau_delta
        ddelta2 = angle_error_torch(delta2_cmd, delta2) / self.tau_delta
        return torch.cat([dx, dy, dpsi, du, dv, dr, dT1, ddelta1, dT2, ddelta2], dim=1)

    def run(self, state, action, disturbance):
        k1 = self.f_c(state, action, disturbance)
        k2 = self.f_c(state + self.Ts / 2.0 * k1, action, disturbance)
        k3 = self.f_c(state + self.Ts / 2.0 * k2, action, disturbance)
        k4 = self.f_c(state + self.Ts * k3, action, disturbance)
        x_next = (state + self.Ts / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        psi = torch.atan2( torch.sin(x_next[:, 2:3]), torch.cos(x_next[:, 2:3]))
        delta1 = torch.atan2( torch.sin(x_next[:, 7:8]), torch.cos(x_next[:, 7:8]))
        delta2 = torch.atan2(torch.sin(x_next[:, 9:10]), torch.cos(x_next[:, 9:10]))
        T1 = torch.clamp(x_next[:, 6:7], self.T_min, self.T_max)
        T2 = torch.clamp(x_next[:, 8:9], self.T_min, self.T_max)
        x_next = torch.cat([x_next[:, 0:2], psi, x_next[:, 3:6], T1, delta1, T2, delta2], dim=1)
        self.t += self.Ts
        return x_next

def angle_normalize(x):
    return (((x + torch.pi) % (2 * torch.pi)) - torch.pi)

def angle_error_torch(target, current):
    return torch.atan2(torch.sin(target - current), torch.cos(target - current))


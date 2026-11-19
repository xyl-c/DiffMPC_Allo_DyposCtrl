import math
import torch
import torch.nn as nn
from qpth.qp import QPFunction

from usv_parameters_6dof import *


class DiffAllocator(nn.Module):
    def __init__(self, Ts, delta_min=DELTA_MIN, delta_max=DELTA_MAX, w_delta_T=1e-1, w_delta_angle=0.5,):
        super().__init__()
        self.Ts = float(Ts)
        self.x1 = float(LX1)
        self.y1 = float(LY1)
        self.x2 = float(LX2)
        self.y2 = float(LY2)
        self.T_min = float(T_MIN)
        self.T_max = float(T_MAX)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.T_rate_max = float(T_RATE_MAX)
        self.delta_rate_max = float(DELTA_RATE_MAX)
        self.dT_max = self.T_rate_max * self.Ts
        self.ddelta_max = self.delta_rate_max * self.Ts
        self.register_buffer("W_tau", torch.diag(torch.tensor([2e3, 2e3, 2e3], dtype=torch.float64)))
        self.register_buffer("W_u", torch.diag(torch.tensor([1e-3, 1e-4, 1e-3, 1e-4], dtype=torch.float64)))
        self.register_buffer("W_delta", torch.diag(torch.tensor([w_delta_T, w_delta_angle, w_delta_T, w_delta_angle,], dtype=torch.float64)))
        self.max_iter = 30
        self.tol = 1e-7
        A_force = torch.tensor(
            [[1.0,       0.0, 1.0,       0.0], [0.0,       1.0, 0.0,       1.0], [-self.y1, self.x1, -self.y2, self.x2],],
            dtype=torch.float64,
        )
        self.register_buffer("A_force", A_force)
        I3 = torch.eye(3, dtype=torch.float64)
        force_pinv = (A_force.transpose(0, 1) @ torch.linalg.inv(A_force @ A_force.transpose(0, 1) + 1e-10 * I3))
        self.register_buffer("force_pinv", force_pinv)

    @staticmethod
    def angle_normalize(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def force_xy_to_T_delta(self, Fx, Fy):
        R = torch.sqrt(Fx**2 + Fy**2 + 1e-12)
        alpha = torch.atan2(Fy, Fx)
        T = R
        delta = alpha
        mask_hi = delta > self.delta_max
        delta = torch.where(mask_hi, delta - math.pi, delta)
        T = torch.where(mask_hi, -T, T)
        mask_lo = delta < self.delta_min
        delta = torch.where(mask_lo, delta + math.pi, delta)
        T = torch.where(mask_lo, -T, T)
        return T, delta


    def geometric_guess(self, tau_des):
        P = self.force_pinv.to(dtype=tau_des.dtype, device=tau_des.device)
        f0 = tau_des @ P.transpose(0, 1)
        Fx1 = f0[:, 0]
        Fy1 = f0[:, 1]
        Fx2 = f0[:, 2]
        Fy2 = f0[:, 3]
        T1, delta1 = self.force_xy_to_T_delta(Fx1, Fy1)
        T2, delta2 = self.force_xy_to_T_delta(Fx2, Fy2)
        return torch.stack([T1, delta1, T2, delta2], dim=1)

    def get_step_bounds(self, z_prev):
        lower_abs = z_prev.new_tensor([self.T_min, self.delta_min, self.T_min, self.delta_min,]).view(1, 4)
        upper_abs = z_prev.new_tensor([self.T_max, self.delta_max, self.T_max, self.delta_max,]).view(1, 4)
        step = z_prev.new_tensor([self.dT_max, self.ddelta_max, self.dT_max, self.ddelta_max,]).view(1, 4)
        lower_rate = z_prev - step
        upper_rate = z_prev + step
        lower = torch.maximum(lower_abs, lower_rate)
        upper = torch.minimum(upper_abs, upper_rate)
        return lower, upper


    def init_guess(self, tau_des, z_prev):
        z_geo = self.geometric_guess(tau_des)
        lower, upper = self.get_step_bounds(z_prev)
        z0 = torch.maximum(lower, torch.minimum(z_geo, upper))
        return z0

    def calc_tau(self, z):
        T1 = z[:, 0]
        delta1 = z[:, 1]
        T2 = z[:, 2]
        delta2 = z[:, 3]
        c1 = torch.cos(delta1)
        s1 = torch.sin(delta1)
        c2 = torch.cos(delta2)
        s2 = torch.sin(delta2)
        Fx1 = T1 * c1
        Fy1 = T1 * s1
        Fx2 = T2 * c2
        Fy2 = T2 * s2
        Fx = Fx1 + Fx2
        Fy = Fy1 + Fy2
        M1 = (self.x1 * Fy1 - self.y1 * Fx1)
        M2 = (self.x2 * Fy2 - self.y2 * Fx2)
        Mz = M1 + M2
        return torch.stack([Fx, Fy, Mz], dim=1)

    def jacobian(self, z):
        B = z.shape[0]
        T1 = z[:, 0]
        delta1 = z[:, 1]
        T2 = z[:, 2]
        delta2 = z[:, 3]
        c1 = torch.cos(delta1)
        s1 = torch.sin(delta1)
        c2 = torch.cos(delta2)
        s2 = torch.sin(delta2)
        J = torch.zeros(B, 3, 4, dtype=z.dtype, device=z.device)
        J[:, 0, 0] = c1
        J[:, 0, 1] = -T1 * s1
        J[:, 0, 2] = c2
        J[:, 0, 3] = -T2 * s2
        J[:, 1, 0] = s1
        J[:, 1, 1] = T1 * c1
        J[:, 1, 2] = s2
        J[:, 1, 3] = T2 * c2
        J[:, 2, 0] = (self.x1 * s1 - self.y1 * c1)
        J[:, 2, 1] = (self.x1 * T1 * c1 + self.y1 * T1 * s1)
        J[:, 2, 2] = (self.x2 * s2 - self.y2 * c2)
        J[:, 2, 3] = (self.x2 * T2 * c2 + self.y2 * T2 * s2)
        return J

    def build_constraints(self, z, z_prev):
        B = z.shape[0]
        lower, upper = self.get_step_bounds(z_prev)
        G = torch.zeros(B, 8, 4, dtype=z.dtype, device=z.device)
        h = torch.zeros(B, 8, dtype=z.dtype, device=z.device)
        for j in range(4):
            G[:, j, j] = 1.0
            h[:, j] = upper[:, j] - z[:, j]
        for j in range(4):
            G[:, 4 + j, j] = -1.0
            h[:, 4 + j] = z[:, j] - lower[:, j]
        return G, h

    def solve_qp(self, H, g, G, h):
        B = H.shape[0]
        n = H.shape[1]
        A = torch.zeros(B, 0, n, dtype=H.dtype, device=H.device)
        b = torch.zeros(B, 0, dtype=H.dtype, device=H.device)
        return QPFunction(verbose=False, check_Q_spd=False,)(H, g, G, h, A, b)

    def forward(self, tau_des, z_prev,):
        if (tau_des.ndim != 2 or tau_des.shape[1] != 3):
            raise ValueError("tau_des must have shape (B,3).")
        if z_prev.ndim == 1:
            z_prev = z_prev.unsqueeze(0)
        z_prev = z_prev.to(dtype=tau_des.dtype, device=tau_des.device)
        if (z_prev.ndim != 2 or z_prev.shape[1] != 4 or z_prev.shape[0] != tau_des.shape[0]):
            raise ValueError("z_prev must have shape (B,4).")
        lower_abs = z_prev.new_tensor([self.T_min, self.delta_min, self.T_min, self.delta_min,]).view(1, 4)
        upper_abs = z_prev.new_tensor([self.T_max, self.delta_max, self.T_max, self.delta_max,]).view(1, 4)
        z_prev = torch.maximum(lower_abs, torch.minimum(z_prev, upper_abs))
        z = self.init_guess(tau_des, z_prev)
        z_ref = torch.zeros_like(z)
        eye = torch.eye(4, dtype=z.dtype, device=z.device).unsqueeze(0)
        B = z.shape[0]
        Wtau = self.W_tau.unsqueeze(0).to(dtype=z.dtype, device=z.device).expand(B, -1, -1)
        Wu = self.W_u.unsqueeze(0).to(dtype=z.dtype, device=z.device).expand(B, -1, -1)
        Wdelta = self.W_delta.unsqueeze(0).to(dtype=z.dtype, device=z.device).expand(B, -1, -1)
        for _ in range(self.max_iter):
            tau = self.calc_tau(z)
            e = tau - tau_des
            J = self.jacobian(z)
            H = torch.bmm(J.transpose(1, 2), torch.bmm(Wtau, J))
            H = (H + Wu + Wdelta + 1e-6 * eye)
            g_tracking = torch.bmm(J.transpose(1, 2), torch.bmm(Wtau, e.unsqueeze(-1))).squeeze(-1)
            g_effort = torch.bmm(Wu, (z - z_ref).unsqueeze(-1)).squeeze(-1)
            g_change = torch.bmm(Wdelta, (z - z_prev).unsqueeze(-1)).squeeze(-1)
            g = (g_tracking + g_effort + g_change)
            G, h = self.build_constraints(z, z_prev)
            dz = self.solve_qp(H, g, G, h)
            z = z + dz
            if (torch.max(torch.norm(dz, dim=1)) < self.tol):
                break
        lower, upper = self.get_step_bounds(z_prev)
        z = torch.maximum(lower, torch.minimum(z, upper))
        return z


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    Ts = 0.05
    allocator = DiffAllocator(Ts=Ts)
    z_prev = torch.zeros(1, 4, dtype=torch.float64)
    tests = [[300.0,   0.0,   0.0], [300.0, 100.0, 100.0], [  0.0, 300.0, 300.0], [-300.0,  0.0, 300.0],]
    for tau_des_np in tests:
        tau_des = torch.tensor([tau_des_np], dtype=torch.float64)
        z = allocator(tau_des, z_prev)
        tau_real = allocator.calc_tau(z)
        dz_cmd = z - z_prev
        print("----------------------------------------")
        print("tau_des =", tau_des)
        print("z_prev  =", z_prev)
        print("z        =", z)
        print("delta z  =", dz_cmd)
        print("tau_real =", tau_real)
        print("tau_err  =", tau_real - tau_des)
        z_prev = z.detach()


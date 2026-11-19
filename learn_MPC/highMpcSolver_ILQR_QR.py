import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import time
import torch
import torch.autograd
from mpc import mpc
from allocate_ctrl import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from envModel_current import *
from ARIMA_pred import *
from usv_parameters_6dof import *
from ESO_ob import *

class HUSVModel(nn.Module):
    def __init__(self, horizon, N) -> None:
        super().__init__()
        self.Ts = horizon / N
        self.N = N
        self.nx_physical = 6
        self.nx = 13
        self.np = 3
        self.nu = 3
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
        self.tau_Fx = 0.20
        self.tau_Fy = 0.20
        self.tau_Mz = 0.20
        self.register_buffer("Q_base", torch.tensor([500, 500, 500, 0.005, 0.005, 0.005], dtype=torch.float64))
        self.register_buffer("R_base", torch.tensor([0.005, 0.005, 0.005], dtype=torch.float64))
        self.register_buffer("R_delta_base", torch.tensor([0.005, 0.005, 0.005], dtype=torch.float64))
        self.log_Q_scale = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
        self.log_R_scale = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.log_R_delta_scale = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def forward_QR(self):
        log_Q_scale = torch.clamp(self.log_Q_scale, min=-10.0, max=10.0)
        log_R_scale = torch.clamp(self.log_R_scale, min=-10.0, max=10.0)
        log_R_delta_scale = torch.clamp(self.log_R_delta_scale, min=-10.0, max=10.0)
        self.Q = self.Q_base * torch.exp(log_Q_scale)
        self.R = self.R_base * torch.exp(log_R_scale)
        self.R_delta = (self.R_delta_base * torch.exp(log_R_delta_scale))
        return (self.Q.reshape(1, -1), self.R.reshape(1, -1), self.R_delta.reshape(1, -1))

    def dis_wave_update(self, wave_pre):
        wave_pre = torch.as_tensor(wave_pre, dtype=torch.float64)
        if wave_pre.ndim != 2 or wave_pre.shape[0] != 3:
            raise ValueError("wave_pre must have shape (3, Np), " "with rows [Xw,Yw,Nw].")
        self.wave_pre = wave_pre

    def dis_wave(self, t):
        wave_pre = self.wave_pre.to(dtype=t.dtype, device=t.device)
        idx = torch.floor(t[:, 0] / self.Ts).long()
        idx = torch.clamp(idx, min=0, max=wave_pre.shape[1] - 1)
        return wave_pre[:, idx].transpose(0, 1)

    def f_c(self, state, action):
        psi = state[:, 2:3]
        u = state[:, 3:4]
        v = state[:, 4:5]
        r = state[:, 5:6]
        t = state[:, 6:7]
        X = state[:, 7:8]
        Y = state[:, 8:9]
        N = state[:, 9:10]
        X_cmd = action[:, 0:1]
        Y_cmd = action[:, 1:2]
        N_cmd = action[:, 2:3]
        tau_wave = self.dis_wave(t)
        Xw = tau_wave[:, 0:1]
        Yw = tau_wave[:, 1:2]
        Nw = tau_wave[:, 2:3]
        dx = u * torch.cos(psi) - v * torch.sin(psi)
        dy = u * torch.sin(psi) + v * torch.cos(psi)
        dpsi = r
        du = (self.m_22 * v * r - self.d_11 * u + X + Xw) / self.m_11
        dv = (-self.m_11 * u * r - self.d_22 * v + Y + Yw) / self.m_22
        dr = ((self.m_11 - self.m_22) * u * v - self.d_33 * r + N + Nw) / self.m_33
        dt = torch.ones_like(t)
        dX = (X_cmd - X) / self.tau_Fx
        dY = (Y_cmd - Y) / self.tau_Fy
        dN = (N_cmd - N) / self.tau_Mz
        dcmd_prev = torch.zeros_like(action)
        rhs = torch.cat([dx, dy, dpsi, du, dv, dr, dt, dX, dY, dN, dcmd_prev], dim=1)
        return rhs

    def forward(self, state, action):
        k1 = self.f_c(state, action)
        k2 = self.f_c(state + (self.Ts / 2.0) * k1, action)
        k3 = self.f_c(state + (self.Ts / 2.0) * k2, action)
        k4 = self.f_c(state + self.Ts * k3, action)
        x_next = state + (self.Ts / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next = torch.cat([x_next[:, 0:2], angle_normalize(x_next[:, 2:3]), x_next[:, 3:10], action,], dim=1)
        return x_next

    def forward_XY(self, state, action, dt):
        k1 = self.f_c(state, action)
        k2 = self.f_c(state + (dt / 2.0) * k1, action)
        k3 = self.f_c(state + (dt / 2.0) * k2, action)
        k4 = self.f_c(state + dt * k3, action)
        x_next = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next = torch.cat([x_next[:, 0:2], angle_normalize(x_next[:, 2:3]), x_next[:, 3:10], action,], dim=1)
        return x_next



class ILQRMPCSolver(object):
    def __init__(self, horizon, N, LQR_ITER, delta_tau_max=(delta_tau_max_X, delta_tau_max_Y, delta_tau_max_M),) -> None:
        self.usvModel = HUSVModel(horizon, N)
        self.N = N
        self.LQR_ITER = LQR_ITER
        self.nx_physical = self.usvModel.nx_physical
        self.nx = self.usvModel.nx
        self.nu = self.usvModel.nu
        delta_tau_max = torch.as_tensor(delta_tau_max, dtype=torch.float64).reshape(-1)
        if delta_tau_max.numel() != 3:
            raise ValueError("delta_tau_max must contain exactly 3 elements: " "[Delta_Xcmd_max, Delta_Ycmd_max, Delta_Ncmd_max].")
        if torch.any(delta_tau_max <= 0):
            raise ValueError("All delta_tau_max values must be positive.")
        self.delta_tau_max = delta_tau_max
        self.u_init = None
        self.prev_tau_cmd = None

    def reset(self):
        self.u_init = None
        self.prev_tau_cmd = None

    def solver(self, ref_traj, wave_pre, prev_tau_cmd=None,):
        ref_traj = torch.as_tensor(ref_traj, dtype=torch.float64)
        N_BATCH = ref_traj.shape[0]
        dtype = ref_traj.dtype
        device = ref_traj.device
        action_low_single = ref_traj.new_tensor([self.usvModel.Fx_min, self.usvModel.Fy_min, self.usvModel.M_min])
        action_high_single = ref_traj.new_tensor([self.usvModel.Fx_max, self.usvModel.Fy_max, self.usvModel.M_max])
        action_low = action_low_single.view(1, 1, self.nu).expand(self.N, N_BATCH, self.nu).clone()
        action_high = action_high_single.view(1, 1, self.nu).expand(self.N, N_BATCH, self.nu).clone()
        self.usvModel.dis_wave_update(wave_pre)
        if prev_tau_cmd is None:
            if (self.prev_tau_cmd is None or self.prev_tau_cmd.shape[0] != N_BATCH):
                prev_tau_cmd = torch.zeros(N_BATCH, 3, dtype=dtype, device=device)
            else:
                prev_tau_cmd = self.prev_tau_cmd.to(dtype=dtype, device=device)
        else:
            prev_tau_cmd = torch.as_tensor(prev_tau_cmd, dtype=dtype, device=device)
            if prev_tau_cmd.ndim == 1:
                prev_tau_cmd = prev_tau_cmd.unsqueeze(0)
            if prev_tau_cmd.shape != (N_BATCH, 3):
                raise ValueError("prev_tau_cmd must have shape (B,3).")
        delta_tau_max = self.delta_tau_max.to(dtype=dtype, device=device).view(1, self.nu)
        rate_low = prev_tau_cmd - delta_tau_max
        rate_high = prev_tau_cmd + delta_tau_max
        first_low = torch.maximum(action_low_single.view(1, self.nu), rate_low)
        first_high = torch.minimum(action_high_single.view(1, self.nu), rate_high)
        if torch.any(first_low > first_high):
            raise RuntimeError("Infeasible first-move bounds: amplitude limits and " "delta_tau_max have an empty intersection.")
        action_low[0, :, :] = first_low
        action_high[0, :, :] = first_high
        state_physical = ref_traj[:, 0:9]
        t_tensor = torch.zeros(N_BATCH, 1, dtype=dtype, device=device)
        state = torch.cat([state_physical[:, 0:6], t_tensor, state_physical[:, 6:9], prev_tau_cmd,], dim=1)
        goal_state = ref_traj[:, 9:15]
        goal_weights = ref_traj[:, 15:21]
        ctrl_penalty = ref_traj[:, 21:24]
        R_delta = ref_traj[:, 24:27]
        nz = self.nx + self.nu
        C_single = torch.zeros(N_BATCH, nz, nz, dtype=dtype, device=device)
        c_single = torch.zeros(N_BATCH, nz, dtype=dtype, device=device)
        for j in range(6):
            C_single[:, j, j] = (2.0 * goal_weights[:, j])
        c_single[:, 0:6] = (-2.0 * goal_weights * goal_state)
        action_start = self.nx  # 13
        prev_start = 10
        for j in range(3):
            ip = prev_start + j
            iu = action_start + j
            r_u = ctrl_penalty[:, j]
            r_d = R_delta[:, j]
            C_single[:, ip, ip] += 2.0 * r_d
            C_single[:, iu, iu] += (2.0 * r_u + 2.0 * r_d)
            C_single[:, ip, iu] += -2.0 * r_d
            C_single[:, iu, ip] += -2.0 * r_d
        P = goal_weights
        C_terminal = torch.zeros(N_BATCH, nz, nz, dtype=dtype, device=device)
        c_terminal = torch.zeros(N_BATCH, nz, dtype=dtype, device=device)
        for j in range(6):
            C_terminal[:, j, j] = (2.0 * P[:, j])
        c_terminal[:, 0:6] = (-2.0 * P * goal_state)
        for j in range(3):
            iu = action_start + j
            C_terminal[:, iu, iu] = (2.0 * ctrl_penalty[:, j])
        if self.N > 1:
            C_stage = C_single.unsqueeze(0).expand(self.N - 1, -1, -1, -1)
            c_stage = c_single.unsqueeze(0).expand(self.N - 1, -1, -1)
            C = torch.cat([C_stage, C_terminal.unsqueeze(0)], dim=0).contiguous()
            c = torch.cat([c_stage, c_terminal.unsqueeze(0)], dim=0).contiguous()
        else:
            C = C_terminal.unsqueeze(0)
            c = c_terminal.unsqueeze(0)
        cost = mpc.QuadCost(C, c)
        ctrl = mpc.MPC(
            self.nx,
            self.nu,
            self.N,
            u_lower=action_low,
            u_upper=action_high,
            lqr_iter=self.LQR_ITER,
            exit_unconverged=False,
            eps=0.1,
            n_batch=N_BATCH,
            backprop=True,
            verbose=0,
            u_init=self.u_init,
            grad_method=mpc.GradMethods.AUTO_DIFF
        )
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, self.usvModel)
        self.u_init = torch.cat((nominal_actions[1:], nominal_actions[-1:]), dim=0).detach()
        U_opt = nominal_actions[0].clone()
        self.prev_tau_cmd = U_opt.detach()
        return U_opt


if __name__ == '__main__':
    device = torch.device("cpu")
    N = 25
    horizon = 5.0
    mpc_solver = ILQRMPCSolver(horizon, N, LQR_ITER=20, delta_tau_max = (delta_tau_max_X, delta_tau_max_Y, delta_tau_max_M),)
    nx = mpc_solver.nx_physical
    nu = mpc_solver.nu
    sim_time = 36.0
    Ts = 0.1
    env = realUSV(Ts, sim_time)
    env.reset()
    allocator = DiffAllocator(Ts=Ts)
    z_prev = torch.zeros(1, 4, dtype=torch.float64, device=device)
    eso = USV_ESO(Ts=Ts, omega_u = 10.0, omega_v = 10.0, omega_r = 10.0)
    steps = int(sim_time / Ts)
    x_sim = np.zeros([nx, steps + 1])
    u_sim = np.zeros([4, steps])
    tau_cmd_sim = np.zeros([3, steps])
    ob_dis_sim = np.zeros([3, steps])
    delta_tau_sim = np.zeros([3, steps])
    x_sim[:, 0] = env.state_3d
    prev_act = None
    each_step_history = np.zeros(steps)
    t_vec = Ts * np.linspace(0, steps, steps+1)
    ref_traj = np.concatenate((0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0]))), axis=0)
    history_T = 10.0
    pre_steps = int(history_T / Ts)
    wave_history = np.zeros([3, steps])
    lumpdis_history = np.zeros([3, steps])
    pre_wave = np.zeros([3, pre_steps])
    Q_s = torch.tensor([500, 500, 500, 0.005, 0.005, 0.005], device=device).reshape(1, -1)
    R_s = torch.tensor([0.005, 0.005, 0.005], device=device).reshape(1, -1)
    R_DELTA_s = torch.tensor([0.005, 0.005, 0.005], device=device).reshape(1, -1)
    prev_tau_cmd = torch.zeros(1, 3, dtype=torch.float64, device=device)
    start_time = time.time()
    for i in range(steps):
        ref_now = torch.tensor(ref_traj[:, i].reshape(1, -1), device=device)
        p_val = torch.cat([torch.tensor(env.state_3d_aug.reshape(1, -1), device=device), ref_now, Q_s, R_s, R_DELTA_s],dim=1)
        start_mpc = time.perf_counter()
        sol = mpc_solver.solver(p_val.double(), pre_wave, prev_tau_cmd)
        delta_tau = sol - prev_tau_cmd
        z = allocator(sol.double(), z_prev)
        end_mpc = time.perf_counter()
        print(f"mpc: {end_mpc - start_mpc:.6f} second")
        act = z[0].detach().cpu().numpy()
        x_aug, wave, lumpdis = env.run(act)
        z_prev = z.detach()
        x0 = env.state_3d
        nu_meas = x0[3:6]
        tau6_actual = env.thruster_generalized_force(env.actuator_state)
        tau_actual = np.array([tau6_actual[0], tau6_actual[1], tau6_actual[5]])
        tau_d_hat = eso.update(nu_meas, tau_actual)
        Xd_hat = tau_d_hat[0]
        Yd_hat = tau_d_hat[1]
        Nd_hat = tau_d_hat[2]
        ob_dis_sim[:, i] = tau_d_hat
        wave_history[:, i] = np.array([wave[0], wave[1], wave[5]])
        lumpdis_history[:, i] = lumpdis
        if i >= int(history_T / Ts - 1):
            pre_wave = arima_pre3d(ob_dis_sim[:, i - int(history_T / Ts - 1):i + 1], horizon/N, horizon, (4, 0, 0))
            # if i == int(20.0 / Ts - 1):
            #     np.savetxt(
            #         "../sim_data/ARNMPC/pre_Data_20.csv",
            #         pre_wave,
            #         delimiter=",",
            #         comments="",
            #         fmt="%.10e"
            #     )
            #     print("pre_Data_20.csv saved successfully.")
        x_sim[:, i + 1] = x0[0:6].reshape(-1)
        u_sim[:, i] = act
        tau_cmd_sim[:, i] = (sol.detach().cpu().numpy().reshape(-1))
        delta_tau_sim[:, i] = (delta_tau.detach().cpu().numpy().reshape(-1))
        prev_tau_cmd = sol.detach()
        POSITION_SCALE = 0.1
        HEADING_SCALE = 0.1
        position_error = x0[0:2]
        heading_error = np.atan2(np.sin(x0[2:3]), np.cos(x0[2:3]))
        position_loss = ((position_error / POSITION_SCALE)** 2).mean()
        heading_loss = ((heading_error / HEADING_SCALE)** 2).mean()
        tracking_loss = (2.0 * position_loss + heading_loss)
        thrust = act[[0, 2]]
        thrust_scale = np.array([allocator.T_max, allocator.T_max])
        control_loss = ((thrust / thrust_scale)** 2).mean()
        if prev_act is None:
            delta_control_loss = np.zeros([])
        else:
            delta_thrust = (thrust - prev_act[[0, 2]]) / thrust_scale
            angle_diff = np.atan2(np.sin(act[[1, 3]] - prev_act[[1, 3]]), np.cos(act[[1, 3]] - prev_act[[1, 3]]) )
            angle_scale = np.pi
            delta_angle = (angle_diff / angle_scale)
            delta_control_loss = ((delta_thrust ** 2).mean())
        prev_act = act
        weight_reg = (mpc_solver.usvModel.log_Q_scale.pow(2).mean() + mpc_solver.usvModel.log_R_scale.pow(
            2).mean() + mpc_solver.usvModel.log_R_delta_scale.pow(2).mean())
        loss = (tracking_loss + 0.5 * delta_control_loss)
        each_step_history[i] = loss.item()
        if i % 50 == 0:
            print(f"Sim time {i * Ts:.2f}s / {sim_time:.1f}s")
    print(np.linalg.norm(x_sim[0:3, int(history_T / Ts - 1):] - ref_traj[0:3, int(history_T / Ts - 1):]))
    end_time = time.time()
    print(end_time - start_time)
    plt.figure()
    plt.plot(np.linspace(0, steps, steps), each_step_history)
    plt.figure()
    plt.plot(x_sim[0, :], x_sim[1, :])
    plt.plot(ref_traj[0, :], ref_traj[1, :], 'r--')
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[0, :].reshape(-1))
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[0, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("x")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[1, :].reshape(-1))
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[1, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("y")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps + 1) * Ts, x_sim[2, :].reshape(-1))
    plt.plot(np.arange(steps + 1) * Ts, ref_traj[2, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("psi")
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(steps) * Ts, u_sim[0, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("T1")
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(steps) * Ts, u_sim[1, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("d1")
    plt.subplot(4, 1, 3)
    plt.plot(np.arange(steps) * Ts, u_sim[2, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("T2")
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(steps) * Ts, u_sim[3, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("d2")
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps) * Ts, lumpdis_history[0, :].reshape(-1))
    plt.plot(np.arange(steps) * Ts, wave_history[0, :].reshape(-1))
    plt.plot(np.arange(steps) * Ts, ob_dis_sim[0, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("X_w")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps) * Ts, lumpdis_history[1, :].reshape(-1))
    plt.plot(np.arange(steps) * Ts, wave_history[1, :].reshape(-1))
    plt.plot(np.arange(steps) * Ts, ob_dis_sim[1, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("Y_w")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps) * Ts, lumpdis_history[2, :].reshape(-1))
    plt.plot(np.arange(steps) * Ts, wave_history[2, :].reshape(-1))
    plt.plot(np.arange(steps) * Ts, ob_dis_sim[2, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("N_w")
    plt.figure()
    tau_names = ["Xcmd", "Ycmd", "Ncmd"]
    for j in range(3):
        plt.subplot(3, 1, j + 1)
        plt.plot(np.arange(steps) * Ts, tau_cmd_sim[j, :])
        plt.ylabel(tau_names[j])
        plt.grid(True)
    plt.xlabel("t(s)")
    plt.figure()
    delta_names = [ "Delta Xcmd", "Delta Ycmd", "Delta Ncmd"]
    for j in range(3):
        plt.subplot(3, 1, j + 1)
        plt.plot( np.arange(steps) * Ts, delta_tau_sim[j, :])
        plt.ylabel(delta_names[j])
        plt.grid(True)
    plt.xlabel("t(s)")
    time_state = np.arange(steps + 1) * Ts

    def pad_last_nan(data):
        data = np.asarray(data).reshape(-1)
        return np.concatenate([data, [np.nan]])
    csv_data = np.column_stack(
        [
            time_state,
            x_sim[0, :],
            x_sim[1, :],
            x_sim[2, :],
            ref_traj[0, :],
            ref_traj[1, :],
            ref_traj[2, :],
            pad_last_nan(each_step_history),
            pad_last_nan(u_sim[0, :]),
            pad_last_nan(u_sim[1, :]),
            pad_last_nan(u_sim[2, :]),
            pad_last_nan(u_sim[3, :]),
            pad_last_nan(lumpdis_history[0, :]),
            pad_last_nan(lumpdis_history[1, :]),
            pad_last_nan(lumpdis_history[2, :]),
            pad_last_nan(wave_history[0, :]),
            pad_last_nan(wave_history[1, :]),
            pad_last_nan(wave_history[2, :]),
            pad_last_nan(ob_dis_sim[0, :]),
            pad_last_nan(ob_dis_sim[1, :]),
            pad_last_nan(ob_dis_sim[2, :]),
            pad_last_nan(tau_cmd_sim[0, :]),
            pad_last_nan(tau_cmd_sim[1, :]),
            pad_last_nan(tau_cmd_sim[2, :]),
        ]
    )
    csv_header = (
        "time_s,"
        "x_m,y_m,psi_rad,"
        "x_ref_m,y_ref_m,psi_ref_rad,"
        "loss,"
        "T1_N,delta1_rad,T2_N,delta2_rad,"
        "Xd_N,Yd_N,Nd_Nm,"
        "Xw_N,Yw_N,Nw_Nm,"
        "Xd_hat_N,Yd_hat_N,Nd_hat_Nm,"
        "Xcmd_N,Ycmd_N,Ncmd_Nm"
    )
    np.savetxt("../sim_data/ARNMPC/ARNMPC_results_1.csv", csv_data, delimiter=",", header=csv_header, comments="", fmt="%.10e")
    print("simulation_results.csv saved successfully.")
    plt.show()

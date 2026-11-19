import numpy as np
import torch

from envModel_current import *
from USVmodel_sur import *
from highMpcSolver_ILQR_QR import *

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cpu")
    N = 25
    horizon = 5.0
    mpc_solver = ILQRMPCSolver(horizon, N, LQR_ITER=20, delta_tau_max=(delta_tau_max_X, delta_tau_max_Y, delta_tau_max_M),)
    nx = mpc_solver.nx
    nu = mpc_solver.nu
    sim_time = 6.0
    num_episode = 6
    Ts = 0.1
    env = realUSV(Ts, sim_time*num_episode)
    env_sur = surUSV(Ts, sim_time*num_episode)
    allocator = DiffAllocator(Ts=Ts)
    z_prev = torch.zeros(1, 4, dtype=torch.float64, device=device)
    eso = USV_ESO(Ts=Ts, omega_u=10.0, omega_v=10.0, omega_r=10.0)
    n_samples = int(sim_time / Ts)
    steps = num_episode*n_samples
    optimizer = torch.optim.Adam(list(mpc_solver.usvModel.parameters()), lr=0.1)
    update_interval = 5
    grad_accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    state = env.reset()
    t_vec = Ts * np.linspace(0, steps, steps + 1)
    ref_traj = np.concatenate((0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0])),
                               0.0 * np.ones((1, t_vec.shape[0]))), axis=0)
    loss_st = np.zeros(num_episode)
    prev_act = None
    history_T = 10.0
    pre_steps = int(history_T / Ts)
    ob_dis_history = np.zeros([3, steps])
    state_history = np.zeros([6, steps+1])
    state_history[:,0] = env.state_3d
    QR_history = np.zeros([12, steps])
    pre_wave = np.zeros([3, pre_steps])
    each_step_history = np.zeros(steps)
    each_step_history_rl = np.zeros(steps)
    u_history = np.zeros([4, steps])
    lumpdis_history = np.zeros([3, steps])
    wave_history = np.zeros([3, steps])
    tau_cmd_history = np.zeros([3, steps])
    prev_tau_cmd = torch.zeros(1, 3, dtype=torch.float64, device=device)
    wavedis_seq = []
    for epoch in range(num_episode):
        total_loss = 0
        mpc_solver.reset()
        for i in range(n_samples):
            ref_now = torch.tensor(ref_traj[:, i].reshape(1,-1), device=device)
            Q_s, R_s, R_DELTA_s = mpc_solver.usvModel.forward_QR()
            p_val = torch.cat([torch.tensor(env.state_3d_aug.reshape(1, -1),device=device),ref_now, Q_s, R_s, R_DELTA_s], dim=1)
            start_mpc = time.perf_counter()
            sol = mpc_solver.solver(p_val.double(), pre_wave, prev_tau_cmd)
            prev_tau_cmd = sol.detach()
            act = allocator(sol.double(), z_prev)
            z_prev = act.detach()
            end_mpc = time.perf_counter()
            act_numpy = act[0].detach().cpu().numpy()
            x_real, wave, lumpdis = env.run(act_numpy)
            x0 = env_sur.run(torch.tensor(env.state_3d_aug_T.reshape(1, -1), device=device), act, pre_wave[:, 0])
            nu_meas = env.state_3d[3:6]
            tau6_actual = env.thruster_generalized_force(env.actuator_state)
            tau_actual = np.array([tau6_actual[0], tau6_actual[1], tau6_actual[5]])
            tau_d_hat = eso.update(nu_meas, tau_actual)
            ob_dis_history[:, n_samples*epoch + i] = tau_d_hat
            state_history[:, n_samples*epoch + i + 1] = env.state_3d
            u_history[:, n_samples*epoch + i] = act_numpy
            QR_history[:, n_samples*epoch + i] = np.concatenate([Q_s.detach().cpu().numpy().reshape(-1), R_s.detach().cpu().numpy().reshape(-1), R_DELTA_s.detach().cpu().numpy().reshape(-1)], axis=0)
            wave_history[:, n_samples*epoch + i] = np.array([wave[0], wave[1], wave[5]])
            lumpdis_history[:, n_samples*epoch + i] = lumpdis
            tau_cmd_history[:, n_samples*epoch + i] = (sol.detach().cpu().numpy().reshape(-1))
            POSITION_SCALE = 0.1
            HEADING_SCALE = 0.1
            position_error = x0[:, 0:2]
            heading_error = torch.atan2(torch.sin(x0[:, 2:3]), torch.cos(x0[:, 2:3]))
            position_loss = (position_error / POSITION_SCALE).pow(2).mean()
            heading_loss = ( heading_error / HEADING_SCALE).pow(2).mean()
            tracking_loss = ( 2.0 * position_loss + heading_loss)
            position_error_rl = x_real[0:2]
            heading_error_rl = np.atan2(np.sin(x_real[5:6]), np.cos(x_real[5:6]))
            position_loss_rl = ((position_error_rl / POSITION_SCALE)**2).mean()
            heading_loss_rl = ((heading_error_rl / HEADING_SCALE)**2).mean()
            tracking_loss_rl = (2.0 * position_loss_rl + heading_loss_rl)
            thrust = act[:, [0, 2]]
            thrust_scale = thrust.new_tensor([allocator.T_max, allocator.T_max])
            control_loss = (thrust / thrust_scale).pow(2).mean()
            if prev_act is None:
                delta_control_loss = torch.zeros((), dtype=act.dtype, device=act.device)
            else:
                delta_thrust = (thrust - prev_act[:, [0, 2]]) / thrust_scale
                angle_diff = torch.atan2(torch.sin(act[:, [1, 3]] - prev_act[:, [1, 3]]),torch.cos(act[:, [1, 3]] - prev_act[:, [1, 3]]))
                angle_scale = np.pi
                delta_angle = (angle_diff / angle_scale)
                delta_control_loss = (delta_thrust.pow(2).mean())
            prev_act = act
            weight_reg = (mpc_solver.usvModel.log_Q_scale.pow(2).mean() + mpc_solver.usvModel.log_R_scale.pow(2).mean() + mpc_solver.usvModel.log_R_delta_scale.pow(2).mean())
            loss = (tracking_loss + 0.5 * delta_control_loss)
            if n_samples*epoch + i >= int(history_T / Ts - 1):
                pre_wave = arima_pre3d(ob_dis_history[:, n_samples*epoch + i - int(history_T / Ts - 1):n_samples*epoch + i + 1], horizon/N, horizon, (4, 0, 0))
            loss.backward()
            grad_accum_count += 1
            global_step = n_samples * epoch + i + 1
            is_update_step = (grad_accum_count == update_interval)
            is_last_step = (global_step == steps)
            if is_update_step or is_last_step:
                for param in mpc_solver.usvModel.parameters():
                    if param.grad is not None:
                        param.grad.div_(grad_accum_count)
                print("Q grad:", mpc_solver.usvModel.log_Q_scale.grad)
                print("R grad:", mpc_solver.usvModel.log_R_scale.grad)
                print("R_delta grad:", mpc_solver.usvModel.log_R_delta_scale.grad)
                torch.nn.utils.clip_grad_norm_(mpc_solver.usvModel.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_accum_count = 0
            total_loss = total_loss + loss.item()
            each_step_history[n_samples*epoch + i] = loss.item()
            each_step_history_rl[n_samples*epoch + i] = tracking_loss_rl.item()
            if i % 50 == 0:
                print(f"Sim time {i * Ts:.2f}s / {sim_time:.1f}s")
            print(
                epoch,
                loss.item(),
                tracking_loss_rl.item(),
                mpc_solver.usvModel.Q.reshape(1, -1).detach(),
                mpc_solver.usvModel.R.reshape(1, -1).detach(),
                mpc_solver.usvModel.R_delta.reshape(1, -1).detach(),
                mpc_solver.usvModel.log_Q_scale.reshape(1, -1).detach(),
                mpc_solver.usvModel.log_R_scale.reshape(1, -1).detach()
            )
            if isinstance(env.state, torch.Tensor):
                env.state_3d_aug = env.state_3d_aug.detach()
                env.state_3d_aug_T = env.state_3d_aug_T.detach()
            if isinstance(prev_act, torch.Tensor):
                prev_act = prev_act.detach()
        total_loss = (total_loss) / n_samples
        loss_st[epoch] = total_loss
        print(total_loss)
    print(np.linalg.norm(state_history[0:3, int(history_T / Ts - 1):] - ref_traj[0:3, int(history_T / Ts - 1):]))
    plt.figure()
    plt.plot(np.linspace(0, num_episode, num_episode), loss_st)
    plt.figure()
    plt.plot(np.linspace(0, steps, steps), each_step_history)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps+1) * Ts, state_history[0, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("x")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps+1) * Ts, state_history[1, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("y")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps+1) * Ts, state_history[2, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("psi")
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps) * Ts, QR_history[0, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("Q1")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps) * Ts, QR_history[1, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("Q2")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps) * Ts, QR_history[2, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("Q3")
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps) * Ts, QR_history[6, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("R1")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps) * Ts, QR_history[7, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("R2")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps) * Ts, QR_history[8, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("R3")
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(steps) * Ts, QR_history[9, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("dR1")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(steps) * Ts, QR_history[10, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("dR2")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(steps) * Ts, QR_history[11, :].reshape(-1))
    plt.xlabel("t(s)")
    plt.ylabel("dR3")
    time_state = np.arange(steps + 1) * Ts
    def pad_last_nan(data):
        data = np.asarray(data).reshape(-1)
        return np.concatenate([data, [np.nan]])
    csv_data = np.column_stack(
        [
            time_state,
            state_history[0, :],
            state_history[1, :],
            state_history[2, :],
            ref_traj[0, :],
            ref_traj[1, :],
            ref_traj[2, :],
            pad_last_nan(each_step_history),
            pad_last_nan(each_step_history_rl),
            pad_last_nan(u_history[0, :]),
            pad_last_nan(u_history[1, :]),
            pad_last_nan(u_history[2, :]),
            pad_last_nan(u_history[3, :]),
            pad_last_nan(lumpdis_history[0, :]),
            pad_last_nan(lumpdis_history[1, :]),
            pad_last_nan(lumpdis_history[2, :]),
            pad_last_nan(wave_history[0, :]),
            pad_last_nan(wave_history[1, :]),
            pad_last_nan(wave_history[2, :]),
            pad_last_nan(ob_dis_history[0, :]),
            pad_last_nan(ob_dis_history[1, :]),
            pad_last_nan(ob_dis_history[2, :]),
            pad_last_nan(tau_cmd_history[0, :]),
            pad_last_nan(tau_cmd_history[1, :]),
            pad_last_nan(tau_cmd_history[2, :]),
            pad_last_nan(QR_history[0, :]),
            pad_last_nan(QR_history[1, :]),
            pad_last_nan(QR_history[2, :]),
            pad_last_nan(QR_history[3, :]),
            pad_last_nan(QR_history[4, :]),
            pad_last_nan(QR_history[5, :]),
            pad_last_nan(QR_history[6, :]),
            pad_last_nan(QR_history[7, :]),
            pad_last_nan(QR_history[8, :]),
            pad_last_nan(QR_history[9, :]),
            pad_last_nan(QR_history[10, :]),
            pad_last_nan(QR_history[11, :]),
        ]
    )
    csv_header = (
        "time_s,"
        "x_m,y_m,psi_rad,"
        "x_ref_m,y_ref_m,psi_ref_rad,"
        "loss,loss_rl,"
        "T1_N,delta1_rad,T2_N,delta2_rad,"
        "Xd_N,Yd_N,Nd_Nm,"
        "Xw_N,Yw_N,Nw_Nm,"
        "Xd_hat_N,Yd_hat_N,Nd_hat_Nm,"
        "Xcmd_N,Ycmd_N,Ncmd_Nm,"
        "Q_1,Q_2,Q_3,Q_4,Q_5,Q_6,"
        "R_1,R_2,R_3,"
        "R_delta_1,R_delta_2,R_delta_3"
    )
    np.savetxt("../sim_data/DANMPC/DANMPC_results_1.csv", csv_data, delimiter=",", header=csv_header, comments="", fmt="%.10e")
    print("DAMPC_results.csv saved successfully.")
    plt.show()

import numpy as np
from usv_parameters_6dof import *


def angle_normalize(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def skew(v):
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z,   y], [z,    0.0, -x], [-y,   x,   0.0]], dtype=np.float64)

def rotation_matrix_321(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps * cth, cps * sth * sphi - sps * cphi, cps * sth * cphi + sps * sphi],
        [sps * cth, sps * sth * sphi + cps * cphi, sps * sth * cphi - cps * sphi],
        [-sth, cth * sphi, cth * cphi]
    ], dtype=np.float64)


def euler_rate_matrix(phi, theta):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    if abs(cth) < 1e-6:
        cth = np.sign(cth) * 1e-6 if cth != 0.0 else 1e-6
    tth = np.sin(theta) / cth
    return np.array([[1.0, sphi * tth, cphi * tth], [0.0, cphi,       -sphi], [0.0, sphi / cth, cphi / cth]], dtype=np.float64)

class JONSWAPWaveLoads:

    def __init__(
        self,
        duration,
        dt,
        hs=1.5,
        tp=5.0,
        gamma=3.3,
        wave_direction=np.deg2rad(30.0),
        n_frequency=200,
        frequency_min=0.02,
        frequency_max=1.0,
        water_depth=None,
        seed=10,
        force_gains=None,
        moment_gains=None,
        omega_ref=None,
    ):
        self.duration = float(duration)
        self.dt = float(dt)
        self.hs = float(hs)
        self.tp = float(tp)
        self.gamma = float(gamma)
        self.wave_direction = float(wave_direction)
        self.water_depth = water_depth
        self.g = 9.81
        self.frequency = np.linspace(frequency_min, frequency_max, n_frequency, dtype=np.float64)
        self.df = (self.frequency[1] - self.frequency[0])
        self.omega = (2.0 * np.pi * self.frequency)
        self.spectrum = (self._jonswap_spectrum(self.frequency, self.hs))
        self.amplitude = np.sqrt(2.0 * self.spectrum * self.df)
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=n_frequency)
        self.wave_number = (self._compute_wave_number(self.omega))
        if force_gains is None:
            force_gains = [350.0, 450.0, 700.0]
        self.force_gains = np.asarray(force_gains, dtype=np.float64).reshape(3)
        if moment_gains is None:
            moment_gains = [500.0, 650.0, 400.0]
        self.moment_gains = np.asarray(moment_gains, dtype=np.float64).reshape(3)
        self.omega_ref = (float(omega_ref) if omega_ref is not None else 2.0 * np.pi / self.tp)
        self.load_phase = np.array([0.0, 0.15 * np.pi, 0.50 * np.pi, 0.35 * np.pi, 0.65 * np.pi, 0.20 * np.pi], dtype=np.float64)

    def _jonswap_spectrum(self, frequency, hs):
        fp = 1.0 / self.tp
        f = np.maximum(np.asarray(frequency, dtype=np.float64), 1e-8)
        sigma = np.where(f <= fp, 0.07, 0.09)
        peak_shape = np.exp(-0.5 * ((f - fp) / (sigma * fp)) ** 2)
        S0 = (self.g**2 * (2.0 * np.pi)**(-4) * f**(-5) * np.exp(-1.25 * (fp / f)**4) * self.gamma**peak_shape)
        target_m0 = hs**2 / 16.0
        current_m0 = np.trapezoid(S0, f)
        if current_m0 <= 0.0:
            raise RuntimeError("JONSWAP spectrum integral is invalid.")
        return (S0 * target_m0 / current_m0)

    def _compute_wave_number(self, omega):
        if self.water_depth is None:
            return (omega**2 / self.g)
        h = float(self.water_depth)
        if h <= 0.0:
            raise ValueError("water_depth must be positive.")
        k = omega**2 / self.g
        for _ in range(30):
            kh = k * h
            tanh_kh = np.tanh(kh)
            cosh_kh = np.cosh(kh)
            F = (self.g * k * tanh_kh - omega**2)
            dF = (self.g * (tanh_kh + kh / (cosh_kh**2)))
            k = (k - F / (dF + 1e-12))
        return k


    def get_current_hs(self, t):
        return self.hs


    def _phase_argument(self, t, x, y):
        direction = np.array([np.cos(self.wave_direction), np.sin(self.wave_direction)])
        s = (x * direction[0] + y * direction[1])
        return (self.wave_number * s - self.omega * t + self.phase)


    def wave_elevation(self, t, x=0.0, y=0.0):
        arg = self._phase_argument(float(t), float(x), float(y))
        eta = np.sum(self.amplitude * np.cos(arg))
        return float(eta)

    def generalized_wave_load(self, t, position, euler):
        x, y, _ = np.asarray(position, dtype=np.float64)
        _, _, psi = np.asarray(euler, dtype=np.float64)
        arg = self._phase_argument(float(t), x, y)
        beta = angle_normalize(self.wave_direction - psi)
        cb = np.cos(beta)
        sb = np.sin(beta)
        freq_shape = (1.0 / np.sqrt(1.0 + (self.omega / (2.0 * self.omega_ref + 1e-12)) ** 4))
        q = (self.amplitude * freq_shape)
        exc0 = np.sum(q * np.cos(arg + self.load_phase[0]))
        exc1 = np.sum(q * np.cos(arg + self.load_phase[1]))
        exc2 = np.sum(q * np.cos(arg + self.load_phase[2]))
        exc3 = np.sum(q * np.cos(arg + self.load_phase[3]))
        exc4 = np.sum(q * np.cos(arg + self.load_phase[4]))
        exc5 = np.sum(q * np.cos(arg + self.load_phase[5]))
        Xw = (self.force_gains[0] * cb * exc0)
        Yw = (self.force_gains[1] * sb * exc1)
        Zw = (self.force_gains[2] * exc2)
        Kw = (self.moment_gains[0] * sb * exc3)
        Mw = (self.moment_gains[1] * cb * exc4)
        Nw = (self.moment_gains[2] * np.sin(2.0 * beta) * exc5)
        return np.array([Xw, Yw, Zw, Kw, Mw, Nw], dtype=np.float64)

class realUSV:

    def __init__(self, Ts, sim_time):
        self.nx = 12
        self.nu = 4
        self.Ts = float(Ts)
        self.sim_time = float(sim_time)
        self.num_steps = int(round(self.sim_time / self.Ts))
        self.M = np.diag([M_11_dis, M_22_dis, M_33_dis, M_44_dis, M_55_dis, M_66_dis]).astype(np.float64)
        self.D_lin = np.diag([D_11_dis, D_22_dis, D_33_dis, D_44_dis, D_55_dis, D_66_dis]).astype(np.float64)
        self.D_quad = np.diag([DQ_11, DQ_22, DQ_33, DQ_44, DQ_55, DQ_66]).astype(np.float64)
        self.K_heave = K_HEAVE
        self.K_roll = K_ROLL
        self.K_pitch = K_PITCH
        self.lx1, self.ly1 = LX1, LY1
        self.lx2, self.ly2 = LX2, LY2
        self.T_min, self.T_max = T_MIN, T_MAX
        self.delta_min, self.delta_max = -np.pi, np.pi
        self.tau_T = TAU_T
        self.tau_delta = TAU_DELTA
        self.T_rate_max = T_RATE_MAX
        self.delta_rate_max = DELTA_RATE_MAX
        self.current_speed = CURRENT_SPEED
        self.current_direction = CURRENT_DIRECTION
        self.wave_env = JONSWAPWaveLoads(
            duration = max(60.0, self.sim_time + self.Ts),
            dt=self.Ts,
            hs=WAVE_HS,
            tp=WAVE_TP,
            gamma=WAVE_GAMMA,
            wave_direction=WAVE_DIRECTION,
            n_frequency=WAVE_N_FREQ,
            frequency_min=WAVE_F_MIN,
            frequency_max=WAVE_F_MAX,
            seed=WAVE_SEED
        )
        self.t = 0.0
        self.actuator_state = np.zeros(4, dtype=np.float64)
        self.wave_load = np.zeros(6, dtype=np.float64)
        self.reset()

    def get_current_velocity(self, eta, t=0.0):
        eta = np.asarray(eta, dtype=np.float64).reshape(6)
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]
        Vc = self.current_speed
        beta_c = self.current_direction
        current_inertial = np.array([Vc * np.cos(beta_c), Vc * np.sin(beta_c), 0.0], dtype=np.float64)
        R = rotation_matrix_321(phi, theta, psi)
        current_body = (R.T @ current_inertial)
        nu_current = np.zeros(6, dtype=np.float64)
        nu_current[0:3] = current_body
        return nu_current

    def _format_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.size != 12:
            raise ValueError(f"state must have 12 elements, got {state.size}.")
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or Inf.")
        return state

    def _format_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 4:
            raise ValueError("action must be [T1_cmd, delta1_cmd, T2_cmd, delta2_cmd].")
        out = action.copy()
        out[0] = np.clip(out[0], self.T_min, self.T_max)
        out[1] = out[1]
        out[2] = np.clip(out[2], self.T_min, self.T_max)
        out[3] = out[3]
        return out

    def _angle_error(self, target, current):
        return np.arctan2(np.sin(target - current), np.cos(target - current))

    def thruster_generalized_force(self, actuator_state):
        T1, delta1, T2, delta2 = np.asarray(actuator_state, dtype=np.float64).reshape(4)
        Fx1 = T1*np.cos(delta1)
        Fy1 = T1*np.sin(delta1)
        Fx2 = T2*np.cos(delta2)
        Fy2 = T2*np.sin(delta2)
        X = Fx1 + Fx2
        Y = Fy1 + Fy2
        N = (self.lx1*Fy1 - self.ly1*Fx1 + self.lx2*Fy2 - self.ly2*Fx2)
        return np.array([X, Y, 0.0, 0.0, 0.0, N], dtype=np.float64)

    def coriolis_matrix(self, nu):
        nu = np.asarray(nu, dtype=np.float64).reshape(6)
        v = nu[:3]
        omega = nu[3:]
        M_t = self.M[:3, :3]
        M_r = self.M[3:, 3:]
        p_lin = M_t @ v
        h_ang = M_r @ omega
        C = np.zeros((6, 6), dtype=np.float64)
        C[:3, 3:] = -skew(p_lin)
        C[3:, :3] = -skew(p_lin)
        C[3:, 3:] = -skew(h_ang)
        return C

    def damping_force(self, nu):
        nu = np.asarray(nu, dtype=np.float64).reshape(6)
        return self.D_lin @ nu + self.D_quad @ (np.abs(nu)*nu)

    def restoring_force(self, eta):
        eta = np.asarray(eta, dtype=np.float64).reshape(6)
        z, phi, theta = eta[2], eta[3], eta[4]
        return np.array([0.0, 0.0, self.K_heave*z, self.K_roll*phi, self.K_pitch*theta, 0.0], dtype=np.float64)

    def kinematics(self, eta, nu):
        x, y, z, phi, theta, psi = eta
        u, v, w, p, q, r = nu
        R = rotation_matrix_321(phi, theta, psi)
        T = euler_rate_matrix(phi, theta)
        pos_dot = R @ np.array([u, v, w], dtype=np.float64)
        eul_dot = T @ np.array([p, q, r], dtype=np.float64)
        return np.concatenate([pos_dot, eul_dot])

    def get_wave_load(self, state, t):
        state = self._format_state(state)
        eta = state[:6]
        return self.wave_env.generalized_wave_load(t=float(t), position=eta[:3], euler=eta[3:6])

    def f_aug(self, aug_state, action_cmd, t):
        aug_state = np.asarray(aug_state, dtype=np.float64).reshape(-1)
        if aug_state.size != 16:
            raise ValueError("aug_state must have 16 elements.")
        action_cmd = self._format_action(action_cmd)
        eta = aug_state[:6]
        nu = aug_state[6:12]
        actuator = aug_state[12:16]
        T1, delta1, T2, delta2 = actuator
        T1_cmd, delta1_cmd, T2_cmd, delta2_cmd = action_cmd
        tau_control = self.thruster_generalized_force(actuator)
        tau_wave = self.wave_env.generalized_wave_load(t=float(t), position=eta[:3], euler=eta[3:6])
        eta_dot = self.kinematics(eta, nu)
        nu_current = self.get_current_velocity(eta, t)
        nu_r = (nu - nu_current)
        C = self.coriolis_matrix(nu_r)
        Dnu = self.damping_force(nu_r)
        g_eta = self.restoring_force(eta)
        rhs = (tau_control + tau_wave - C @ nu_r - Dnu - g_eta)
        nu_dot = np.linalg.solve(self.M, rhs)
        dT1_raw = (T1_cmd - T1) / self.tau_T
        dT2_raw = (T2_cmd - T2) / self.tau_T
        ddelta1_raw = (self._angle_error(delta1_cmd, delta1) / self.tau_delta)
        ddelta2_raw = (self._angle_error(delta2_cmd, delta2) / self.tau_delta)
        dT1 = np.clip(dT1_raw, -self.T_rate_max, self.T_rate_max)
        dT2 = np.clip(dT2_raw, -self.T_rate_max, self.T_rate_max)
        ddelta1 = np.clip(ddelta1_raw, -self.delta_rate_max, self.delta_rate_max)
        ddelta2 = np.clip(ddelta2_raw, -self.delta_rate_max, self.delta_rate_max)
        return np.concatenate([eta_dot, nu_dot, np.array([dT1, ddelta1, dT2, ddelta2], dtype=np.float64)])

    def run(self, action_cmd):
        action_cmd = self._format_action(action_cmd)
        aug = np.concatenate([self.state, self.actuator_state])
        t0 = self.t
        dt = self.Ts
        k1 = self.f_aug(aug, action_cmd, t0)
        k2 = self.f_aug(aug + 0.5 * dt * k1, action_cmd, t0 + 0.5 * dt)
        k3 = self.f_aug(aug + 0.5 * dt * k2, action_cmd, t0 + 0.5 * dt)
        k4 = self.f_aug(aug + dt * k3, action_cmd, t0 + dt)
        aug_next = ( aug + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        aug_next[3] = angle_normalize(aug_next[3])
        aug_next[4] = angle_normalize(aug_next[4])
        aug_next[5] = angle_normalize(aug_next[5])
        aug_next[13] = angle_normalize(aug_next[13])
        aug_next[15] = angle_normalize(aug_next[15])
        aug_next[12] = np.clip(aug_next[12], self.T_min, self.T_max)
        aug_next[14] = np.clip(aug_next[14], self.T_min, self.T_max)
        self.state = aug_next[:12].copy()
        self.actuator_state = (aug_next[12:16].copy())
        self.t = t0 + dt
        tau_control = ( self.thruster_generalized_force(self.actuator_state))
        self.wave_load = self.get_wave_load(self.state, self.t)
        eta_6 = self.state[:6].copy()
        nu_6 = self.state[6:12].copy()
        C_6 = self.coriolis_matrix(nu_6)
        Dnu_6 = self.damping_force(nu_6)
        g_6 = self.restoring_force(eta_6)
        rhs_6 = (tau_control + self.wave_load - C_6 @ nu_6 - Dnu_6 - g_6)
        nu_dot_6 = np.linalg.solve(self.M, rhs_6)
        idx_3 = [0, 1, 5]
        nu_3 = nu_6[idx_3].copy()
        nu_dot_3 = nu_dot_6[idx_3].copy()
        tau_control_3 = ( tau_control[idx_3].copy())
        M_3 = self.M[ np.ix_(idx_3, idx_3)].copy()
        nu_nominal_6 = np.array(
            [
                nu_3[0],  # u
                nu_3[1],  # v
                0.0,  # w
                0.0,  # p
                0.0,  # q
                nu_3[2]  # r
            ],
            dtype=np.float64
        )
        C_nominal_6 = (self.coriolis_matrix(nu_nominal_6))
        C_3 = C_nominal_6[np.ix_(idx_3, idx_3)]
        D_lin_3 = self.D_lin[np.ix_(idx_3, idx_3)]
        D_quad_3 = self.D_quad[np.ix_(idx_3, idx_3)]
        Dnu_3 = (D_lin_3 @ nu_3 + D_quad_3 @ (np.abs(nu_3) * nu_3))
        tau_disturbance_true = (M_3 @ nu_dot_3 + C_3 @ nu_3 + Dnu_3 - tau_control_3)
        self.total_disturbance_true = (tau_disturbance_true.copy())
        self.state_3d = aug_next[[0, 1, 5, 6, 7, 11]].copy()
        self.state_3d_aug = np.concatenate([self.state_3d, tau_control[[0, 1, 5]]])
        self.state_aug = np.concatenate([self.state, tau_control[[0, 1, 5]]])
        self.state_3d_aug_T = np.concatenate([self.state_3d, self.actuator_state])
        return (self.state.copy(), self.wave_load.copy(), self.total_disturbance_true.copy())

    def reset(self, state=None):
        self.t = 0.0
        if state is None:
            self.state = np.array([
                1.3, 1.3, 0.0,                 # x,y,z
                0.0, 0.0, np.deg2rad(30.0),    # phi,theta,psi
                0.0, 0.0, 0.0,                 # u,v,w
                0.0, 0.0, 0.0                  # p,q,r
            ], dtype=np.float64)
            self.state_3d = self.state[[0, 1, 5, 6, 7, 11]].copy()
        else:
            self.state = self._format_state(state).copy()
            self.state_3d = self.state[[0, 1, 5, 6, 7, 11]].copy()
        self.state[3:6] = np.array([angle_normalize(a) for a in self.state[3:6]], dtype=np.float64)
        self.actuator_state = np.zeros(4, dtype=np.float64)
        self.wave_load = self.get_wave_load(self.state, self.t)
        tau_control = self.thruster_generalized_force(self.actuator_state)
        self.state_3d_aug = np.concatenate([self.state_3d, tau_control[[0, 1, 5]]])
        self.state_aug = np.concatenate([self.state, tau_control[[0, 1, 5]]])
        self.state_3d_aug_T = np.concatenate([self.state_3d, self.actuator_state])
        return self.state.copy()



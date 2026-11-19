import numpy as np
from usv_parameters_6dof import *

class USV_ESO:
    def __init__(self, Ts, omega_u=5.0, omega_v=5.0, omega_r=5.0):
        self.Ts = float(Ts)
        self.m11 = float(M_11)
        self.m22 = float(M_22)
        self.m33 = float(M_66)
        self.d11 = float(D_11)
        self.d22 = float(D_22)
        self.d33 = float(D_66)
        self.M = np.diag([self.m11, self.m22, self.m33])
        self.omega = np.array([omega_u, omega_v, omega_r], dtype=np.float64)
        self.beta1 = 2.0 * self.omega
        self.beta2 = self.omega ** 2
        self.nu_hat = np.zeros(3, dtype=np.float64)
        self.tau_d_hat = np.zeros(3, dtype=np.float64)
        self.initialized = False

    def nominal_acceleration(self, nu, tau):
        u, v, r = nu
        X, Y, N = tau
        du = (self.m22 * v * r - self.d11 * u + X) / self.m11
        dv = (-self.m11 * u * r - self.d22 * v + Y) / self.m22
        dr = ((self.m11 - self.m22) * u * v - self.d33 * r + N) / self.m33
        return np.array([du, dv, dr], dtype=np.float64)

    def eso_dynamics(self, observer_state, nu_meas, tau_actual):
        nu_hat = observer_state[0:3]
        tau_d_hat = observer_state[3:6]
        e = nu_meas - nu_hat
        f_nom = self.nominal_acceleration(nu_meas, tau_actual)
        disturbance_acc = np.array([tau_d_hat[0] / self.m11, tau_d_hat[1] / self.m22, tau_d_hat[2] / self.m33])
        nu_hat_dot = (f_nom + disturbance_acc + self.beta1 * e)
        tau_d_hat_dot = np.array([self.m11 * self.beta2[0] * e[0], self.m22 * self.beta2[1] * e[1], self.m33 * self.beta2[2] * e[2]])
        return np.concatenate([nu_hat_dot, tau_d_hat_dot])

    def update(self, nu_meas, tau_actual):
        nu_meas = np.asarray(nu_meas, dtype=np.float64).reshape(3)
        tau_actual = np.asarray(tau_actual, dtype=np.float64).reshape(3)
        if not self.initialized:
            self.nu_hat = nu_meas.copy()
            self.tau_d_hat[:] = 0.0
            self.initialized = True
            return self.tau_d_hat.copy()
        x_obs = np.concatenate([self.nu_hat, self.tau_d_hat])
        dt = self.Ts
        k1 = self.eso_dynamics(x_obs, nu_meas, tau_actual)
        k2 = self.eso_dynamics(x_obs + 0.5 * dt * k1, nu_meas, tau_actual)
        k3 = self.eso_dynamics(x_obs + 0.5 * dt * k2, nu_meas, tau_actual)
        k4 = self.eso_dynamics(x_obs + dt * k3, nu_meas, tau_actual)
        x_obs_next = (x_obs + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        self.nu_hat = x_obs_next[0:3]
        self.tau_d_hat = x_obs_next[3:6]
        return self.tau_d_hat.copy()

    def reset(self):
        self.nu_hat[:] = 0.0
        self.tau_d_hat[:] = 0.0
        self.initialized = False


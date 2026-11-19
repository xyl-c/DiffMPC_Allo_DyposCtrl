import time
from collections import namedtuple

import torch
from torch.autograd import Function, Variable

from . import util, mpc
from .pnqp import pnqp

LqrBackOut = namedtuple("lqrBackOut", "n_total_qp_iter")
LqrForOut = namedtuple("lqrForOut", "objs full_du_norm alpha_du_norm mean_alphas costs",)


def LQRStep(
    n_state,
    n_ctrl,
    T,
    u_lower=None,
    u_upper=None,
    u_zero_I=None,
    delta_u=None,
    linesearch_decay=0.2,
    max_linesearch_iter=10,
    true_cost=None,
    true_dynamics=None,
    delta_space=True,
    current_x=None,
    current_u=None,
    verbose=0,
    back_eps=1e-3,
    no_op_forward=False,
):
    """A batch-safe box-constrained iLQR step.

    Supported bound shapes:
        scalar
        [n_ctrl]
        [n_batch, n_ctrl]
        [T, n_ctrl]
        [T, n_batch, n_ctrl]

    All internal state/control tensors keep the batch dimension, including
    the special case ``n_batch == 1``.
    """

    def _as_batch_control(value, t, n_batch, ref, name):
        """Return a control quantity with shape [n_batch, n_ctrl]."""
        if value is None:
            return None
        if isinstance(value, (float, int)):
            return torch.full((n_batch, n_ctrl), float(value), dtype=ref.dtype, device=ref.device,)
        value = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
        if value.ndim == 0:
            return value.expand(n_batch, n_ctrl)
        if value.ndim == 1:
            if value.numel() == 1:
                return value.reshape(1, 1).expand(n_batch, n_ctrl)
            if value.shape[0] != n_ctrl:
                raise ValueError(f"{name} with one dimension must have shape [{n_ctrl}], " f"but got {tuple(value.shape)}.")
            return value.unsqueeze(0).expand(n_batch, -1)
        if value.ndim == 2:
            # [n_batch, n_ctrl]
            if value.shape == (n_batch, n_ctrl):
                return value
            # [1, n_ctrl]
            if value.shape == (1, n_ctrl):
                return value.expand(n_batch, -1)
            # [T, n_ctrl]
            if value.shape[0] in (1, T) and value.shape[1] == n_ctrl:
                index = 0 if value.shape[0] == 1 else t
                return value[index].unsqueeze(0).expand(n_batch, -1)
            raise ValueError(f"Unsupported {name} shape {tuple(value.shape)}. Expected " f"[{n_batch}, {n_ctrl}], [1, {n_ctrl}], or [{T}, {n_ctrl}].")
        if value.ndim == 3:
            # [T or 1, n_batch or 1, n_ctrl]
            if value.shape[0] not in (1, T):
                raise ValueError(f"The time dimension of {name} must be 1 or {T}, " f"but got {value.shape[0]}.")
            if value.shape[1] not in (1, n_batch):
                raise ValueError(f"The batch dimension of {name} must be 1 or {n_batch}, " f"but got {value.shape[1]}.")
            if value.shape[2] != n_ctrl:
                raise ValueError(f"The control dimension of {name} must be {n_ctrl}, " f"but got {value.shape[2]}.")
            index = 0 if value.shape[0] == 1 else t
            result = value[index]
            if result.shape[0] == 1:
                result = result.expand(n_batch, -1)
            return result
        raise ValueError(f"Unsupported {name} rank {value.ndim}; shape={tuple(value.shape)}.")

    def _bound(side, t, n_batch, ref):
        if side == "lower":
            value = u_lower
        elif side == "upper":
            value = u_upper
        else:
            raise ValueError(f"Unknown bound side: {side}")
        return _as_batch_control(value, t, n_batch, ref, f"u_{side}")

    def _zero_mask(t, n_batch, ref):
        mask = _as_batch_control(u_zero_I, t, n_batch, ref, "u_zero_I")
        return None if mask is None else mask.bool()

    def lqr_backward(ctx, C, c, F, f):
        n_batch = C.size(1)
        u = ctx.current_u
        Ks = []
        ks = []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1, 2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
                propagated_v = Ft_T.bmm(vtp1.unsqueeze(-1)).squeeze(-1)
                if f is None or f.nelement() == 0:
                    qt = c[t] + propagated_v
                else:
                    ft = f[t]
                    qt = (c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(-1)).squeeze(-1) + propagated_v)
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]
            if u_lower is None:
                active_mask = _zero_mask(t, n_batch, Qt_uu)
                if n_ctrl == 1 and active_mask is None:
                    Kt = -(1.0 / Qt_uu) * Qt_ux
                    kt = -(1.0 / Qt_uu.squeeze(-1)) * qt_u
                elif active_mask is None:
                    # Batch-safe pseudo-inverse: [B, nu, nu].
                    Qt_uu_inv = torch.linalg.pinv(Qt_uu)
                    Kt = -Qt_uu_inv.bmm(Qt_ux)
                    kt = util.bmv(-Qt_uu_inv, qt_u)
                else:
                    # Constrain active controls to zero without collapsing B=1.
                    qt_u_ = qt_u.masked_fill(active_mask, 0.0)
                    free_mask = ~active_mask
                    free_outer = util.bger(free_mask.to(Qt_uu.dtype), free_mask.to(Qt_uu.dtype),).bool()
                    Qt_uu_ = Qt_uu.masked_fill(~free_outer, 0.0)
                    # Put a small positive value on active diagonal entries.
                    # diag_embed preserves [B, nu, nu], including B=1.
                    active_diag = torch.diag_embed(active_mask.to(Qt_uu.dtype))
                    Qt_uu_ = Qt_uu_ + 1e-8 * active_diag
                    state_mask = active_mask.unsqueeze(-1).expand(-1, -1, Qt_ux.size(-1))
                    Qt_ux_ = Qt_ux.masked_fill(state_mask, 0.0)
                    if n_ctrl == 1:
                        Kt = -(1.0 / Qt_uu_) * Qt_ux_
                        kt = -(1.0 / Qt_uu_.squeeze(-1)) * qt_u_
                    else:
                        # solve(A, B) is batch-safe and avoids deprecated LU APIs.
                        Kt = -torch.linalg.solve(Qt_uu_, Qt_ux_)
                        kt = -torch.linalg.solve(Qt_uu_, qt_u_.unsqueeze(-1)).squeeze(-1)
            else:
                assert delta_space
                lb = _bound("lower", t, n_batch, u[t]) - u[t]
                ub = _bound("upper", t, n_batch, u[t]) - u[t]
                if delta_u is not None:
                    lb = torch.clamp(lb, min=-float(delta_u))
                    ub = torch.clamp(ub, max=float(delta_u))
                kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(Qt_uu, qt_u, lb, ub, x_init=prev_kt, n_iter=20,)
                if verbose > 1:
                    print("  + n_qp_iter: ", n_qp_iter + 1)
                n_total_qp_iter += 1 + n_qp_iter
                prev_kt = kt
                fixed_mask = (~If.bool()).unsqueeze(-1).expand(-1, -1, Qt_ux.size(-1))
                Qt_ux_ = Qt_ux.masked_fill(fixed_mask, 0.0)
                if n_ctrl == 1:
                    # Qt_uu_free_LU is not an LU factor in the scalar case.
                    Kt = -(1.0 / Qt_uu_free_LU) * Qt_ux_
                else:
                    # pnqp in this repository returns the legacy LU tuple.
                    Kt = -torch.lu_solve(Qt_ux_, *Qt_uu_free_LU)
            Kt_T = Kt.transpose(1, 2)
            Ks.append(Kt)
            ks.append(kt)
            Vtp1 = (Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt))
            vtp1 = (
                qt_x
                + Qt_xu.bmm(kt.unsqueeze(-1)).squeeze(-1)
                + Kt_T.bmm(qt_u.unsqueeze(-1)).squeeze(-1)
                + Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(-1)).squeeze(-1)
            )
        return Ks, ks, n_total_qp_iter

    def lqr_forward(ctx, x_init, C, c, F, f, Ks, ks):
        x = ctx.current_x
        u = ctx.current_u
        n_batch = C.size(1)
        old_cost = util.get_cost(T, u, true_cost, true_dynamics, x=x)
        current_cost = None
        alphas = torch.ones(n_batch, dtype=C.dtype, device=C.device)
        full_du_norm = None
        i = 0
        while (current_cost is None or (old_cost is not None and torch.any(current_cost > old_cost).item())) and i < max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(T):
                t_rev = T - 1 - t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                ut = u[t]
                dxt = dx[t]
                # [B, 1] * [B, nu] is safer than diag(alphas) @ kt.
                new_ut = util.bmv(Kt, dxt) + ut + alphas.unsqueeze(-1) * kt
                assert not ((delta_u is not None) and (u_lower is None))
                active_mask = _zero_mask(t, n_batch, new_ut)
                if active_mask is not None:
                    new_ut = new_ut.masked_fill(active_mask, 0.0)
                if u_lower is not None:
                    lb_limit = _bound("lower", t, n_batch, new_ut)
                    ub_limit = _bound("upper", t, n_batch, new_ut)
                    if delta_u is not None:
                        lb = torch.maximum(u[t] - float(delta_u), lb_limit)
                        ub = torch.minimum(u[t] + float(delta_u), ub_limit)
                    else:
                        lb, ub = lb_limit, ub_limit
                    new_ut = util.eclamp(new_ut, lb, ub)
                new_u.append(new_ut)
                new_xut = torch.cat((new_xt, new_ut), dim=-1)
                if t < T - 1:
                    if isinstance(true_dynamics, mpc.LinDx):
                        dyn_F, dyn_f = true_dynamics.F, true_dynamics.f
                        new_xtp1 = util.bmv(dyn_F[t], new_xut)
                        if dyn_f is not None and dyn_f.nelement() > 0:
                            new_xtp1 = new_xtp1 + dyn_f[t]
                    else:
                        new_xtp1 = true_dynamics(new_xt, new_ut)
                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t + 1])
                if isinstance(true_cost, mpc.QuadCost):
                    cost_C, cost_c = true_cost.C, true_cost.c
                    obj = 0.5 * util.bquad(new_xut, cost_C[t]) + util.bdot(new_xut, cost_c[t])
                else:
                    obj = true_cost(new_xut)
                objs.append(obj)
            objs = torch.stack(objs, dim=0)
            current_cost = torch.sum(objs, dim=0)
            new_u = torch.stack(new_u, dim=0)
            new_x = torch.stack(new_x, dim=0)
            if full_du_norm is None:
                full_du_norm = ((u - new_u) .transpose(1, 2) .contiguous() .reshape(n_batch, -1) .norm(2, dim=1))
            if old_cost is not None:
                alphas = torch.where(current_cost > old_cost, alphas * linesearch_decay, alphas,)
            i += 1
        if old_cost is not None:
            failed = current_cost > old_cost
            alphas = torch.where(failed, alphas / linesearch_decay, alphas,)
        alpha_du_norm = ((u - new_u) .transpose(1, 2) .contiguous() .reshape(n_batch, -1) .norm(2, dim=1))
        return new_x, new_u, LqrForOut(objs, full_du_norm, alpha_du_norm, torch.mean(alphas), current_cost,)

    class LQRStepFn(Function):
        @staticmethod
        def forward(ctx, x_init, C, c, F, f=None):
            if no_op_forward:
                ctx.save_for_backward(x_init, C, c, F, f, current_x, current_u)
                ctx.current_x, ctx.current_u = current_x, current_u
                return current_x, current_u
            if delta_space:
                assert current_x is not None
                assert current_u is not None
                c_back = []
                for t in range(T):
                    xt = current_x[t]
                    ut = current_u[t]
                    xut = torch.cat((xt, ut), dim=-1)
                    c_back.append(util.bmv(C[t], xut) + c[t])
                c_back = torch.stack(c_back, dim=0)
                f_back = None
            else:
                raise NotImplementedError("Only delta_space=True is supported.")
            ctx.current_x = current_x
            ctx.current_u = current_u
            Ks, ks, n_total_qp_iter = lqr_backward(ctx, C, c_back, F, f_back)
            new_x, new_u, for_out = lqr_forward(ctx, x_init, C, c, F, f, Ks, ks)
            ctx.save_for_backward(x_init, C, c, F, f, new_x, new_u)
            qp_iter_tensor = C.new_tensor([n_total_qp_iter])
            return (new_x, new_u, qp_iter_tensor, for_out.costs, for_out.full_du_norm, for_out.mean_alphas,)

        @staticmethod
        def backward(ctx, dl_dx, dl_du, temp=None, temp2=None, temp3=None, temp4=None):
            start = time.time()
            x_init, C, c, F, f, new_x, new_u = ctx.saved_tensors
            n_batch = C.size(1)
            r = torch.cat((dl_dx, dl_du), dim=-1)
            if u_lower is None:
                active_I = None
            else:
                lower = torch.stack([_bound("lower", t, n_batch, new_u[t]) for t in range(T)], dim=0,)
                upper = torch.stack([_bound("upper", t, n_batch, new_u[t]) for t in range(T)], dim=0,)
                active_I = torch.isclose(new_u, lower, atol=1e-8, rtol=0.0) | torch.isclose(new_u, upper, atol=1e-8, rtol=0.0)
            dx_init_zero = torch.zeros_like(x_init)
            _mpc = mpc.MPC(
                n_state,
                n_ctrl,
                T,
                u_zero_I=active_I,
                u_init=None,
                lqr_iter=1,
                verbose=-1,
                n_batch=n_batch,
                delta_u=None,
                exit_unconverged=False,
                eps=back_eps,
            )
            dx, du, _ = _mpc(dx_init_zero, mpc.QuadCost(C, -r), mpc.LinDx(F, None),)
            # Detach the adjoint solve; gradients are assembled explicitly below.
            dx = dx.detach()
            du = du.detach()
            dxu = torch.cat((dx, du), dim=-1)
            xu = torch.cat((new_x, new_u), dim=-1)
            dC = torch.zeros_like(C)
            for t in range(T):
                xut = torch.cat((new_x[t], new_u[t]), dim=-1)
                dxut = dxu[t]
                dC[t] = -0.5 * (util.bger(dxut, xut) + util.bger(xut, dxut))
            dc = -dxu
            lams = []
            prev_lam = None
            for t in range(T - 1, -1, -1):
                Ct_xx = C[t, :, :n_state, :n_state]
                Ct_xu = C[t, :, :n_state, n_state:]
                ct_x = c[t, :, :n_state]
                xt = new_x[t]
                ut = new_u[t]
                lamt = (util.bmv(Ct_xx, xt) + util.bmv(Ct_xu, ut) + ct_x)
                if prev_lam is not None:
                    Fxt = F[t, :, :, :n_state].transpose(1, 2)
                    lamt = lamt + util.bmv(Fxt, prev_lam)
                lams.append(lamt)
                prev_lam = lamt
            lams = list(reversed(lams))
            dlams = []
            prev_dlam = None
            for t in range(T - 1, -1, -1):
                Ct_xx = C[t, :, :n_state, :n_state]
                Ct_xu = C[t, :, :n_state, n_state:]
                drt_x = -r[t, :, :n_state]
                dxt = dx[t]
                dut = du[t]
                dlamt = (util.bmv(Ct_xx, dxt) + util.bmv(Ct_xu, dut) + drt_x)
                if prev_dlam is not None:
                    Fxt = F[t, :, :, :n_state].transpose(1, 2)
                    dlamt = dlamt + util.bmv(Fxt, prev_dlam)
                dlams.append(dlamt)
                prev_dlam = dlamt
            dlams = torch.stack(list(reversed(dlams)), dim=0)
            dF = torch.zeros_like(F)
            for t in range(T - 1):
                xut = xu[t]
                lamt = lams[t + 1]
                dxut = dxu[t]
                dlamt = dlams[t + 1]
                dF[t] = -(util.bger(dlamt, xut) + util.bger(lamt, dxut))
            if f.nelement() > 0:
                df = -dlams[1:]
                if df.shape != f.shape:
                    raise RuntimeError(f"df shape {tuple(df.shape)} does not match " f"f shape {tuple(f.shape)}.")
            else:
                df = torch.empty_like(f)
            dx_init_grad = -dlams[0]
            _ = time.time() - start
            # Gradients correspond to forward inputs: x_init, C, c, F, f.
            return dx_init_grad, dC, dc, dF, df
    return LQRStepFn.apply


"""Quasi-static 2D catheter model.

Representation
--------------
The catheter is a chain of knots p_0..p_N starting at the fixed entry point
p_0. Segment i has angle theta_i and length l_i, so the rod is inextensible by
construction and the base position is fixed while its orientation theta_0 is
free (with an optional rotational spring towards the entry direction). The
displayed / logged centreline is the arc-length cubic spline through the knots.

Insertion is kinematic: material enters through p_0 by growing the base
segment l_0. When it exceeds `segment_length` it is split; when retracted to
zero it is removed. Knots 1..N are therefore fixed material points counted
from the tip, which is what the friction model needs.

Energy (minimised every step with Gauss-Newton + line search)
-------------------------------------------------------------
  bending : sum_j EI_j / (2 ds_j) * (theta_j - theta_{j-1} - phi_j)^2
            phi_j = -kappa_cmd * (length of the tip arc around knot j)
            i.e. the tip is a piecewise-constant-curvature arc of length
            `tip_length` (kappa>0 clockwise, kappa<0 counter-clockwise).
  base    : k_base / 2 * (theta_0 - theta_entry)^2
  contact : sum_c k_wall / 2 * max(0, sdf(x_c) + r)^2
  friction: sum_c mu * N_c * (sqrt(|u_c|^2 + eps^2) - eps)
            u_c = tangential slip since the last step, N_c = lagged normal
            force (incremental-potential / IPC-style Coulomb friction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .config import CatheterParams
from .vessel import VesselModel


def _modified_newton_step(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Newton step on a positive-definite H; if H is indefinite (the rod is past
    its buckling load) use a shifted Newton step plus the most negative
    curvature direction, so symmetric unstable equilibria (a perfectly straight
    rod pushed into a dead end) still buckle."""
    try:
        L = np.linalg.cholesky(H)
        return np.linalg.solve(L.T, np.linalg.solve(L, -g)), True
    except np.linalg.LinAlgError:
        pass
    w, V = np.linalg.eigh(H)
    tau = max(-w[0], 0.0) + 1e-6 * (np.abs(w).max() + 1e-12)
    step = V @ (-(V.T @ g) / (w + tau))
    v = V[:, 0]
    if g @ v > 0 or (abs(g @ v) < 1e-12 and v[np.argmax(np.abs(v))] < 0):
        v = -v
    return step + 0.05 * v / np.abs(v).max(), False


@dataclass
class ContactInfo:
    points: np.ndarray      # (P,2) contact sample positions
    force: np.ndarray       # (P,2) wall normal force on catheter [N]
    friction: np.ndarray    # (P,2) friction force on catheter [N]
    active: np.ndarray      # (P,) bool


class Catheter:
    def __init__(self, params: CatheterParams, vessel: VesselModel):
        self.p = params
        self.vessel = vessel
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, length: float | None = None) -> None:
        p = self.p
        L = float(np.clip(length if length is not None else p.initial_length,
                          p.min_length, p.max_length))
        ds = p.segment_length
        n = max(int(np.ceil(L / ds - 1e-9)), 1)
        self.lens = np.full(n, ds)
        self.lens[0] = L - (n - 1) * ds
        self.theta = np.full(n, self.vessel.entry_angle)
        self.kappa = 0.0
        self.base = self.vessel.entry_pos.copy()
        P = 2 * n - 1
        self._prev_x = self._contact_points(self.theta)
        self._lag_N = np.zeros(P)
        self._lag_n = np.zeros((P, 2))
        self.last_contact = ContactInfo(self._prev_x, np.zeros((P, 2)), np.zeros((P, 2)),
                                        np.zeros(P, bool))
        self.insertion_force = 0.0
        self.solver_iters = 0

    _STATE = ("theta", "lens", "base", "_prev_x", "_lag_N", "_lag_n")

    def get_state(self) -> dict:
        """Full solver state (enough to re-simulate deterministically)."""
        st = {k.lstrip("_"): np.array(getattr(self, k), np.float64) for k in self._STATE}
        st["kappa"] = np.float64(self.kappa)
        return st

    def set_full_state(self, st: dict) -> None:
        for k in self._STATE:
            setattr(self, k, np.array(st[k.lstrip("_")], np.float64))
        self.kappa = float(st["kappa"])

    @property
    def length(self) -> float:
        return float(self.lens.sum())

    @property
    def n_seg(self) -> int:
        return len(self.theta)

    # -------------------------------------------------------------- kinematics
    def knots(self, theta: np.ndarray | None = None) -> np.ndarray:
        th = self.theta if theta is None else theta
        v = self.lens[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
        return np.vstack([self.base, self.base + np.cumsum(v, 0)])

    def _contact_layout(self):
        """Contact samples: knot 1..N plus mid-points of segments 1..N-1.

        Returns (seg index j, fraction a) for x_c = p_j + a * l_j * e_j,
        ordered base->tip. Midpoint of segment 0 is skipped because it is not a
        material point (segment 0 changes length during insertion).
        """
        n = self.n_seg
        js = np.concatenate([[0], np.repeat(np.arange(1, n), 2)])
        fr = np.concatenate([[1.0], np.tile([0.5, 1.0], n - 1)])
        return js, fr

    def _contact_points(self, theta):
        js, fr = self._contact_layout()
        k = self.knots(theta)
        e = np.stack([np.cos(theta), np.sin(theta)], 1)
        return k[js] + (fr * self.lens[js])[:, None] * e[js]

    def _weights(self):
        """W[c,i] = d x_c / d(l_i e_i) scale: 1 for i<j, a for i==j."""
        js, fr = self._contact_layout()
        n = self.n_seg
        idx = np.arange(n)[None, :]
        W = (idx < js[:, None]).astype(float)
        W[np.arange(len(js)), js] = fr
        return W

    # --------------------------------------------------------------- rest shape
    def _bending_terms(self):
        """Per-joint stiffness K_j = EI_j/ds_j and rest angle phi_j, j=1..n-1."""
        p, l, n = self.p, self.lens, self.n_seg
        if n < 2:
            return np.zeros(0), np.zeros(0)
        d_tip = np.cumsum(l[::-1])[::-1]          # distance from tip to start of seg i
        j = np.arange(1, n)
        dj = d_tip[j]                              # knot j distance from tip
        lo = dj - l[j] / 2
        lo[-1] = 0.0                               # last joint also owns the tip half-segment
        hi = dj + l[j - 1] / 2
        ov = np.clip(np.minimum(hi, p.tip_length) - np.maximum(lo, 0), 0, None)
        frac = ov / np.maximum(hi - lo, 1e-9)
        EI = frac * p.EI_tip + (1 - frac) * p.EI_shaft
        dsj = np.maximum((l[j - 1] + l[j]) / 2, 0.5 * p.segment_length)
        # positive kappa = clockwise = negative turning angle in a y-up frame
        return EI / dsj, -self.kappa * ov

    # ----------------------------------------------------------------- energy
    def _energy(self, theta, K, phi, prev_x, lagN, lagn, need_grad=True):
        p = self.p
        r = 0.5 * p.diameter
        dth = theta[1:] - theta[:-1] - phi
        E = 0.5 * np.sum(K * dth ** 2)
        db = theta[0] - self.vessel.entry_angle
        E += 0.5 * p.base_rot_stiffness * db ** 2

        x = self._contact_points(theta)
        sd, nrm = self.vessel.query(x)
        pen = np.maximum(sd + r, 0.0)
        E += 0.5 * p.wall_stiffness * np.sum(pen ** 2)

        # friction (lagged normal force / direction)
        fr_mask = lagN > 0
        gx_fric = np.zeros_like(x)
        A_fric = None
        if p.friction > 0 and fr_mask.any():
            dx = x - prev_x
            u = dx - np.sum(dx * lagn, 1, keepdims=True) * lagn      # tangential slip
            un = np.sqrt(np.sum(u ** 2, 1) + p.friction_eps ** 2)
            muN = p.friction * lagN
            E += np.sum(muN * (un - p.friction_eps))
            gx_fric = (muN / un)[:, None] * u
            A_fric = (muN / un)
        if not need_grad:
            return E

        gx_wall = (p.wall_stiffness * pen)[:, None] * nrm
        gx = gx_wall + gx_fric
        n = len(theta)
        W = self._weights()
        perp = self.lens[:, None] * np.stack([-np.sin(theta), np.cos(theta)], 1)   # (n,2)
        g = (W * (gx @ perp.T)).sum(0)             # sum_c W_ci (perp_i . gx_c)
        g[1:] += K * dth
        g[:-1] -= K * dth
        g[0] += p.base_rot_stiffness * db

        # Gauss-Newton Hessian
        H = np.zeros((n, n))
        if n > 1:
            ii = np.arange(n - 1)
            H[ii, ii] += K
            H[ii + 1, ii + 1] += K
            H[ii, ii + 1] -= K
            H[ii + 1, ii] -= K
        H[0, 0] += p.base_rot_stiffness
        # exact second-derivative (geometric stiffness) term: d2x_c/dtheta_i^2 = -W_ci l_i e_i
        G = W.T @ gx                                                # (n,2)
        H[np.diag_indices(n)] -= self.lens * (np.cos(theta) * G[:, 0] + np.sin(theta) * G[:, 1])
        act = pen > 0
        A = np.zeros((len(x), 2, 2))
        A[act] += p.wall_stiffness * nrm[act, :, None] * nrm[act, None, :]
        if A_fric is not None:
            T = np.eye(2)[None] - lagn[:, :, None] * lagn[:, None, :]
            A[fr_mask] += A_fric[fr_mask, None, None] * T[fr_mask]
            act = act | fr_mask
        if act.any():
            Wa = W[act]                                            # (a,n)
            Jc = Wa[:, None, :] * perp.T[None, :, :]               # (a,2,n)
            H += np.einsum("cai,cab,cbj->ij", Jc, A[act], Jc, optimize=True)
        info = (x, pen, nrm, gx_wall, gx_fric)
        return E, g, H, info

    # ------------------------------------------------------------------ solve
    def _solve(self):
        p = self.p
        K, phi = self._bending_terms()
        theta = self.theta.copy()
        prev_x, lagN, lagn = self._prev_x, self._lag_N, self._lag_n
        it = 0
        for it in range(1, p.newton_iters + 1):
            E, g, H, info = self._energy(theta, K, phi, prev_x, lagN, lagn)
            step, stable = _modified_newton_step(H, g)
            if stable and np.abs(g).max() < p.newton_tol:
                break
            # limit any single rotation per iteration
            mx = np.abs(step).max()
            if mx > 0.2:
                step *= 0.2 / mx
            slope = g @ step
            if slope >= 0:
                step, slope = -g * 1e-3, -(g @ g) * 1e-3
            alpha = 1.0
            while alpha > 1e-4:
                En = self._energy(theta + alpha * step, K, phi, prev_x, lagN, lagn, need_grad=False)
                if En <= E + 1e-4 * alpha * slope:
                    break
                alpha *= 0.5
            theta = theta + alpha * step
            if np.abs(alpha * step).max() < 1e-9:
                break
        self.theta = theta
        self.solver_iters = it

        # Update contact state and lagged friction quantities for the next step
        _, g, _, (x, pen, nrm, gx_wall, gx_fric) = self._energy(theta, K, phi, prev_x, lagN, lagn)
        self.last_contact = ContactInfo(x, -gx_wall, -gx_fric, pen > 0)
        e0 = np.array([np.cos(theta[0]), np.sin(theta[0])])
        # axial force needed at the entry to hold the current insertion
        self.insertion_force = float(np.sum((gx_wall + gx_fric) @ e0))
        self._prev_x = x
        self._lag_N = p.wall_stiffness * pen
        self._lag_n = nrm

    # ------------------------------------------------------------- insertion
    def _grow(self, dl: float) -> None:
        ds = self.p.segment_length
        self.lens[0] += dl
        while self.lens[0] > ds + 1e-9:          # split base segment
            extra = self.lens[0] - ds
            self.lens = np.concatenate([[extra, ds], self.lens[1:]])
            self.theta = np.concatenate([[self.theta[0]], self.theta])
            self._prepend_contacts()
        while self.lens[0] <= 1e-9 and self.n_seg > 1:   # merge
            rem = self.lens[0]
            self.lens = self.lens[1:].copy()
            self.theta = self.theta[1:].copy()
            self.lens[0] += rem
            self._prev_x = self._prev_x[2:]
            self._lag_N = self._lag_N[2:]
            self._lag_n = self._lag_n[2:]

    def _prepend_contacts(self) -> None:
        x = self._contact_points(self.theta)
        self._prev_x = np.vstack([x[:2], self._prev_x])
        self._lag_N = np.concatenate([[0.0, 0.0], self._lag_N])
        self._lag_n = np.vstack([np.zeros((2, 2)), self._lag_n])

    def step(self, d_length: float, d_kappa: float) -> None:
        """Advance by an insertion increment [mm] and curvature increment [1/mm]."""
        p = self.p
        L = self.length
        d_length = float(np.clip(L + d_length, p.min_length, p.max_length) - L)
        k_new = float(np.clip(self.kappa + d_kappa, -p.kappa_max, p.kappa_max))
        d_kappa = k_new - self.kappa
        n_sub = int(max(np.ceil(abs(d_length) / p.max_substep),
                        np.ceil(abs(d_kappa) / 0.01), 1))
        for _ in range(n_sub):
            if d_length:
                self._grow(d_length / n_sub)
            self.kappa += d_kappa / n_sub
            self._solve()
        self.kappa = k_new

    def set_state(self, length: float, kappa: float) -> None:
        self.step(length - self.length, kappa - self.kappa)

    # ---------------------------------------------------------------- outputs
    def spline(self, step: float = 0.5) -> np.ndarray:
        """Centreline sampled from the arc-length cubic spline through the knots."""
        k = self.knots()
        s = np.concatenate([[0], np.cumsum(self.lens)])
        keep = np.concatenate([[True], np.diff(s) > 1e-6])
        k, s = k[keep], s[keep]
        if len(s) < 3:
            return k
        cs = CubicSpline(s, k, bc_type="natural")
        return cs(np.linspace(0, s[-1], max(int(s[-1] / step), 2) + 1))

    def tip_pose(self):
        k = self.knots()
        return k[-1].copy(), float(self.theta[-1])

    def measured_tip_curvature(self) -> float:
        """Mean curvature over the bending section (+ = clockwise)."""
        d_tip = np.cumsum(self.lens[::-1])[::-1]
        inside = np.where(d_tip <= self.p.tip_length + 1e-6)[0]
        if len(inside) < 2:
            return 0.0
        i0 = inside[0]
        arc = self.lens[i0:].sum()
        return float(-(self.theta[-1] - self.theta[i0]) / max(arc - self.lens[-1] / 2, 1e-6))

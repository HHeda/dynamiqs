# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import replace
from itertools import product

import diffrax as dx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...qarrays.qarray import QArray, TimeQArray
from ...utils.operators import eye_like
from .diffrax_integrator import MESolveDiffraxIntegrator


class AbstractRouchonTerm(dx.AbstractTerm):
    # this class bypasses the typical Diffrax term implementation, as Rouchon schemes
    # don't match the vf/contr/prod structure

    kraus_map: Callable[[RealScalarLike, RealScalarLike, Y], [Y, Y]]
    # should be defined as `kraus_map(t0, t1, y0) -> y1, error`

    def vf(self, t, y, args):
        pass

    def contr(self, t0, t1, **kwargs):
        pass

    def prod(self, vf, control):
        pass


class RouchonDXSolver(dx.AbstractSolver):
    _order: int
    term_structure = AbstractRouchonTerm
    interpolation_cls = LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        pass

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1, error = terms.term.kraus_map(t0, t1, y0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, dx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        pass

    def order(self, terms):
        return self._order


class AdaptiveRouchonDXSolver(dx.AbstractAdaptiveSolver, RouchonDXSolver):
    pass


def cholesky_normalize(Ms: Sequence[QArray], rho: QArray) -> jax.Array:
    # To normalize the scheme, we compute
    #   S = sum_k Mk^† @ Mk
    # and replace
    #   Mk by ~Mk = Mk @ S^{-1/2}
    # such that
    #   sum_k ~Mk^† @ ~Mk = S^{†(-1/2)} @ (sum_k Mk^† @ Mk) @ S^{-1/2}
    #                   = S^{†(-1/2)} @ S @ S^{-1/2}
    #                   = I
    # To (i) keep sparse matrices and (ii) have a generic implementation that also
    # works for time-dependent systems, we use a Cholesky decomposition at each step
    # instead of computing S^{-1/2} explicitly. We write S = T @ T^† with T lower
    # triangular, and we replace
    #   Mk by ~Mk = Mk @ S^{-1/2}
    # such that
    #   #   sum_k ~Mk^† @ ~Mk = T^{-1} @ (sum_k Mk^† @ Mk) @ T^{†(-1)}
    #                       = T^{-1} @ T @ T^† @ T^{(-1)}
    #                       = I
    # In practice we directly replace rho_k by T^{†(-1)} @ rho_k @ T^{-1} instead of
    # computing all ~Mks.

    S = sum([compute_partial_S(rho, Mss) for Mss in Ms])
    T = jnp.linalg.cholesky(S.to_jax())  # T lower triangular

    # we want T^{†(-1)} @ y0 @ T^{-1}
    rho = rho.to_jax()
    # solve T^† @ x = rho => x = T^{†(-1)} @ rho
    rho = jax.lax.linalg.triangular_solve(
        T, rho, lower=True, transpose_a=True, conjugate_a=True
    )
    # solve x @ T = rho => x = rho @ T^{-1}
    return jax.lax.linalg.triangular_solve(T, rho, lower=True, left_side=True)


def _expm_taylor(A: QArray, order: int) -> QArray:
    I = eye_like(A)
    out = I
    powers_of_A = I
    for i in range(1, order + 1):
        powers_of_A = A @ powers_of_A
        out += 1 / jax.scipy.special.factorial(i) * powers_of_A

    return out

def apply_nested_map(rho: QArray, Mss: Sequence[Sequence[QArray]]) -> QArray:
    """Applies the partial Kraus map defined by the operators Mss to the density matrix rho recursively."""
    res = rho
    for Msss in Mss:
        res = sum([M @ res @ M.dag() for M in Msss])
    return res

def compute_partial_S(rho: QArray, Mss: Sequence[Sequence[QArray]]) -> QArray:
    """Computes the corresponding operator S = Mk^† @ Mk for the Kraus operators Mss."""
    S = eye_like(rho)
    for Msss in Mss:
        S = sum([M.dag() @ S @ M for M in Msss])
    return S


def dense_RK3(Gt: TimeArray, t0: float, dt: float):
    """Third-order Runge–Kutta (Kutta's RK3) step for U' = G(t) @ U with U0 = I.
    Returns U1, interp where interp(t) is a quadratic (2nd-order) dense output.
    """
    t1 = t0 + dt
    U0 = eye_like(Gt(t0))

    # Sample generators
    G0 = Gt(t0)
    Gmid = Gt(t0 + 0.5 * dt)
    G1 = Gt(t1)

    # Stages exploiting U0 = I:
    # k1 = G0
    # k2 = Gmid @ (I + (dt/2) k1) = Gmid + (dt/2) (Gmid @ G0)
    # k3 = G1 @ (I + dt(-k1 + 2 k2)) = G1 + G1 @ (-dt G0 + 2 dt Gmid + dt^2 (Gmid @ G0))
    k1 = G0
    k2 = Gmid + (dt / 2) * (Gmid @ G0)
    k3 = G1 + G1 @ (-dt * G0 + 2 * dt * Gmid + dt**2 * (Gmid @ G0))

    # Kutta's RK3 weights: b = [1/6, 2/3, 1/6]
    U1 = U0 + dt / 6 * (k1 + 4 * k2 + k3)

    # Derivative at t0 for dense output
    f0 = k1

    # Quadratic interpolant matching U(t0)=U0, U'(t0)=f0, U(t1)=U1 (2nd-order accurate)
    def interp(t: float) -> QArray:
        theta = (t - t0) / dt
        return U0 + theta * (dt * f0) + theta**2 * (U1 - U0 - dt * f0)

    return U1, interp

def dense_RK4(Gt: Callable[[float], QArray], t0: float, dt: float):
    """Fourth-order Runge–Kutta step for U' = G(t) @ U assuming U0 is the identity.
    Returns U1, interp where interp(t) gives cubic (3rd order) Hermite dense output.
    """
    t1 = t0 + dt
    U0 = eye_like(Gt(t0))
    # Precompute generators
    G0 = Gt(t0)
    Gmid = Gt(t0 + 0.5 * dt)
    G1 = Gt(t1)

    # Stages (use U0 = I to skip needless multiplications)
    k1 = G0
    k2 = Gmid + Gmid @ (0.5 * dt * k1)
    k3 = Gmid + Gmid @ (0.5 * dt * k2)
    k4 = G1 + G1 @ (dt * k3)

    U1 = U0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Derivatives at endpoints
    f0 = k1
    f1 = G1 @ U1

    def interp(t: float) -> QArray:
        theta = (t - t0) / dt
        h00 = 1 + theta * (-3 + 2 * theta)
        h10 = theta * (1 - 2 * theta + theta**2)
        h01 = theta**2 * (3 - 2 * theta)
        h11 = (-theta**2 + theta**3)
        return h00 * U0 + h01 * U1 + dt * (h10 * f0 + h11 * f1)

    return U1, interp

def dense_RK5(Gt: Callable[[float], QArray], t0: float, dt: float):
    """Fifth-order Runge–Kutta (Dormand–Prince) step for U' = G(t) @ U with U0 = I.
    Uses stages exploiting U0 = I; returns (U1, interp) where interp(t) is a 4th-order
    Lagrange dense output through selected internal stage reconstructions.
    """
    t1 = t0 + dt
    U0 = eye_like(Gt(t0))

    # Sample generators
    G1 = Gt(t0)  # c1 = 0
    G2 = Gt(t0 + (1/5) * dt)
    G3 = Gt(t0 + (3/10) * dt)
    G4 = Gt(t0 + (4/5) * dt)
    G5 = Gt(t0 + (8/9) * dt)
    G6 = Gt(t1)          # c6 = 1
    G7 = G6              # final stage uses same time

    # Stages k_i = G_i @ (U0 + dt * Σ a_ij k_j) expanded as k_i = G_i + G_i @ (dt * Σ a_ij k_j)
    k1 = G1
    k2 = G2 + G2 @ (dt * (1/5) * k1)
    k3 = G3 + G3 @ (dt * ((3/40) * k1 + (9/40) * k2))
    k4 = G4 + G4 @ (dt * ((44/45) * k1 + (-56/15) * k2 + (32/9) * k3))
    k5 = G5 + G5 @ (dt * ((19372/6561) * k1 + (-25360/2187) * k2 + (64448/6561) * k3 + (-212/729) * k4))
    k6 = G6 + G6 @ (dt * ((9017/3168) * k1 + (-355/33) * k2 + (46732/5247) * k3 + (49/176) * k4 + (-5103/18656) * k5))
    k7 = G7 + G7 @ (dt * ((35/384) * k1 + (500/1113) * k3 + (125/192) * k4 + (-2187/6784) * k5 + (11/84) * k6))

    # 5th-order solution coefficients (Dormand–Prince)
    U1 = U0 + dt * ((35/384) * k1 + (500/1113) * k3 + (125/192) * k4 + (-2187/6784) * k5 + (11/84) * k6)

    # Reconstruct internal stage approximations for dense output polynomial
    U_c2 = U0 + dt * (1/5) * k1
    U_c3 = U0 + dt * ((3/40) * k1 + (9/40) * k2)
    U_c4 = U0 + dt * ((44/45) * k1 + (-56/15) * k2 + (32/9) * k3)

    nodes = jnp.array([0.0, 1/5, 3/10, 4/5, 1.0])
    Us = [U0, U_c2, U_c3, U_c4, U1]

    # Precompute constant denominators for Lagrange basis once
    inv_den = []
    for i in range(len(nodes)):
        xi = nodes[i]
        den = 1.0
        for j, xj in enumerate(nodes):
            if j == i:
                continue
            den *= (xi - xj)
        inv_den.append(1.0 / den)

    def interp(t: float) -> QArray:
        theta = (t - t0) / dt
        # Lagrange basis over nodes using precomputed denominators
        coeffs = []
        for i in range(len(nodes)):
            xi = nodes[i]
            num = 1.0
            for j, xj in enumerate(nodes):
                if j == i:
                    continue
                num *= (theta - xj)
            coeffs.append(num * inv_den[i])
        out = 0
        for c, U in zip(coeffs, Us):
            out = out + c * U
        return out

    return U1, interp


class MESolveFixedRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using a
    fixed step Rouchon method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            # The Rouchon update for a single loss channel is:
            #   rho_{k+1} = sum_k Mk @ rho_k @ Mk^†
            # See comment of `cholesky_normalize()` for the normalization.

            rho = y0
            t = (t0 + t1) / 2
            dt = t1 - t0
            Ms = self._kraus_ops(t, dt)

            if self.method.normalize:
                rho = cholesky_normalize(Ms, rho)

            # for fixed step size, we return None for the error estimate
            return sum([apply_nested_map(rho, M) for M in Ms]), None

        return AbstractRouchonTerm(kraus_map)

    def _kraus_ops(self, t: float, dt: float) -> Sequence[TimeQArray]:
        # L, H = self.L(t), self.H(t)
        return self.Ms(self.H, self.L, dt, self.method.exact_expm)

    @staticmethod
    @abstractmethod
    def Ms(H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool) -> Sequence[QArray]:
        pass


class MESolveFixedRouchon1Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 1 method.
    """

    @staticmethod
    def Ms(H: QArray, L: Sequence[QArray], t: float, dt: float, exact_expm: bool) -> Sequence[Sequence[QArray]]:
        # M0 = I - (iH + 0.5 sum_k Lk^† @ Lk) dt
        # Mk = Lk sqrt(dt)
        LdL = sum([_L.dag() @ _L for _L in L(t)])
        G = -1j * H - 0.5 * LdL
        e1 = (dt * G).expm() if exact_expm else _expm_taylor(dt * G, 1)
        return [[[e1]]] + [[[jnp.sqrt(dt) * _L for _L in L(t)]]]


mesolve_rouchon1_integrator_constructor = (
    lambda **kwargs: MESolveFixedRouchon1Integrator(
        **kwargs, diffrax_solver=RouchonDXSolver(1), fixed_step=True
    )
)


class MESolveFixedRouchon2Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 2 method.
    """

    @staticmethod
    def Ms(H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool) -> Sequence[Sequence[Sequence[QArray]]]:
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1 = (dt * G).expm() if exact_expm else _expm_taylor(dt * G, 2)

        return (
            [[[e1]]]
            + [[[jnp.sqrt(dt / 2) * e1 @ _L for _L in L]]]
            + [[[jnp.sqrt(dt / 2) * _L @ e1 for _L in L]]]
            + [[[jnp.sqrt(dt**2 / 2) * _L1 for _L1 in L], [_L2 for _L2 in L]]]
        )


class MESolveFixedRouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 3 method.
    """

    @staticmethod
    def Ms(H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool) -> Sequence[Sequence[Sequence[QArray]]]:
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1o3 = (dt / 3 * G).expm() if exact_expm else _expm_taylor(dt / 3 * G, 3)
        e2o3 = e1o3 @ e1o3
        e3o3 = e2o3 @ e1o3

        return (
            [[[e3o3]]]
            + [[[jnp.sqrt(3 * dt / 4) * e1o3 @ _L @ e2o3 for _L in L]]]
            + [[[jnp.sqrt(dt / 4) * e3o3 @ _L for _L in L]]]
            + [[
                [jnp.sqrt(dt**2 / 2) * e1o3 @ _L1 for _L1 in L],  [e1o3 @ _L2 @ e1o3 for _L2 in L]
            ]]
            + [[
                [jnp.sqrt(dt**3 / 6) * _L1 for _L1 in L],  [_L2 for _L2 in L],  [_L3 for _L3 in L]
            ]]
        )
    
class MESolveFixed_RK3_Rouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 3 method.
    """

    @staticmethod
    def Ms(H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool) -> Sequence[Sequence[Sequence[QArray]]]:
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1o3 = (dt / 3 * G).expm() if exact_expm else _expm_taylor(dt / 3 * G, 3)
        e2o3 = e1o3 @ e1o3
        e3o3 = e2o3 @ e1o3

        return (
            [[[e3o3]]]
            + [[[jnp.sqrt(3 * dt / 4) * e1o3 @ _L @ e2o3 for _L in L]]]
            + [[[jnp.sqrt(dt / 4) * e3o3 @ _L for _L in L]]]
            + [[
                [jnp.sqrt(dt**2 / 2) * e1o3 @ _L1 for _L1 in L],  [e1o3 @ _L2 @ e1o3 for _L2 in L]
            ]]
            + [[
                [jnp.sqrt(dt**3 / 6) * _L1 for _L1 in L],  [_L2 for _L2 in L],  [_L3 for _L3 in L]
            ]]
        )
    

class MESolveFixedRouchon4Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 4 method.
    """

    @staticmethod
    def Ms(H: QArray, L: Sequence[QArray], dt: float, exact_expm: bool) -> Sequence[Sequence[Sequence[QArray]]]:
        LdL = sum([_L.dag() @ _L for _L in L])
        G = -1j * H - 0.5 * LdL
        e1o4 = (dt / 4 * G).expm() if exact_expm else _expm_taylor(dt / 4 * G, 4)
        e2o4 = e1o4 @ e1o4
        e3o4 = e2o4 @ e1o4
        e4o4 = e3o4 @ e1o4
        e1o3m = (dt * (3-jnp.sqrt(3)) / 6 * G).expm() if exact_expm else _expm_taylor(dt * (3-jnp.sqrt(3)) / 6 * G, 3)
        e1o3p = (dt * (3+jnp.sqrt(3)) / 6 * G).expm() if exact_expm else _expm_taylor(dt * (3+jnp.sqrt(3)) / 6 * G, 3)

        return (
            [[[e4o4]]]
            + [[[jnp.sqrt(dt / 2) * e1o3m @ _L @ e1o3p for _L in L]]]
            + [[[jnp.sqrt(dt / 2) * e1o3p @ _L @ e1o3m for _L in L]]]
            + [[
                [jnp.sqrt(dt**2 / 9) * e3o4 @ _L1 for _L1 in L],  [e1o4 @ _L2 for _L2 in L]
            ]]
            + [[
                [jnp.sqrt(dt**2 / 3) * e1o4 @ _L1 for _L1 in L],  [e1o4 @ _L2 @ e2o4 for _L2 in L]
            ]]
            + [[
                [jnp.sqrt(dt**2 / 18) *  _L1 @ e4o4 for _L1 in L],  [_L2 for _L2 in L]
            ]]
            + [[
                [jnp.sqrt(dt**3 / 6) * e1o4 @ _L1 for _L1 in L],  [ e1o4 @ _L2 for _L2 in L],  [ e1o4 @ _L3 @ e1o4 for _L3 in L]
            ]]
            + [[
                [jnp.sqrt(dt**4 / 24) * _L1 for _L1 in L],  [_L2 for _L2 in L],  [_L3 for _L3 in L], [_L4 for _L4 in L]
            ]]
        )


class MESolveAdaptiveRouchonIntegrator(MESolveDiffraxIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using an
    adaptive Rouchon method.
    """

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        # todo: can we do better?
        stepsize_controller = super().stepsize_controller
        # fix incorrect default linear interpolation by stepping exactly at all times
        # in tsave, so interpolation is bypassed
        return replace(stepsize_controller, step_ts=self.ts)


class MESolveAdaptiveRouchon2Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 1-2 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = (t0 + t1) / 2
            dt = t1 - t0

            L, H = self.L(t), self.H(t)

            # === first order
            Ms_1 = MESolveFixedRouchon1Integrator.Ms(H, L, dt, self.method.exact_expm)
            rho_1 = cholesky_normalize(Ms_1, rho) if self.method.normalize else rho
            rho_1 = sum([apply_nested_map(rho, Mss) for Mss in Ms_1])

            # === second order
            Ms_2 = MESolveFixedRouchon2Integrator.Ms(H, L, dt, self.method.exact_expm)
            rho_2 = cholesky_normalize(Ms_2, rho) if self.method.normalize else rho
            rho_2 = sum([apply_nested_map(rho, Mss) for Mss in Ms_2])

            return rho_2, 0.5 * (rho_2 - rho_1)

        return AbstractRouchonTerm(kraus_map)


class MESolveAdaptiveRouchon3Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 2-3 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = (t0 + t1) / 2
            dt = t1 - t0

            L, H = self.L(t), self.H(t)

            # === second order
            Ms_2 = MESolveFixedRouchon2Integrator.Ms(H, L, dt, self.method.exact_expm)
            rho_2 = cholesky_normalize(Ms_2, rho) if self.method.normalize else rho
            rho_2 = sum([apply_nested_map(rho, Mss) for Mss in Ms_2])

            # === third order
            Ms_3 = MESolveFixedRouchon3Integrator.Ms(H, L, dt, self.method.exact_expm)
            rho_3 = cholesky_normalize(Ms_3, rho) if self.method.normalize else rho
            rho_3 = sum([apply_nested_map(rho, Mss) for Mss in Ms_3])
            return rho_3, 0.5 * (rho_3 - rho_2)

        return AbstractRouchonTerm(kraus_map)


def mesolve_rouchon2_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon2 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon2Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(2), fixed_step=True
        )
    return MESolveAdaptiveRouchon2Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(2), fixed_step=False
    )


def mesolve_rouchon3_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon3 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon3Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(3), fixed_step=True
        )
    return MESolveAdaptiveRouchon3Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(3), fixed_step=False
    )

mesolve_rouchon4_integrator_constructor = (
    lambda **kwargs: MESolveFixedRouchon4Integrator(
        **kwargs, diffrax_solver=RouchonDXSolver(4), fixed_step=True
    )
)
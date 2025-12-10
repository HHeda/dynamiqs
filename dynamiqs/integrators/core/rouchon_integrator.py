# ruff: noqa: ANN001, ANN201, ARG002
# we mostly ignore type hinting in this file for readability purposes

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace

import diffrax as dx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax._custom_types import RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation

from ...qarrays.qarray import QArray
from ...utils.operators import asqarray, eye_like
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


def cholesky_normalize(
    Msss: Sequence[Sequence[Sequence[QArray]]], rho: QArray
) -> jax.Array:
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

    S = sum([compute_partial_S(rho, Mss) for Mss in Msss])
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
    """Applies the partial Kraus map defined by the operators
    Mss to the density matrix rho recursively.
    """
    res = rho
    for Ms in reversed(Mss):
        res = sum([M @ res @ M.dag() for M in Ms])
    return res


def compute_partial_S(rho: QArray, Mss: Sequence[Sequence[QArray]]) -> QArray:
    """Computes the corresponding operator S = Mk^† @ Mk for the Kraus operators Mss."""
    S = eye_like(rho)
    for Ms in Mss:
        S = sum([M.dag() @ S @ M for M in Ms])
    return S


def interp3(
    theta: float, dt: float, U0: QArray, U1: QArray, f0: QArray, f1: QArray
) -> QArray:
    # Cubic Hermite interpolation: p(theta) with constraints:
    # p(0) = U0, p(1) = U1, p'(0) = dt*f0, p'(1) = dt*f1
    # Using standard Hermite basis functions
    h00 = 1 - 3 * theta**2 + 2 * theta**3
    h10 = theta - 2 * theta**2 + theta**3
    h01 = 3 * theta**2 - 2 * theta**3
    h11 = -(theta**2) + theta**3
    return h00 * U0 + h01 * U1 + dt * (h10 * f0 + h11 * f1)


def interp2(theta: float, dt: float, U0: QArray, U1: QArray, f0: QArray) -> QArray:
    # Quadratic Hermite interpolation: p(theta) = a0 + a1*theta + a2*theta^2
    # Constraints: p(0)=U0, p(1)=U1, p'(0)=dt*f0
    a0 = U0
    a1 = dt * f0
    a2 = U1 - U0 - dt * f0
    return a0 + theta * a1 + theta**2 * a2


def interp4(theta, y_n, y_np1, k1, k3, k4, k5, k6, k7):
    """Interpolant dense-output du Dormand-Prince 4(5) (ode45 / Numerical Recipes).

    Paramètres
    ----------
    theta : float
        (t - t_n) / h  dans [0,1]
    y_n : array
        valeur y_n
    y_np1 : array
        y_{n+1}
    k1, k3, k4, k5, k6, k7 : arrays
        stages k_i = h * f(...) déjà calculés dans le pas.
        k1 = h f_n
        k7 = h f_{n+1}

    Retour
    ------
    Y(theta) : array
        Valeur interpolée.
    """
    # Coefficients of the dense output (Dormand-Prince 4(5))
    d1 = -12715105075.0 / 11282082432.0
    d3 = 87487479700.0 / 32700410799.0
    d4 = -10690763975.0 / 1880347072.0
    d5 = 701980252875.0 / 199316789632.0
    d6 = -1453857185.0 / 822651844.0
    d7 = 69997945.0 / 29380423.0

    # r1 à r4
    r1 = y_n
    r2 = y_np1 - y_n
    r3 = y_n + k1 - y_np1
    r4 = 2.0 * (y_np1 - y_n) - (k1 + k7)

    # r5 = stages combination
    r5 = d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7

    # Factorized evaluation
    return r1 + theta * (r2 + (1 - theta) * (r3 + theta * (r4 + (1 - theta) * r5)))


def propagator(U1, U2):
    """Compute the propagator from t2 to t1 using LU factorization.

    U1: propagator from 0 to t1
    U2: propagator from 0 to t2
    Returns: propagator from t2 to t1
    """
    # U1 = U(t2->t1) @ U2, so U(t2->t1) = U1 @ U2^{-1}
    # Compute U2^{-1} using LU factorization
    return asqarray(jnp.linalg.solve(U2.to_jax().T, U1.to_jax().T).T, dims=U1.dims)


# Dunavant degree 4 quadrature rule on the triangle with 6 points
# Reference: Dunavant, D.A. (1985). High degree efficient symmetrical Gaussian
# quadrature rules for the triangle. International Journal for Numerical Methods
# in Engineering, 21(6), 1129-1148.

# Barycentric coordinates (λ1, λ2, λ3) with λ1 + λ2 + λ3 = 1
# Points and weights for degree 4 precision (6 points)

# the values at which evaluate the jump operators for 1 jump
thetas_L1 = [[(1 - jnp.sqrt(3 / 5)) / 2], [1 / 2], [(1 + jnp.sqrt(3 / 5)) / 2]]

# the values at which evaluate the evolution operators for 1 jump
thetas_U1 = np.array(
    [
        [[1, (1 - jnp.sqrt(3 / 5)) / 2], [(1 - jnp.sqrt(3 / 5)) / 2, 0]],
        [[1, 1 / 2], [1 / 2, 0]],
        [[1, (1 + jnp.sqrt(3 / 5)) / 2], [(1 + jnp.sqrt(3 / 5)) / 2, 0]],
    ]
)
weights_1 = [5 / 18, 8 / 18, 5 / 18]

dunavant_degree4_points_barycentric = np.array(
    [
        [0.10810301816807022736, 0.44594849091596488632, 0.44594849091596488632],
        [0.44594849091596488632, 0.10810301816807022736, 0.44594849091596488632],
        [0.44594849091596488632, 0.44594849091596488632, 0.10810301816807022736],
        [0.81684757298045851308, 0.09157621350977074346, 0.09157621350977074346],
        [0.09157621350977074346, 0.81684757298045851308, 0.09157621350977074346],
        [0.09157621350977074346, 0.09157621350977074346, 0.81684757298045851308],
    ],
    dtype=np.float64,
).T

# conversion to Cartesian coordinates for the reference triangle with vertices
# at (0,0), (1,0), (1,1)

barycentric_to_cartesian2 = np.array([[0.0, 1, 1], [0.0, 0.0, 1.0]], dtype=np.float64)
dunavant_degree4_points_cartesian = (
    barycentric_to_cartesian2 @ dunavant_degree4_points_barycentric
)
# helper to compute the thetas for the evolution operators
dunavant_degree4_points_cartesian_extended = np.concatenate(
    (np.ones((1, 6)), dunavant_degree4_points_cartesian, np.zeros((1, 6))), axis=0
)

dunavant_degree4_weights = (
    np.array(
        [
            0.22338158967801146570,
            0.22338158967801146570,
            0.22338158967801146570,
            0.10995174365532186764,
            0.10995174365532186764,
            0.10995174365532186764,
        ],
        dtype=np.float64,
    )
    / 2.0
)  # divide by 2 for  triangle area

# helper to compute the thetas for the evolution operators
dunavant_degree4_points_cartesian_extended = np.concatenate(
    (np.ones((1, 6)), dunavant_degree4_points_cartesian, np.zeros((1, 6))), axis=0
)

# the values at which evaluate the evolution operators for 2 jumps
thetas_U2 = np.array(
    [
        dunavant_degree4_points_cartesian_extended.T[:, :-1],
        dunavant_degree4_points_cartesian_extended.T[:, 1:],
    ]
).transpose(1, 2, 0)
# the values at which evaluate the jump operators for 2 jumps
thetas_L2 = dunavant_degree4_points_cartesian.T
weights_2 = dunavant_degree4_weights


def get_tetra_degree2_quadrature():
    a = 0.5854101966249685
    b = 0.1381966011250105
    barycentric = np.array([[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]])

    points = barycentric  # @ vertices
    weights = np.full(4, 1 / 24)

    return points, weights


tetra_points, tetra_weights = get_tetra_degree2_quadrature()

barycentric_to_cartesian3 = np.array(
    [[0.0, 1, 1, 1], [0.0, 0, 1, 1], [0.0, 0.0, 0, 1.0]], dtype=np.float64
)
tetra_points_cartesian = barycentric_to_cartesian3 @ tetra_points.T
tetra_points_cartesian_extended = np.concatenate(
    (np.ones((1, 4)), tetra_points_cartesian, np.zeros((1, 4))), axis=0
)
# the values at which evaluate the evolution operators for 3 jumps
thetas_U3 = np.array(
    [
        tetra_points_cartesian_extended.T[:, :-1],
        tetra_points_cartesian_extended.T[:, 1:],
    ]
).transpose(1, 2, 0)
# the values at which evaluate the jump operators for 3 jumps
thetas_L3 = tetra_points_cartesian.T
weights_3 = tetra_weights


# -- collect all θ utilisés dans thetas_U{1,2,3} --
def flatten_thetas(arr):
    return [float(x) for x in np.array(arr).reshape(-1)]


theta_values = []
theta_values += flatten_thetas(thetas_U1)
theta_values += flatten_thetas(thetas_U2)
theta_values += flatten_thetas(thetas_U3)

unique_thetas = []
seen = set()
for th in theta_values:
    if th not in seen:
        seen.add(th)
        unique_thetas.append(th)

theta_to_idx = {th: i for i, th in enumerate(unique_thetas)}


# -- collect all (θ₁, θ₂) demanded --
def collect_pairs(thetas):
    pairs = []
    arr = np.array(thetas)
    if arr.ndim < 3:
        arr = arr[:, None, :]
    for block in arr:
        for row in block:
            pairs.append((float(row[0]), float(row[1])))
    return pairs


pair_list = []
for src in (thetas_U1, thetas_U2, thetas_U3):
    for p in collect_pairs(src):
        if p not in pair_list:
            pair_list.append(p)

pair_to_idx = {p: i for i, p in enumerate(pair_list)}


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
            t = t0
            dt = t1 - t0
            Msss = self._kraus_ops(t, dt)

            if self.method.normalize:
                rho = cholesky_normalize(Msss, rho)

            # for fixed step size, we return None for the error estimate
            return sum([apply_nested_map(rho, Mss) for Mss in Msss]), None

        return AbstractRouchonTerm(kraus_map)

    def _kraus_ops(self, t: float, dt: float) -> Sequence[QArray]:
        return self.Msss(self.H, self.L, t, dt, self.method.exact_expm)

    @staticmethod
    @abstractmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[QArray]:
        pass


class MESolveFixedRouchon1Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 1 method.
    """

    @staticmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:
        if exact_expm:
            pass
        Lmid, Hmid = L(t + dt / 2), H(t + dt / 2)
        U0 = eye_like(Hmid)
        # M0 = I - (iH + 0.5 sum_k Lk^† @ Lk) dt
        # Mk = Lk sqrt(dt)
        LdL = sum([_L.dag() @ _L for _L in Lmid])
        G = -1j * Hmid - 0.5 * LdL
        e1 = (dt * G).expm() if exact_expm else U0 + dt * G
        return [[[e1]], [[jnp.sqrt(dt) * _L for _L in Lmid]]]


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
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:
        if exact_expm:
            pass
        U0 = eye_like(H(t))
        # Sample generators
        G0 = -1j * H(t) - 0.5 * sum([_L.dag() @ _L for _L in L(t)])
        Gmid = -1j * H(t + 0.5 * dt) - 0.5 * sum(
            [_L.dag() @ _L for _L in L(t + 0.5 * dt)]
        )

        # RK2 stages exploiting U0 = I:
        # k1 = G0
        # k2 = Gmid @ (I + (dt/2) k1)
        k1 = U0 + dt / 2 * G0
        U1 = U0 + dt * Gmid @ (k1)
        emid = (U0 + U1) / 2
        # RK2 weights: b = [0, 1]
        # U1 = U0 + dt * k2

        e1 = U1
        Lmid = L(t + 0.5 * dt)
        J0 = [[[e1]]]
        J1 = [[[jnp.sqrt(dt) * propagator(U1, emid) @ _L @ emid for _L in Lmid]]]
        J2 = [[[jnp.sqrt(dt**2 / 2) * _L1 for _L1 in Lmid], Lmid]]
        return [*J0, *J1, *J2]


class MESolveFixedRouchon3Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 3 method.
    """

    @staticmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:
        if exact_expm:
            pass
        U0 = eye_like(H(t))
        # Sample generators
        G0 = -1j * H(t) - 0.5 * sum([_L.dag() @ _L for _L in L(t)])
        Gmid = -1j * H(t + 0.5 * dt) - 0.5 * sum(
            [_L.dag() @ _L for _L in L(t + 0.5 * dt)]
        )
        G1 = -1j * H(t + dt) - 0.5 * sum([_L.dag() @ _L for _L in L(t + dt)])

        k1 = G0
        k2 = Gmid @ (U0 + (dt / 2) * k1)
        k3 = G1 @ (U0 - dt * k1 + 2 * dt * k2)

        # Kutta's RK3 weights: b = [1/6, 2/3, 1/6]
        U1 = U0 + dt / 6 * (k1 + 4 * k2 + k3)

        # Derivative at t0 for dense output
        f0 = k1

        e1o3 = interp2(1 / 3, dt, U0, U1, f0)
        e2o3 = interp2(2 / 3, dt, U0, U1, f0)
        e3o3 = U1
        L0o3 = L(t)
        L1o3 = L(t + 1 / 3 * dt)
        L2o3 = L(t + 2 / 3 * dt)
        L1o4 = L(t + dt / 4)
        L2o4 = L(t + dt / 2)
        L3o4 = L(t + 3 * dt / 4)
        # L3o3 = self.L(t+dt)
        e2o3_to_e3o3 = propagator(U1, e2o3)

        J0 = [[[e3o3]]]
        J1a = [[[(jnp.sqrt(3 * dt / 4) * e2o3_to_e3o3 @ _L @ e2o3) for _L in L2o3]]]
        J1b = [[[jnp.sqrt(dt / 4) * e3o3 @ _L for _L in L0o3]]]
        J2 = [
            [
                [jnp.sqrt(dt**2 / 2) * e2o3_to_e3o3 @ _L1 for _L1 in L2o3],
                [propagator(e2o3, e1o3) @ _L2 @ e1o3 for _L2 in L1o3],
            ]
        ]
        J3 = [[[jnp.sqrt(dt**3 / 6) * _L1 for _L1 in L3o4], L2o4, L1o4]]
        return [*J0, *J1a, *J1b, *J2, *J3]


class MESolveFixedRouchon4Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 4 method.
    """

    @staticmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:
        if exact_expm:
            pass
        U0 = eye_like(H(t))
        # Sample generators
        G0 = -1j * H(t) - 0.5 * sum([_L.dag() @ _L for _L in L(t)])
        Gmid = -1j * H(t + 0.5 * dt) - 0.5 * sum(
            [_L.dag() @ _L for _L in L(t + 0.5 * dt)]
        )
        G1 = -1j * H(t + dt) - 0.5 * sum([_L.dag() @ _L for _L in L(t + dt)])

        k1 = G0
        k2 = Gmid @ (U0 + (dt / 2) * k1)
        k3 = Gmid @ (U0 + (dt / 2) * k2)
        k4 = G1 @ (U0 + dt * k3)

        U1 = U0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Derivative at t0 for dense output
        f0 = k1
        # Derivative at t1 for dense output
        f1 = G1 @ U1

        e1o4 = interp3(1 / 4, dt, U0, U1, f0, f1)
        e2o4 = interp3(2 / 4, dt, U0, U1, f0, f1)
        e3o4 = interp3(3 / 4, dt, U0, U1, f0, f1)
        e4o4 = U1
        o3m = (3 - jnp.sqrt(3)) / 6
        o3p = (3 + jnp.sqrt(3)) / 6
        e1o3m = interp3(o3m, dt, U0, U1, f0, f1)
        e1o3p = interp3(o3p, dt, U0, U1, f0, f1)
        e3o4_to_e4o4 = propagator(U1, e3o4)
        e2o4_to_e3o4 = propagator(e3o4, e2o4)

        Lt0 = L(t)
        Lt1o3p = L(t + o3p * dt)
        Lt1o3m = L(t + o3m * dt)
        Lt1o4 = L(t + dt / 4)
        Lt2o4 = L(t + dt / 2)
        Lt3o4 = L(t + 3 * dt / 4)
        Lt1o5 = L(t + dt / 5)
        Lt2o5 = L(t + 2 * dt / 5)
        Lt3o5 = L(t + 3 * dt / 5)
        Lt4o5 = L(t + 4 * dt / 5)
        Lt1 = L(t + dt)

        # 0 jump: 1 operator
        J0 = [[[e4o4]]]
        # 1 jumps: 2 operators
        J1a = [
            [[jnp.sqrt(dt / 2) * propagator(U1, e1o3p) @ _L @ e1o3p for _L in Lt1o3p]]
        ]
        J1b = [
            [[jnp.sqrt(dt / 2) * propagator(U1, e1o3m) @ _L @ e1o3m for _L in Lt1o3m]]
        ]
        # 2 jumps: 3 operators
        J2a = [
            [
                [jnp.sqrt(dt**2 / 9) * propagator(U1, e1o4) @ _L1 for _L1 in Lt1o4],
                [e1o4 @ _L2 for _L2 in Lt0],
            ]
        ]
        J2b = [
            [
                [jnp.sqrt(dt**2 / 3) * e3o4_to_e4o4 @ _L1 for _L1 in Lt3o4],
                [e2o4_to_e3o4 @ _L2 @ e2o4 for _L2 in Lt2o4],
            ]
        ]
        J2c = [[[jnp.sqrt(dt**2 / 18) * _L1 @ e4o4 for _L1 in Lt1], Lt0]]
        # 3 jumps: 1 operators
        J3 = [
            [
                [jnp.sqrt(dt**3 / 6) * e3o4_to_e4o4 @ _L1 for _L1 in Lt3o4],
                [e2o4_to_e3o4 @ _L2 for _L2 in Lt2o4],
                [propagator(e2o4, e1o4) @ _L3 @ e1o4 for _L3 in Lt1o4],
            ]
        ]
        # 4 jumps: 1 operators
        J4 = [[[jnp.sqrt(dt**4 / 24) * _L1 for _L1 in Lt4o5], Lt3o5, Lt2o5, Lt1o5]]

        return [*J0, *J1a, *J1b, *J2a, *J2b, *J2c, *J3, *J4]


class MESolveFixedRouchon5Integrator(MESolveFixedRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    fixed step Rouchon 4 method.
    """

    @staticmethod
    def Msss(
        H: Callable[[RealScalarLike], QArray],
        L: Callable[[RealScalarLike], Sequence[QArray]],
        t: float,
        dt: float,
        exact_expm: bool,
    ) -> Sequence[Sequence[Sequence[QArray]]]:
        if exact_expm:
            pass
        U0 = eye_like(H(t))

        # define time-dependent generator G(tau) = -i H(t+tau) - 0.5 sum L^† L
        def G_at(theta: float) -> QArray:
            return -1j * H(t + theta * dt) - 0.5 * sum(
                [_L.dag() @ _L for _L in L(t + theta * dt)]
            )

        # Dormand-Prince 5(4) (DOPRI5) coefficients for an explicit RK5
        c2, c3, c4, c5, c6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0
        a21 = 1 / 5
        a31, a32 = 3 / 40, 9 / 40
        a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
        a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
        a61, a62, a63, a64, a65 = (
            9017 / 3168,
            -355 / 33,
            46732 / 5247,
            49 / 176,
            -5103 / 18656,
        )
        a71, a72, a73, a74, a75, a76 = (
            35 / 384,
            0.0,
            500 / 1113,
            125 / 192,
            -2187 / 6784,
            11 / 84,
        )
        b1, b3, b4, b5, b6 = a71, a73, a74, a75, a76

        # evaluate generators at needed nodes
        G0 = G_at(0.0)
        G2 = G_at(c2)
        G3 = G_at(c3)
        G4 = G_at(c4)
        G5 = G_at(c5)
        G6 = G_at(c6)  # same as G_at(dt)
        G1 = G_at(1.0)

        # derivatives for RK5 (dU/ds = G(t+s) @ U(s))
        d1 = G0
        d2 = G2 @ (U0 + dt * (a21 * d1))
        d3 = G3 @ (U0 + dt * (a31 * d1 + a32 * d2))
        d4 = G4 @ (U0 + dt * (a41 * d1 + a42 * d2 + a43 * d3))
        d5 = G5 @ (U0 + dt * (a51 * d1 + a52 * d2 + a53 * d3 + a54 * d4))
        d6 = G6 @ (U0 + dt * (a61 * d1 + a62 * d2 + a63 * d3 + a64 * d4 + a65 * d5))
        d7 = G1 @ (
            U0 + dt * (a71 * d1 + a72 * d2 + a73 * d3 + a74 * d4 + a75 * d5 + a76 * d6)
        )

        U1 = U0 + dt * (b1 * d1 + b3 * d3 + b4 * d4 + b5 * d5 + b6 * d6)

        def U_at(theta: float) -> QArray:
            return interp4(
                theta, U0, U1, dt * d1, dt * d3, dt * d4, dt * d5, dt * d6, dt * d7
            )

        stacked_U = jnp.stack([U_at(th).to_jax() for th in unique_thetas])

        def solve_prop(th1: float, th2: float) -> jnp.ndarray:
            U1 = stacked_U[theta_to_idx[th1]]
            if th2 == 0:
                return U_at(th1).to_jax()
            U2 = stacked_U[theta_to_idx[th2]]
            return jnp.linalg.solve(U2.T, U1.T).T

        stacked_prop = jnp.stack([solve_prop(a, b) for (a, b) in pair_list])
        sample_dims = U0.dims
        stacked_prop_q = [asqarray(mat, dims=sample_dims) for mat in stacked_prop]

        def get_prop(theta1: float, theta2: float) -> QArray:
            return stacked_prop_q[pair_to_idx[(float(theta1), float(theta2))]]

        # 0 jump: 1 operator
        J0 = [[[U1]]]
        # 1 jump: 3 operators
        J1 = [
            [
                [
                    (
                        jnp.sqrt(dt * w)
                        * get_prop(theta_U[0, 0], theta_U[0, 1])
                        @ _L
                        @ get_prop(theta_U[1, 0], theta_U[1, 1])
                    )
                    for _L in L(t + theta_L[0] * dt)
                ]
            ]
            for (w, theta_U, theta_L) in zip(
                weights_1, thetas_U1, thetas_L1, strict=True
            )
        ]
        # 2 jumps: 6 operators
        J2 = [
            [
                [
                    (
                        jnp.sqrt(dt**2 * w)
                        * get_prop(theta_U[0, 0], theta_U[0, 1])
                        @ _L1
                        @ get_prop(theta_U[1, 0], theta_U[1, 1])
                    )
                    for _L1 in L(t + theta_L[0] * dt)
                ],
                [
                    (_L2 @ get_prop(theta_U[2, 0], theta_U[2, 1]))
                    for _L2 in L(t + theta_L[1] * dt)
                ],
            ]
            for (w, theta_U, theta_L) in zip(
                weights_2, thetas_U2, thetas_L2, strict=True
            )
        ]
        # 3 jumps: 4 operators
        J3 = [
            [
                [
                    (
                        jnp.sqrt(dt**3 * w)
                        * get_prop(theta_U[0, 0], theta_U[0, 1])
                        @ _L1
                        @ get_prop(theta_U[1, 0], theta_U[1, 1])
                    )
                    for _L1 in L(t + theta_L[0] * dt)
                ],
                [
                    (_L2 @ get_prop(theta_U[2, 0], theta_U[2, 1]))
                    for _L2 in L(t + theta_L[1] * dt)
                ],
                [
                    _L3 @ get_prop(theta_U[3, 0], theta_U[3, 1])
                    for _L3 in L(t + theta_L[2] * dt)
                ],
            ]
            for (w, theta_U, theta_L) in zip(
                weights_3, thetas_U3, thetas_L3, strict=True
            )
        ]
        # 4 jumps: 1 operator
        J4 = [
            [
                [
                    (
                        jnp.sqrt(dt**4 / 24)
                        * propagator(U1, U_at(4 / 5))
                        @ _L1
                        @ propagator(U_at(4 / 5), U_at(3 / 5))
                    )
                    for _L1 in L(t + 3 / 5 * dt)
                ],
                [
                    _L2 @ propagator(U_at(3 / 5), U_at(2 / 5))
                    for _L2 in L(t + 3 / 5 * dt)
                ],
                [
                    _L3 @ propagator(U_at(2 / 5), U_at(1 / 5))
                    for _L3 in L(t + 2 / 5 * dt)
                ],
                [_L4 @ U_at(1 / 5) for _L4 in L(t + 1 / 5 * dt)],
            ]
        ]
        # 5 jumps: 1 operator
        J5 = [
            [
                [jnp.sqrt(dt**5 / 120) * _L1 for _L1 in L(t + 5 / 6 * dt)],
                L(t + 4 / 6 * dt),
                L(t + 3 / 6 * dt),
                L(t + 2 / 6 * dt),
                L(t + 1 / 6 * dt),
            ]
        ]
        return (
            # No jump:
            [*J0, *J1, *J2, *J3, *J4, *J5]
        )
        # One jump: three points
        # Two jumps: six points

        # Total number of Kraus terms: 1 + 3 + 6 + 4 + 1 + 1 = 16


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
            t = t0
            dt = t1 - t0

            L, H = self.L, self.H

            # === first order
            Msss_1 = MESolveFixedRouchon1Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_1 = cholesky_normalize(Msss_1, rho) if self.method.normalize else rho
            rho_1 = sum([apply_nested_map(rho_1, Mss) for Mss in Msss_1])

            # === second order
            Msss_2 = MESolveFixedRouchon2Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_2 = cholesky_normalize(Msss_2, rho) if self.method.normalize else rho
            rho_2 = sum([apply_nested_map(rho_2, Mss) for Mss in Msss_2])

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
            t = t0
            dt = t1 - t0

            L, H = self.L, self.H

            # === second order
            Msss_2 = MESolveFixedRouchon2Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_2 = cholesky_normalize(Msss_2, rho) if self.method.normalize else rho
            rho_2 = sum([apply_nested_map(rho_2, Mss) for Mss in Msss_2])

            # === third order
            Msss_3 = MESolveFixedRouchon3Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_3 = cholesky_normalize(Msss_3, rho) if self.method.normalize else rho
            rho_3 = sum([apply_nested_map(rho_3, Mss) for Mss in Msss_3])
            return rho_3, 0.5 * (rho_3 - rho_2)

        return AbstractRouchonTerm(kraus_map)


class MESolveAdaptiveRouchon4Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 3-4 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = t0
            dt = t1 - t0

            L, H = self.L, self.H

            # === third order
            Msss_3 = MESolveFixedRouchon3Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_3 = cholesky_normalize(Msss_3, rho) if self.method.normalize else rho
            rho_3 = sum([apply_nested_map(rho_3, Mss) for Mss in Msss_3])

            # === fourth order
            Msss_4 = MESolveFixedRouchon4Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_4 = cholesky_normalize(Msss_4, rho) if self.method.normalize else rho
            rho_4 = sum([apply_nested_map(rho_4, Mss) for Mss in Msss_4])
            return rho_4, 0.5 * (rho_4 - rho_3)

        return AbstractRouchonTerm(kraus_map)


class MESolveAdaptiveRouchon5Integrator(MESolveAdaptiveRouchonIntegrator):
    """Integrator computing the time evolution of the Lindblad master equation using the
    adaptive Rouchon 4-5 method.
    """

    @property
    def terms(self) -> dx.AbstractTerm:
        def kraus_map(t0, t1, y0):  # noqa: ANN202
            rho = y0
            t = t0
            dt = t1 - t0

            L, H = self.L, self.H

            # === fourth order
            Msss_4 = MESolveFixedRouchon4Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_4 = cholesky_normalize(Msss_4, rho) if self.method.normalize else rho
            rho_4 = sum([apply_nested_map(rho_4, Mss) for Mss in Msss_4])

            # === fifth order
            Msss_5 = MESolveFixedRouchon5Integrator.Msss(
                H, L, t, dt, self.method.exact_expm
            )
            rho_5 = cholesky_normalize(Msss_5, rho) if self.method.normalize else rho
            rho_5 = sum([apply_nested_map(rho_5, Mss) for Mss in Msss_5])
            return rho_5, 0.5 * (rho_5 - rho_4)

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


def mesolve_rouchon4_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon4 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon4Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(4), fixed_step=True
        )
    return MESolveAdaptiveRouchon4Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(4), fixed_step=False
    )


def mesolve_rouchon5_integrator_constructor(**kwargs):
    """Factory function to create a Rouchon5 integrator."""
    if kwargs['method'].dt is not None:
        return MESolveFixedRouchon5Integrator(
            **kwargs, diffrax_solver=RouchonDXSolver(5), fixed_step=True
        )
    return MESolveAdaptiveRouchon5Integrator(
        **kwargs, diffrax_solver=AdaptiveRouchonDXSolver(5), fixed_step=False
    )

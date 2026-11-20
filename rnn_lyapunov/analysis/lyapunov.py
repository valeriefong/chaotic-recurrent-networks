from typing import Any
import numpy as np
import torch
import time
import math
import logging
from typing import Optional
from dataclasses import dataclass
from rnn_lyapunov.models.base import RNN

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass
class LyapunovResult:
    '''
    Output of LyapunovSpectrum.compute().

    exponents: Lyapunov exponents, one per tracked direction, in QR (Gram-Schmidt) order.
    local_exponents: per-interval (finite-time) exponents, shape (number of intervals, n_LE).
    participation_ratio: per-interval participation ratio of the first QR direction.
    '''
    exponents: np.ndarray
    local_exponents: np.ndarray
    participation_ratio: np.ndarray

    @property
    def spectrum(self) -> np.ndarray:
        '''
        The exponents as a PHYSICAL SPECTRUM: descending rank order, largest first.

        `exponents` is NOT this. It comes out of the QR algorithm in Gram-Schmidt direction order,
        and Gram-Schmidt only orders directions by growth rate on average over infinite time -- at
        finite T_sim, near-degenerate exponents routinely come back out of order. Index i of
        `exponents` therefore means "the i-th QR direction", not "the i-th largest exponent".
        Anything order-dependent (largest exponent, Kaplan-Yorke) must read through here.

        `exponents` is deliberately left in raw solver order so it stays column-aligned with
        `local_exponents`; sorting it in place would put the two out of step without any error.
        '''
        return np.sort(np.asarray(self.exponents, dtype=float))[::-1]

    @property
    def max_exponent(self) -> float:
        '''Largest Lyapunov exponent lambda_1.'''
        return float(self.spectrum[0])

    @property
    def entropy(self) -> float:
        # Kolmogorov-Sinai entropy rate: sum of all positive exponents. Order-independent, but read
        # through .spectrum anyway so all three scalars have exactly one definition of their input.
        e = self.spectrum
        return float(np.sum(e[e > 0]))

    @property
    def lyapunov_dimension(self) -> float:
        # Kaplan-Yorke attractor dimension: D = M + (sum_{i<=M} lambda_i) / |lambda_{M+1}|, where
        # M = max{j : sum_{i<=j} lambda_i >= 0} and the exponents are in descending order.
        e = self.spectrum
        if np.allclose(e, 0.0):
            # Degenerate case: every exponent is exactly zero, so there is no attractor structure
            # to measure. The walk below would run off the end and return N.
            return 0.0
        cumsum = np.cumsum(e)
        if cumsum[0] < 0:
            # Nothing grows: fixed point.
            return 0.0
        below = np.flatnonzero(cumsum < 0)
        if below.size == 0:
            # Running sum never turns negative -- the whole spectrum is needed.
            return float(len(e))
        # Descending order => cumsum is concave => it crosses zero exactly once, so the first
        # negative index ends the non-negative prefix.
        j = int(below[0]) - 1
        return float((j + 1) + cumsum[j] / abs(e[j + 1]))

class LyapunovSpectrum:
    '''
    Lyapunov spectrum of an RNN model by the QR (Benettin) method.

    compute() is adapted from demo_lypunov_spectra_chaotic_ratenetwork.py by Rainer Engelken
    (https://github.com/RainerEngelken/RNN-LyapunovSpectra, commit fb81838), the reference code for
    Engelken, Wolf & Abbott, "Lyapunov spectra of chaotic recurrent neural networks",
    Phys. Rev. Research 5, 043044 (2023). See compute() for the license note.
    '''
    def __init__(self, model:RNN):
        self.model = model

    def compute(
            self,
            T_sim: float,
            dt: float,
            T_transient: float = 100.0,
            n_LE: Optional[int] = None,
            T_ONS: float = 1.0,
            seed_IC: int = 42,
            seed_ONS: int = 42,
            verbose: bool = False,
            integrator: str = 'euler',
            T_ONS_warmup: Optional[float] = None,
    ):
        '''
        Full (or partial) Lyapunov spectrum by evolving an orthonormal set of tangent vectors along
        a trajectory and re-orthonormalizing it with a QR decomposition every T_ONS.

        Adapted from demo_lypunov_spectra_chaotic_ratenetwork.py by Rainer Engelken
        (https://github.com/RainerEngelken/RNN-LyapunovSpectra, commit fb81838; Engelken, Wolf &
        Abbott, Phys. Rev. Research 5, 043044, 2023): same loop structure, step counting, QR
        accumulation and participation ratio, ported to PyTorch, with an RK4 integrator and its
        exact tangent map added. The original code has no license; this function is adapted from
        it and is not covered by this repository's MIT license.

        T_sim: time over which exponents are accumulated (units of tau).
        dt: integration step.
        T_transient: time the state is evolved first, so the trajectory reaches the attractor.
        n_LE: number of exponents to compute; None computes all N.
        T_ONS: time between re-orthonormalizations.
        seed_IC, seed_ONS: seeds for the initial state and the initial orthonormal basis.
        verbose: print progress every 1000 steps.
        integrator: 'euler' (default) or 'rk4' (fixed-step 4-stage Runge-Kutta with its exact
            tangent map). 'rk4' costs 4x per step and has far smaller discretization bias at
            the same dt. An adaptive step size is deliberately not offered: a state-dependent step
            makes the tangent map not a fixed map, so its exponents are not cleanly the flow's.
        T_ONS_warmup: time the orthonormal system is evolved after the transient, before exponents
            are accumulated, so it aligns with the dominant directions. None uses T_sim/10, which
            ties the warm-up to T_sim; pass a value to set the two independently.

        Returns a LyapunovResult.
        '''

        # start timing here
        t_start = time.time()

        N = self.model.N
        if n_LE is None:
            n_LE = N  # Compute full Lyapunov spectrum by default
        assert n_LE <= N

        n_step = int(T_sim / dt) # Total number of steps
        n_step_ONS = max(1, int(T_ONS / dt)) # Steps between orthonormalizations
        n_step_transient = int(T_transient / dt) # Steps during initial transient
        # Steps of orthonormal-system warm-up: T_sim/10 by default, or T_ONS_warmup if given.
        n_step_transient_ONS = (n_step // 10 if T_ONS_warmup is None
                                else int(T_ONS_warmup / dt))
        n_ONS = math.ceil((n_step_transient_ONS + n_step)/n_step_ONS)-1  # Number of ONS steps

        # Gain of the model. Not used below: the reference demo scales the initial state by
        # (g - 1); this code draws it from a standard normal instead.
        g = getattr(self.model, 'g', 1.0) # default gain is 1.0

        torch.manual_seed(seed_IC) # Set seed for initial state
        x = torch.randn(N) # Initial network state

        # Initialize matrix of random orthonormal perturbation vectors
        torch.manual_seed(seed_ONS) # Set seed for orthonormal system
        Q, _ = torch.linalg.qr(torch.randn(N, n_LE))  # Initial ONS has shape (N, n_LE)

        LS_cumulative = torch.zeros(n_LE)  # Lyapunov exponents : accumulated log stretching
        t = 0.0 # accumulated time (only after both transients)

        LS_all = torch.zeros((n_ONS, n_LE))  # Local exponents, one row per orthonormalization interval
        p_ratio_all = torch.zeros(n_ONS)
        lsidx = -1

        total_steps = n_step_transient + n_step_transient_ONS + n_step

        for n in range(total_steps - 1):
            # Main loop over simulation steps. During the state transient only the state moves.
            # After it, the state and the orthonormal system move together, and the system is
            # re-orthonormalized every n_step_ONS steps. Exponents are accumulated only after the
            # orthonormal-system warm-up as well.

            if (n+1) > n_step_transient:
                # Advance BOTH the state and the orthonormal system by one step of the
                # chosen integrator, sharing stage evaluations. x_next is held back and
                # assigned after the QR block, so the block sees the pre-step state.
                x_next, Q = self.model.step_and_tangent(x, dt, Q, integrator)

                # Reorthonormalization every n_step_ONS steps
                if (n+1) % n_step_ONS == 0:
                    # QR decomposition: Q is the new orthonormal system; the diagonal of R holds
                    # how much each direction was stretched since the last orthonormalization.
                    Q, R = torch.linalg.qr(Q)
                    # Participation ratio of the first (most expanding) direction q, a unit
                    # vector: 1 / sum_i q_i^4. About 1 if q sits on a single unit and about N if
                    # it is spread evenly over all units, so it counts how many units take part
                    # in the leading unstable direction.
                    if n_LE == 1:
                        Q4 = Q*Q*Q*Q
                    else:
                        Q1 = Q[:, 0]
                        Q4 = Q1*Q1*Q1*Q1

                    lsidx += 1

                    p_ratio_all[lsidx] = 1.0/sum(Q4)

                    # Local (finite-time) exponents over this interval: log stretching per unit
                    # time. Recorded for every interval, including the warm-up, to show convergence.
                    LS_all[lsidx, :] = torch.log(torch.abs(torch.diag(R))) / n_step_ONS / dt

                    # Accumulate only after both the state transient and the warm-up
                    if (n + 1) > n_step_transient_ONS + n_step_transient:
                        LS_cumulative += torch.log(torch.abs(torch.diag(R)))  # sum of log stretching factors
                        t += n_step_ONS * dt  # increment time

                step_display = 1000
                if (n + 1) % step_display == 0:  # progress report
                    if (n + 1) > n_step_transient + n_step_transient_ONS:
                        percentage_finished = (n + 1 - n_step_transient - n_step_transient_ONS)*100/n_step
                        elapsed = time.time() - t_start
                        msg = (f'{round(percentage_finished)}% done after {elapsed:.1f}s, '
                               f'SimTime: {round(dt*(n+1))} tau, std(x)={torch.std(x).item():.2f}')
                        logger.info(msg)
                        if verbose:
                            print(msg)

            # Evolve current state so that next loop iteration applies Jacobian at next state.
            # Past the state transient the step was already taken above, sharing integrator
            # stages with the tangent update; during the transient only the state moves.
            if (n+1) > n_step_transient:
                x = x_next
            else:
                x = self.model.step_with(x, dt, integrator)


        LS = LS_cumulative/t  # Exponents: accumulated log stretching divided by accumulated time
        return LyapunovResult(
                exponents=LS.numpy(),
                local_exponents=LS_all.numpy(),
                participation_ratio=p_ratio_all.numpy(),
            )

    def dns(
                self,
                T_sim: float,
                dt: float,
                perturbation: float = 1e-4,
                seed_IC: int = 42,
                seed_ONS: int = 42,
                T_ONS: float = 1.0,
    ):
        '''
        Largest Lyapunov exponent by direct numerical simulation, as an independent check on
        compute(). Instead of applying the Jacobian to tangent vectors, simulate a reference and a
        slightly perturbed trajectory, measure how fast they separate, and rescale the separation
        back to `perturbation` every T_ONS. Uses forward Euler only, with a fixed 100-time-unit
        transient and the same step counting as compute()'s default warm-up.

        Returns (lambda_max, per-interval estimates).
        '''
        # Gain of the model (not used; see compute())
        g = getattr(self.model, 'g', 1.0) # default gain is 1.0

        torch.manual_seed(seed_IC) # Set seed for initial state
        x = torch.randn(self.model.N) # Initial network state

        torch.manual_seed(seed_ONS) # Set seed for orthonormal system
        Q, _ = torch.linalg.qr(torch.randn(self.model.N, self.model.N))  # Random orthonormal basis

        x_perturbed = x + perturbation * Q[:, 0]/torch.linalg.norm(Q[:, 0])

        # Number of time steps: matched to compute() with its default warm-up
        n_step = int(T_sim / dt) # Total number of steps
        n_step_ONS = max(1, int(T_ONS / dt)) # Steps between renormalizations
        n_step_transient = int(100 / dt) # Steps during initial transient
        n_step_transient_ONS = n_step // 10 # Warm-up steps
        n_ONS = math.ceil((n_step_transient_ONS + n_step)/n_step_ONS)-1  # Number of renormalizations

        # time steps for simulating trajectories
        total_steps = n_step_transient + n_step_transient_ONS + n_step

        norm_dx_all = torch.zeros(n_ONS)  # Initialize
        ls_idx = -1
        for n in range(total_steps - 1):
            # Evolve both trajectories forward so we measure how they diverge under the real dynamics
            x = self.model.step(x, dt)
            x_perturbed = self.model.step(x_perturbed, dt)

            dx = x_perturbed - x  # Difference of perturbed and unperturbed state
            norm_dx = torch.norm(dx)  # Norm of difference

            if (n+1) % n_step_ONS == 0:
                x_perturbed = x + dx * perturbation / norm_dx
                if (n + 1) > n_step_transient:
                    ls_idx += 1
                    norm_dx_all[ls_idx] = norm_dx

        lambda_max = torch.sum(torch.log(norm_dx_all/perturbation)) / (n_step + n_step_transient_ONS) / dt
        lambda_max_all = torch.log(norm_dx_all/perturbation) / n_step_ONS / dt
        return lambda_max, lambda_max_all

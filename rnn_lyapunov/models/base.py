# Abstract base class for RNN models: integrators (Euler, RK4) and tangent maps.
from abc import ABC, abstractmethod
import torch
from torch.func import jacrev


class RNN(ABC):
    def __init__(self, N:int):
        self.N = N
        

    @abstractmethod
    def dynamics(self, x:torch.Tensor) -> torch.Tensor:
        """
        Compute the dynamics f(x) of the RNN at state x (time derivative of x).
        Returns a tensor with shape (N,)
        """

    def step(self, x:torch.Tensor, dt:float) -> torch.Tensor:
        """
        Advance state by one step using Euler's method.
        Returns state at the next time step with shape (N,)
        """
        return x + dt * self.dynamics(x)
    

    def jacobian(self, x:torch.Tensor, dt:float) -> torch.Tensor:
        """
        Compute Jacobian of the step function at state x i.e. step(x, dt).
        Returns Jacobian matrix with shape (N, N)
        """
        return jacrev(lambda x_: self.step(x_, dt))(x)

    def _eye_like(self, x:torch.Tensor) -> torch.Tensor:
        """
        Cached (N, N) identity matching x's dtype. RK4 builds four Jacobians per step, so a
        fresh torch.eye on each one is a measurable O(N^2) allocation cost at N=1000.
        """
        e = getattr(self, '_eye_cache', None)
        if e is None or e.shape[0] != self.N or e.dtype != x.dtype:
            e = torch.eye(self.N, dtype=x.dtype)
            self._eye_cache = e
        return e

    def flow_jacobian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Jacobian of the VECTOR FIELD, df/dx at state x. Returns shape (N, N).

        This is the continuous-time Jacobian and carries no dt: it is a property of the
        model, not of any integrator. `jacobian(x, dt)` above is a different object --
        the Jacobian of the discrete STEP map -- and for forward Euler the two are
        related by jacobian(x, dt) = I + dt * flow_jacobian(x).

        The default differentiates `dynamics` by automatic differentiation. Models with a
        closed-form Jacobian should override this; `jacobian` can then be left to the
        Euler relation above, or overridden separately.
        """
        return jacrev(self.dynamics)(x)

    def step_rk4(self, x:torch.Tensor, dt:float) -> torch.Tensor:
        """
        Advance state by one classical 4-stage, 4th-order Runge-Kutta step.
        Returns state at the next time step with shape (N,).
        """
        k1 = self.dynamics(x)
        k2 = self.dynamics(x + 0.5 * dt * k1)
        k3 = self.dynamics(x + 0.5 * dt * k2)
        k4 = self.dynamics(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step_and_tangent(self, x:torch.Tensor, dt:float, Q:torch.Tensor,
                         integrator:str = 'euler'):
        """
        Advance the state AND the tangent basis Q by one step of `integrator`, sharing
        stage evaluations between the two. Returns (x_next, Q_next).

        Q has shape (N, n_LE); Q_next is the exact tangent map of the chosen step map
        applied to the columns of Q -- not a finite-difference approximation to it. For
        'euler' this is exactly `jacobian(x, dt) @ Q` and `step(x, dt)`, so the choice of
        integrator is the only thing that changes any number.

        For 'rk4' the tangent map is the derivative of the 4-stage step, propagated
        stage by stage:
            d1 = J(x1) Q,                      x1 = x
            d2 = J(x2) (Q + dt/2 d1),          x2 = x + dt/2 k1
            d3 = J(x3) (Q + dt/2 d2),          x3 = x + dt/2 k2
            d4 = J(x4) (Q + dt   d3),          x4 = x + dt   k3
            Q_next = Q + dt/6 (d1 + 2 d2 + 2 d3 + d4)
        This costs 4 flow-Jacobian builds and 4 applications per step, i.e. exactly 4x
        the Euler per-step cost.
        """
        if integrator == 'euler':
            Df = self.jacobian(x, dt)
            return self.step(x, dt), Df @ Q

        if integrator == 'rk4':
            k1 = self.dynamics(x)
            x2 = x + 0.5 * dt * k1
            k2 = self.dynamics(x2)
            x3 = x + 0.5 * dt * k2
            k3 = self.dynamics(x3)
            x4 = x + dt * k3
            k4 = self.dynamics(x4)

            d1 = self.flow_jacobian(x)  @ Q
            d2 = self.flow_jacobian(x2) @ (Q + 0.5 * dt * d1)
            d3 = self.flow_jacobian(x3) @ (Q + 0.5 * dt * d2)
            d4 = self.flow_jacobian(x4) @ (Q + dt * d3)

            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Q_next = Q + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            assert Q_next.shape == Q.shape, f"tangent shape drift: {Q_next.shape} vs {Q.shape}"
            return x_next, Q_next

        raise ValueError(f"unknown integrator {integrator!r}; expected 'euler' or 'rk4'")

    def step_with(self, x:torch.Tensor, dt:float, integrator:str = 'euler') -> torch.Tensor:
        """
        Advance the state only, using `integrator`. Used for the pure-transient phase,
        where no tangent basis is being carried but the trajectory must still be the one
        the chosen integrator produces.
        """
        if integrator == 'euler':
            return self.step(x, dt)
        if integrator == 'rk4':
            return self.step_rk4(x, dt)
        raise ValueError(f"unknown integrator {integrator!r}; expected 'euler' or 'rk4'")

    def simulate(self, x0:torch.Tensor, T:float, dt:float) -> torch.Tensor:
        """
        Simulate trajectory starting from x0 for time T (units of tau) with time step dt,
        using forward Euler (`step`).
        Returns trajectory with shape (T/dt + 1, N)
        """
        steps = int(T / dt)
        trajectory = [x0]
        x = x0.clone()
        for _ in range(steps):
            x = self.step(x, dt)
            trajectory.append(x)
        return torch.stack(trajectory)
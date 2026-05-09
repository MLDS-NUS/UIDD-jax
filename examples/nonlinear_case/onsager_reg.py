"""Local OnsagerNet variant with optional regularization on the potential V.

Hooks into the ``regu`` / ``regularization(x)`` interface that
``UIDD._losses.MLELoss`` already checks: when ``model.regu`` is truthy,
``MLELoss`` adds ``model.regularization(x)`` to the per-sample loss. We support
two families that act on the potential V only:

  * **L2** on the parameters of V (sum of squares of all eqx float-array
    leaves of ``self.potential``). Penalises large MLP weights.
  * **Sobolev**: ``||Hess V(x)||_F^2``, evaluated at the current sample x.
    Encourages a smooth potential (suppresses spurious second-derivative
    oscillations).

Each family has its own non-negative weight; both default to 0 (i.e. no
regularization). Suggested starting values for the nonlinear_case problem
(V_target ~ 100, ||Hess V_target||_F^2 ~ 1e4, MLE loss ~ 5):

    OnsagerNetReg(..., l2_weight=1e-3, sobolev_weight=1e-4)

Since regularization is added per-sample inside ``MLELoss``, the L2 part does
not depend on ``x``: the per-sample loss becomes ``MLE + L2 + Sobolev(x)``,
and averaging over (samples, timesteps) leaves L2 untouched.
"""

import sys
import jax
import jax.numpy as jnp
import equinox as eqx

from jax import Array
from jax.typing import ArrayLike
from typing import Optional

sys.path.append("../..")
from UIDD.dynamics import SDE


# Suggested starting weights (non-zero values to copy if you want to enable
# regularization). Defaults in __init__ are still 0 so the loss is unchanged
# unless the user opts in.
DEFAULT_L2_WEIGHT = 1.0e-3
DEFAULT_SOBOLEV_WEIGHT = 1.0e-4


class OnsagerNetReg(SDE):
    r"""OnsagerNet with optional L2/Sobolev regularization on V.

    Drift / diffusion are identical to ``UIDD.dynamics.OnsagerNet``:

    .. math::
        dX_t = -\big[ M(X_t) + W(X_t) \big] \nabla V(X_t, args)\,dt
               + \sqrt{\epsilon}\,\sigma(X_t, args)\,dW_t .

    Regularization:

    .. math::
        R(x) = \lambda_{L2}\,\sum_{\theta\in V}\|\theta\|^2
             + \lambda_{Sob}\,\big\|\nabla^2 V(x, args_0)\big\|_F^2.

    ``args_0`` is the constant ``default_args`` stored on the model (kept
    static so it does not receive gradients from the regularizer).
    """

    potential: eqx.Module
    dissipation: eqx.Module
    conservation: eqx.Module
    diffusion_func: eqx.Module

    regu: bool = eqx.field(static=True)
    l2_weight: float = eqx.field(static=True)
    sobolev_weight: float = eqx.field(static=True)
    default_args: tuple = eqx.field(static=True)

    def __init__(
        self,
        potential: eqx.Module,
        dissipation: eqx.Module,
        conservation: eqx.Module,
        diffusion: eqx.Module,
        l2_weight: float = 0.0,
        sobolev_weight: float = 0.0,
        default_args: Optional[tuple] = None,
    ) -> None:
        """Constructor.

        Args:
            potential, dissipation, conservation, diffusion: same as
                ``UIDD.dynamics.OnsagerNet``.
            l2_weight: coefficient of the L2 penalty on V's weights.
            sobolev_weight: coefficient of the Frobenius-squared Hessian
                penalty on V evaluated at the current sample.
            default_args: tuple of floats used as the ``args`` argument when
                evaluating Hessian(V); the first entry is the temperature.
                Stored statically so it does not receive gradients. Defaults
                to ``(1.0,)``.
        """
        self.potential = potential
        self.dissipation = dissipation
        self.conservation = conservation
        self.diffusion_func = diffusion

        self.l2_weight = float(l2_weight)
        self.sobolev_weight = float(sobolev_weight)
        self.regu = (self.l2_weight > 0.0) or (self.sobolev_weight > 0.0)

        if default_args is None:
            default_args = (1.0,)
        self.default_args = tuple(float(a) for a in default_args)

    # ---- SDE interface (drift / diffusion: copied from OnsagerNet) ----

    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        dvdx = jax.grad(self.potential, argnums=0)(x, args)
        return -(self.dissipation(x) + self.conservation(x)) @ dvdx

    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        temperature = args[0]
        return jnp.sqrt(temperature) * self.diffusion_func(x, args)

    # ---- Regularization hook (UIDD._losses.MLELoss reads model.regu) ----

    def regularization(self, x: ArrayLike) -> Array:
        """Combined L2 + Sobolev regularizer evaluated at sample x."""
        reg = jnp.zeros((), dtype=x.dtype)

        if self.l2_weight > 0.0:
            leaves = jax.tree.leaves(eqx.filter(self.potential, eqx.is_inexact_array))
            l2 = sum(jnp.sum(p ** 2) for p in leaves)
            reg = reg + self.l2_weight * l2

        if self.sobolev_weight > 0.0:
            args0 = jnp.asarray(self.default_args, dtype=x.dtype)
            H = jax.hessian(self.potential, argnums=0)(x, args0)
            reg = reg + self.sobolev_weight * jnp.sum(H ** 2)

        return reg

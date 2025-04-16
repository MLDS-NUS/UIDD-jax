import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

# ------------------------- Typing imports ------------------------- #

from typing import Callable
from jax.typing import ArrayLike
from jax import Array
from .transformations import Encoder, Decoder

DynamicCallable = Callable[[ArrayLike, ArrayLike, ArrayLike], Array]

"""
Add by aiqing
"""
from .dynamics import SDE
# ------------------------------------------------------------------ #
#        OnsagerNet based on Helmholtz Decomposition       #
# ------------------------------------------------------------------ #


class OnsagerNetHD(SDE):
    potential: eqx.Module
    shared: eqx.nn.Shared
    Hamiltonian: eqx.Module
    J: Array
    
    def __init__(
        self, D: int, potential: eqx.Module, dissipation: eqx.Module, Hamiltonian: eqx.Module
    ) -> None:
        r"""Stochastic OnsagerNet model based on Helmholtz Decomposition.

        This is a modified version of the Stochastic OnsagerNet model.
        Let $X(t) \in \mathbb{R}^d$. This model is defined by the SDE
        $$
            dX(t) = -
            \left[
                -M(x) \nabla V(x) + \epsilon \nabla\cdot M(x) + \gamma(x)
            \right] dt
            + \sqrt{2 \epsilon} [M(x(t)]^\frac{1}{2}dW(t)
        $$

        $$
            \gamma = \sum_{d=1}^{D-1} J_d \nabla H_d  - \sum_{d=1}^{D-1}H_d J_d \nabla V.
        $$

        where

        - $M : \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the dissipation matrix,
          which is symmetric positive semi-definite for all $x$
        - $V : \mathbb{R}^{d} \to \mathbb{R}$ is the potential function
        - $\gamma : \mathbb{R}^{d} \to \mathbb{R}^{d}$ is the Helmholtz decomposition term
        - $H : \mathbb{R}^{d} \to \mathbb{R}^{d-1}$. H e^{-V} is the Hamiltonian decomposition of \gamma e^{-V}.
        - $J_d$ is a $D\times D$ matrix with only two non-zero elements: 
            a $1$ at the position $(d, d+1)$ and a $-1$ at the position $(d+1, d)$.
        
        Notice that the main difference with `OnsagerNet` is that the
        diffusion matrix is now given by a (positive semi-definite) square root of the dissipation matrix.

        Args:
            potential (eqx.Module): potential function $V$
            dissipation (eqx.Module): dissipation matrix $M$
            Hamiltonian (eqx.Module): Hamiltonian functions $H$
        """ 
        self.potential = potential
        self.Hamiltonian = Hamiltonian 

        #define the J matrix for computing the gamma term
        self.J = jnp.zeros((D - 1, D, D))
        for d in range(D - 1):
            self.J = self.J.at[d, d, d + 1].set(1)
            self.J = self.J.at[d, d + 1, d].set(-1)

        # Share the dissipation module
        dissipation_drift = dissipation
        dissipation_diffusion = dissipation
        where = lambda shared_layers: shared_layers[0]
        get = lambda shared_layers: shared_layers[1]
        self.shared = eqx.nn.Shared(
            (dissipation_drift, dissipation_diffusion), where, get
        )

    @property
    def dissipation(self) -> eqx.Module:
        """Dissipation matrix wrapper

        Returns:
            eqx.Module: dissipation matrix module
        """
        return self.shared()[0]

    def _matrix_div(self, M: eqx.Module, x: ArrayLike) -> Array:
        r"""Computes the matrix divergence of a matrix function $M(x)$.

        This is defined in component form as
        $$
            [\nabla \cdot M(x)]_i = \sum_j \frac{\partial M_{ij}}{\partial x_j}.
        $$

        Args:
            M (eqx.Module): matrix function
            x (ArrayLike): state

        Returns:
            Array: \nabla \cdot M(x)
        """
        jac_M_x = jax.jacfwd(M)(x)
        return jnp.trace(jac_M_x, axis1=1, axis2=2)
    
    
    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: drift vector field
        """
        temperature = args[0]
        dissipation = self.shared()[0]
        dvdx = jax.grad(self.potential, argnums=0)(x, args)

        #compution of the gamma term
        H = self.Hamiltonian(x)
        grad_H = jax.jacfwd(self.Hamiltonian, argnums=0)(x)
        gamma = jnp.einsum('dab,db->a', self.J, grad_H) - jnp.einsum('d,dab,b->a', H, self.J, dvdx)


        return - dissipation(x) @ dvdx + temperature * self._matrix_div(dissipation, x) + gamma



    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: diffusion matrix
        """
        temperature = args[0]
        dissipation = self.shared()[1]
        M_x = dissipation(x)
        sqrt_M_x = jnp.linalg.cholesky(M_x)
        return jnp.sqrt(2.0 * temperature) * sqrt_M_x



class OnsagerNetHD2(SDE):
    shared: eqx.nn.Shared
    potential: eqx.Module
    Hamiltonian: eqx.Module
    J: Array
    
    def __init__(
        self, D: int, potential: eqx.Module, diffusion: eqx.Module, Hamiltonian: eqx.Module
    ) -> None:
        """Second way for Stochastic OnsagerNet model based on Helmholtz Decomposition.

        This is a modified version of the Stochastic OnsagerNet model.
        Let $X(t) \in \mathbb{R}^d$. This model is defined by the SDE
        $$
            dX(t) = -
            \left[
                -M(x) \nabla V(x) + \epsilon \nabla\cdot M(x) + \gamma(x)
            \right] dt
            + \sqrt{2 \epsilon} Diff(x) dW(t)
        $$
        
        $$
            M(x) = Diff(x) Diff(x)^T
        $$

        $$
            \gamma = \sum_{d=1}^{D-1} J_d \nabla H_d  - \sum_{d=1}^{D-1}H_d J_d \nabla V.
        $$

        where

        - $Diff: \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the diffusion part,
          which is symmetric positive semi-definite for all $x$
        - $V : \mathbb{R}^{d} \to \mathbb{R}$ is the potential function
        - $\gamma : \mathbb{R}^{d} \to \mathbb{R}^{d}$ is the Helmholtz decomposition term
        - $H : \mathbb{R}^{d} \to \mathbb{R}^{d-1}$. H e^{-V} is the Hamiltonian decomposition of \gamma e^{-V}.
        - $J_d$ is a $D\times D$ matrix with only two non-zero elements: 
            a $1$ at the position $(d, d+1)$ and a $-1$ at the position $(d+1, d)$.
        
        Notice that the main difference with `OnsagerNet` is that the
        diffusion matrix is now given by a (positive semi-definite) square root of the dissipation matrix.

        Args:
            potential (eqx.Module): potential function $V$
            diffusion (eqx.Module): diffusion function $Diff$
            Hamiltonian (eqx.Module): Hamiltonian functions $H$
        """ 
        self.potential = potential
        self.Hamiltonian = Hamiltonian 

        #define the J matrix for computing the gamma term
        
        self.J = jnp.zeros((D - 1, D, D))
        for d in range(D - 1):
            self.J = self.J.at[d, d, d + 1].set(1)
            self.J = self.J.at[d, d + 1, d].set(-1)

        # Share the dissipation module
        diffusion_drift = diffusion
        diffusion_diffusion = diffusion
        where = lambda shared_layers: shared_layers[0]
        get = lambda shared_layers: shared_layers[1]
        self.shared = eqx.nn.Shared(
            (diffusion_drift, diffusion_diffusion), where, get
        )

    def dissipation(self, x: ArrayLike)  -> Array:
        """Dissipation matrix wrapper

        Returns:
            Array: dissipation matrix
        """
        sqrtdissipation = self.shared()[1]
        M_x = sqrtdissipation(x)
        return M_x @ M_x.T

    def _matrix_div(self, M: eqx.Module, x: ArrayLike) -> Array:
        r"""Computes the matrix divergence of a matrix function $M(x)$.

        This is defined in component form as
        $$
            [\nabla \cdot M(x)]_i = \sum_j \frac{\partial M_{ij}}{\partial x_j}.
        $$

        Args:
            M (eqx.Module): matrix function
            x (ArrayLike): state

        Returns:
            Array: \nabla \cdot M(x)
        """
        jac_M_x = jax.jacfwd(M)(x)
        return jnp.trace(jac_M_x, axis1=1, axis2=2)
    
    
    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: drift vector field
        """

        temperature = args[0]
        dissipation = self.dissipation
        dvdx = jax.grad(self.potential, argnums=0)(x, args)

        #compution of the gamma term
        H = self.Hamiltonian(x)
        grad_H = jax.jacfwd(self.Hamiltonian, argnums=0)(x)
        gamma = jnp.einsum('dab,db->a', self.J, grad_H) - jnp.einsum('d,dab,b->a', H, self.J, dvdx)

        return - dissipation(x) @ dvdx #+ gamma + temperature * self._matrix_div(dissipation, x) 


    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: diffusion matrix
        """
        temperature = args[0] 
        diffusion = self.shared()[1]
        return jnp.sqrt(2.0 * temperature) * diffusion(x)
    
class OnsagerNetHD3(SDE):
    shared: eqx.nn.Shared
    potential: eqx.Module
    Hamiltonian: eqx.Module
    J: Array
    
    def __init__(
        self, D: int, potential: eqx.Module, diffusion: eqx.Module, Hamiltonian: eqx.Module
    ) -> None:
        """Second way for Stochastic OnsagerNet model based on Helmholtz Decomposition.

        This is a modified version of the Stochastic OnsagerNet model.
        Let $X(t) \in \mathbb{R}^d$. This model is defined by the SDE
        $$
            dX(t) = -
            \left[
                -M(x) \nabla V(x) + \epsilon \nabla\cdot M(x) + \gamma(x)
            \right] dt
            + \sqrt{2 \epsilon} Diff(x) dW(t)
        $$
        
        $$
            M(x) = Diff(x) Diff(x)^T
        $$

        $$
            \gamma = \sum_{d=1}^{D-1} J_d \nabla H_d  - \sum_{d=1}^{D-1}H_d J_d \nabla V.
        $$

        where

        - $Diff: \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the diffusion part,
          which is symmetric positive semi-definite for all $x$
        - $V : \mathbb{R}^{d} \to \mathbb{R}$ is the potential function
        - $\gamma : \mathbb{R}^{d} \to \mathbb{R}^{d}$ is the Helmholtz decomposition term
        - $H : \mathbb{R}^{d} \to \mathbb{R}^{d-1}$. H e^{-V} is the Hamiltonian decomposition of \gamma e^{-V}.
        - $J_d$ is a $D\times D$ matrix with only two non-zero elements: 
            a $1$ at the position $(d, d+1)$ and a $-1$ at the position $(d+1, d)$.
        
        Notice that the main difference with `OnsagerNet` is that the
        diffusion matrix is now given by a (positive semi-definite) square root of the dissipation matrix.

        Args:
            potential (eqx.Module): potential function $V$
            diffusion (eqx.Module): diffusion function $Diff$
            Hamiltonian (eqx.Module): Hamiltonian functions $H$
        """ 
        self.potential = potential
        self.Hamiltonian = Hamiltonian 

        #define the J matrix for computing the gamma term
        
        self.J = jnp.zeros((D, D, D))
        for d in range(D - 1):
            self.J = self.J.at[d, d, d + 1].set(1)
            self.J = self.J.at[d, d + 1, d].set(-1)
        self.J = self.J.at[D-1, D-1, 0].set(1)
        self.J = self.J.at[D-1, 0, D-1].set(-1)
        

        # Share the dissipation module
        diffusion_drift = diffusion
        diffusion_diffusion = diffusion
        where = lambda shared_layers: shared_layers[0]
        get = lambda shared_layers: shared_layers[1]
        self.shared = eqx.nn.Shared(
            (diffusion_drift, diffusion_diffusion), where, get
        )

    def dissipation(self, x: ArrayLike)  -> Array:
        """Dissipation matrix wrapper

        Returns:
            Array: dissipation matrix
        """
        sqrtdissipation = self.shared()[1]
        M_x = sqrtdissipation(x)
        return M_x @ M_x.T

    def _matrix_div(self, M: eqx.Module, x: ArrayLike) -> Array:
        r"""Computes the matrix divergence of a matrix function $M(x)$.

        This is defined in component form as
        $$
            [\nabla \cdot M(x)]_i = \sum_j \frac{\partial M_{ij}}{\partial x_j}.
        $$

        Args:
            M (eqx.Module): matrix function
            x (ArrayLike): state

        Returns:
            Array: \nabla \cdot M(x)
        """
        jac_M_x = jax.jacfwd(M)(x)
        return jnp.trace(jac_M_x, axis1=1, axis2=2)
    
    
    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: drift vector field
        """

        temperature = args[0]
        dissipation = self.dissipation
        dvdx = jax.grad(self.potential, argnums=0)(x, args)

        #compution of the gamma term
        H = self.Hamiltonian(x)
        grad_H = jax.jacfwd(self.Hamiltonian, argnums=0)(x)
        gamma = jnp.einsum('dab,db->a', self.J, grad_H) - jnp.einsum('d,dab,b->a', H, self.J, dvdx)
        return - dissipation(x) @ dvdx + gamma + temperature * self._matrix_div(dissipation, x) 


    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: diffusion matrix
        """
        temperature = args[0] 
        diffusion = self.shared()[1]
        return jnp.sqrt(2.0 * temperature) * diffusion(x) 
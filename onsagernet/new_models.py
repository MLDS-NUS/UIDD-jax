import jax
import jax.numpy as jnp
import equinox as eqx

from ._activations import get_activation
from ._layers import ConstantLayer

# ------------------------- Typing imports ------------------------- #

from jax import Array
from jax.typing import ArrayLike
from typing import Callable
from jax.random import PRNGKey

####Add by aiqing
from .new_layers import ConstantTensorLayer
from .models import MLP

class HamiltonianResMLP(MLP):
    r"""Hamiltonian network with a residual connection."""

    alpha: float
    gamma_layer: eqx.nn.Linear
    gamma_out_layer: eqx.nn.Linear
    dim: int
    param_dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        n_pot: int,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        r"""Potential network with a residual connection.

        This implements the modified potential function
        $$
            V(x, args) = \alpha \|(x, args)\|^2
            + \frac{1}{2}
            \| \text{MLP}(x, args)+ \Gamma (x, args) \|^2
        $$
        where

        - $\phi$ is a MLP of dim + param_dim -> n_pot
        - $\Gamma$ ia matrix of size [n_pot, dim + para_dim]
        - $\alpha > 0$ is a scalar regulariser

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            n_pot (int): size of the MLP part of the potential
            alpha (float): regulariser
            param_dim (int, optional): dimension of the parameters. Defaults to 0.
        """
        self.dim = dim
        self.param_dim = param_dim
        units = units + [n_pot]
        mlp_key, gamma_key, gamma_out_key = jax.random.split(key, 3)
        super().__init__(mlp_key, dim + param_dim, units, activation)
        self.alpha = alpha
        self.gamma_layer = eqx.nn.Linear(
            dim + param_dim, n_pot, key=gamma_key, use_bias=False
        )
        self.gamma_out_layer = eqx.nn.Linear(
            n_pot+1, dim-1, key=gamma_out_key, use_bias=False
        )

    def __call__(self, x: ArrayLike, args: ArrayLike=None) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        output_phi = super().__call__(x)
        output_gamma = self.gamma_layer(x) 
        output_combined = jnp.concatenate([(output_phi + output_gamma)**2, jnp.array([self.alpha * (x @ x)])], axis=0)

        return self.gamma_out_layer(output_combined)

class PotentialResMLP_scale(MLP):
    r"""Potential network with a residual connection and a scale parameter."""

    alpha: float
    scale: float
    gamma_layer: eqx.nn.Linear
    dim: int
    param_dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        n_pot: int,
        alpha: float,
        scale: float,
        param_dim: int = 0,
    ) -> None:
        r"""Potential network with a residual connection.

        This implements the modified potential function
        $$
            V(x, args) = \alpha \|(x, args)\|^2
            + \frac{1}{2}
            \| \text{MLP}(x, args)+ \Gamma (x, args) \|^2 * scale
        $$
        where

        - $\phi$ is a MLP of dim + param_dim -> n_pot
        - $\Gamma$ ia matrix of size [n_pot, dim + para_dim]
        - $\alpha > 0$ is a scalar regulariser
        - $scale > 0$ is a scalar regulariser for the potential

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            n_pot (int): size of the MLP part of the potential
            alpha (float): regulariser
            param_dim (int, optional): dimension of the parameters. Defaults to 0.
        """
        self.dim = dim
        self.param_dim = param_dim
        units = units + [n_pot]
        mlp_key, gamma_key = jax.random.split(key)
        super().__init__(mlp_key, dim + param_dim, units, activation)
        self.alpha = alpha
        self.scale = scale
        self.gamma_layer = eqx.nn.Linear(
            dim + param_dim, n_pot, key=gamma_key, use_bias=False
        )

    def __call__(self, x: ArrayLike, args: ArrayLike) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        output_phi = super().__call__(x)
        output_gamma = self.gamma_layer(x)
        output_combined = (output_phi + output_gamma) @ (output_phi + output_gamma)* self.scale
        regularisation = self.alpha * (x @ x)
        return 0.5 * output_combined + regularisation


class PotentialResMLP_extend(MLP):
    r"""Potential network with a residual connection and additional network."""

    alpha: float
    scale_net: MLP
    gamma1_layer: eqx.nn.Linear
    gamma2_layer: eqx.nn.Linear
    dim: int
    param_dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        n_pot: int,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        r"""Potential network with a residual connection.

        This implements the modified potential function
        $$
            V(x, args) = \alpha \|(x, args)\|^2
            + \frac{1}{2}
            \| \text{MLP}(x, args)+ \Gamma (x, args) \|^2
        $$
        where

        - $\phi$ is a MLP of dim + param_dim -> n_pot
        - $\Gamma$ ia matrix of size [n_pot, dim + para_dim]
        - $\alpha > 0$ is a scalar regulariser

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            n_pot (int): size of the MLP part of the potential
            alpha (float): regulariser
            param_dim (int, optional): dimension of the parameters. Defaults to 0.
        """
        self.dim = dim
        self.param_dim = param_dim
        units = units + [n_pot]
        mlp_key, gamma1_key, gamma2_key, scale_key = jax.random.split(key, 4)
        super().__init__(mlp_key, dim + param_dim, units, activation)
        self.alpha = alpha
        self.scale_net = MLP(scale_key, dim + param_dim, units, activation)
        self.gamma1_layer = eqx.nn.Linear(
            dim + param_dim, n_pot, key=gamma1_key, use_bias=False
        )
        self.gamma2_layer = eqx.nn.Linear(
            dim + param_dim, n_pot, key=gamma2_key, use_bias=False
        )

    def __call__(self, x: ArrayLike, args: ArrayLike) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        output_phi = super().__call__(x)
        output_gamma1 = self.gamma1_layer(x)
        output_gamma2 = self.gamma2_layer(x)
        output_scale = self.scale_net(x)
        output_combined_a = (output_phi + output_gamma1)*(output_scale  + output_gamma2)
        output_combined = (output_combined_a) @ (output_combined_a)*10
        regularisation = self.alpha * (x @ x)
        return 0.5 * output_combined + regularisation 

class DiffusionConstant(eqx.Module):
    """Diagonal diffusion matrix network based on a constant tensor layer."""

    alpha: float 
    diag_constant_layer: ConstantLayer
    tril_constant_layer: ConstantTensorLayer
    dim: int

    def __init__(self, key: PRNGKey, dim: int, alpha: float) -> None:
        r"""Diagonal diffusion matrix network based on a constant layer.

        This implements the diffusion matrix function that is constant
        $$
            \sigma(x, args) = \text{diag}(\alpha + \text{Constant}^2)^{\frac{1}{2}}.
        $$
        where $\text{Constant}$ is a vector of size `dim`.
        Note that by constant we mean that it does not depend on the input $x$ or the parameters $args$,
        but it can be trained.

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            alpha (float): regulariser
        """
        self.dim = dim
        self.alpha = alpha
        self.diag_constant_layer = ConstantLayer(dim, key)
        self.tril_constant_layer = ConstantTensorLayer((dim, dim), key)

    def __call__(self, x: ArrayLike = None, args: ArrayLike = None) -> Array:
        sigma_diag = self.diag_constant_layer()
        sigma_squared_regularised = (jnp.sqrt(self.alpha + sigma_diag**2))
        sigma_tril = self.tril_constant_layer()*0.1
        return jnp.diag(sigma_squared_regularised) + jnp.tril(sigma_tril, k=-1)

class DiffusionCholeskyMLP(eqx.Module):
    """Diffusion matrix network based on a multi-layer perceptron and cholesky factorization."""

    alpha: float
    dim: int
    param_dim: int
    diagnet: MLP
    trilnet: MLP

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        r"""Diffusion matrix network based on a multi-layer perceptron and cholesky factorization.

        This implements the diffusion matrix function
        $$
            \sigma(x, args) = \text{diag}(\alpha + \text{MLP}(x, args)^2)^{\frac{1}{2}}.
        $$
        Here, MLP maps $(x, args)$ of dimension `dim` + `param_dim` to a vector of size `dim`.

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            alpha (float): regulariser
            param_dim (int, optional): dimension of the parameters. Defaults to 0.
        """
        self.dim = dim
        self.param_dim = param_dim 
        d_key, t_key = jax.random.split(key, 2)
        self. diagnet = MLP(d_key, dim + param_dim, units + [dim], activation)
        self. trilnet = MLP(t_key, dim + param_dim, units + [dim**2], activation)
        self.alpha = alpha

    def __call__(self, x: ArrayLike, args: ArrayLike = None) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        sigma_diag = self. diagnet(x)
        sigma_diag_regularised = (jnp.sqrt(self.alpha + sigma_diag**2))


        return (jnp.diag(sigma_diag_regularised)
                + jnp.tril(self. trilnet(x).reshape([self.dim, self.dim]), k=-1)*0.1
                )
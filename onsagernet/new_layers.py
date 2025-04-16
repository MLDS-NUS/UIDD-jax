import jax
import equinox as eqx
from math import sqrt
from ._utils import default_floating_dtype
from jax import Array
from jax.random import PRNGKey
from typing import Optional
from jax.typing import DTypeLike


####Add by aiqing
class ConstantTensorLayer(eqx.Module):
    """Constant Tensor layer."""

    weight: Array
    shape: tuple

    def __init__(
        self, shape: tuple, key: PRNGKey, dtype: Optional[DTypeLike] = None
    ) -> None:
        """Constant Tensor layer.
        Returns a constant, trainable tensor.

        Args:
            shape (tuple): shape of the constant tensor
            key (PRNGKey): random key
            dtype (Optional[DTypeLike], optional): data type. Defaults to None.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        lim = 1 / sqrt(shape[0])
        self.shape = shape
        self.weight = jax.random.uniform(
            key, shape, minval=-lim, maxval=lim, dtype=dtype
        )

    def __call__(self, *, key: Optional[PRNGKey] = None) -> Array:
        return self.weight
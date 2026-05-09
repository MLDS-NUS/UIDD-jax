"""nonlinear case."""

import os 
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import sys
sys.path.append('..')
from utils.sde import SDEIntegrator
from utils.data import shrink_trajectory_len

import hydra
import logging
from omegaconf import DictConfig
from datasets import Dataset, Features, Array2D

sys.path.append('../..')
from UIDD.dynamics import OnsagerNetFD, OnsagerNet, UIDD2, SDEfromFunc
from UIDD.models import PotentialMLP, DissipationMatrixMLP, ConservationMatrixMLP
from UIDD.trainers import MLETrainer
from UIDD.models import PotentialMLP_scale, DiffusionConstant
from UIDD.models import MLP

from onsager_reg import OnsagerNetReg
# ------------------------- Typing imports ------------------------- #

from jax import Array
from jax.typing import ArrayLike


class DriftMLP(eqx.Module):
    net: MLP

    def __init__(self, key: Array, dim: int, units: list[int], activation: str) -> None:
        self.net = MLP(
            key=key,
            dim=dim,
            units=units + [dim],
            activation=activation,
        )

    def __call__(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        return self.net(x)


class DiffusionConst(eqx.Module):
    net: DiffusionConstant

    def __init__(self, key: Array, dim: int, alpha: float) -> None:
        self.net = DiffusionConstant(key=key, dim=dim, alpha=alpha)

    def __call__(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        temperature = args[0]
        return jnp.sqrt(temperature) * self.net(x, args)


def build_targets(kappa: float) -> OnsagerNetFD:
    r"""Builds a nonlinear 2D target dynamics with a bimodal invariant law.

    The target potential is
    $$
    V(x, y)=0.25 (x^2-a^2)^2 + 2.5 y^2 + \epsilon (1+\sin x) y^2,
    $$
    with constant dissipation and constant anti-symmetric circulation.

    This yields a nonlinear drift of the form
    $$
    dX_t = -[M + W]\nabla V(X_t)\,dt + \sqrt{2T}\,M^{1/2}dW_t,
    $$
    whose invariant distribution is proportional to $\exp(-V/T)$.

    Args:
        kappa (float): strength of the anti-symmetric circulation term

    Returns:
        OnsagerNetFD: target nonlinear dynamics
    """
    a = 1.0
    b = 1.0
    eps= 0.01
    M = jnp.eye(2)
    J = kappa * jnp.array([[0.0, -1.0], [1.0, 0.0]])

    def dissipation(x: ArrayLike) -> Array:
        return M*eps

    def conservation(x: ArrayLike) -> Array:
        return J*eps

    def potential(x: ArrayLike, args: ArrayLike) -> Array:
        x1, x2 = x
        return (0.25 * (x1**2 - a**2) ** 2 + 2.5 * x2**2 + b * (1.0 + jnp.sin(x1)) * x2**2)/eps

    return OnsagerNetFD(
        potential=potential, dissipation=dissipation, conservation=conservation
    )


def load_data(config: DictConfig) -> Dataset:
    """Load the dataset for the test case

    Args:
        config (DictConfig): configuration object

    Returns:
        Dataset: huggingface dataset
    """
    if config.dim != 2:
        raise ValueError("The nonlinear target model is defined only for dim=2.")

    kappa = float(config.data.var)
    print(f"Using nonlinear target with a=1.0, b=1.0, eps=0.01, kappa={kappa}")
    target_SDE = build_targets(kappa)
    integrator = SDEIntegrator(model=target_SDE, state_dim=config.dim)

    init_key, bm_key = jr.split(jr.PRNGKey(config.data.seed), 2)
    bm_keys = jr.split(bm_key, config.data.num_runs + config.data.num_runs_test)
    init_min = jnp.asarray(config.data.init_min, dtype=jnp.float64)
    init_max = jnp.asarray(config.data.init_max, dtype=jnp.float64)
    init_conditions = jr.uniform(
        key=init_key,
        shape=(config.data.num_runs + config.data.num_runs_test, config.dim),
        minval=init_min,
        maxval=init_max,
    )
    substeps = 10
    sol = integrator.parallel_solve(
        initial_conditions=init_conditions,
        key=bm_keys,
        t0=config.data.t0,
        t1=config.data.t1,
        dt=config.dt / substeps,
        args=[config.temperature],
    )
    ts_rec = sol.ts[:, ::substeps]
    ys_rec = sol.ys[:, ::substeps]
    traj_length = ts_rec.shape[1]
    features = Features(
        {
            "t": Array2D(shape=(traj_length, 1), dtype="float64"),
            "x": Array2D(shape=(traj_length, config.dim), dtype="float64"),
            "args": Array2D(shape=(traj_length, 1), dtype="float64"),
        }
    )

    dataset = Dataset.from_dict(
        {
            "t": ts_rec[:config.data.num_runs, :, None],
            "x": ys_rec[:config.data.num_runs],
            "args": config.temperature * jnp.ones_like(ts_rec[:config.data.num_runs, :, None]),
        },
        features=features,
    )
    test_dataset = Dataset.from_dict(
        {
            "t": ts_rec[config.data.num_runs:, :, None],
            "x": ys_rec[config.data.num_runs:],
            "args": config.temperature * jnp.ones_like(ts_rec[config.data.num_runs:, :, None]),
        },
        features=features,
    )
    return dataset.with_format("jax"), test_dataset.with_format("jax")

from typing import Any
import jax.tree_util as tree
def get_filter_spec(model) -> Any:
    filter_spec = tree.tree_map(lambda _: True, model)

    # set 'name' as False
    filter_spec = eqx.tree_at(
        lambda m: m.J,  # path is model.name
        filter_spec,
        replace=False      # False -> static
    )
    return filter_spec

def build_UIDD2(config: DictConfig) -> UIDD2:
    """Build the UIDD2 model to learn the target dynamics

    Args:
        config (DictConfig): configuration object

    Returns:
        UIDD2: UIDD2 model
    """

    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, m_key, w_key = jax.random.split(init_keys, 3)

    potential = PotentialMLP_scale(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        alpha=config.model.potential.alpha,
        scale=50
    )
    Diffusion = DiffusionConstant(
        key=m_key,
        dim=config.dim,
        alpha=1,
    )
    Hamiltonian = MLP(
        key=w_key,
        dim=config.dim,
        units=config.model.hamiltonian.units + [config.dim-1],
        activation=config.model.hamiltonian.activation,
    )
    return UIDD2(config.dim, potential, Diffusion, Hamiltonian) 

def build_OnsagerNet(config: DictConfig) -> OnsagerNet:
    """Build the OnsagerNet model to learn the target dynamics

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNet: OnsagerNet model
    """

    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, m_key, w_key = jax.random.split(init_keys, 3)

    potential = PotentialMLP(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        alpha=config.model.potential.alpha,
    )
    dissipation = DissipationMatrixMLP(
        key=m_key,
        dim=config.dim,
        units=config.model.dissipation.units,
        activation=config.model.dissipation.activation,
        alpha=config.model.dissipation.alpha,
        is_bounded=config.model.dissipation.is_bounded,
    )
    conservation = ConservationMatrixMLP(
        key=w_key,
        dim=config.dim,
        units=config.model.conservation.units,
        activation=config.model.conservation.activation,
        is_bounded=config.model.dissipation.is_bounded,
    )
    Diffusion = DiffusionConstant(
        key=m_key,
        dim=config.dim,
        alpha=1,
    )
    return OnsagerNet(potential, dissipation, conservation, Diffusion)


def build_OnsagerNetReg(config: DictConfig) -> OnsagerNetReg:
    """Build the OnsagerNetReg model (OnsagerNet + V regularization).

    Reads ``config.model.potential.l2_weight`` and
    ``config.model.potential.sobolev_weight`` if present (default 0 → no
    regularization). ``model.regu`` is auto-set when any weight is positive,
    which is what ``UIDD._losses.MLELoss`` checks before adding
    ``model.regularization(x)`` to the loss.
    """

    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, m_key, w_key = jax.random.split(init_keys, 3)

    potential = PotentialMLP(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        alpha=config.model.potential.alpha,
    )
    dissipation = DissipationMatrixMLP(
        key=m_key,
        dim=config.dim,
        units=config.model.dissipation.units,
        activation=config.model.dissipation.activation,
        alpha=config.model.dissipation.alpha,
        is_bounded=config.model.dissipation.is_bounded,
    )
    conservation = ConservationMatrixMLP(
        key=w_key,
        dim=config.dim,
        units=config.model.conservation.units,
        activation=config.model.conservation.activation,
        is_bounded=config.model.dissipation.is_bounded,
    )
    Diffusion = DiffusionConstant(
        key=m_key,
        dim=config.dim,
        alpha=1,
    )
    l2_weight = float(config.model.potential.get("l2_weight", 0.0))
    sobolev_weight = float(config.model.potential.get("sobolev_weight", 0.0))
    return OnsagerNetReg(
        potential, dissipation, conservation, Diffusion,
        l2_weight=l2_weight,
        sobolev_weight=sobolev_weight,
        default_args=(float(config.temperature),),
    )


def build_SDEfromFunc(config: DictConfig) -> SDEfromFunc:
    """Build a generic unstructured SDE model.

    Both drift and diffusion are learned directly by neural networks,
    without any Onsager/UIDD structure.
    """

    init_keys = jax.random.PRNGKey(config.model.seed)
    drift_key, diffusion_key = jax.random.split(init_keys, 2)

    drift_func = DriftMLP(
        key=drift_key,
        dim=config.dim,
        units=config.model.drift.units,
        activation=config.model.drift.activation,
    )
    diffusion_func = DiffusionConst(
        key=diffusion_key,
        dim=config.dim,
        alpha=1,
    )

    return SDEfromFunc(drift_func=drift_func, diffusion_func=diffusion_func)


@hydra.main(config_path="./config", config_name="nonlinear_case", version_base=None)
def train_model(config: DictConfig) -> None:
    """Main training routine

    Args:
        config (DictConfig): configuration object
    """

    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(runtime_dir)

    logger = logging.getLogger(__name__)

    logger.info(f"Loading dataset...")
    train_dataset, test_dataset = load_data(config)

    train_traj_len = config.train.get("train_traj_len", None)
    if train_traj_len is not None:
        train_dataset = shrink_trajectory_len(
            train_dataset, train_traj_len
        )  # change the trajectory length to improve GPU usage
        test_dataset = shrink_trajectory_len(
            test_dataset, train_traj_len
        )  

    logger.info(f"Building model...")
    if config.Model_name == 'Onsager':
        model = build_OnsagerNet(config)
        filter_spec=None
    elif config.Model_name == 'OnsagerReg':
        model = build_OnsagerNetReg(config)
        filter_spec = None
    elif config.Model_name == 'HD2':
        model = build_UIDD2(config)
        filter_spec = get_filter_spec(model)
    elif config.Model_name == 'SDE':
        model = build_SDEfromFunc(config)
        filter_spec = None
    else:
        raise ValueError("wrong model")
    trainer = MLETrainer(opt_options=config.train.opt, rop_options=config.train.rop)

    logger.info(f"Training {config.Model_name} for {config.train.num_epochs} epochs...")
    # print(dataset) 
    trained_model, _, _ = trainer.train(
        model=model,
        dataset=train_dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        test_dataset= test_dataset,
        logger=logger,
        filter_spec=filter_spec,
        checkpoint_dir=runtime_dir,
        checkpoint_every=config.train.checkpoint_every,
        print_every=config.train.print_every,
    )

    logger.info(f"Saving output to {runtime_dir}")
    eqx.tree_serialise_leaves(os.path.join(runtime_dir, "model.eqx"), trained_model)


if __name__ == "__main__":
    train_model()

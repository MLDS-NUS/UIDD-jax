"""linear case."""

import os 
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.18"

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
from onsagernet.dynamics import OnsagerNetFD, OnsagerNet,  OnsagerNetHD2
from onsagernet.models import PotentialMLP, DissipationMatrixMLP, ConservationMatrixMLP
from onsagernet.trainers import MLETrainer
from onsagernet.models import PotentialMLP_scale, DiffusionConstant
from onsagernet.models import MLP
# ------------------------- Typing imports ------------------------- #

from jax import Array
from jax.typing import ArrayLike


def build_targets(M: ArrayLike, J: ArrayLike, S: ArrayLike) -> OnsagerNetFD:
    r"""Builds the target linear dynamics 
    $$ d X(t) =M S X(t) - J S X(t) d t + \sqrt{2M} d \omega(t)$$
    where 
    $M$ and $S$ are symmetric positive definite matrices, 
    $J$ is a antisymmetric matrix
    Returns:
        OnsagerNetFD: target dynamics
    """
    def dissipation(x: ArrayLike) -> Array:
        return M

    def conservation(x: ArrayLike) -> Array:
        return J

    def potential(x: ArrayLike, args: ArrayLike) -> Array:
        return 0.5 * (x @ S @ x)

    return OnsagerNetFD(
        potential=potential, dissipation=dissipation, conservation=conservation
    )

def generate_coefficients(dim: int, key: Array) -> Array:
    """Generate the coefficients for the linear dynamics
    """
    key1, key2, key3, key4, key5 = jax.random.split(key, 5) 

    M = jax.random.normal(key1, (dim, dim))
    J = jax.random.normal(key2, (dim, dim))
    S = jax.random.normal(key3, (dim, dim))
    M = M/jnp.max(jnp.abs(M)) 
    J = J/jnp.max(jnp.abs(J)) 
    S = S/jnp.max(jnp.abs(S))
    alpha_M=jax.random.normal(key4, dim)
    alpha_S=jax.random.normal(key5, dim) 
    # return (jnp.abs(alpha_M) * jnp.eye(dim) + M@M.T), (J-J.T), (jnp.abs(alpha_S) * jnp.eye(dim) + S@S.T)
    return (jnp.abs(alpha_M) * jnp.eye(dim) + M@M.T)/50, (J-J.T)*0.4, (jnp.abs(alpha_S) * jnp.eye(dim) + S@S.T)*10


def load_data(config: DictConfig) -> Dataset:
    """Load the dataset for the test case

    Args:
        config (DictConfig): configuration object

    Returns:
        Dataset: huggingface dataset
    """
    M, J, S = generate_coefficients(config.dim, jr.PRNGKey(config.data.seed))
    print(M,'\n', J,'\n',  S) 
    target_SDE = build_targets(M, J*config.data.var, S)
    integrator = SDEIntegrator(model=target_SDE, state_dim=config.dim)

    init_key, bm_key = jr.split(jr.PRNGKey(config.data.seed), 2)
    bm_keys = jr.split(bm_key, config.data.num_runs + config.data.num_runs_test)
    init_conditions = config.data.init_scale * jr.normal(
        key=init_key, shape=(config.data.num_runs + config.data.num_runs_test, config.dim)
    ) 
    sol = integrator.parallel_solve(
        initial_conditions=init_conditions,
        key=bm_keys,
        t0=config.data.t0,
        t1=config.data.t1,
        dt=config.dt,
        args=[config.temperature],
    ) 
    traj_length = sol.ts.shape[1]
    features = Features(
        {
            "t": Array2D(shape=(traj_length, 1), dtype="float64"),
            "x": Array2D(shape=(traj_length, config.dim), dtype="float64"),
            "args": Array2D(shape=(traj_length, 1), dtype="float64"),
        }
    )

    dataset = Dataset.from_dict(
        {
            "t": sol.ts[:config.data.num_runs, :, None],
            "x": sol.ys[:config.data.num_runs],
            "args": config.temperature * jnp.ones_like(sol.ts[:config.data.num_runs, :, None]),
        },
        features=features,
    )
    test_dataset = Dataset.from_dict(
        {
            "t": sol.ts[config.data.num_runs:, :, None],
            "x": sol.ys[config.data.num_runs:],
            "args": config.temperature * jnp.ones_like(sol.ts[config.data.num_runs:, :, None]),
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

def build_OnsagerNetHD2(config: DictConfig) -> OnsagerNetHD2:
    """Build the OnsagerNetHD2 model to learn the target dynamics

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNetHD2: OnsagerNetHD2 model
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
    return OnsagerNetHD2(config.dim, potential, Diffusion, Hamiltonian) 

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


@hydra.main(config_path="./config", config_name="linear_case", version_base=None)
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
    elif config.Model_name == 'HD2':
        model = build_OnsagerNetHD2(config)
        filter_spec = get_filter_spec(model)
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

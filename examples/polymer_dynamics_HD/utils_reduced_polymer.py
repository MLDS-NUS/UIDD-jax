"""utils for reduced polymer dynamics."""

import os
import pickle
import jax
import equinox as eqx
import sys
sys.path.append('..')

from omegaconf import DictConfig
from datasets import Dataset, Features, Array2D
from sklearn.model_selection import train_test_split

sys.path.append('../..')
from onsagernet.dynamics import OnsagerNet, OnsagerNetHD2
from onsagernet.models import PotentialResMLP_scale
from onsagernet.models import MLP
from onsagernet.models import (
    PotentialResMLP,
    DissipationMatrixMLP,
    ConservationMatrixMLP,
    DiffusionDiagonalConstant,
)
# ------------------------- Typing imports ------------------------- #



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

def load_data(dataset_name: str) -> Dataset:
    """Load the dataset for the test case

    Args:
        config (DictConfig): configuration object

    Returns:
        Dataset: huggingface dataset
    """
    load_path = os.path.join('OriginalData', 'dataset.pkl')
    # load dataset
    with open(load_path, 'rb') as f:
        loaded_dataset = pickle.load(f)

    dataset = loaded_dataset[dataset_name]
    x_train, x_test, t_train, t_test, args_train, args_test = train_test_split(dataset['x'], dataset['t'], dataset['args'], test_size=0.2, random_state=42)
    traj_length = int(len(dataset['t'][0]))
    features = Features(
        {
            "t": Array2D(shape=(traj_length, 1), dtype="float32"),
            "x": Array2D(shape=(traj_length, 3), dtype="float32"),
            "args": Array2D(shape=(traj_length, 2), dtype="float32"),
        }
    )

    dataset = Dataset.from_dict(
        {
            "t": t_train,
            "x": x_train,
            "args": args_train,
        },
        features=features,
    )
    test_dataset = Dataset.from_dict(
        {
            "t": t_test,
            "x": x_test,
            "args": args_test,
        },
        features=features,
    )
    return dataset.with_format("jax"), test_dataset.with_format("jax")

def build_OnsagerNetHD2_scale(config: DictConfig, scale:float = 1.0) -> OnsagerNetHD2:
    """Build the OnsagerNetHD2 model to learn the target dynamics

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNetHD2: OnsagerNetHD2 model
    """

    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, d_key, h_key = jax.random.split(init_keys, 3)

    potential = PotentialResMLP_scale(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
        scale=scale,
    )

    Diffusion = DiffusionDiagonalConstant(
        key=d_key,
        dim=config.dim,
        alpha=config.model.diffusion.alpha,
    )
    Hamiltonian = MLP(
        key=h_key,
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
    v_key, m_key, w_key, d_key = jax.random.split(init_keys, 4)

    potential = PotentialResMLP(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
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

    Diffusion = DiffusionDiagonalConstant(
        key=d_key,
        dim=config.dim,
        alpha=config.model.diffusion.alpha,
    ) 
    return OnsagerNet(potential, dissipation, conservation, Diffusion)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
def PlotV(net, config, level_end =500.0, Title = None):
    z1_min, z1_max = -3, 3
    z2_min, z2_max = -2, 2
    z3_min, z3_max = -6, 6

    num_samples_aux_dimension = 128
    def V(z1, z2, z3):
        z = jnp.array([z1, z2, z3])
        return net.potential(z, [config.temperature])
    @jax.vmap
    @jax.jit
    def V_12(z1, z2):
        z3_test_range = jnp.linspace(z3_min, z3_max, num_samples_aux_dimension)
        return jnp.min(jax.vmap(V, (None, None, 0))(z1, z2, z3_test_range))

    @jax.vmap
    @jax.jit
    def V_13(z1, z3):
        z2_test_range = jnp.linspace(z2_min, z2_max, num_samples_aux_dimension)
        return jnp.min(jax.vmap(V, (None, 0, None))(z1, z2_test_range, z3))

    @jax.vmap
    @jax.jit
    def V_23(z2, z3):
        z1_test_range = jnp.linspace(z1_min, z1_max, num_samples_aux_dimension)
        return jnp.min(jax.vmap(V, (0, None, None))(z1_test_range, z2, z3))
    num_grid = 100
    num_levels = 50
    off_set = 1
    level_start = 0.0
    

    custom_levels = jnp.linspace(level_start, level_end, num_levels)
    # custom_levels = None

    # Define the range and number of points in each dimension
    z1_range = jnp.linspace(z1_min - off_set, z1_max + off_set, num_grid)
    z2_range = jnp.linspace(z2_min - off_set, z2_max + off_set, num_grid)
    z3_range = jnp.linspace(z3_min - off_set, z3_max + off_set, num_grid)

    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    # V12
    Z1, Z2 = jnp.meshgrid(z1_range, z2_range)
    
    V12_grid = V_12(Z1.ravel(), Z2.ravel()).reshape(Z1.shape)
    V12_0 = jnp.min(V12_grid) 
    contour1 = ax[0].contour(Z1, Z2, V12_grid - V12_0, levels=custom_levels)
    ax[0].set_title(r'$V_{12}$')
    ax[0].set_xlabel(r'$Z_1$')
    ax[0].set_ylabel(r'$Z_2$')

    # V13
    Z1, Z3 = jnp.meshgrid(z1_range, z3_range)
    V13_grid = V_13(Z1.ravel(), Z3.ravel()).reshape(Z1.shape)
    V13_0 = jnp.min(V13_grid) 
    contour2 = ax[1].contour(Z1, Z3, V13_grid - V13_0, levels=custom_levels)
    ax[1].set_title(r'$V_{13}$')
    ax[1].set_xlabel(r'$Z_1$')
    ax[1].set_ylabel(r'$Z_3$')

    # V23
    Z2, Z3 = jnp.meshgrid(z2_range, z3_range)
    V23_grid = V_23(Z2.ravel(), Z3.ravel()).reshape(Z1.shape)
    V23_0 = jnp.min(V23_grid) 
    contour3 = ax[2].contour(Z2, Z3, V23_grid - V23_0, levels=custom_levels)
    ax[2].set_title(r'$V_{23}$')
    ax[2].set_xlabel(r'$Z_2$')
    ax[2].set_ylabel(r'$Z_3$')
    cbar = fig.colorbar(contour3, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    if Title is not None:
        fig.suptitle(Title)
    return fig, ax

import jax.random as jr 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import os  
from examples.utils.sde import SDEIntegrator

import pickle

def entropy_production_polymer(net, temperature=1.0, end_time=2.0, dt=0.0005):
    z1_min, z1_max = -3, 3
    z2_min, z2_max = -2, 2
    z3_min, z3_max = -6, 6

    init_unit = jr.uniform(
        key=jr.PRNGKey(123), shape=(1000,3)
    )
    bounds = jnp.array([[z1_min, z1_max],[z2_min, z2_max],[z3_min, z3_max]])
    init_conditions = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * init_unit 

    integrator = SDEIntegrator(model=net, state_dim=3)

    split_number=5
    for i in range(split_number):
        key = jr.PRNGKey(i) 
        bm_keys = jr.split(key, 1000) 
        sol = integrator.parallel_solve(
            key=bm_keys,
            initial_conditions=init_conditions,
            t0=0.0,
            t1=end_time/split_number,
            dt=dt,
            args=[temperature],
        )
        init_conditions = sol.ys[:, -1, :]

    predicted_trajectories = sol.ys

    def gamma_net(net):
        def gamma(x):
            args=[1]  
            dvdx = jax.grad(net.potential, argnums=0)(x, args)

            #compution of the gamma term
            H = net.Hamiltonian(x)
            grad_H = jax.jacfwd(net.Hamiltonian, argnums=0)(x)
            gamma_value = jnp.einsum('dab,db->a', net.J, grad_H) - jnp.einsum('d,dab,b->a', H, net.J, dvdx)
            return gamma_value
        return gamma

    def ep_term(gamma, dissipation):
        def ep_term_x(x):
            return gamma(x)@(jnp.linalg.inv(dissipation(x))@gamma(x))
        return ep_term_x

    ep_term = ep_term(gamma_net(net), net.dissipation)
    x= predicted_trajectories[:,-1000:,].reshape(-1,3)
    ep = jax.vmap(ep_term)(x)
    
    return ep.mean(), predicted_trajectories

def PlotTraj(net, config, dataset_name="F23_10_T1"):
    load_path = os.path.join('OriginalData', 'dataset.pkl')
    # load dataset
    with open(load_path, 'rb') as f:
        loaded_dataset = pickle.load(f)

    dataset_fast = loaded_dataset[dataset_name+'_fast'].with_format("jax")
    dataset_middle = loaded_dataset[dataset_name+'_med'].with_format("jax")
    dataset_slow = loaded_dataset[dataset_name+'_slow'].with_format("jax")

    integrator = SDEIntegrator(model=net, state_dim=3)


    test_z = [dataset_fast["x"], dataset_middle["x"], dataset_slow["x"]]
    # num_runs = test_data_fast.shape[0]
    num_runs = 500


    keys  = jr.split(jr.PRNGKey(config.model.seed + 123), 3)
    predicted_trajectories = []


    init_conditions = jnp.repeat(test_z[0][0, 0, :][None, :], num_runs, axis=0)

    for key, traj in zip(keys, test_z):
        init_conditions = jnp.repeat(traj[0, 0, :][None, :], num_runs, axis=0)
        bm_keys = jr.split(key, num_runs)
        sol = integrator.parallel_solve(
            key=bm_keys,
            initial_conditions=init_conditions,
            t0=0.0,
            t1=config.dt * test_z[0].shape[1],
            dt=config.dt,
            args=[config.temperature],
        )
        predicted_trajectories.append(sol.ys)
    return test_z, predicted_trajectories
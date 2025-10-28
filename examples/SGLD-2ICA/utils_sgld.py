import os
import pickle
import jax
import equinox as eqx
import sys
sys.path.append('..') 

from omegaconf import DictConfig
from datasets import Dataset, Features, Array2D, load_from_disk 

sys.path.append('../..') 
from onsagernet.dynamics import OnsagerNetHD2 
from onsagernet.models import DiffusionCholeskyMLP, PotentialResMLP_scaleV2
from onsagernet.models import MLP

# ------------------------- Typing imports ------------------------- #

from typing import Any
import jax.tree_util as tree
def get_filter_spec(model) -> Any:
    filter_spec = tree.tree_map(lambda _: True, model)

    # set 'name' as Falss, to represent static
    filter_spec = eqx.tree_at(
        lambda m: m.J,  # path is model.name
        filter_spec,
        replace=False      # False -> static
    )
    return filter_spec

def load_data(cache_dir: str) -> Dataset:
    """Load the dataset for the test case

    Args:
        config (DictConfig): configuration object

    Returns:
        Dataset: huggingface dataset
    """
    load_path = os.path.join(cache_dir, "train")
    # load dataset
    dataset = load_from_disk(load_path)
    load_path = os.path.join(cache_dir, "test")
    test_dataset = load_from_disk(load_path)
    return dataset.with_format("jax"), test_dataset.with_format("jax")  


def build_OnsagerNetHD2(config: DictConfig) -> OnsagerNetHD2:
    """Build the OnsagerNetHD2 model to learn the target dynamics

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNetHD2: OnsagerNetHD2 model
    """

    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, d_key, h_key = jax.random.split(init_keys, 3)
 
    potential = PotentialResMLP_scaleV2(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
        scale=1 
    ) 
    Diffusion = DiffusionCholeskyMLP( 
        key = d_key,
        dim=config.dim,
        units= config.model.diffusion.units,
        activation= config.model.diffusion.activation,
        alpha= config.model.diffusion.alpha,
    )
    Hamiltonian = MLP(
        key=h_key,
        dim=config.dim,
        units=config.model.hamiltonian.units + [config.dim-1],
        activation=config.model.hamiltonian.activation,
    ) 
    return OnsagerNetHD2(config.dim, potential, Diffusion, Hamiltonian)



############### ICA Implementation ############
#  
# --------------------------------------------------------
import numpy as np
from datasets import Dataset, Features, Array2D, load_from_disk
def ICA(A, w0=None, block_size=1, max_iter=1000, lr=0.01, lam=1, n_samples=100):
    """
    SGLD implementation for ICA
    """
    m, n = A.shape
    w = np.random.uniform(size=(n,n))*4-2 if w0 is None else w0.copy()
 
    W=[w.copy().reshape(-1)]
    for k in range(max_iter):
        i = np.random.choice(m, size=block_size, replace=True)  # 随机选择行索引
        xi = A[i]
        Y = w @ xi.T 
        phi_Y = np.tanh(Y)
        
        # update with projection
        dw = (np.eye(n)*n - (phi_Y @ Y.T)  / block_size) @ w - lam * w@ w.T@ w 
        w = w + lr/2 * dw + np.random.normal(size=(n,n))@np.linalg.cholesky(w.T@ w) * (lr/n_samples) ** 0.5  
        W.append(w.copy().reshape(-1))
    return W



def process(X, step=5):
    X_np=np.array(X)
    X0=X_np[:-1:step, :]
    X1=X_np[1::step, :]
    return np.stack([X0,X1],axis=-2).tolist()

def generate_data(total_seed = 2000, batch_size=1, step=10, max_iter=1000, lr=0.05, W0=None) -> Dataset:
    np.random.seed(0)
    n_samples = 100       # number of data samples
    df_t = 1 
    heavy = np.random.standard_t(df_t, size=(n_samples, 1))
    heavy = heavy / np.std(heavy, axis=0, keepdims=True) 
    normal = np.random.normal(0, 1, size=(n_samples, 1))
    A= np.hstack([heavy, (normal)])
    # run ICA randomly
    Data_list=[]
    for seed in range(total_seed):
        np.random.default_rng()
        w0=None if W0 is None else W0[seed].reshape(2,2)
        W=ICA(A, w0=w0, block_size=batch_size, max_iter=max_iter, lr=lr, lam=1)
        Data_list.extend(process(W,step=step))

    time=[[[0,],[lr]]]*len(Data_list)
    args=[[[1.],[1.]]]*len(Data_list)
    data = {
        'x': Data_list,
        't': time,
        'args': args
    }
    features = Features(
        {
        "t": Array2D(shape=(2, 1), dtype="float32"),
        "x": Array2D(shape=(2, 4), dtype="float32"),
        "args": Array2D(shape=(2, 1), dtype="float32"),
        }
        )
    dataset = Dataset.from_dict(data, features=features)
    return dataset 


def full_batch_ICA(A, w0=None,  max_iter=1000, lr=0.01, lam=1, n_samples=100):
    """
    Full-batch ICA implementation
    """
    m, n = A.shape
    w = np.random.uniform(size=(n,n))*4-2 if w0 is None else w0.copy()

    W=[w.copy().reshape(-1)]
    for k in range(max_iter):
        xi = A
        Y = w @ xi.T 
        phi_Y = np.tanh(Y)
        
        # # 投影更新
        dw = (np.eye(n)*n - (phi_Y @ Y.T)  / m) @ w - lam * w@ w.T@ w 
        w = w + lr/2 * dw + np.random.normal(size=(n,n))@np.linalg.cholesky(w.T@ w) * (lr/n_samples) ** 0.5  
        W.append(w.copy().reshape(-1))
    return W

def generate_full_batch_data(total_seed = 2000, batch_size=1, step=10, max_iter=1000, lr=0.05, W0=None) -> Dataset:
    np.random.seed(0)
    n_samples = 100       # number of data samples
    df_t = 1 
    heavy = np.random.standard_t(df_t, size=(n_samples, 1))
    heavy = heavy / np.std(heavy, axis=0, keepdims=True) 
    normal = np.random.normal(0, 1, size=(n_samples, 1))
    A= np.hstack([heavy, (normal)])
    # run ICA randomly
    Data_list=[]
    for seed in range(total_seed):
        np.random.default_rng()
        w0=None if W0 is None else W0[seed].reshape(2,2)
        W=full_batch_ICA(A, w0=w0, max_iter=max_iter, lr=lr, lam=1)
        Data_list.extend(process(W,step=step))

    time=[[[0,],[lr]]]*len(Data_list)
    args=[[[1.],[1.]]]*len(Data_list)
    data = {
        'x': Data_list,
        't': time,
        'args': args
    }
    features = Features(
        {
        "t": Array2D(shape=(2, 1), dtype="float32"),
        "x": Array2D(shape=(2, 4), dtype="float32"),
        "args": Array2D(shape=(2, 1), dtype="float32"),
        }
        )
    dataset = Dataset.from_dict(data, features=features)
    return dataset 


### Entropy Production Calculation for ICA ###
import jax.numpy as jnp
import jax.random as jr
from examples.utils.sde import SDEIntegrator

def entropy_production_sgld(net, temperature=1.0, end_time=2.0, dt=0.005):

    init_unit = jr.uniform(
        key=jr.PRNGKey(123), shape=(1000,4)
    )
    init_conditions = init_unit 

    integrator = SDEIntegrator(model=net, state_dim=4)

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
            args=[temperature]
            dvdx = jax.grad(net.potential, argnums=0)(x, args)
            # compution of the time-irreversible drift
            H = net.Hamiltonian(x)
            grad_H = jax.jacfwd(net.Hamiltonian, argnums=0)(x)
            gamma_value = temperature*jnp.einsum('dab,db->a', net.J, grad_H) - jnp.einsum('d,dab,b->a', H, net.J, dvdx)
            return gamma_value
        return gamma

    def ep_term(gamma, dissipation):
        def ep_term_x(x):
            return gamma(x)@(jnp.linalg.inv(dissipation(x))@gamma(x))
        return ep_term_x

    ep_term = ep_term(gamma_net(net), net.dissipation)
    x= predicted_trajectories[:,-200:,].reshape(-1,4)
    ep = jax.vmap(ep_term)(x)
    
    return ep.mean(), predicted_trajectories
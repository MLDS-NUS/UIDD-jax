import os
import jax
import equinox as eqx
import sys
sys.path.append('..')
from tqdm import tqdm
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


def build_OnsagerNetHD2(config: DictConfig, seed:int) -> OnsagerNetHD2:
    """Build the OnsagerNetHD2 model to learn the target dynamics

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNetHD2: OnsagerNetHD2 model
    """

    init_keys = jax.random.PRNGKey(seed)
    v_key, d_key, h_key = jax.random.split(init_keys, 3)

    potential = PotentialResMLP_scaleV2(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
        scale=50 if config.data.batch_size < 50 else config.data.batch_size,  # scale by batch size
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



############### Kaczmarz Method Implementation ############
# Kaczmarz method for solving linear systems
# --------------------------------------------------------

import numpy as np
import scipy.io
from datasets import Dataset, Features, Array2D, load_from_disk
def randomized_kaczmarz(A, b, x0=None, block_size=1, max_iter=1000, lr=0.01):
    """
    Solves Ax = b using the randomized Kaczmarz method with Gaussion noise. 
    Parameters:
        A : ndarray of shape (m, n)
            Coefficient matrix.
        b : ndarray of shape (m,)
            Right-hand side vector.
        x0 : ndarray of shape (n,), optional
            Initial guess for the solution. Defaults to the zero vector.
        max_iter : int
            Maximum number of iterations.

    Returns:
        x : ndarray of shape (n,)
            Final solution after iterations.
        residuals : list of float
            Relative residuals at each iteration.
        X : list of ndarray
            Solution estimates at each iteration.
    """

    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()

    residuals = []
    X=[x]
    for k in range(max_iter):
        i = np.random.choice(m, size=block_size, replace=True) 
        ai = A[i]
        bi = b[i]
        # update with projection
        delta=0 
        delta1 = ((bi - ai.dot(x)))[:, None] * ai 
        delta = delta1.mean(axis=0) 
        x = x + lr * delta + np.random.randn(n) * (0.1 * lr) ** 0.5

        # residual monitoring
        rel_res = np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)
        residuals.append(rel_res)
        X.append(x)


    return x, residuals, X

def process(X, step=50):
    X_np=np.array(X)/5-1
    X0=X_np[:-1:step, :]
    X1=X_np[1::step, :]
    return np.stack([X0,X1],axis=-2).tolist()

def def_lstsq_1():
    mat_data = scipy.io.loadmat('OriginalData/relat4.mat')
    sparse_matrix = mat_data['Problem'][0,0][2]
    A = sparse_matrix.toarray()
    A = A + np.random.randn(A.shape[0], A.shape[1])

    # right hand side b is constructed as a random vector
    x = np.random.rand(A.shape[1])*6+2
    b1 = np.random.randn(A.shape[0])*2
    b = A @ x + b1
    x_ls, residuals_ls, _, _ = np.linalg.lstsq(A, b, rcond=None)
    print("Least Squares Solution:", x_ls, np.linalg.norm(A.dot(x_ls) - b) / np.linalg.norm(b))
    return A, b

def generate_data(total_seed = 2000, batch_size=1, test=False, max_iter=1000, lr=0.01):
    np.random.seed(0)
    # construct target problem Ax = b
    A, b = def_lstsq_1()

    # run Kaczmarz
    Data_list=[]
    for s in tqdm(range(total_seed), miniters=10000, desc="Generating data", mininterval=60.0, maxinterval=600.0):
        x0 = np.random.normal(loc=5, scale=3, size=A.shape[1])
        x_est, residuals, X = randomized_kaczmarz(A, b, x0=x0, block_size=batch_size, max_iter=max_iter, lr=lr)
        Data_list.extend(process(X))

    time=[[[0,],[0.01]]]*len(Data_list)
    args=[[[1.],[1.]]]*len(Data_list)
    data = {
        'x': Data_list,
        't': time,
        'args': args
    }
    features = Features(
        {
        "t": Array2D(shape=(2, 1), dtype="float32"),
        "x": Array2D(shape=(2, 12), dtype="float32"),
        "args": Array2D(shape=(2, 1), dtype="float32"),
        }
        )
    dataset = Dataset.from_dict(data, features=features)
    return dataset


def determitic_kaczmarz(A, b, x0=None, block_size=1, max_iter=1000, lr=0.01):
    """
    Solves Ax = b using the full-batch Kaczmarz method with Gaussion noise. 

    Parameters:
        A : ndarray of shape (m, n)
            Coefficient matrix.
        b : ndarray of shape (m,)
            Right-hand side vector.
        x0 : ndarray of shape (n,), optional
            Initial guess for the solution. Defaults to the zero vector.
        max_iter : int
            Maximum number of iterations.

    Returns:
        x : ndarray of shape (n,)
            Final solution after iterations.
        residuals : list of float
            Relative residuals at each iteration.
        X : list of ndarray
            Solution estimates at each iteration.
    """
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()

    residuals = []
    X=[x]
    for k in range(max_iter):
        # update with projection
        delta=0
        delta1 = ((b - A.dot(x)))[:, None] * A
        delta = delta1.mean(axis=0) 
        x = x + lr * delta + np.random.randn(n) * (0.1 * lr) ** 0.5

        # residual monitoring
        rel_res = np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)
        residuals.append(rel_res)
        X.append(x)


    return x, residuals, X


### Entropy Production Calculation for Kaczmarz ###
import jax.numpy as jnp
import jax.random as jr
from examples.utils.sde import SDEIntegrator

def entropy_production_Kaczmarz(net, temperature=1.0, end_time=2.0, dt=0.0005):

    init_unit = jr.uniform(
        key=jr.PRNGKey(123), shape=(1000,12)
    )
    init_conditions = init_unit * 2 - 1

    integrator = SDEIntegrator(model=net, state_dim=12)

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
    x= predicted_trajectories[:,-200:,].reshape(-1,12)
    ep = jax.vmap(ep_term)(x)
    
    return ep.mean(), predicted_trajectories
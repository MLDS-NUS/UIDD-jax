# Identifiable learning of dissipative dynamics

This repository contains an implementation of UIDD, a deep learning framework for learning not only stable and interpretable, but also **Identifiable** dissipative dynamics from data. 

## Requirements
- Python 3.8+
- JAX and related libraries
- Equinox for neural network modules
- Optax for optimization
- Scikit-learn for PCA
- Hugging Face datasets for data handling

or just copy our environment using conda:
```
conda env create -f environment.yml
```

## Key Features
 
- **Identifiable Learning**: key components are **Identifiable**
- **Stable Dynamics**: Guarantees stability through symmetric positive semi-definite dissipation and anti-symmetric conservation matrices
- **Flexible Components**: Modular design with customizable potential, Hamiltonian(conservation), and diffusion/dissipation functions
- **Invariant Distributaion**: The learned potential corresponds to the stationary density of the system
- **Entropy Production Rate**: The drift admits an equivalent decomposition into time-reversible and time-irreversible components. This property directly enables computation of the EPR for the learned system

## Code of UIDD

The dynamics of UIDD is of the form:

$ dZ_t = - \left[ [M(Z_t)+W(Z_t)] \nabla V(Z_t) + \nabla\cdot M(Z_t) + \nabla\cdot W(Z_t) \right]dt + \sigma(Z_t) dW(t).$
 
We provide two versions of the UIDD implementation:
| Version | Path | Input Matrix | Computation |
|--------|------|---------------|-------------|
| **1** | `onsagernet/dynamics.py/OnsagerNetHD` | $W$, $V$, Dissipation matrix $M $ | Compute $\sigma = \sqrt{2M}$ |
| **2** | `onsagernet/dynamics.py/OnsagerNetHD2` |  $W$, $V$, Diffusion matrix $\sigma$ | Compute $M = \sigma \sigma^\top/2$ |

> Version 1 takes the **dissipation matrix** as input and computes its square root (Cholesky decomposition) to obtain the diffusion matrix.  
> Version 2 takes the **diffusion matrix** as input and computes the dissipation matrix by multiplying it with its transpose.

**Note:** The default version is **Version 2**.
 
 

## Main Applications

### Polymer Dynamics Modeling
The primary application demonstrates learning macroscopic dynamics of a 300-bead linear polymer (900 spatial dimensions) from microscopic trajectory observations, reducing to 3D macroscopic coordinates:

1. **Chain Extension**: Primary macroscopic coordinate representing polymer stretching
2. **Energy Landscape**: super-linear scaling of barrier heights with the strain rate
3. **Entropy Production Rate**: sub-linear scaling of entropy production rates with the strain rate

### Stochastic gradient langevin dynamics.  
The primary application demonstrates how mini-batching introduces state-dependent noise that breaks detailed balance and drives the system out of equilibrium. 

1. **Entropy Production Rate** the EPR decreases with the batch size before plateauing, indicating a suppression of non-equilibrium behavior as stochasticity is reduced
2. **Diagnostic for the Sampling Quality** sampling error follows a consistent trend of EPR

### General Dynamical Systems
The framework can be applied to other complex dynamical systems:
- Linear system 
- General stochastic differential equations (SDEs)

## How to Use
**NOTE: We recommend customizing the GPU memory settings and output directories to match your environment.**


### Running the Polymer Dynamics Example
1. **Training**:
   ```bash
   cd examples/polymer_dynamics_HD
   Download.ipynb #download data
   sh train.sh # train models
   ```

2. **Configuration**:
   - Modify parameters in `config/main_reduced_polymer.yaml`
   - Adjust network architectures, training epochs, learning rates, etc.


3. **Analysis**
   - Results are saved in timestamped directories under `outputs/`.
   - Use `plot_traj.ipynb` to visualize the learned dynamics.
   - Use `plot_var.ipynb` to compute variations in the learned energy functions.
   - Use `plot_potential.ipynb` to verify the learned potential landscape.
   - Use `barrier_height.ipynb` to compute the energy barrier height.
   - Use `results.ipynb` to evaluate the global entropy production rate (EPR).
   - Use `plot_localEPR.ipynb` to analyze the local EPR distribution.
   - Use `polymer_exp_val/EPR_30V.ipynb` and `polymer_exp_val/EPR_60V.ipynb` to validate results against experimental data.


4.  Dimensionality Reduction (This step is separate from network training, and the Huggingface data has already undergone dimensionality reduction)
    - **PCA-ResNet**: Combines PCA with residual neural networks for effective model reduction
    - **Closure Modeling**: Learns mappings between microscopic and macroscopic descriptions
    - **Multi-scale Modeling**: Handles systems with high-dimensional microscopic states (e.g., 900 dimensions) and low-dimensional macroscopic dynamics (e.g., 3 dimensions)



### Running the SGLD Example
1. **Training**:
   ```bash
   cd examples/SGLD-1Kaczmarz # the first linear SGLD example
   sh train.sh # train models
   ```
   ```bash
   cd examples/SGLD-2ICA # the second non-linear SGLD example
   sh train.sh # train models
   ```
2. **Configuration**:
   - Modify parameters in `config/main.yaml`
   - Adjust network architectures, training epochs, learning rates, etc.

3. **Analysis** 
the results are found in the notebook:
    - `SGLD-1Kaczmarz/results.ipynb` 
    - `SGLD-2ICA/results.ipynb`


### Custom Applications
1. **Define Components**:
   - Create custom potential, dissipation, conservation, and diffusion functions
   - Use provided base classes in `onsagernet/models.py`

2. **Set Up Dynamics**:
   - Assemble components using `OnsagerNetHD` or `OnsagerNetHD2` class in `onsagernet/dynamics.py`
   - For reduced dynamics, use `ReducedSDE` with appropriate encoders/decoders

3. **Training**:
   - Use trainers from `onsagernet/trainers.py`
   - Implement custom loss functions if needed


### Model Flexibility
- Supports various activation functions (tanh, srequ, etc.)
- Configurable network architectures
- Extensible to different physical systems

## Repository Structure
```
├── onsagernet/              # Core framework modules
│   ├── dynamics.py          # SDE models and UIDD
│   ├── models.py            # Neural network components
│   ├── transformations.py   # Dimensionality reduction tools
│   ├── trainers.py          # Training routines
│   └── ...                  # Auxiliary modules
├── examples/                # Example applications
    ├── linear_case/         # Linear test example
    ├── polymer_dynamics_HD/ # Main polymer dynamics example
    ├── SGLD-1Kaczmarz/      # First linear SGLD (Kaczmarz) example
    ├── SGLD-2ICA/           # Second non-linear SGLD (ICA) example
    └── ...                  # Other examples
```

## References
- https://github.com/MLDS-NUS/onsagernet-jax
- Aiqing Zhu, Beatrice W. Soh, Grigorios A. Pavliotis, and Qianxiao Li, Identifiable learning of dissipative dynamics



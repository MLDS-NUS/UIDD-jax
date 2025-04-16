# onsagernet-jax-hhd
OnsagerNet via Helmholtz-Hodge decomposition

---

In the `onsagernet/new_dynamics.py` module, two variants of OnsagerNet are implemented using the Helmholtz–Hodge decomposition:

- **OnsagerNetHD**: This model parameterizes the diffusion matrix $M$ directly and computes its square root $\sqrt{M}$.

- **OnsagerNetHD2**: This model parameterizes the square root of the diffusion matrix $\sqrt{M}$ and subsequently computes $M$.

In practice we use **OnsagerNetHD2** with scaled potential function **PotentialResMLP_scale**, which can be found in ``onsagernet/new_models.py`

- **PotentialResMLP_scale** : comparing with previous version of **PotentialResMLP**, a scale parameter was included and multiplied with the output.

The function to build the network can be found in `examples/utils_reduced_polymer.py`; it's named **build_OnsagerNetHD2_scale**".

---




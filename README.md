# onsagernet-jax-hhd
OnsagerNet via Helmholtz-Hodge decomposition

---

In the `onsagernet/new_dynamics.py` module, two variants of OnsagerNet are implemented using the Helmholtz–Hodge decomposition:

- **OnsagerNetHD**: This model parameterizes the diffusion matrix $M$ directly and computes its square root $\sqrt{M}$.

- **OnsagerNetHD2**: This model parameterizes the square root of the diffusion matrix $\sqrt{M}$ and subsequently computes $M$.

In practice we use **OnsagerNetHD2** with scaled potential function (In the previous version of **PotentialResMLP**, a scale parameter was included and multiplied with the output)

---




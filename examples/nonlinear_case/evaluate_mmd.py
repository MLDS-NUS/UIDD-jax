"""Evaluate trained nonlinear-case models via MMD vs time.

For every (seed, model) pair, loads `outputs/nonlinear<seed>_<Model>_dt<dt>/<ckpt>`,
simulates trajectories from both the trained model and the exact target SDE
on freshly drawn (independent) initial conditions, and computes the
time-evolution of the Gaussian-kernel MMD^2 against the target ensemble.

Each ensemble is integrated with an inner solver step `dt / solve_substeps`
(default substeps=10) but states are saved every `dt`. ICs and Brownian
motions are re-sampled for every (seed, model) evaluation.

Within each model class, the per-seed curves are aggregated into mean +- std,
plotted as one band per model on a single summary figure. A target-vs-target
curve from two independent target draws is included as a finite-sample noise
floor reference.

Usage:
    python evaluate_mmd.py                       # seeds=[0,1,12,123,1234], dt=0.005, kappa=0.8
    python evaluate_mmd.py --seeds 1 12 123      # custom seeds
    python evaluate_mmd.py --dt 0.005 --kappa 0.8 --num-trajs 1500 --solve-substeps 20
"""

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.18")

import argparse
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from omegaconf import OmegaConf

sys.path.append("..")
sys.path.append("../..")

from utils.sde import SDEIntegrator
from nonlinear_case import (
    build_targets,
    build_OnsagerNet,
    build_OnsagerNetReg,
    build_UIDD2,
    build_SDEfromFunc,
)


# ------------------------------------------------------------------ #
#                              MMD                                   #
# ------------------------------------------------------------------ #


def _sq_pdist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Squared pairwise Euclidean distances. x:[n,d], y:[m,d] -> [n,m]."""
    xx = jnp.sum(x ** 2, axis=-1, keepdims=True)
    yy = jnp.sum(y ** 2, axis=-1, keepdims=True)
    return jnp.maximum(xx + yy.T - 2.0 * x @ y.T, 0.0)


def median_heuristic_bandwidth(samples: jnp.ndarray) -> jnp.ndarray:
    """Pick sigma^2 as the median of squared pairwise distances on `samples`."""
    n = samples.shape[0]
    d2 = _sq_pdist(samples, samples)
    iu = jnp.triu_indices(n, k=1)
    med = jnp.median(d2[iu])
    return jnp.where(med > 0, med, 1.0)


def gaussian_mmd2(x: jnp.ndarray, y: jnp.ndarray, bandwidth: jnp.ndarray) -> jnp.ndarray:
    """Standard (biased) Gaussian MMD^2: mean(Kxx) + mean(Kyy) - 2*mean(Kxy).

    Always >= 0 (clamped to 0 to absorb tiny FP underflow).
    """
    Kxx = jnp.exp(-_sq_pdist(x, x) / bandwidth)
    Kyy = jnp.exp(-_sq_pdist(y, y) / bandwidth)
    Kxy = jnp.exp(-_sq_pdist(x, y) / bandwidth)
    return jnp.maximum(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean(), 0.0)


# ------------------------------------------------------------------ #
#                  Structural errors (M, W, grad V, drift)           #
# ------------------------------------------------------------------ #


def _grad_V(potential_fn, x, args):
    return jax.grad(potential_fn, argnums=0)(x, args)


def _W_gradV_target(target, x, args):
    """Target's W(x) @ ∇V(x) vector."""
    return target.conservation(x) @ _grad_V(target.potential, x, args)


def _W_gradV_onsager(model, x, args):
    return model.conservation(x) @ _grad_V(model.potential, x, args)


def _W_gradV_uidd2(model, x, args):
    """UIDD2's "W·∇V" vector = einsum('d,dab,b->a', H(x), J, ∇V(x))."""
    H = model.Hamiltonian(x)
    dvdx = _grad_V(model.potential, x, args)
    return jnp.einsum("d,dab,b->a", H, model.J, dvdx)


def _rel_err_from_vecs(v_m: jnp.ndarray, v_t: jnp.ndarray) -> jnp.ndarray:
    """sqrt( sum ||v_m - v_t||^2 / sum ||v_t||^2 ) over the leading sample axis."""
    num = jnp.sum((v_m - v_t) ** 2)
    den = jnp.sum(v_t ** 2)
    return jnp.sqrt(num / jnp.maximum(den, 1e-30))


def _drift_relerr(model, target, x_samples, args, t):
    f_m = jax.vmap(lambda x: model.drift(t, x, args))(x_samples)
    f_t = jax.vmap(lambda x: target.drift(t, x, args))(x_samples)
    return _rel_err_from_vecs(f_m, f_t)


def _gradV_relerr(model_pot, target_pot, x_samples, args):
    g_m = jax.vmap(lambda x: _grad_V(model_pot, x, args))(x_samples)
    g_t = jax.vmap(lambda x: _grad_V(target_pot, x, args))(x_samples)
    return _rel_err_from_vecs(g_m, g_t)


def _MgradV_relerr(model_M_fn, target_M_fn,
                   model_pot, target_pot, x_samples, args):
    """Relative error on the M·∇V vector."""
    g_m = jax.vmap(lambda x: _grad_V(model_pot, x, args))(x_samples)
    g_t = jax.vmap(lambda x: _grad_V(target_pot, x, args))(x_samples)
    M_m = jax.vmap(model_M_fn)(x_samples)
    M_t = jax.vmap(target_M_fn)(x_samples)
    v_m = jnp.einsum("nij,nj->ni", M_m, g_m)
    v_t = jnp.einsum("nij,nj->ni", M_t, g_t)
    return _rel_err_from_vecs(v_m, v_t)


def _WgradV_relerr(model_Wfn, target_Wfn, x_samples, args):
    v_m = jax.vmap(lambda x: model_Wfn(x, args))(x_samples)
    v_t = jax.vmap(lambda x: target_Wfn(x, args))(x_samples)
    return _rel_err_from_vecs(v_m, v_t)


def compute_structural_errors(model, model_name: str, target,
                              x_samples: jnp.ndarray, args) -> dict:
    """Relative error of drift / ∇V / M·∇V / W·∇V vs target.

    SDE only has the drift entry; M·∇V and W·∇V are skipped.
    Each entry is e_f = sqrt( Σ_x ||f_θ(x) - f(x)||^2 / Σ_x ||f(x)||^2 ).
    """
    t0 = jnp.asarray(0.0, dtype=x_samples.dtype)
    out = {"drift": _drift_relerr(model, target, x_samples, args, t0)}
    if model_name == "SDE":
        return out
    out["gradV"] = _gradV_relerr(model.potential, target.potential, x_samples, args)
    out["MgradV"] = _MgradV_relerr(
        model.dissipation, target.dissipation,
        model.potential, target.potential, x_samples, args,
    )
    if hasattr(model, "Hamiltonian"):  # HD2 / UIDD2
        Wfn = lambda x, a: _W_gradV_uidd2(model, x, a)
    else:  # Onsager / OnsagerReg* (use conservation matrix)
        Wfn = lambda x, a: _W_gradV_onsager(model, x, a)
    out["WgradV"] = _WgradV_relerr(
        Wfn, lambda x, a: _W_gradV_target(target, x, a), x_samples, args,
    )
    return out


# ------------------------------------------------------------------ #
#                       Model build / load                           #
# ------------------------------------------------------------------ #


def build_model(config, model_name: str):
    if model_name == "Onsager":
        return build_OnsagerNet(config)
    if model_name.startswith("OnsagerReg"):
        # Any tag with the OnsagerReg* prefix (e.g. OnsagerRegL2, OnsagerRegL2V2,
        # OnsagerRegSobolevV2) shares the same dynamic structure; reg fields are
        # static so any weight choice yields a deserialization-compatible template.
        return build_OnsagerNetReg(config)
    if model_name == "HD2":
        return build_UIDD2(config)
    if model_name == "SDE":
        return build_SDEfromFunc(config)
    raise ValueError(f"Unknown Model_name '{model_name}'")


def parse_dir_name(output_dir: str) -> Tuple[int, str, float]:
    """Parse 'nonlinear<seed>_<Model_name>_dt<dt>'."""
    base = os.path.basename(os.path.normpath(output_dir))
    parts = base.split("_")
    if len(parts) < 3 or not parts[0].startswith("nonlinear") or not parts[2].startswith("dt"):
        raise ValueError(
            f"Cannot parse output dir name '{base}'. "
            "Expected 'nonlinear<seed>_<Model_name>_dt<dt>'."
        )
    seed = int(parts[0][len("nonlinear"):])
    model_name = parts[1]
    dt = float(parts[2][len("dt"):])
    return seed, model_name, dt


def load_trained_model(output_dir: str, config_path: str, checkpoint: str):
    config = OmegaConf.load(config_path)
    seed, model_name, dt = parse_dir_name(output_dir)
    config.model.seed = seed
    config.Model_name = model_name
    config.dt = dt

    template = build_model(config, model_name)
    ckpt = os.path.join(output_dir, checkpoint)
    model = eqx.tree_deserialise_leaves(ckpt, template)
    return model, config, model_name


# ------------------------------------------------------------------ #
#                       Trajectory sampling                          #
# ------------------------------------------------------------------ #


def simulate(model, config, inits: jnp.ndarray, key: jnp.ndarray,
             solve_substeps: int = 10):
    """Integrate `model` with inner step config.dt / solve_substeps and subsample
    every `solve_substeps`-th saved point to recover a record interval of config.dt.
    Returns (ts[T], ys[n, T, dim])."""
    integrator = SDEIntegrator(model=model, state_dim=int(config.dim))
    keys = jr.split(key, inits.shape[0])
    sol = integrator.parallel_solve(
        initial_conditions=inits,
        key=keys,
        t0=float(config.data.t0),
        t1=float(config.data.t1)*4,
        dt=float(config.dt) / solve_substeps,
        args=[config.temperature],
    )
    return sol.ts[0, ::solve_substeps], sol.ys[:, ::solve_substeps]


# ------------------------------------------------------------------ #
#                              Main                                  #
# ------------------------------------------------------------------ #


DEFAULT_SEEDS = [0, 1, 12, 123, 1234]
DEFAULT_MODELS = ["HD2", "Onsager", "SDE"]
MODEL_COLORS = {
    "HD2":               "tab:blue",
    "Onsager":           "tab:orange",
    "SDE":               "tab:green",
    "OnsagerReg":        "tab:red",
    "OnsagerRegL2":      "tab:purple",
    "OnsagerRegSobolev": "tab:brown",
}
DEFAULT_INIT_MIN = [-2.0, -1.0]
DEFAULT_INIT_MAX = [2.0, 1.0]


def _sample_inits(key: jnp.ndarray, n: int, dim: int,
                  init_min: jnp.ndarray, init_max: jnp.ndarray) -> jnp.ndarray:
    return jr.uniform(key, shape=(n, dim), minval=init_min, maxval=init_max)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"model.seed values to evaluate (default: {DEFAULT_SEEDS}).")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help=(f"Model_name tags to evaluate; the dir prefix used for lookup. "
                              f"Default: {DEFAULT_MODELS}. Use e.g. "
                              "'OnsagerRegL2 OnsagerRegSobolev' to evaluate the regularized runs."))
    parser.add_argument("--dt", type=float, default=None,
                        help="Save dt; also the dir-name tag. Defaults to config.dt.")
    parser.add_argument("--kappa", type=float, default=None,
                        help="kappa (data.var) of the target SDE. Defaults to config.data.var.")
    parser.add_argument("--outputs-root", default="outputs",
                        help="Root dir holding nonlinear<seed>_<Model>_dt<dt> folders.")
    parser.add_argument("--config", default="config/nonlinear_case.yaml")
    parser.add_argument("--checkpoint", default="model.eqx",
                        help="Checkpoint file inside each output dir.")
    parser.add_argument("--num-trajs", type=int, default=1000,
                        help="Number of trajectories sampled per ensemble.")
    parser.add_argument("--solve-substeps", type=int, default=10,
                        help="Inner solver step = config.dt / solve_substeps; states saved every config.dt.")
    parser.add_argument("--init-min", type=float, nargs="+", default=DEFAULT_INIT_MIN,
                        help="Lower corner of the IC sampling box (default: [-2, -1]).")
    parser.add_argument("--init-max", type=float, nargs="+", default=DEFAULT_INIT_MAX,
                        help="Upper corner of the IC sampling box (default: [2, 1]).")
    parser.add_argument("--eval-seed", type=int, default=20260430,
                        help="Master random seed for ICs and Brownian motions.")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to write npz/png. Defaults to <outputs-root>.")
    parser.add_argument("--out-name", default=None,
                        help="Base name for npz/png. Default: 'mmd_summary_dt<dt>'.")
    args = parser.parse_args()

    save_dir = args.save_dir or args.outputs_root
    os.makedirs(save_dir, exist_ok=True)

    # ---- Target SDE: shared across all (seed, model) evaluations ----
    config0 = OmegaConf.load(args.config)
    if args.dt is not None:
        config0.dt = args.dt
    if args.kappa is not None:
        config0.data.var = args.kappa
    dt_tag = float(config0.dt)
    kappa = float(config0.data.var)
    out_name = args.out_name or f"mmd_summary_dt{dt_tag}"
    target_SDE = build_targets(kappa)

    init_min = jnp.asarray(args.init_min, dtype=jnp.float64)
    init_max = jnp.asarray(args.init_max, dtype=jnp.float64)
    if init_min.shape != (config0.dim,) or init_max.shape != (config0.dim,):
        raise ValueError(f"init_min/init_max must each have {config0.dim} entries.")
    n = args.num_trajs
    dim = int(config0.dim)
    print(f"IC sampling box: min={list(args.init_min)}, max={list(args.init_max)}; "
          f"solve_dt = config.dt / {args.solve_substeps}")

    master_key = jr.PRNGKey(args.eval_seed)

    # ---- Bandwidth: pick once on a fixed reference target draw ----
    bw_key = jr.fold_in(master_key, 0)
    k_inits, k_bm = jr.split(bw_key, 2)
    bw_inits = _sample_inits(k_inits, n, dim, init_min, init_max)
    print(f"Simulating bandwidth-reference target ensemble (kappa={kappa}, n={n}) ...")
    ts, ys_bw = simulate(target_SDE, config0, bw_inits, k_bm,
                         solve_substeps=args.solve_substeps)
    pooled = ys_bw.reshape(-1, dim)
    if pooled.shape[0] > 4000:
        idx = jr.choice(jr.PRNGKey(0), pooled.shape[0], shape=(4000,), replace=False)
        pooled = pooled[idx]
    bandwidth = median_heuristic_bandwidth(pooled)
    print(f"Median-heuristic bandwidth (sigma^2) = {float(bandwidth):.4f}")
    ts_np = np.asarray(ts)

    # ---- Eval sample for structural M / W / grad-V / drift errors ----
    n_struct = min(2000, pooled.shape[0])
    x_eval = pooled[:n_struct]
    args_const = jnp.asarray([float(config0.temperature)], dtype=jnp.float64)
    print(f"Structural-error sample size: {n_struct}")

    @jax.jit
    def mmd_traj(ys_a: jnp.ndarray, ys_b: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda a, b: gaussian_mmd2(a, b, bandwidth),
                        in_axes=(1, 1))(ys_a, ys_b)

    # ---- Noise floor: two independent fresh target draws ----
    floor_key = jr.fold_in(master_key, 1)
    k_a, k_b = jr.split(floor_key, 2)
    inits_a = _sample_inits(jr.fold_in(k_a, 0), n, dim, init_min, init_max)
    inits_b = _sample_inits(jr.fold_in(k_a, 1), n, dim, init_min, init_max)
    print("Simulating noise-floor target draws ...")
    _, ys_a = simulate(target_SDE, config0, inits_a, jr.fold_in(k_b, 0),
                       solve_substeps=args.solve_substeps)
    _, ys_b = simulate(target_SDE, config0, inits_b, jr.fold_in(k_b, 1),
                       solve_substeps=args.solve_substeps)
    mmd_ref = np.asarray(mmd_traj(ys_a, ys_b))

    # ---- Pre-sample per-seed ICs and target trajectories (shared across models) ----
    # All 3 models at the same seed reuse the same inits_tgt/inits_mdl/BMs,
    # so MMD^2(t=0) is identical across HD2/Onsager/SDE per seed.
    seed_inputs = {}
    for seed in args.seeds:
        seed_key = jr.fold_in(jr.fold_in(master_key, 2), seed)
        k_inits_t, k_bm_t, k_inits_m, k_bm_m = jr.split(seed_key, 4)
        inits_tgt = _sample_inits(k_inits_t, n, dim, init_min, init_max)
        inits_mdl = _sample_inits(k_inits_m, n, dim, init_min, init_max)
        print(f"Simulating target SDE for seed={seed} (shared across models) ...")
        _, ys_target_i = simulate(target_SDE, config0, inits_tgt, k_bm_t,
                                  solve_substeps=args.solve_substeps)
        seed_inputs[seed] = (ys_target_i, inits_mdl, k_bm_m)

    # ---- Loop over (model, seed) ----
    per_seed_curves = {m: {} for m in args.models}
    per_seed_metrics = {m: {} for m in args.models}
    for mi, model_name in enumerate(args.models):
        for seed in args.seeds:
            tag = f"nonlinear{seed}_{model_name}_dt{dt_tag}"
            out_dir = os.path.join(args.outputs_root, tag)
            ckpt = os.path.join(out_dir, args.checkpoint)
            if not os.path.isfile(ckpt):
                print(f"[skip] missing {ckpt}")
                continue
            try:
                model, config, _ = load_trained_model(out_dir, args.config, args.checkpoint)
            except Exception as e:
                print(f"[skip] failed loading {out_dir}: {e}")
                continue

            ys_target_i, inits_mdl, k_bm_m = seed_inputs[seed]
            print(f"Simulating {tag} ...")
            _, ys_model = simulate(model, config, inits_mdl, k_bm_m,
                                   solve_substeps=args.solve_substeps)
            ys_model_np = np.asarray(ys_model)
            if not np.all(np.isfinite(ys_model_np)):
                bad_t = int(np.argmax(~np.isfinite(ys_model_np).all(axis=(0, 2))))
                print(f"[skip] {tag}: trajectories diverged (NaN/Inf from t-index {bad_t}).")
                continue
            per_seed_curves[model_name][seed] = np.asarray(mmd_traj(ys_target_i, ys_model))

            # Structural errors (drift, plus gradV/MgradV/WgradV for HD2/Onsager).
            try:
                metrics = compute_structural_errors(
                    model, model_name, target_SDE, x_eval, args_const,
                )
                metrics = {k: float(v) for k, v in metrics.items()}
                per_seed_metrics[model_name][seed] = metrics
            except Exception as e:
                print(f"[warn] {tag}: structural-error computation failed: {e}")

    # ---- Aggregate per model ----
    summary = {}  # model_name -> (mean, std, n_seeds)
    for model_name, by_seed in per_seed_curves.items():
        if not by_seed:
            print(f"[warn] no runs found for {model_name}")
            continue
        stack = np.stack([by_seed[s] for s in sorted(by_seed)], axis=0)  # (n_seeds, T)
        summary[model_name] = (stack.mean(0), stack.std(0), stack.shape[0])

    # ---- Aggregate structural errors per model ----
    metric_keys = ["drift", "gradV", "MgradV", "WgradV"]
    metric_summary = {m: {} for m in args.models}
    for model_name, by_seed in per_seed_metrics.items():
        if not by_seed:
            continue
        for k in metric_keys:
            vals = [by_seed[s][k] for s in by_seed if k in by_seed[s]]
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            metric_summary[model_name][k] = (arr.mean(), arr.std(), len(arr))

    # ---- Save ----
    npz_payload = {"ts": ts_np, "bandwidth": float(bandwidth), "mmd_ref": mmd_ref,
                   "kappa": kappa, "dt": dt_tag}
    for model_name, (mean, std, n) in summary.items():
        npz_payload[f"{model_name}_mean"] = mean
        npz_payload[f"{model_name}_std"] = std
        npz_payload[f"{model_name}_n_seeds"] = n
    for model_name, by_seed in per_seed_curves.items():
        for seed, mmd in by_seed.items():
            npz_payload[f"{model_name}_seed{seed}"] = mmd
    for model_name in args.models:
        for k, (mean, std, n_) in metric_summary[model_name].items():
            npz_payload[f"err_{model_name}_{k}_mean"] = mean
            npz_payload[f"err_{model_name}_{k}_std"] = std
            npz_payload[f"err_{model_name}_{k}_n"] = n_
        for seed, m in per_seed_metrics[model_name].items():
            for k, v in m.items():
                npz_payload[f"err_{model_name}_seed{seed}_{k}"] = v
    npz_path = os.path.join(save_dir, f"{out_name}.npz")
    np.savez(npz_path, **npz_payload)
    print(f"Saved MMD traces to {npz_path}")

    # ---- Print summary table ----
    print("\n=== Relative error summary (mean ± std across seeds) ===")
    metric_titles = {"drift": "drift rel err", "gradV": "∇V rel err",
                     "MgradV": "M·∇V rel err", "WgradV": "W·∇V rel err"}
    print(f"{'Model':<10}" + "".join(f"{metric_titles[k]:>22}" for k in metric_keys))
    for model_name in args.models:
        line = f"{model_name:<10}"
        for k in metric_keys:
            if k in metric_summary[model_name]:
                mean, std, _ = metric_summary[model_name][k]
                cell = f"{mean:.3e} ± {std:.1e}"
            else:
                cell = "-"
            line += f"{cell:>22}"
        print(line)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, yscale in zip(axes, ["linear", "log"]):
        ref_to_plot = mmd_ref if yscale == "linear" else np.where(mmd_ref > 0, mmd_ref, np.nan)
        ax.plot(ts_np, ref_to_plot, ls="--", color="k", lw=1.5,
                label="target vs target (noise floor)")
        for model_name, by_seed in per_seed_curves.items():
            if model_name not in summary:
                continue
            color = MODEL_COLORS.get(model_name, None)
            mean, std, n = summary[model_name]
            for seed in sorted(by_seed):
                curve = by_seed[seed]
                if yscale == "log":
                    curve = np.where(curve > 0, curve, np.nan)
                ax.plot(ts_np, curve, color=color, lw=0.8, alpha=0.35)
            mean_to_plot = mean if yscale == "linear" else np.where(mean > 0, mean, np.nan)
            ax.plot(ts_np, mean_to_plot, color=color, lw=2.2,
                    label=f"{model_name} mean ({n} seeds)")
            lower = mean - std
            upper = mean + std
            if yscale == "log":
                mask = upper > 0
                lower = np.where(mask, np.maximum(lower, 1e-15), np.nan)
                upper = np.where(mask, upper, np.nan)
            ax.fill_between(ts_np, lower, upper, color=color, alpha=0.18)
        ax.set_xlabel("t")
        ax.set_ylabel(r"MMD$^2$ (Gaussian kernel)")
        ax.set_yscale(yscale)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"y-scale: {yscale}")
    axes[0].legend(loc="best", fontsize=9)
    seeds_str = ",".join(str(s) for s in args.seeds)
    fig.suptitle(
        f"MMD$^2$ vs time -- dt={dt_tag}, kappa={kappa}, "
        f"seeds=[{seeds_str}], n_traj={args.num_trajs}",
        y=1.02,
    )
    fig.tight_layout()
    png_path = os.path.join(save_dir, f"{out_name}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {png_path}")

    # ---- Structural-error bar chart ----
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    for ax, k in zip(axes2, metric_keys):
        means, stds, labels, colors = [], [], [], []
        for m in args.models:
            if k in metric_summary[m]:
                mean, std, _ = metric_summary[m][k]
                means.append(mean)
                stds.append(std)
                labels.append(m)
                colors.append(MODEL_COLORS.get(m, "gray"))
        if not means:
            ax.set_axis_off()
            ax.set_title(metric_titles[k] + " (no data)")
            continue
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_title(metric_titles[k])
        ax.grid(True, which="both", alpha=0.3, axis="y")
    fig2.suptitle(
        f"Relative errors $e_f=\\sqrt{{\\sum\\|f_\\theta-f\\|^2/\\sum\\|f\\|^2}}$ "
        f"-- dt={dt_tag}, kappa={kappa}, n_eval={n_struct}",
        y=1.02,
    )
    fig2.tight_layout()
    metrics_png = os.path.join(save_dir, f"{out_name}_metrics.png")
    fig2.savefig(metrics_png, dpi=150, bbox_inches="tight")
    print(f"Saved structural-error plot to {metrics_png}")


if __name__ == "__main__":
    main()

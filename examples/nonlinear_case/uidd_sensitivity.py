"""Run and evaluate a UIDD hyperparameter sensitivity sweep.

This script deliberately lives next to nonlinear_case.py and uses Hydra
overrides instead of editing the original training code.

Default design:
  - one-factor-at-a-time variations around the existing HD2/UIDD baseline
  - 4 GPUs, up to 2 Python training processes per GPU
  - structural errors are evaluated with evaluate_mmd.compute_structural_errors

Typical usage:
  python uidd_sensitivity.py plan
  python uidd_sensitivity.py train --gpus 0 1 2 3 --max-per-gpu 2
  python uidd_sensitivity.py evaluate
  python uidd_sensitivity.py all --gpus 0 1 2 3 --max-per-gpu 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "uidd_sensitivity"
DEFAULT_LOG_ROOT = SCRIPT_DIR / "running_log" / "uidd_sensitivity"
DEFAULT_SEEDS = [0, 1, 12, 123, 1234]
DEFAULT_GPUS = [0, 1, 2, 3]
DEFAULT_DT = 0.01
TRAIN_TRAJECTORIES = 9000
TRAIN_T1 = 0.5
TRAIN_TRAJ_LEN = 2


def cuda_library_dirs() -> list[str]:
    """CUDA wheel library directories needed by jax[cuda12] in non-login shells."""
    roots = glob.glob(str(Path(sys.prefix) / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib"))
    roots.extend(glob.glob(str(Path(sys.prefix) / "lib" / "site-packages" / "nvidia" / "*" / "lib")))
    roots.append("/usr/local/cuda-12.8/targets/x86_64-linux/lib")
    seen = set()
    out = []
    for root in roots:
        if root not in seen and Path(root).is_dir():
            out.append(root)
            seen.add(root)
    return out


def add_cuda_library_path(env: dict[str, str]) -> dict[str, str]:
    libs = cuda_library_dirs()
    if not libs:
        return env
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(libs + ([existing] if existing else []))
    return env


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    factor: str
    level: str
    potential_units: tuple[int, ...]
    hamiltonian_units: tuple[int, ...]
    learning_rate: float
    batch_size: int
    num_epochs: int


@dataclass(frozen=True)
class EvalSpec:
    run_id: str
    factor: str
    level: str
    train_run_id: str
    checkpoint: str
    potential_units: tuple[int, ...]
    hamiltonian_units: tuple[int, ...]
    learning_rate: float
    batch_size: int
    num_epochs: int
    log_epoch: int | None = None


BASE_5000_SPEC = RunSpec(
    run_id="epochs_5000",
    factor="num_epochs",
    level="5000",
    potential_units=(64, 32),
    hamiltonian_units=(32, 16),
    learning_rate=1e-3,
    batch_size=100000,
    num_epochs=5000,
)


def effective_examples(dt: float) -> int:
    """Examples after shrink_trajectory_len for the default nonlinear dataset."""
    saved_points = int(round(TRAIN_T1 / dt)) + 1
    return TRAIN_TRAJECTORIES * (saved_points // TRAIN_TRAJ_LEN)


def estimated_cost(spec: RunSpec, dt: float) -> float:
    """Rough scheduling weight in update steps, with a small net-size factor."""
    steps_per_epoch = max(1, math.ceil(effective_examples(dt) / spec.batch_size))
    width_factor = (
        sum(spec.potential_units) / (64 + 32)
        + sum(spec.hamiltonian_units) / (32 + 16)
    ) / 2.0
    return spec.num_epochs * steps_per_epoch * max(width_factor, 0.5)


def make_train_plan() -> list[RunSpec]:
    """Actual training jobs. Epoch sensitivity is trained once at 5000 epochs."""
    base_3000 = replace(BASE_5000_SPEC, num_epochs=3000)
    return [
        BASE_5000_SPEC,
        replace(
            base_3000,
            run_id="net_small",
            factor="network",
            level="small",
            potential_units=(32, 16),
            hamiltonian_units=(16, 8),
        ),
        replace(
            base_3000,
            run_id="net_wide",
            factor="network",
            level="wide",
            potential_units=(128, 64),
            hamiltonian_units=(64, 32),
        ),
        replace(
            base_3000,
            run_id="net_deep",
            factor="network",
            level="deep",
            potential_units=(64, 64, 32),
            hamiltonian_units=(32, 32, 16),
        ),
        replace(base_3000, run_id="lr_5e-4", factor="learning_rate", level="5e-4", learning_rate=5e-4),
        replace(base_3000, run_id="lr_2e-3", factor="learning_rate", level="2e-3", learning_rate=2e-3),
        replace(base_3000, run_id="batch_25000", factor="batch_size", level="25000", batch_size=25000),
        replace(base_3000, run_id="batch_50000", factor="batch_size", level="50000", batch_size=50000),
        replace(base_3000, run_id="batch_200000", factor="batch_size", level="200000", batch_size=200000),
    ]


def _eval_spec(
    *,
    run_id: str,
    factor: str,
    level: str,
    train_spec: RunSpec,
    checkpoint: str,
    num_epochs: int,
    log_epoch: int | None = None,
) -> EvalSpec:
    return EvalSpec(
        run_id=run_id,
        factor=factor,
        level=level,
        train_run_id=train_spec.run_id,
        checkpoint=checkpoint,
        potential_units=train_spec.potential_units,
        hamiltonian_units=train_spec.hamiltonian_units,
        learning_rate=train_spec.learning_rate,
        batch_size=train_spec.batch_size,
        num_epochs=num_epochs,
        log_epoch=log_epoch,
    )


def make_eval_plan() -> list[EvalSpec]:
    """Evaluation rows. Epoch rows reuse checkpoints from epochs_5000."""
    train_by_id = {spec.run_id: spec for spec in make_train_plan()}
    base = BASE_5000_SPEC
    rows = [
        _eval_spec(
            run_id="baseline",
            factor="baseline",
            level="3000",
            train_spec=base,
            checkpoint="model_epoch_03000.eqx",
            num_epochs=3000,
            log_epoch=3000,
        )
    ]
    for run_id in [
        "net_small",
        "net_wide",
        "net_deep",
        "lr_5e-4",
        "lr_2e-3",
        "batch_25000",
        "batch_50000",
        "batch_200000",
    ]:
        spec = train_by_id[run_id]
        rows.append(
            _eval_spec(
                run_id=spec.run_id,
                factor=spec.factor,
                level=spec.level,
                train_spec=spec,
                checkpoint="model.eqx",
                num_epochs=spec.num_epochs,
            )
        )
    for epoch in [1000, 2000, 3000]:
        rows.append(
            _eval_spec(
                run_id=f"epochs_{epoch}",
                factor="num_epochs",
                level=str(epoch),
                train_spec=base,
                checkpoint=f"model_epoch_{epoch:05d}.eqx",
                num_epochs=epoch,
                log_epoch=epoch,
            )
        )
    rows.append(
        _eval_spec(
            run_id="epochs_5000",
            factor="num_epochs",
            level="5000",
            train_spec=base,
            checkpoint="model.eqx",
            num_epochs=5000,
        )
    )
    return rows


def fmt_units(units: Iterable[int]) -> str:
    return "[" + ",".join(str(u) for u in units) + "]"


def output_dir(output_root: Path, spec: RunSpec, seed: int) -> Path:
    return output_root / spec.run_id / f"seed{seed}"


def model_path(output_root: Path, spec: RunSpec, seed: int) -> Path:
    return output_dir(output_root, spec, seed) / "model.eqx"


def training_complete(output_root: Path, spec: RunSpec, seed: int) -> bool:
    paths = [model_path(output_root, spec, seed)]
    if spec.run_id == "epochs_5000":
        paths.extend(
            output_dir(output_root, spec, seed) / f"model_epoch_{epoch:05d}.eqx"
            for epoch in [1000, 2000, 3000]
        )
    return all(path.is_file() for path in paths)


def eval_checkpoint_path(output_root: Path, spec: EvalSpec, seed: int) -> Path:
    return output_root / spec.train_run_id / f"seed{seed}" / spec.checkpoint


def eval_config_path(output_root: Path, spec: EvalSpec, seed: int) -> Path:
    return output_root / spec.train_run_id / f"seed{seed}" / ".hydra" / "config.yaml"


def hydra_run_dir(output_root: Path, spec: RunSpec, seed: int) -> str:
    try:
        rel = output_dir(output_root, spec, seed).relative_to(SCRIPT_DIR)
        return "./" + rel.as_posix()
    except ValueError:
        return output_dir(output_root, spec, seed).as_posix()


def train_plan_rows(seeds: list[int], dt: float) -> list[dict[str, str]]:
    rows = []
    for spec in make_train_plan():
        rows.append(
            {
                "run_id": spec.run_id,
                "factor": spec.factor,
                "level": spec.level,
                "V_units": fmt_units(spec.potential_units),
                "H_units": fmt_units(spec.hamiltonian_units),
                "lr": f"{spec.learning_rate:g}",
                "batch_size": str(spec.batch_size),
                "epochs": str(spec.num_epochs),
                "jobs": str(len(seeds)),
                "cost": f"{estimated_cost(spec, dt):.0f}",
            }
        )
    return rows


def eval_plan_rows(seeds: list[int]) -> list[dict[str, str]]:
    rows = []
    for spec in make_eval_plan():
        rows.append(
            {
                "run_id": spec.run_id,
                "factor": spec.factor,
                "level": spec.level,
                "source_run": spec.train_run_id,
                "checkpoint": spec.checkpoint,
                "epochs": str(spec.num_epochs),
                "rows": str(len(seeds)),
            }
        )
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    headers = list(rows[0])
    widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))


def train_command(
    spec: RunSpec,
    seed: int,
    *,
    output_root: Path,
    dt: float,
    kappa: float,
    print_every: int,
    checkpoint_every: int,
) -> list[str]:
    return [
        sys.executable,
        "nonlinear_case.py",
        "Model_name=HD2",
        f"model.seed={seed}",
        f"data.var={kappa:g}",
        f"dt={dt:g}",
        f"model.potential.units={fmt_units(spec.potential_units)}",
        f"model.hamiltonian.units={fmt_units(spec.hamiltonian_units)}",
        f"train.opt.learning_rate={spec.learning_rate:g}",
        f"train.batch_size={spec.batch_size}",
        f"train.num_epochs={spec.num_epochs}",
        f"train.print_every={print_every}",
        f"train.checkpoint_every={checkpoint_every}",
        f"hydra.run.dir={hydra_run_dir(output_root, spec, seed)}",
    ]


@dataclass
class ActiveJob:
    proc: subprocess.Popen
    gpu: int
    spec: RunSpec
    seed: int
    log_path: Path
    started_at: float


def run_train(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    plan = make_train_plan()
    selected = []
    if args.only:
        requested = set(args.only)
        wanted = {
            spec.train_run_id
            for spec in make_eval_plan()
            if spec.run_id in requested or spec.train_run_id in requested
        }
    else:
        wanted = None
    for spec in plan:
        if wanted is not None and spec.run_id not in wanted:
            continue
        for seed in args.seeds:
            if training_complete(output_root, spec, seed) and not args.force:
                print(f"[skip] {spec.run_id} seed={seed}: required checkpoint(s) already exist")
                continue
            selected.append((spec, seed))

    selected.sort(key=lambda item: estimated_cost(item[0], args.dt), reverse=True)
    print(f"Training jobs to run: {len(selected)}")
    if args.dry_run:
        for spec, seed in selected:
            cmd = train_command(
                spec,
                seed,
                output_root=output_root,
                dt=args.dt,
                kappa=args.kappa,
                print_every=args.print_every,
                checkpoint_every=args.checkpoint_every,
            )
            print(f"[dry-run] seed={seed} run_id={spec.run_id}: {' '.join(cmd)}")
        return 0

    pending = list(selected)
    active: list[ActiveJob] = []
    failures: list[ActiveJob] = []
    gpu_slots = {gpu: 0 for gpu in args.gpus}
    scheduler_log = log_root / "scheduler.tsv"
    with scheduler_log.open("a", encoding="utf-8") as sched:
        sched.write(f"# start {time.strftime('%F %T')} jobs={len(selected)} gpus={args.gpus} max_per_gpu={args.max_per_gpu}\n")

    def available_gpu() -> int | None:
        for gpu in sorted(gpu_slots, key=lambda g: (gpu_slots[g], g)):
            if gpu_slots[gpu] < args.max_per_gpu:
                return gpu
        return None

    while pending or active:
        still_active = []
        for job in active:
            ret = job.proc.poll()
            if ret is None:
                still_active.append(job)
                continue
            gpu_slots[job.gpu] -= 1
            elapsed = time.time() - job.started_at
            status = "ok" if ret == 0 else f"failed:{ret}"
            print(
                f"[done] {job.spec.run_id} seed={job.seed} gpu={job.gpu} "
                f"status={status} elapsed={elapsed / 60:.1f} min"
            )
            with scheduler_log.open("a", encoding="utf-8") as sched:
                sched.write(
                    f"{time.strftime('%F %T')}\tdone\t{job.spec.run_id}\t{job.seed}\t"
                    f"{job.gpu}\t{ret}\t{elapsed:.1f}\t{job.log_path}\n"
                )
            if ret != 0:
                failures.append(job)
        active = still_active

        while pending:
            gpu = available_gpu()
            if gpu is None:
                break
            spec, seed = pending.pop(0)
            out_dir = output_dir(output_root, spec, seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{spec.run_id}_seed{seed}.log"
            cmd = train_command(
                spec,
                seed,
                output_root=output_root,
                dt=args.dt,
                kappa=args.kappa,
                print_every=args.print_every,
                checkpoint_every=args.checkpoint_every,
            )
            env = os.environ.copy()
            env = add_cuda_library_path(env)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n===== start {time.strftime('%F %T')} gpu={gpu} seed={seed} run_id={spec.run_id} =====\n")
                log.write(" ".join(cmd) + "\n")
                proc = subprocess.Popen(
                    cmd,
                    cwd=SCRIPT_DIR,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            gpu_slots[gpu] += 1
            job = ActiveJob(proc=proc, gpu=gpu, spec=spec, seed=seed, log_path=log_path, started_at=time.time())
            active.append(job)
            print(f"[start] {spec.run_id} seed={seed} gpu={gpu} pid={proc.pid} log={log_path}")
            with scheduler_log.open("a", encoding="utf-8") as sched:
                sched.write(
                    f"{time.strftime('%F %T')}\tstart\t{spec.run_id}\t{seed}\t"
                    f"{gpu}\t{proc.pid}\t0\t{log_path}\n"
                )

        if pending or active:
            time.sleep(args.poll_seconds)

    if failures:
        print("Failed jobs:")
        for job in failures:
            print(f"  {job.spec.run_id} seed={job.seed} log={job.log_path}")
        return 1
    print("All requested training jobs finished.")
    return 0


def import_eval_modules():
    add_cuda_library_path(os.environ)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    from omegaconf import OmegaConf

    from evaluate_mmd import compute_structural_errors, simulate
    from nonlinear_case import build_UIDD2, build_targets

    return eqx, jax, jnp, jr, np, OmegaConf, compute_structural_errors, simulate, build_UIDD2, build_targets


def parse_losses(log_file: Path, target_epoch: int | None = None) -> tuple[str, str, str]:
    if not log_file.is_file():
        return "", "", ""
    pattern = re.compile(
        r"epoch=(?P<epoch>\d+), train_loss=(?P<train>[-+0-9.eE]+), "
        r"test_loss=(?P<test>[-+0-9.eE]+), lr_scale=(?P<scale>[-+0-9.eE]+)"
    )
    last = None
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            current = match.groupdict()
            if target_epoch is not None and int(current["epoch"]) == target_epoch:
                return current["train"], current["test"], current["scale"]
            last = current
    if last is None:
        return "", "", ""
    return last["train"], last["test"], last["scale"]


def run_evaluate(args: argparse.Namespace) -> int:
    eqx, jax, jnp, jr, np, OmegaConf, compute_structural_errors, simulate, build_UIDD2, build_targets = import_eval_modules()

    output_root = Path(args.output_root).resolve()
    save_dir = output_root / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    config0 = OmegaConf.load(SCRIPT_DIR / "config" / "nonlinear_case.yaml")
    config0.dt = args.dt
    config0.data.var = args.kappa
    target = build_targets(float(args.kappa))

    init_min = jnp.asarray(config0.data.init_min, dtype=jnp.float64)
    init_max = jnp.asarray(config0.data.init_max, dtype=jnp.float64)
    key = jr.PRNGKey(args.eval_seed)
    init_key, bm_key, sample_key = jr.split(key, 3)
    inits = jr.uniform(
        init_key,
        shape=(args.num_eval_trajs, int(config0.dim)),
        minval=init_min,
        maxval=init_max,
    )
    print(f"Sampling target trajectory cloud for structural errors: n_traj={args.num_eval_trajs}")
    _, ys = simulate(target, config0, inits, bm_key, solve_substeps=args.solve_substeps)
    pooled = ys.reshape(-1, int(config0.dim))
    if pooled.shape[0] > args.max_eval_samples:
        idx = jr.choice(sample_key, pooled.shape[0], shape=(args.max_eval_samples,), replace=False)
        x_eval = pooled[idx]
    else:
        x_eval = pooled
    args_const = jnp.asarray([float(config0.temperature)], dtype=jnp.float64)

    rows: list[dict[str, str]] = []
    failures = 0
    requested = set(args.only) if args.only else None
    for spec in make_eval_plan():
        if requested and spec.run_id not in requested:
            continue
        for seed in args.seeds:
            out_dir = output_root / spec.train_run_id / f"seed{seed}"
            ckpt = eval_checkpoint_path(output_root, spec, seed)
            cfg_path = eval_config_path(output_root, spec, seed)
            if not ckpt.is_file() or not cfg_path.is_file():
                print(f"[skip] missing checkpoint/config: {spec.run_id} seed={seed} ({ckpt.name})")
                continue
            try:
                config = OmegaConf.load(cfg_path)
                template = build_UIDD2(config)
                model = eqx.tree_deserialise_leaves(ckpt, template)
                metrics = compute_structural_errors(model, "HD2", target, x_eval, args_const)
                metrics = {k: float(v) for k, v in metrics.items()}
                train_loss, test_loss, lr_scale = parse_losses(out_dir / "nonlinear_case.log", spec.log_epoch)
                row = {
                    "run_id": spec.run_id,
                    "factor": spec.factor,
                    "level": spec.level,
                    "source_run": spec.train_run_id,
                    "checkpoint": spec.checkpoint,
                    "seed": str(seed),
                    "potential_units": fmt_units(spec.potential_units),
                    "hamiltonian_units": fmt_units(spec.hamiltonian_units),
                    "learning_rate": f"{spec.learning_rate:g}",
                    "batch_size": str(spec.batch_size),
                    "num_epochs": str(spec.num_epochs),
                    "drift": f"{metrics['drift']:.10g}",
                    "gradV": f"{metrics['gradV']:.10g}",
                    "MgradV": f"{metrics['MgradV']:.10g}",
                    "WgradV": f"{metrics['WgradV']:.10g}",
                    "last_train_loss": train_loss,
                    "last_test_loss": test_loss,
                    "last_lr_scale": lr_scale,
                    "output_dir": out_dir.as_posix(),
                }
                rows.append(row)
                print(
                    f"[ok] {spec.run_id} seed={seed}: "
                    f"drift={metrics['drift']:.3e}, gradV={metrics['gradV']:.3e}, "
                    f"MgradV={metrics['MgradV']:.3e}, WgradV={metrics['WgradV']:.3e}"
                )
            except Exception as exc:
                failures += 1
                print(f"[warn] failed evaluating {spec.run_id} seed={seed}: {exc}")

    if not rows:
        print("No completed runs were found to evaluate.")
        return 1

    by_seed_csv = save_dir / "metrics_by_seed.csv"
    headers = list(rows[0])
    with by_seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-seed metrics: {by_seed_csv}")

    summary_rows = summarize_rows(rows)
    summary_csv = save_dir / "metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary metrics: {summary_csv}")

    write_markdown_summary(save_dir / "sensitivity_summary.md", summary_rows, args)
    write_plots(save_dir, summary_rows)
    if failures:
        print(f"Evaluation completed with {failures} failed run(s).")
    return 0 if failures == 0 else 1


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metric_names = ["drift", "gradV", "MgradV", "WgradV"]
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["run_id"], []).append(row)

    summary = []
    spec_by_id = {spec.run_id: spec for spec in make_eval_plan()}
    for run_id, group in groups.items():
        spec = spec_by_id[run_id]
        out = {
            "run_id": run_id,
            "factor": spec.factor,
            "level": spec.level,
            "n": str(len(group)),
            "source_run": spec.train_run_id,
            "checkpoint": spec.checkpoint,
            "potential_units": fmt_units(spec.potential_units),
            "hamiltonian_units": fmt_units(spec.hamiltonian_units),
            "learning_rate": f"{spec.learning_rate:g}",
            "batch_size": str(spec.batch_size),
            "num_epochs": str(spec.num_epochs),
        }
        for metric in metric_names:
            vals = [float(r[metric]) for r in group if r[metric] != ""]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
            out[f"{metric}_mean"] = f"{mean:.10g}"
            out[f"{metric}_std"] = f"{math.sqrt(var):.10g}"
        summary.append(out)
    summary.sort(key=lambda row: (row["factor"], float(row["drift_mean"])))
    return summary


def write_markdown_summary(path: Path, summary_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    metric_cols = ["drift", "gradV", "MgradV", "WgradV"]
    lines = [
        "# UIDD Sensitivity Summary",
        "",
        f"- dt: {args.dt:g}",
        f"- kappa: {args.kappa:g}",
        f"- seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- evaluation trajectories: {args.num_eval_trajs}",
        f"- max structural samples: {args.max_eval_samples}",
        "",
        "Relative error is sqrt(sum ||f_theta - f||^2 / sum ||f||^2).",
        "Metrics are computed by reusing evaluate_mmd.compute_structural_errors.",
        "",
    ]
    for factor in ["baseline", "network", "learning_rate", "batch_size", "num_epochs"]:
        rows = [r for r in summary_rows if r["factor"] == factor]
        if not rows:
            continue
        lines.append(f"## {factor}")
        lines.append("")
        lines.append("| run_id | level | n | drift | gradV | MgradV | WgradV |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            cells = []
            for metric in metric_cols:
                cells.append(f"{float(row[f'{metric}_mean']):.3e} +/- {float(row[f'{metric}_std']):.1e}")
            lines.append(
                f"| {row['run_id']} | {row['level']} | {row['n']} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Markdown summary: {path}")


def write_plots(save_dir: Path, summary_rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[warn] plotting skipped: {exc}")
        return

    metrics = ["drift", "gradV", "MgradV", "WgradV"]
    factors = ["network", "learning_rate", "batch_size", "num_epochs"]
    for factor in factors:
        if factor == "num_epochs":
            rows = [r for r in summary_rows if r["factor"] == factor]
        else:
            rows = [r for r in summary_rows if r["factor"] in ("baseline", factor)]
        if not rows:
            continue
        rows.sort(key=_plot_order_key)
        labels = [r["level"] if r["factor"] != "baseline" else "baseline" for r in rows]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, metric in zip(axes, metrics):
            means = np.array([float(r[f"{metric}_mean"]) for r in rows])
            stds = np.array([float(r[f"{metric}_std"]) for r in rows])
            xs = np.arange(len(rows))
            ax.bar(xs, means, yerr=stds, capsize=3, alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_yscale("log")
            ax.set_title(metric)
            ax.grid(True, which="both", axis="y", alpha=0.3)
        fig.suptitle(f"UIDD sensitivity: {factor}")
        fig.tight_layout()
        png = save_dir / f"{factor}_metrics.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {png}")


def _plot_order_key(row: dict[str, str]) -> tuple[int, float | str]:
    if row["factor"] == "baseline":
        return (0, 0.0)
    if row["factor"] == "num_epochs":
        return (1, float(row["num_epochs"]))
    if row["factor"] == "batch_size":
        return (1, float(row["batch_size"]))
    if row["factor"] == "learning_rate":
        return (1, float(row["learning_rate"]))
    network_order = {"small": 0, "wide": 1, "deep": 2}
    if row["factor"] == "network":
        return (1, network_order.get(row["level"], row["level"]))
    return (1, row["run_id"])


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--kappa", type=float, default=0.8)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT.as_posix())
    parser.add_argument("--only", nargs="+", default=None, help="Optional run_id subset.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Print the sweep matrix.")
    add_common(p_plan)

    p_train = sub.add_parser("train", help="Run the training sweep.")
    add_common(p_train)
    p_train.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS)
    p_train.add_argument("--max-per-gpu", type=int, default=2)
    p_train.add_argument("--poll-seconds", type=float, default=20.0)
    p_train.add_argument("--print-every", type=int, default=200)
    p_train.add_argument("--checkpoint-every", type=int, default=1000)
    p_train.add_argument("--force", action="store_true")
    p_train.add_argument("--dry-run", action="store_true")

    p_eval = sub.add_parser("evaluate", help="Evaluate completed checkpoints.")
    add_common(p_eval)
    p_eval.add_argument("--eval-seed", type=int, default=20260501)
    p_eval.add_argument("--num-eval-trajs", type=int, default=1000)
    p_eval.add_argument("--max-eval-samples", type=int, default=4000)
    p_eval.add_argument("--solve-substeps", type=int, default=10)

    p_all = sub.add_parser("all", help="Train, then evaluate.")
    add_common(p_all)
    p_all.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS)
    p_all.add_argument("--max-per-gpu", type=int, default=2)
    p_all.add_argument("--poll-seconds", type=float, default=20.0)
    p_all.add_argument("--print-every", type=int, default=200)
    p_all.add_argument("--checkpoint-every", type=int, default=1000)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--dry-run", action="store_true")
    p_all.add_argument("--eval-seed", type=int, default=20260501)
    p_all.add_argument("--num-eval-trajs", type=int, default=1000)
    p_all.add_argument("--max-eval-samples", type=int, default=4000)
    p_all.add_argument("--solve-substeps", type=int, default=10)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "plan":
        print(f"dt={args.dt:g}; effective short-trajectory samples ~= {effective_examples(args.dt)}")
        print("\nTraining plan:")
        print_table(train_plan_rows(args.seeds, args.dt))
        print("\nEvaluation plan:")
        print_table(eval_plan_rows(args.seeds))
        total_train_jobs = len(make_train_plan()) * len(args.seeds)
        total_eval_rows = len(make_eval_plan()) * len(args.seeds)
        print(f"\nTraining jobs: {total_train_jobs} ({len(make_train_plan())} settings x {len(args.seeds)} seeds)")
        print(f"Evaluation rows: {total_eval_rows} ({len(make_eval_plan())} metric rows x {len(args.seeds)} seeds)")
        print("Default GPU plan: 4 GPUs x 2 processes/GPU = 8 concurrent training jobs.")
        return 0
    if args.command == "train":
        return run_train(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "all":
        train_status = run_train(args)
        if args.dry_run:
            return train_status
        eval_status = run_evaluate(args)
        return train_status or eval_status
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

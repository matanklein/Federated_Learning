"""
real fed(all data split into P1 + P2):
  - P1's view:  norm_improvement(P1+P2) vs P1 local
  - P2's view:  norm_improvement(P1+P2) vs P2 local

P1 self-division simulation (P1 split into P11 + P12):
  - P11's view: norm_improvement(P11+P12) vs P11 local
  - P12's view: norm_improvement(P11+P12) vs P12 local

P2 self-division simulation (P2 split into P21 + P22):
  - P21's view: norm_improvement(P21+P22) vs P21 local
  - P22's view: norm_improvement(P21+P22) vs P22 local

Normalized improvement = (theta - Theta) / |o - theta|
  where o     = oracle loss (initial before any training)
        theta = local training loss
        Theta = federated training loss

Each of the 6 pairs produced TWICE: once for Suppression, once for DP.
= 12 heatmaps total.

Note: run from project directory.

Quick run (full data, 1 seed, 3x3 grids, 50 rounds):
    PYTHONPATH=/home/pejo_balazs/COL-Drug/mina python src/main.py \
    --data_path /home/pejo_balazs/COL-Drug/mina/data/ \
    --n_samples 50000 \
    --seed_range 3 \
    --rounds 50 \
    --data_root src/data_50k/ \
    --results src/results_50k/ \
    --plots src/plots_50k/

Full run:
    PYTHONPATH=/home/pejo_balazs/COL-Drug/mina python src/main.py --seed_range 10 --rounds 200
"""

import packages.utils.data_utils as du
import numpy as np
import torch
import CollaborativeLearning as CoL
import sparsechem as sc
import csv, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
from split_data import split_with_overlap

INPUT_SIZE  = 32000
OUTPUT_SIZE = 2808
OVERLAP     = 2808

DP_NOISE_LEVELS_FULL  = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
DP_NOISE_LEVELS_QUICK = [0.00, 0.50, 2.00]

SUP_LEVELS_FULL  = [round(0.1 * i, 1) for i in range(11)]
SUP_LEVELS_QUICK = [0.0, 0.45, 0.90]

DATA_PATH    = "/home/pejo_balazs/COL-Drug/mina/data/"
DATA_ROOT    = "experiment_data/"
RESULTS_ROOT = "results/"
PLOTS_ROOT   = "plots/"



def parse_args():
    p = argparse.ArgumentParser(
        description="DP + Suppression federated privacy experiment"
    )
    p.add_argument("--seed_range", type=int, default=10,
        help="Number of seeds. Use 1-3 for a quick check.")
    p.add_argument("--rounds", type=int, default=200,
        help="FL training rounds per experiment. Use 50 for a quick check.")
    p.add_argument("--n_samples", type=int, default=None,
        help="Subsample N rows from training data. Omit to use all data.")
    p.add_argument("--overlap", type=int, default=OVERLAP,
        help=f"Number of shared tasks between clients (default {OVERLAP}).")
    p.add_argument("--quick", action="store_true",
        help="Use reduced 3x3 parameter grids (~1-2 hours).")
    p.add_argument("--data_path", type=str, default=DATA_PATH,
        help="Path to raw data directory.")
    p.add_argument("--data_root", type=str, default=DATA_ROOT,
        help="Root directory for persisted splits.")
    p.add_argument("--results", type=str, default=RESULTS_ROOT,
        help="Directory for CSV result files.")
    p.add_argument("--plots", type=str, default=PLOTS_ROOT,
        help="Directory for heatmap PNG files.")
    p.add_argument("--clip_batches", type=int, default=20,
        help="Number of batches to use for auto clip estimation (default 20).")
    return p.parse_args()


# Protects dp_noise_std / dp_clip_norm / dp_scope from being silently reset
# by ModelConfig.__setattr__ when output_size or batch_size is updated
# inside CollaborativeLearning.run_server.
class ConfigWrapper:
    def __init__(self, base_config, dp_noise, dp_clip):
        self._base        = base_config
        self.dp_noise_std = dp_noise
        self.dp_clip_norm = dp_clip
        self.dp_scope     = "none"

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        if name in ['_base', 'dp_noise_std', 'dp_clip_norm', 'dp_scope']:
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def subsample(ecfp, ic50, n, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(ecfp.shape[0], size=n, replace=False))
    return ecfp[idx], ic50[idx]


def load_split_xy(split_dir, client_idx, subset):
    path = os.path.join(split_dir, "data_2_split") + "/"
    return du.load_ratio_split_data(path, client_idx, train=(subset == "train"))


def prepare_fixed_splits(ecfp_tr, ic50_tr, overlap, root):
    """
    Create (once) the three fixed two-client splits using split_with_overlap.

    Layout:
        <root>/full/data_2_split/  -> P1  + P2   (from all training rows)
        <root>/p1/data_2_split/   -> P11 + P12  (from P1 training rows)
        <root>/p2/data_2_split/   -> P21 + P22  (from P2 training rows)
    """
    paths = {
        "full": os.path.join(root, "full"),
        "p1":   os.path.join(root, "p1"),
        "p2":   os.path.join(root, "p2"),
    }

    marker_full = os.path.join(paths["full"], "data_2_split", "0_train")
    if not os.path.exists(marker_full):
        print("  Creating P1/P2 split ...")
        set_seed(0)
        split_with_overlap(1, ecfp_tr, ic50_tr,
                           root_dir=paths["full"], overlap=overlap)
    else:
        print("  P1/P2 split already exists, skipping.")

    x_p1_tr, y_p1_tr = load_split_xy(paths["full"], 0, "train")
    x_p2_tr, y_p2_tr = load_split_xy(paths["full"], 1, "train")

    marker_p1 = os.path.join(paths["p1"], "data_2_split", "0_train")
    if not os.path.exists(marker_p1):
        print("  Creating P11/P12 split ...")
        set_seed(0)
        split_with_overlap(1, x_p1_tr, y_p1_tr,
                           root_dir=paths["p1"], overlap=overlap)
    else:
        print("  P11/P12 split already exists, skipping.")

    marker_p2 = os.path.join(paths["p2"], "data_2_split", "0_train")
    if not os.path.exists(marker_p2):
        print("  Creating P21/P22 split ...")
        set_seed(0)
        split_with_overlap(1, x_p2_tr, y_p2_tr,
                           root_dir=paths["p2"], overlap=overlap)
    else:
        print("  P21/P22 split already exists, skipping.")

    return paths



def make_base_conf(n_train_total, rounds):
    """Base ModelConfig without DP params. Use ConfigWrapper to add those."""
    batch_size = max(64, int((n_train_total / 2) * 0.02))
    c = sc.ModelConfig(
        input_size         = INPUT_SIZE,
        hidden_sizes       = [40],
        output_size        = OUTPUT_SIZE,
        batch_size         = batch_size,
        lr                 = 1e-3,
        last_dropout       = 0.2,
        weight_decay       = 1e-5,
        non_linearity      = "relu",
        last_non_linearity = "relu",
        optimizer          = "ADAM",
    )
    c.rounds = rounds
    return c


def make_wrapped_conf(base_conf, dp_noise, dp_clip):
    """Wrap base_conf with DP parameters, protected from ModelConfig.__setattr__."""
    return ConfigWrapper(
        deepcopy(base_conf),
        dp_noise = dp_noise,
        dp_clip  = dp_clip if dp_noise > 0.0 else None,
    )


# AUTO CLIP ESTIMATION
# Runs n_batches forward+backward passes and picks the 75th percentile
# of observed trunk gradient norms as the clip value.
# - p75 means ~75% of batches get clipped (clips the noisiest quarter)
# - adapts automatically to dataset size, model size, and batch size

def estimate_dp_clip(base_conf, data_dir, n_batches=20):
    """
    Estimate a good DP clip norm from actual gradient magnitudes.
    Returns the 75th percentile of trunk gradient norms over n_batches.
    """
    from packages.collaborative.participant import Client

    print("\nEstimating gradient norms for DP clip selection ...")

    path = os.path.join(data_dir, "data_2_split") + "/"
    X, Y = du.load_ratio_split_data(path, 0, train=True)

    dataset    = sc.SparseDataset(X, Y)
    X_va, Y_va = du.load_ratio_split_data(path, 0, train=False)
    dataset_va = sc.SparseDataset(X_va, Y_va)

    c = deepcopy(base_conf)
    c.output_size = Y.shape[1]
    c.batch_size  = max(64, int(X.shape[0] * 0.02))

    trunk  = sc.Trunk(c)
    model  = sc.TrunkAndHead(conf=c, trunk=trunk)
    client = Client(model, conf=c, dataset=dataset, dataset_va=dataset_va)

    norms = []
    for _ in range(n_batches):
        batch = client.get_next_batch()
        client.train(batch)

        total_norm = 0.0
        for p in model.trunk.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        norms.append(total_norm)

        client.zero_grad()

    norms = np.array(norms)
    clip  = float(np.percentile(norms, 75))

    print(f"  Gradient norms over {n_batches} batches:")
    print(f"    min    = {norms.min():.6f}")
    print(f"    median = {np.median(norms):.6f}")
    print(f"    p75    = {clip:.6f}   ← selected as dp_clip")
    print(f"    max    = {norms.max():.6f}")

    return clip



# NORMALIZED ACCURACY IMPROVEMENT
# (theta - Theta) / |o - theta|
#   o     = oracle loss (initial, before any training)
#   theta = local training loss
#   Theta = federated training loss
# Positive = federation helped, negative = federation hurt.

def norm_improvement(oracle, local, federated, eps=1e-8):
    return (local - federated) / (abs(oracle - local) + eps)



def run_local(client_idx, data_dir, ecfp_va, ic50_va, base_conf):
    """Train one client alone (no DP). Returns (oracle_loss, local_loss)."""
    c = make_wrapped_conf(base_conf, dp_noise=0.0, dp_clip=None)
    res, accs = CoL.run_server(
        group            = [client_idx],
        hide             = [0],
        conf_list        = [c],
        rounds           = base_conf.rounds,
        server_data      = (ecfp_va, ic50_va),
        client_data_path = data_dir.rstrip("/") + "/",
        same_head        = True,
    )
    return res[0], accs[0]


def run_joint_dp(data_dir, ecfp_va, ic50_va, base_conf,
                 noise_c0, noise_c1, dp_clip):
    """Federated run with per-client DP noise. Returns (loss_c0, loss_c1)."""
    c0 = make_wrapped_conf(base_conf, noise_c0, dp_clip)
    c1 = make_wrapped_conf(base_conf, noise_c1, dp_clip)
    res, accs = CoL.run_server(
        group            = [0, 1],
        hide             = [0, 0],
        conf_list        = [c0, c1],
        rounds           = base_conf.rounds,
        server_data      = (ecfp_va, ic50_va),
        client_data_path = data_dir.rstrip("/") + "/",
        same_head        = True,
    )
    return accs[0], accs[1]


def run_joint_sup(data_dir, ecfp_va, ic50_va, base_conf, hide_c0, hide_c1):
    """Federated run with suppression. Returns (loss_c0, loss_c1)."""
    c0 = make_wrapped_conf(base_conf, dp_noise=0.0, dp_clip=None)
    c1 = make_wrapped_conf(base_conf, dp_noise=0.0, dp_clip=None)
    res, accs = CoL.run_server(
        group            = [0, 1],
        hide             = [hide_c0, hide_c1],
        conf_list        = [c0, c1],
        rounds           = base_conf.rounds,
        server_data      = (ecfp_va, ic50_va),
        client_data_path = data_dir.rstrip("/") + "/",
        same_head        = True,
    )
    return accs[0], accs[1]


def run_scenario_dp(data_dir, ecfp_va, ic50_va, base_conf, seed,
                    dp_noise_levels, dp_clip):
    set_seed(seed)
    oracle0, local0 = run_local(0, data_dir, ecfp_va, ic50_va, base_conf)
    set_seed(seed)
    oracle1, local1 = run_local(1, data_dir, ecfp_va, ic50_va, base_conf)

    print(f"      Baselines — "
          f"c0: oracle={oracle0:.6f} local={local0:.6f} | "
          f"c1: oracle={oracle1:.6f} local={local1:.6f}")

    n = len(dp_noise_levels)
    grid_c0, grid_c1 = np.zeros((n, n)), np.zeros((n, n))

    for i, n0 in enumerate(dp_noise_levels):
        for j, n1 in enumerate(dp_noise_levels):
            set_seed(seed)
            j0, j1 = run_joint_dp(data_dir, ecfp_va, ic50_va,
                                   base_conf, n0, n1, dp_clip)
            grid_c0[i, j] = norm_improvement(oracle0, local0, j0)
            grid_c1[i, j] = norm_improvement(oracle1, local1, j1)
            print(f"      DP ({n0:.2f},{n1:.2f}) → "
                  f"c0={grid_c0[i,j]:+.4f}  c1={grid_c1[i,j]:+.4f}")

    return grid_c0, grid_c1


def run_scenario_sup(data_dir, ecfp_va, ic50_va, base_conf, seed, sup_levels):
    set_seed(seed)
    oracle0, local0 = run_local(0, data_dir, ecfp_va, ic50_va, base_conf)
    set_seed(seed)
    oracle1, local1 = run_local(1, data_dir, ecfp_va, ic50_va, base_conf)

    print(f"      Baselines — "
          f"c0: oracle={oracle0:.6f} local={local0:.6f} | "
          f"c1: oracle={oracle1:.6f} local={local1:.6f}")

    n = len(sup_levels)
    grid_c0, grid_c1 = np.zeros((n, n)), np.zeros((n, n))

    for i, h0 in enumerate(sup_levels):
        for j, h1 in enumerate(sup_levels):
            set_seed(seed)
            j0, j1 = run_joint_sup(data_dir, ecfp_va, ic50_va,
                                    base_conf, h0, h1)
            grid_c0[i, j] = norm_improvement(oracle0, local0, j0)
            grid_c1[i, j] = norm_improvement(oracle1, local1, j1)
            print(f"      SUP ({h0:.2f},{h1:.2f}) → "
                  f"c0={grid_c0[i,j]:+.4f}  c1={grid_c1[i,j]:+.4f}")

    return grid_c0, grid_c1


def grid_to_csv(grid, param_grid, param_name, scenario,
                client_label, seed, path):
    rows = []
    for i, p0 in enumerate(param_grid):
        for j, p1 in enumerate(param_grid):
            rows.append({
                "scenario":         scenario,
                "client":           client_label,
                "seed":             seed,
                "param_name":       param_name,
                f"{param_name}_c0": p0,
                f"{param_name}_c1": p1,
                "norm_improvement": float(grid[i, j]),
            })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)



def plot_heatmap(mean_grid, std_grid, param_grid, param_name,
                 title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    abs_max = max(float(np.abs(mean_grid).max()), 1e-6)
    im = ax.imshow(mean_grid, cmap="RdYlGn",
                   vmin=-abs_max, vmax=abs_max,
                   aspect="auto", origin="upper")

    tick_labels = [f"{p:.2f}" if isinstance(p, float) else str(p)
                   for p in param_grid]
    ax.set_xticks(range(len(param_grid)))
    ax.set_yticks(range(len(param_grid)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel(f"Client 1  {param_name}", fontsize=11)
    ax.set_ylabel(f"Client 0  {param_name}", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)

    for i in range(len(param_grid)):
        for j in range(len(param_grid)):
            val   = float(mean_grid[i, j])
            std   = float(std_grid[i, j])
            color = "white" if abs(val) > abs_max * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}\n±{std:.3f}",
                    ha="center", va="center", fontsize=6.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Normalised Accuracy Improvement\n(theta − Theta) / |o − theta|",
        fontsize=10
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")




def run_all(ecfp_tr, ic50_tr, ecfp_va, ic50_va,
            splits, base_conf, seed_range,
            dp_noise_levels, sup_levels, dp_clip,
            results_root, plots_root):

    scenarios = {
        "real":   (splits["full"], "P1",  "P2"),
        "sim_p1": (splits["p1"],   "P11", "P12"),
        "sim_p2": (splits["p2"],   "P21", "P22"),
    }

    accum = {
        sname: {"dp": [[], []], "sup": [[], []]}
        for sname in scenarios
    }

    for seed in seed_range:
        print(f"\n{'='*60}  SEED {seed}  {'='*60}")

        for sname, (data_dir, lbl0, lbl1) in scenarios.items():
            print(f"\n  [{sname}]  {lbl0} vs {lbl1}")

            print("    DP grid ...")
            g0_dp, g1_dp = run_scenario_dp(
                data_dir, ecfp_va, ic50_va, base_conf, seed,
                dp_noise_levels, dp_clip
            )
            accum[sname]["dp"][0].append(g0_dp)
            accum[sname]["dp"][1].append(g1_dp)
            for grid, lbl in [(g0_dp, lbl0), (g1_dp, lbl1)]:
                grid_to_csv(grid, dp_noise_levels, "noise",
                            sname, lbl, seed,
                            os.path.join(results_root, f"dp_{sname}.csv"))

            print("    Suppression grid ...")
            g0_sup, g1_sup = run_scenario_sup(
                data_dir, ecfp_va, ic50_va, base_conf, seed, sup_levels
            )
            accum[sname]["sup"][0].append(g0_sup)
            accum[sname]["sup"][1].append(g1_sup)
            for grid, lbl in [(g0_sup, lbl0), (g1_sup, lbl1)]:
                grid_to_csv(grid, sup_levels, "suppression",
                            sname, lbl, seed,
                            os.path.join(results_root, f"sup_{sname}.csv"))

    # plot
    print(f"\n{'='*60}")
    print("  Plotting 12 heatmaps ...")

    method_meta = [
        ("dp",  dp_noise_levels, "DP Noise σ"),
        ("sup", sup_levels,      "Suppression ratio"),
    ]

    for sname, (_, lbl0, lbl1) in scenarios.items():
        for method, param_grid, param_label in method_meta:
            for c_idx, lbl in enumerate([lbl0, lbl1]):
                stack     = np.stack(accum[sname][method][c_idx], axis=0)
                mean_grid = stack.mean(axis=0)
                std_grid  = stack.std(axis=0)

                n_seeds = len(list(seed_range))
                title = (
                    f"{lbl}'s View  ·  {method.upper()}  ·  {sname.upper()}\n"
                    f"Norm. Improvement = (theta − Theta) / |o − theta|\n"
                    f"mean ± std  over {n_seeds} seed{'s' if n_seeds > 1 else ''}"
                )
                plot_heatmap(
                    mean_grid, std_grid, param_grid, param_label,
                    title,
                    out_path=os.path.join(plots_root, method,
                                          f"{sname}_{lbl}_{method}.png"),
                )

    print("\nAll 12 heatmaps saved.")


# ENTRY POINT

if __name__ == "__main__":
    args = parse_args()

    # select full or quick exp run
    use_quick = args.quick or (args.n_samples is not None)
    dp_noise_levels = DP_NOISE_LEVELS_QUICK if use_quick else DP_NOISE_LEVELS_FULL
    sup_levels      = SUP_LEVELS_QUICK      if use_quick else SUP_LEVELS_FULL
    print(f"Grid mode : {'QUICK (3x3)' if use_quick else 'FULL (9x9 / 11x11)'}")
    print(f"DP  grid  : {dp_noise_levels}")
    print(f"SUP grid  : {sup_levels}")

    for d in [args.data_root, args.results,
              os.path.join(args.plots, "dp"),
              os.path.join(args.plots, "sup")]:
        os.makedirs(d, exist_ok=True)

    # load raw data
    print("\nLoading data ...")
    ecfp_tr, ic50_tr, ecfp_va, ic50_va = du.load_data(args.data_path)
    ecfp_tr = du.fold_input(ecfp_tr, INPUT_SIZE)
    ecfp_va = du.fold_input(ecfp_va, INPUT_SIZE)

    if args.n_samples is not None:
        print(f"Subsampling to {args.n_samples:,} training rows ...")
        ecfp_tr, ic50_tr = subsample(ecfp_tr, ic50_tr,
                                     n=args.n_samples, seed=0)

    n_train = ecfp_tr.shape[0]
    print(f"Training set : {n_train:,} samples  "
          f"{ecfp_tr.shape[1]:,} features  "
          f"{ic50_tr.shape[1]:,} tasks")

    # fixed splits
    print("\nPreparing fixed data splits ...")
    splits = prepare_fixed_splits(
        ecfp_tr, ic50_tr,
        overlap = args.overlap,
        root    = args.data_root,
    )

    base_conf = make_base_conf(n_train, rounds=args.rounds)

    # estimate dp clip
    dp_clip = estimate_dp_clip(
        base_conf,
        data_dir    = splits["full"],
        n_batches   = args.clip_batches,
    )

    # summ
    dp_cells   = len(dp_noise_levels) ** 2
    sup_cells  = len(sup_levels) ** 2
    # 2 baselines per scenario per seed (local c0 + local c1)
    # baselines are shared between DP and SUP grids so counted once
    total_runs = args.seed_range * 3 * (dp_cells + sup_cells + 4)
    est_minutes = total_runs * 2 * 66 / 60   # 2 clients * ~66s per cell
    print(f"\nConfiguration:")
    print(f"  seeds        : {args.seed_range}")
    print(f"  rounds       : {args.rounds}")
    print(f"  overlap      : {args.overlap}")
    print(f"  batch_size   : {base_conf.batch_size}")
    print(f"  n_train      : {n_train:,}")
    print(f"  dp_clip      : {dp_clip:.6f}  (auto-estimated)")
    print(f"  DP  grid     : {len(dp_noise_levels)}×{len(dp_noise_levels)} = {dp_cells} cells")
    print(f"  SUP grid     : {len(sup_levels)}×{len(sup_levels)} = {sup_cells} cells")
    print(f"  Total CoL runs : ~{total_runs:,}")
    print(f"  Est. runtime   : ~{est_minutes:.0f} minutes")

    run_all(
        ecfp_tr, ic50_tr, ecfp_va, ic50_va,
        splits          = splits,
        base_conf       = base_conf,
        seed_range      = range(args.seed_range),
        dp_noise_levels = dp_noise_levels,
        sup_levels      = sup_levels,
        dp_clip         = dp_clip,
        results_root    = args.results,
        plots_root      = args.plots,
    )
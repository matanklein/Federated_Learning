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

Quick run (full data, 3 seed, 3x3 grids, 50 rounds):
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

import argparse
import os
import csv
import copy
import numpy as np
import torch
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flwr as fl
import sparsechem as sc
import packages.utils.data_utils as du
from split_data import split_with_overlap
from dataset import get_client_datasets
from client import DrugDiscoveryClient

# --- Configuration Constants ---
INPUT_SIZE  = 32000
OUTPUT_SIZE = 2808
OVERLAP     = 2808

DP_NOISE_LEVELS_FULL  = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
DP_NOISE_LEVELS_QUICK = [0.00, 0.50, 2.00]

SUP_LEVELS_FULL  = [round(0.1 * i, 1) for i in range(11)]
SUP_LEVELS_QUICK = [0.0, 0.45, 0.90]

DATA_PATH    = "../data/"
DATA_ROOT    = "experiment_data/"
RESULTS_ROOT = "results/"
PLOTS_ROOT   = "plots/"

# --- Argument Parsing ---
def parse_args():
    p = argparse.ArgumentParser(description="DP + Suppression federated privacy experiment with Flower")
    p.add_argument("--seed_range", type=int, default=10, help="Number of seeds. Use 1-3 for a quick check.")
    p.add_argument("--rounds", type=int, default=200, help="FL training rounds per experiment. Use 50 for a quick check.")
    p.add_argument("--n_samples", type=int, default=None, help="Subsample N rows from training data. Omit to use all.")
    p.add_argument("--overlap", type=int, default=OVERLAP, help=f"Number of shared tasks between clients (default {OVERLAP}).")
    p.add_argument("--quick", action="store_true", help="Use reduced 3x3 parameter grids (~1-2 hours).")
    p.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to raw data directory.")
    p.add_argument("--data_root", type=str, default=DATA_ROOT, help="Root directory for persisted splits.")
    p.add_argument("--results", type=str, default=RESULTS_ROOT, help="Directory for CSV result files.")
    p.add_argument("--plots", type=str, default=PLOTS_ROOT, help="Directory for heatmap PNG files.")
    p.add_argument("--clip_batches", type=int, default=20, help="Number of batches to use for auto clip estimation.")
    return p.parse_args()

# --- Utility Functions ---
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
    """Creates the three fixed two-client splits for Real, Sim_P1, and Sim_P2."""
    paths = {
        "full": os.path.join(root, "full"),
        "p1":   os.path.join(root, "p1"),
        "p2":   os.path.join(root, "p2"),
    }

    marker_full = os.path.join(paths["full"], "data_2_split", "0_train")
    if not os.path.exists(marker_full):
        print("  Creating P1/P2 split ...")
        set_seed(0)
        split_with_overlap(1, ecfp_tr, ic50_tr, root_dir=paths["full"], overlap=overlap)
    else:
        print("  P1/P2 split already exists, skipping.")

    x_p1_tr, y_p1_tr = load_split_xy(paths["full"], 0, "train")
    x_p2_tr, y_p2_tr = load_split_xy(paths["full"], 1, "train")

    marker_p1 = os.path.join(paths["p1"], "data_2_split", "0_train")
    if not os.path.exists(marker_p1):
        print("  Creating P11/P12 split ...")
        set_seed(0)
        split_with_overlap(1, x_p1_tr, y_p1_tr, root_dir=paths["p1"], overlap=overlap)

    marker_p2 = os.path.join(paths["p2"], "data_2_split", "0_train")
    if not os.path.exists(marker_p2):
        print("  Creating P21/P22 split ...")
        set_seed(0)
        split_with_overlap(1, x_p2_tr, y_p2_tr, root_dir=paths["p2"], overlap=overlap)

    return paths

def make_base_conf(n_train_total, rounds):
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
    )
    c.rounds = rounds
    return c

def estimate_dp_clip(base_conf, data_dir, device, n_batches=20):
    """Auto-estimates DP clip norm from the 75th percentile of actual gradient norms."""
    print("\nEstimating gradient norms for DP clip selection ...")
    X, Y = load_split_xy(data_dir, 0, "train")
    dataset = sc.SparseDataset(X, Y)

    c = copy.deepcopy(base_conf)
    c.output_size = Y.shape[1]
    c.batch_size = max(64, int(X.shape[0] * 0.02))

    model = sc.TrunkAndHead(conf=c, trunk=sc.Trunk(c)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    loader = torch.utils.data.DataLoader(dataset, batch_size=c.batch_size, shuffle=True, collate_fn=sc.sparse_collate)

    norms = []
    model.train()
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        
        # Construct the PyTorch sparse tensor for the input
        b_x = torch.sparse_coo_tensor(
            batch["x_ind"], 
            batch["x_data"], 
            size=[batch["batch_size"], c.input_size]
        ).to(device)
        
        y_ind = batch["y_ind"].to(device)
        y_data = batch["y_data"].to(device)

        optimizer.zero_grad()
        
        # Pass the sparse tensor to the model
        logits = model(b_x)
        
        # Compute loss ONLY on the known sparse labels to save memory
        logits_subset = logits[y_ind[0], y_ind[1]]
        loss = loss_fn(logits_subset, y_data).mean()
        loss.backward()

        total_norm = 0.0
        for p in model.trunk.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        norms.append(total_norm ** 0.5)

    norms = np.array(norms)
    clip = float(np.percentile(norms, 75))
    print(f"  Gradient norms over {n_batches} batches:")
    print(f"    p75    = {clip:.6f}   ← selected as dp_clip")
    return clip

def norm_improvement(oracle, local, federated, eps=1e-8):
    return (local - federated) / (abs(oracle - local) + eps)

# --- Flower FL Execution Engine ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, target_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None
        self.target_rounds = target_rounds

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None and server_round == self.target_rounds:
            self.final_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

def evaluate_global_model(model, test_dataset, conf, loss_fn, device):
    """Calculates evaluation loss. The model retains its personalized Head while evaluating."""
    model.to(device)
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, collate_fn=sc.sparse_collate)
    
    total_loss, samples = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            # Construct the PyTorch sparse tensor
            b_x = torch.sparse_coo_tensor(
                batch["x_ind"], 
                batch["x_data"], 
                size=[batch["batch_size"], conf.input_size]
            ).to(device)
            
            y_ind = batch["y_ind"].to(device)
            y_data = batch["y_data"].to(device)
            
            logits = model(b_x)
            
            # Extract only the active target predictions
            logits_subset = logits[y_ind[0], y_ind[1]]
            if logits_subset.numel() > 0:
                loss = loss_fn(logits_subset, y_data)
                total_loss += loss.sum().item()
                samples += logits_subset.numel()
                
    return total_loss / max(samples, 1)

def run_fl_experiment(data_dir, group, privacy_mode, privacy_params, base_conf, dp_clip, args, loss_fn, device, seed):
    """Executes a Flower simulation and returns final losses for the given clients."""
    path_suffix = os.path.join(data_dir, "data_2_split/")
    head_dir = args.results 
    
    # Clean up old persistent heads before starting a fresh simulation
    for k in group:
        hp = os.path.join(head_dir, f"head_{k}.pt")
        op = os.path.join(head_dir, f"optim_{k}.pt")
        if os.path.exists(hp): os.remove(hp)
        if os.path.exists(op): os.remove(op)

    # FIX 1: Pre-load datasets to ensure suppression consistency across rounds!
    set_seed(seed)
    client_datasets = {}
    for flower_id, k in enumerate(group):
        p_val = privacy_params[flower_id]
        client_datasets[k] = get_client_datasets(path_suffix, k, p_val, privacy_mode, base_conf)

    # FIX 2: Ensure all clients start from the EXACT SAME initial weights as the Oracle
    set_seed(seed)
    initial_model = sc.TrunkAndHead(conf=base_conf, trunk=sc.Trunk(base_conf))
    
    # Extract trunk parameters to give to the Flower Server
    initial_trunk_params = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.trunk.state_dict().items()]
    )
    # Extract head parameters to inject into clients
    initial_head_state = {k: v for k, v in initial_model.state_dict().items() if 'trunk' not in k}

    def client_fn(cid: str) -> fl.client.Client:
            flower_id = int(cid) 
            k = group[flower_id]
            p_val = privacy_params[flower_id]
            
            # Pull fixed dataset from closure
            train_ds, test_ds = client_datasets[k]
            
            trunk = sc.Trunk(base_conf)
            model = sc.TrunkAndHead(conf=base_conf, trunk=trunk)
            
            # Force deterministic Head initialization on Round 1
            hp = os.path.join(head_dir, f"head_{k}.pt")
            if not os.path.exists(hp):
                model.load_state_dict(initial_head_state, strict=False)
            
            return DrugDiscoveryClient(
                model, train_ds, test_ds, base_conf, loss_fn, privacy_mode, p_val, dp_clip, k, head_dir
            ).to_client()

    strategy = SaveModelStrategy(
        target_rounds=args.rounds,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=len(group), min_evaluate_clients=len(group), min_available_clients=len(group),
        initial_parameters=initial_trunk_params  # FIX 3: Push deterministic trunk to server
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(group),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if device.type == 'cpu' else 1.0},
    )
        
    # Calculate performance using the global trunk + personalized local head
    final_losses = []
    for k in group:
        _, test_ds = get_client_datasets(path_suffix, k, 0.0, 'suppression', base_conf)
        trunk = sc.Trunk(base_conf)
        model = sc.TrunkAndHead(conf=base_conf, trunk=trunk)
        
        hp = os.path.join(head_dir, f"head_{k}.pt")
        if os.path.exists(hp):
            model.load_state_dict(torch.load(hp), strict=False)
            
        if strategy.final_parameters is not None:
            params_dict = zip(model.trunk.state_dict().keys(), strategy.final_parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.trunk.load_state_dict(state_dict, strict=True)
            
        loss = evaluate_global_model(model, test_ds, base_conf, loss_fn, device)
        final_losses.append(loss)
        
    return final_losses

def run_local(client_idx, data_dir, base_conf, dp_clip, args, loss_fn, device, seed):
    """Evaluates untouched (Oracle) model, then trains a local model via 1-client FL instance."""
    path_suffix = os.path.join(data_dir, "data_2_split/")
    _, test_ds = get_client_datasets(path_suffix, client_idx, 0.0, 'suppression', base_conf)
    
    # Untrained model for Oracle
    set_seed(seed)
    untrained_model = sc.TrunkAndHead(conf=base_conf, trunk=sc.Trunk(base_conf))
    oracle_loss = evaluate_global_model(untrained_model, test_ds, base_conf, loss_fn, device)
    
    # Standard local training wrapped in Flower engine (1 round equivalent local passes)
    losses = run_fl_experiment(data_dir, [client_idx], 'suppression', [0.0], base_conf, dp_clip, args, loss_fn, device, seed)
    return oracle_loss, losses[0]

# --- Output Orchestration ---
def grid_to_csv(grid, param_grid, param_name, scenario, client_label, seed, path):
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

def plot_heatmap(mean_grid, std_grid, param_grid, param_name, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    abs_max = max(float(np.abs(mean_grid).max()), 1e-6)
    im = ax.imshow(mean_grid, cmap="RdYlGn", vmin=-abs_max, vmax=abs_max, aspect="auto", origin="upper")

    tick_labels = [f"{p:.2f}" if isinstance(p, float) else str(p) for p in param_grid]
    ax.set_xticks(range(len(param_grid)))
    ax.set_yticks(range(len(param_grid)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel(f"Client 1  {param_name}", fontsize=11)
    ax.set_ylabel(f"Client 0  {param_name}", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)

    for i in range(len(param_grid)):
        for j in range(len(param_grid)):
            val = float(mean_grid[i, j])
            std = float(std_grid[i, j])
            color = "white" if abs(val) > abs_max * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}\n±{std:.3f}", ha="center", va="center", fontsize=6.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Accuracy Improvement\n(theta − Theta) / |o − theta|", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")

def run_all(splits, base_conf, seed_range, dp_noise_levels, sup_levels, dp_clip, args, loss_fn, device):
    scenarios = {
        "real":   (splits["full"], "P1",  "P2"),
        "sim_p1": (splits["p1"],   "P11", "P12"),
        "sim_p2": (splits["p2"],   "P21", "P22"),
    }

    accum = {sname: {"dp": [[], []], "sup": [[], []]} for sname in scenarios}

    for seed in seed_range:
        print(f"\n{'='*60}  SEED {seed}  {'='*60}")
        
        for sname, (data_dir, lbl0, lbl1) in scenarios.items():
            print(f"\n  [{sname}]  {lbl0} vs {lbl1}")
            set_seed(seed)
            oracle0, local0 = run_local(0, data_dir, base_conf, dp_clip, args, loss_fn, device, seed)
            set_seed(seed)
            oracle1, local1 = run_local(1, data_dir, base_conf, dp_clip, args, loss_fn, device, seed)

            print(f"      Baselines — c0: oracle={oracle0:.6f} local={local0:.6f} | c1: oracle={oracle1:.6f} local={local1:.6f}")

            # DP Grid
            print("    DP grid ...")
            n = len(dp_noise_levels)
            grid_c0_dp, grid_c1_dp = np.zeros((n, n)), np.zeros((n, n))
            for i, n0 in enumerate(dp_noise_levels):
                for j, n1 in enumerate(dp_noise_levels):
                    set_seed(seed)
                    j0, j1 = run_fl_experiment(data_dir, [0, 1], 'dp', [n0, n1], base_conf, dp_clip, args, loss_fn, device, seed)
                    grid_c0_dp[i, j] = norm_improvement(oracle0, local0, j0)
                    grid_c1_dp[i, j] = norm_improvement(oracle1, local1, j1)
            
            accum[sname]["dp"][0].append(grid_c0_dp)
            accum[sname]["dp"][1].append(grid_c1_dp)
            grid_to_csv(grid_c0_dp, dp_noise_levels, "noise", sname, lbl0, seed, os.path.join(args.results, f"dp_{sname}.csv"))
            grid_to_csv(grid_c1_dp, dp_noise_levels, "noise", sname, lbl1, seed, os.path.join(args.results, f"dp_{sname}.csv"))

            # Suppression Grid
            print("    Suppression grid ...")
            m = len(sup_levels)
            grid_c0_sup, grid_c1_sup = np.zeros((m, m)), np.zeros((m, m))
            for i, h0 in enumerate(sup_levels):
                for j, h1 in enumerate(sup_levels):
                    set_seed(seed)
                    j0, j1 = run_fl_experiment(data_dir, [0, 1], 'suppression', [h0, h1], base_conf, dp_clip, args, loss_fn, device, seed)
                    grid_c0_sup[i, j] = norm_improvement(oracle0, local0, j0)
                    grid_c1_sup[i, j] = norm_improvement(oracle1, local1, j1)

            accum[sname]["sup"][0].append(grid_c0_sup)
            accum[sname]["sup"][1].append(grid_c1_sup)
            grid_to_csv(grid_c0_sup, sup_levels, "suppression", sname, lbl0, seed, os.path.join(args.results, f"sup_{sname}.csv"))
            grid_to_csv(grid_c1_sup, sup_levels, "suppression", sname, lbl1, seed, os.path.join(args.results, f"sup_{sname}.csv"))

    # Plot 12 Heatmaps
    print(f"\n{'='*60}\n  Plotting 12 heatmaps ...")
    method_meta = [("dp", dp_noise_levels, "DP Noise σ"), ("sup", sup_levels, "Suppression ratio")]

    for sname, (_, lbl0, lbl1) in scenarios.items():
        for method, param_grid, param_label in method_meta:
            for c_idx, lbl in enumerate([lbl0, lbl1]):
                stack = np.stack(accum[sname][method][c_idx], axis=0)
                mean_grid = stack.mean(axis=0)
                std_grid = stack.std(axis=0)

                n_seeds = len(list(seed_range))
                title = (f"{lbl}'s View  ·  {method.upper()}  ·  {sname.upper()}\n"
                         f"Norm. Improvement = (theta − Theta) / |o − theta|\n"
                         f"mean ± std  over {n_seeds} seed{'s' if n_seeds > 1 else ''}")
                
                plot_heatmap(mean_grid, std_grid, param_grid, param_label, title, 
                             out_path=os.path.join(args.plots, method, f"{sname}_{lbl}_{method}.png"))
    print("\nAll 12 heatmaps saved.")

# --- Main Entry Point ---
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    use_quick = args.quick or (args.n_samples is not None)
    dp_noise_levels = DP_NOISE_LEVELS_QUICK if use_quick else DP_NOISE_LEVELS_FULL
    sup_levels = SUP_LEVELS_QUICK if use_quick else SUP_LEVELS_FULL
    
    print(f"Grid mode : {'QUICK (3x3)' if use_quick else 'FULL (9x9 / 11x11)'}")
    print(f"DP  grid  : {dp_noise_levels}")
    print(f"SUP grid  : {sup_levels}")

    for d in [args.data_root, args.results, os.path.join(args.plots, "dp"), os.path.join(args.plots, "sup")]:
        os.makedirs(d, exist_ok=True)

    print("\nLoading data ...")
    ecfp_tr, ic50_tr, ecfp_va, ic50_va = du.load_data(args.data_path)
    ecfp_tr, ecfp_va = du.fold_input(ecfp_tr, INPUT_SIZE), du.fold_input(ecfp_va, INPUT_SIZE)

    if args.n_samples is not None:
        print(f"Subsampling to {args.n_samples:,} training rows ...")
        ecfp_tr, ic50_tr = subsample(ecfp_tr, ic50_tr, n=args.n_samples, seed=0)

    n_train = ecfp_tr.shape[0]
    print(f"Training set : {n_train:,} samples | {ecfp_tr.shape[1]:,} features | {ic50_tr.shape[1]:,} tasks")

    print("\nPreparing fixed data splits ...")
    splits = prepare_fixed_splits(ecfp_tr, ic50_tr, overlap=args.overlap, root=args.data_root)

    base_conf = make_base_conf(n_train, rounds=args.rounds)
    dp_clip = estimate_dp_clip(base_conf, data_dir=splits["full"], device=device, n_batches=args.clip_batches)

    dp_cells, sup_cells = len(dp_noise_levels)**2, len(sup_levels)**2
    total_runs = args.seed_range * 3 * (dp_cells + sup_cells + 4)
    print(f"\nConfiguration:\n  seeds        : {args.seed_range}\n  rounds       : {args.rounds}\n  overlap      : {args.overlap}")
    print(f"  batch_size   : {base_conf.batch_size}\n  n_train      : {n_train:,}\n  dp_clip      : {dp_clip:.6f}  (auto-estimated)")
    print(f"  DP  grid     : {len(dp_noise_levels)}×{len(dp_noise_levels)} = {dp_cells} cells")
    print(f"  SUP grid     : {len(sup_levels)}×{len(sup_levels)} = {sup_cells} cells")
    print(f"  Total Flower FL runs : ~{total_runs:,}")

    run_all(splits, base_conf, range(args.seed_range), dp_noise_levels, sup_levels, dp_clip, args, loss_fn, device)
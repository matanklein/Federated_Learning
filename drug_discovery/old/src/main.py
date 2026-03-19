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

import argparse
import flwr as fl
import sparsechem as sc
import torch
import numpy as np
import csv
import os
from dataset import get_client_datasets
from client import DrugDiscoveryClient, set_parameters
from utils.split_data import split_with_overlap 
import utils.data_utils as du

# --- 1. Custom Strategy to Extract Global Model ---
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

# --- 2. Standalone Evaluation Function ---
def evaluate_global_model(model, test_dataset, conf, loss_fn, device):
    model.to(device)
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=conf.batch_size, collate_fn=test_dataset.collate
    )
    total_loss = 0.0
    samples = 0
    with torch.no_grad():
        for batch in test_loader:
            b_x, b_y = batch["ecfp"].to(device), batch["ic50"].to(device)
            logits = model(b_x)
            mask = ~torch.isnan(b_y)
            if mask.sum() > 0:
                loss = loss_fn(logits[mask], b_y[mask])
                total_loss += loss.sum().item()
                samples += mask.sum().item()
    return total_loss / max(samples, 1)

# --- 3. Flower Simulation Wrapper ---
def run_fl_experiment(group, hidings, conf, args, loss_fn, device):
    def client_fn(cid: str) -> fl.client.Client:
        k = int(cid)
        h = hidings[group.index(k)]
        train_ds, test_ds = get_client_datasets(args.data_root, k, h, conf)
        
        trunk = sc.Trunk(conf)
        model = sc.TrunkAndHead(conf=conf, trunk=trunk)
        return DrugDiscoveryClient(model, train_ds, test_ds, conf, loss_fn).to_client()

    strategy = SaveModelStrategy(
        target_rounds=args.rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(group),
        min_available_clients=len(group),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(group),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if device.type == 'cpu' else 1.0},
    )
    
    return strategy.final_parameters

# --- Main CLI & Grid Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Price of Privacy Simulation")
    parser.add_argument("--data_path", type=str, default="../data/", help="Path to raw data")
    parser.add_argument("--n_samples", type=int, default=236182, help="Number of training samples")
    parser.add_argument("--seed_range", type=int, default=3, help="Number of seeds to run")
    parser.add_argument("--rounds", type=int, default=200, help="Federated learning rounds")
    parser.add_argument("--data_root", type=str, default="src/data_split/", help="Path for split data shards")
    parser.add_argument("--results", type=str, default="src/results/", help="Path to save CSVs")
    parser.add_argument("--plots", type=str, default="src/plots/", help="Path to save heatmaps")
    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    results_csv = os.path.join(args.results, "ratio_both_hide_flower.csv")
    client_num = 2
    batch_size = int((args.n_samples / client_num) * 0.02)

    conf = sc.ModelConfig(
        input_size         = 32000,
        hidden_sizes       = [40],
        output_size        = 2808,
        batch_size         = batch_size,
        lr                 = 1e-3,
        last_dropout       = 0.2,
        weight_decay       = 1e-5,
        non_linearity      = "relu",
        last_non_linearity = "relu",
    )

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratios = [1]
    
    # Pre-load global raw data for the split utility
    print(f"Loading raw data from {args.data_path}...")
    ecfp_tr, ic50_tr, ecfp_va, ic50_va = du.load_data(args.data_path)
    ecfp_tr, ecfp_va = du.fold_input(ecfp_tr, 32000), du.fold_input(ecfp_va, 32000)

    for seed_init in range(args.seed_range):
        for ratio in ratios:
            np.random.seed(seed_init)
            torch.manual_seed(seed_init)
            
            print(f"\n=== Preparing Data (Seed {seed_init}, Ratio {ratio}) ===")
            # 1. Execute the split utility to generate fresh shards in args.data_root
            split_with_overlap(ratio, ecfp_tr, ic50_tr, root_dir=args.data_root, overlap=2808)

            # Pre-load unsuppressed datasets to evaluate o, theta, and Theta
            _, test_ds_1 = get_client_datasets(args.data_root, k=0, h=0.0, conf=conf)
            _, test_ds_2 = get_client_datasets(args.data_root, k=1, h=0.0, conf=conf)

            # Initialize an untrained model to calculate `o` (Oracle Loss before training)
            base_model = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf))
            o_1 = evaluate_global_model(base_model, test_ds_1, conf, loss_fn, device)
            o_2 = evaluate_global_model(base_model, test_ds_2, conf, loss_fn, device)

            print("\n--- Training Alone (Baselines) ---")
            params_theta_1 = run_fl_experiment([0], [0.0], conf, args, loss_fn, device)
            set_parameters(base_model, params_theta_1)
            theta_1 = evaluate_global_model(base_model, test_ds_1, conf, loss_fn, device)

            params_theta_2 = run_fl_experiment([1], [0.0], conf, args, loss_fn, device)
            set_parameters(base_model, params_theta_2)
            theta_2 = evaluate_global_model(base_model, test_ds_2, conf, loss_fn, device)

            # Define hiding grids (you can adjust this for the 'quick run' vs 'full run')
            hidings = [0.0, 0.2, 0.4, 0.6, 0.8]
            p1_h = len(hidings) * hidings
            p2_h = [h for h in hidings for _ in range(len(hidings))]

            for h1, h2 in zip(p1_h, p2_h):
                print(f"\n--- Collaborative Training -> User-1 hides: {h1*100}%, User-2 hides: {h2*100}% ---")
                
                params_Theta = run_fl_experiment([0, 1], [h1, h2], conf, args, loss_fn, device)
                
                set_parameters(base_model, params_Theta)
                Theta_1 = evaluate_global_model(base_model, test_ds_1, conf, loss_fn, device)
                Theta_2 = evaluate_global_model(base_model, test_ds_2, conf, loss_fn, device)

                # Accuracy Improvement (Absolute value used in denominator as per docstring)
                norm_acc_impr_1 = (theta_1 - Theta_1) / abs(o_1 - theta_1)
                norm_acc_impr_2 = (theta_2 - Theta_2) / abs(o_2 - theta_2)

                print(f"(1) Accuracy improvement: {norm_acc_impr_1}")
                print(f"(2) Accuracy improvement: {norm_acc_impr_2}")

                results = {
                    "ratio": f"{ratio}:1", "hidden": h1, 
                    "o": o_1, "theta": theta_1, "Theta": Theta_1, 
                    "accuracy_improvement": norm_acc_impr_1
                }
                results_2 = {
                    "ratio": f"1:{ratio}", "hidden": h2, 
                    "o": o_2, "theta": theta_2, "Theta": Theta_2, 
                    "accuracy_improvement": norm_acc_impr_2
                }

                with open(results_csv, "a+", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(results.keys()))
                    f.seek(0, os.SEEK_END)
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(results)
                    writer.writerow(results_2)
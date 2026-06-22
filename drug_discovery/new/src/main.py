import os
import sys
import argparse
import csv
import time
import datetime
import numpy as np
import torch
import flwr as fl
import flwr.common as flc
from flwr.server.strategy import FedAvg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Inject local paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

import sparsechem as sc
from client import DrugDiscoveryClient, get_parameters, set_parameters
from dataset import get_client_datasets

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
MECHANISMS = ["sup", "dp"]
SCENARIOS = ["full", "p1", "p2"]

PRIVACY_PARAMS_BY_MECH = {
    "dp": [0.0, 0.001, 0.01, 0.1, 1.0],
    "sup": [0.0, 0.2, 0.4, 0.6, 0.8],
}

# Name mapping for clean outputs
SCENARIO_MAP = {
    "full": "real",
    "p1": "sim_p1",
    "p2": "sim_p2"
}

CLIENT_MAP = {
    "full": {0: "P1", 1: "P2"},
    "p1": {0: "P11", 1: "P12"},
    "p2": {0: "P21", 1: "P22"}
}

class DummyClientProxy(fl.server.client_proxy.ClientProxy):
    def __init__(self, cid):
        try:
            super().__init__(cid)
        except Exception:
            self.cid = cid
    def get_properties(self, ins, timeout, group_id=0): pass
    def get_parameters(self, ins, timeout, group_id=0): pass
    def fit(self, ins, timeout, group_id=0): pass
    def evaluate(self, ins, timeout, group_id=0): pass
    def reconnect(self, ins, timeout, group_id=0): pass

def make_base_conf():
    c = sc.ModelConfig(
        input_size=32000,
        hidden_sizes=[40], # Optimal capacity
        output_size=2808,
        batch_size=64,
        lr=1e-3,
        last_dropout=0.2,
        weight_decay=1e-5,
        non_linearity="relu",
        last_non_linearity="relu",
    )
    c.epochs = 5 # Local epochs per FL round
    return c

def train_and_eval_local_baseline(data_path, client_id, seed, rounds):
    """Establishes Oracle and Alone baselines."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    conf = make_base_conf()
    train_ds, test_ds = get_client_datasets(data_path, client_id, 0.0, 'none', conf, seed)
    
    model = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    dummy_client = DrugDiscoveryClient(model, train_ds, test_ds, conf, loss_fn, 'none', 0.0, 1.0, client_id, True)
    _, _, oracle_metrics = dummy_client.evaluate(get_parameters(model), {})
    
    # Simulate equivalent standalone training (rounds * local_epochs)
    conf.epochs = rounds * 5 
    dummy_client.fit(get_parameters(model), {})
    _, _, alone_metrics = dummy_client.evaluate(get_parameters(dummy_client.model), {})
    
    del model, dummy_client
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        "oracle_acc": oracle_metrics["accuracy"],
        "oracle_loss": oracle_metrics["loss"],
        "alone_acc": alone_metrics["accuracy"],
        "alone_loss": alone_metrics["loss"]
    }

def worker_task(kwargs):
    """Isolated worker process for FL Grid execution."""
    # EXTREMELY IMPORTANT FOR MULTIPROCESSING RUNTIME: 
    # Prevent PyTorch from spawning hundreds of background threads that gridlock the CPU
    torch.set_num_threads(1)
    
    seed = kwargs["seed"]
    scenario = kwargs["scenario"]
    mech = kwargs["mech"]
    p1_val = kwargs["p1_val"]
    p2_val = kwargs["p2_val"]
    data_path = kwargs["data_path"]
    args = kwargs["args"]
    baselines = kwargs["baselines"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    conf = make_base_conf()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    t1, te1 = get_client_datasets(data_path, 0, p1_val, mech, conf, seed)
    t2, te2 = get_client_datasets(data_path, 1, p2_val, mech, conf, seed)
    
    model1 = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf))
    model2 = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf))
    
    c1 = DrugDiscoveryClient(model1, t1, te1, conf, loss_fn, mech, p1_val, 1.0, "0", True)
    c2 = DrugDiscoveryClient(model2, t2, te2, conf, loss_fn, mech, p2_val, 1.0, "1", True)
    
    global_weights = get_parameters(model1)
    strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
    
    proxy1, proxy2 = DummyClientProxy("0"), DummyClientProxy("1")
    
    for rnd in range(1, args.rounds + 1):
        c1.set_parameters(global_weights)
        c2.set_parameters(global_weights)
        
        w1, num1, _ = c1.fit(global_weights, {})
        w2, num2, _ = c2.fit(global_weights, {})
        
        if hasattr(flc, "Status"):
            res1 = flc.FitRes(status=flc.Status(code=flc.Code.OK, message=""), parameters=flc.ndarrays_to_parameters(w1), num_examples=num1, metrics={})
            res2 = flc.FitRes(status=flc.Status(code=flc.Code.OK, message=""), parameters=flc.ndarrays_to_parameters(w2), num_examples=num2, metrics={})
        else:
            res1, res2 = flc.FitRes(parameters=flc.ndarrays_to_parameters(w1), num_examples=num1, metrics={}), flc.FitRes(parameters=flc.ndarrays_to_parameters(w2), num_examples=num2, metrics={})
            
        agg_params, _ = strategy.aggregate_fit(server_round=rnd, results=[(proxy1, res1), (proxy2, res2)], failures=[])
        if agg_params is not None:
            global_weights = flc.parameters_to_ndarrays(agg_params)

    _, _, fed_metrics_1 = c1.evaluate(global_weights, {})
    _, _, fed_metrics_2 = c2.evaluate(global_weights, {})
    
    records = []
    for c_id, fed_metrics in zip([0, 1], [fed_metrics_1, fed_metrics_2]):
        b = baselines[c_id]
        
        acc_denom = abs(b["oracle_acc"] - b["alone_acc"])
        gain_acc = (fed_metrics["accuracy"] - b["alone_acc"]) / acc_denom if acc_denom != 0 else 0.0
        
        loss_denom = abs(b["oracle_loss"] - b["alone_loss"])
        gain_loss = (b["alone_loss"] - fed_metrics["loss"]) / loss_denom if loss_denom != 0 else 0.0
        
        records.append({
            "Seed": seed,
            "Scenario": scenario,
            "Mechanism": mech,
            "Client": c_id,
            "P1_Param": p1_val,
            "P2_Param": p2_val,
            "Acc_Oracle": b["oracle_acc"],
            "Acc_Alone": b["alone_acc"],
            "Acc_Fed": fed_metrics["accuracy"],
            "Gain_Acc": gain_acc,
            "Loss_Oracle": b["oracle_loss"],
            "Loss_Alone": b["alone_loss"],
            "Loss_Fed": fed_metrics["loss"],
            "Gain_Loss": gain_loss
        })
        
    del model1, model2, c1, c2, t1, t2, te1, te2
    gc.collect()
    return records

def generate_outputs(df, csv_dir, plot_dir):
    """
    Splits CSVs by case and generates Mean+STD Heatmaps exclusively for Accuracy.
    Implements a custom Red-White-Green colormap with independent domain scaling 
    to properly visualize the Nash Equilibrium boundary of collaborative learning.
    """
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(os.path.join(plot_dir, "dp"), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, "sup"), exist_ok=True)
    
    # --- DEFINE CUSTOM COLORMAP: Red -> White -> Green ---
    # White centers cleanly at 0.0, highlighting beneficial collaboration (Green) 
    # and privacy-induced utility collapse (Red) without muddy yellow artifacts.
    rwg_cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"])
    
    for mech in df['Mechanism'].unique():
        for scenario in df['Scenario'].unique():
            # 1. Output Individual CSVs
            sub_df = df[(df['Mechanism'] == mech) & (df['Scenario'] == scenario)]
            if sub_df.empty: 
                continue
            
            clean_scenario = SCENARIO_MAP[scenario]
            csv_filename = os.path.join(csv_dir, f"{mech}_{clean_scenario}.csv")
            sub_df.to_csv(csv_filename, index=False)
            print(f"📁 Saved CSV: {csv_filename}")
            
            # 2. Output Heatmaps for Accuracy Gain (Mean + STD)
            for client in df['Client'].unique():
                client_sub = sub_df[sub_df['Client'] == client]
                if client_sub.empty: 
                    continue
                
                # Calculate Mean and Standard Deviation across the 10 seeds
                pivot_mean = client_sub.pivot_table(index='P1_Param', columns='P2_Param', values='Gain_Acc', aggfunc='mean')
                pivot_std = client_sub.pivot_table(index='P1_Param', columns='P2_Param', values='Gain_Acc', aggfunc='std').fillna(0)
                
                # --- DYNAMIC TWO-SLOPE NORMALIZATION ---
                v_min = pivot_mean.values.min()
                v_max = pivot_mean.values.max()
                
                # TwoSlopeNorm strictly requires vmin < vcenter < vmax. 
                # We inject a negligible mathematical buffer if a specific client's 
                # matrix happens to be entirely positive or entirely negative.
                if v_min >= 0:
                    v_min = -1e-3
                if v_max <= 0:
                    v_max = 1e-3
                    
                norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
                # ---------------------------------------

                # Create custom text annotations: "Mean \n ±STD"
                annot_matrix = np.empty(pivot_mean.shape, dtype=object)
                for i in range(pivot_mean.shape[0]):
                    for j in range(pivot_mean.shape[1]):
                        annot_matrix[i, j] = f"{pivot_mean.iloc[i, j]:.3f}\n±{pivot_std.iloc[i, j]:.3f}"
                
                plt.figure(figsize=(9, 7))
                
                # Render heatmap with custom palette and independent domain scaling
                sns.heatmap(
                    pivot_mean, 
                    annot=annot_matrix, 
                    fmt="", 
                    cmap=rwg_cmap,
                    norm=norm,
                    annot_kws={"size": 10},
                    cbar_kws={'label': 'Empirical Accuracy Gain'}
                )
                
                clean_client = CLIENT_MAP[scenario][client]
                
                plt.title(f"Accuracy Gain - {clean_scenario.upper()} - {clean_client} ({mech.upper()})\n(Mean ± STD over 10 Seeds)")
                plt.xlabel(f"P2 Privacy Parameter ({mech.upper()})")
                plt.ylabel(f"P1 Privacy Parameter ({mech.upper()})")
                
                # Retaining your target nomenclature structure
                plot_filename = os.path.join(plot_dir, mech, f"{clean_scenario}_{clean_client}_{mech}.png")
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Saved Plot: {plot_filename}")

def save_runtime_report(runtime_path, formatted_time, total_seconds):
    os.makedirs(os.path.dirname(runtime_path), exist_ok=True)
    with open(runtime_path, "w", encoding="utf-8") as handle:

        handle.write(f"Total Runtime: {formatted_time} (HH:MM:SS)\n")
        handle.write(f"Total Seconds: {total_seconds}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_csv_dir", type=str, default="src/Results/CSVs/")
    parser.add_argument("--out_plot_dir", type=str, default="src/Results/Plots/")
    parser.add_argument("--out_runtime_file", type=str, default="src/Results/runtime.txt")
    parser.add_argument("--seed_range", type=int, default=10) # Set to 10 seeds
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    start_time = time.time()
    print(f"===========================================================")
    print(f" 🚀 STARTING FULL EXPERIMENT SUITE ({args.seed_range} Seeds)")
    print(f"===========================================================")

    tasks = []
    for seed in range(1, args.seed_range + 1):
        for scenario in SCENARIOS:
            data_path = os.path.join(args.data_root, f"{scenario}/data_2_split/")
            if not os.path.exists(data_path):
                print(f"Skipping {scenario}: Path {data_path} does not exist.")
                continue
                
            print(f"\n--- Computing Local Baselines [Seed {seed}/10] [{scenario}] ---")
            baselines = {
                0: train_and_eval_local_baseline(data_path, 0, seed, args.rounds),
                1: train_and_eval_local_baseline(data_path, 1, seed, args.rounds)
            }
            
            for mech in MECHANISMS:
                privacy_params = PRIVACY_PARAMS_BY_MECH[mech]
                for p1_val in privacy_params:
                    for p2_val in privacy_params:
                        tasks.append({
                            "seed": seed, "scenario": scenario, "mech": mech,
                            "p1_val": p1_val, "p2_val": p2_val,
                            "data_path": data_path, "args": args, "baselines": baselines
                        })

    if not tasks:
        print("No tasks generated. Check your --data_root path.")
        return

    all_records = []
    print(f"\n⚡ Dispatching {len(tasks)} FL combinations across {args.workers} CPUs...")
    
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            try:
                records = future.result()
                all_records.extend(records)
                completed += 1
                if completed % 50 == 0:
                    print(f"   ... Progress: {completed}/{len(tasks)} simulations completed.")
            except Exception as e:
                print(f"Grid Task Failed: {e}")

    if all_records:
        df = pd.DataFrame(all_records)
        generate_outputs(df, args.out_csv_dir, args.out_plot_dir)

    # Calculate and Output Total Runtime
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    formatted_time = str(datetime.timedelta(seconds=total_seconds))
    save_runtime_report(args.out_runtime_file, formatted_time, total_seconds)
    
    print(f"\n===========================================================")
    print(f" 🎉 EXPERIMENT COMPLETE")
    print(f" ⏱️  Total Runtime: {formatted_time} (HH:MM:SS)")
    print(f"===========================================================")

if __name__ == "__main__":
    main()
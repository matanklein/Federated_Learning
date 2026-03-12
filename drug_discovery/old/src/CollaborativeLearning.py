import sparsechem as sc
from packages.collaborative.participant import Server, Client
import packages.utils.data_utils as du
import random
import numpy as np
from tqdm import tqdm
import torch
from types import SimpleNamespace
import packages.utils.data_utils as du
import numpy as np
import torch
import CollaborativeLearning as CoL
import sparsechem as sc
import csv, os
from itertools import product

def apply_dp_to_gradients(model, conf):
    """
    Apply differential privacy (gradient clipping + noise) to trunk gradients only.
    Pure PyTorch implementation - NO OPACUS DEPENDENCY.
    """
    if not hasattr(conf, 'dp_clip_norm') or not hasattr(conf, 'dp_noise_std'):
        return
    
    clip_norm = getattr(conf, 'dp_clip_norm', None)
    noise_std = getattr(conf, 'dp_noise_std', 0.0)
    
    # Skip if no DP configured
    if clip_norm is None or noise_std == 0.0:
        return
    
    print(f"  [DP] Applying noise_std={noise_std}, clip_norm={clip_norm}")
    
    # Get trunk parameters with gradients
    trunk_params_with_grad = [p for p in model.trunk.parameters() if p.grad is not None]
    if not trunk_params_with_grad:
        return
    
    # Step 1: Compute L2 norm of trunk gradients
    total_norm = 0.0
    for p in trunk_params_with_grad:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = (total_norm ** 0.5)
    
    # Step 2: Clip gradients if needed
    clip_coef = clip_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in trunk_params_with_grad:
            p.grad.data.mul_(clip_coef)
    
    # Step 3: Add Gaussian noise to clipped gradients
    for p in trunk_params_with_grad:
        noise = torch.randn_like(p.grad.data) * noise_std * clip_norm
        p.grad.data.add_(noise)
def hide_h_percent(X_train, Y_train, h, seed=None):
    """Hide h% of samples (suppression)."""
    if seed is not None:
        np.random.seed(seed)
    
    X_train_num = X_train.shape[0]
    print(f"All data: {X_train_num}")
    remaining_data_num = int((1 - h) * X_train_num)
    shared_data_indices = random.sample(range(X_train_num), remaining_data_num)
    print(f"After hiding {int(h * 100)}%: {remaining_data_num}")
    return X_train[shared_data_indices], Y_train[shared_data_indices]



def run_server(group, hide, conf_list, rounds, server_data, client_data_path, same_head=False):
    """
    Run federated learning server with multiple clients.
    
    Args:
        group: List of client IDs
        hide: List of hiding percentages for each client (suppression)
        conf_list: List of configurations for each client, or single config for all
        rounds: Number of training rounds
        server_data: Tuple of (ecfp_va, ic50_va) for server validation
        client_data_path: Path to client data
        same_head: Whether clients share the same head
    
    Returns:
        loss_o_arr: Initial losses for each client
        loss_arr: Final losses for each client
    """
    
    # Handle single config case
    if isinstance(conf_list, sc.ModelConfig):
        conf_list = [conf_list] * len(group)
    
    if len(conf_list) != len(group):
        raise ValueError(f"Expected {len(conf_list)} configs for {len(group)} clients")
    
    # Use first config for trunk initialization
    trunk_conf = conf_list[0]
    trunk = sc.Trunk(trunk_conf)
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    # Initialize clients
    clients = []
    train_path = client_data_path + "data_2_split/"
    
    
    # Create shared model if same_head
    if same_head:
        model = sc.TrunkAndHead(conf=trunk_conf, trunk=trunk)
    
    for i, (k, h) in enumerate(zip(group, hide)):
        client_conf = conf_list[i]
        
        # Load client train data
        X_train, Y_train = du.load_ratio_split_data(train_path, k, train=True)
        if h != 0:
            X_train, Y_train = hide_h_percent(X_train, Y_train, h)
        dataset = sc.SparseDataset(X_train, Y_train)
        
        # Load client test data
        X_data, Y_data = du.load_ratio_split_data(train_path, k, train=False)
        dataset_va = sc.SparseDataset(X_data, Y_data)
        
        # Update client configuration
        client_conf.output_size = Y_train.shape[1]
        client_conf.batch_size = int(X_train.shape[0] * 0.02)
        print(f"Client {k} - Batch size: {client_conf.batch_size}")
        
        # Create model (reuse shared model if same_head)
        if not same_head:
            model = sc.TrunkAndHead(conf=client_conf, trunk=trunk)
        
        print(f"Client {k} - DP noise: {getattr(client_conf, 'dp_noise_std', 0.0)}")
        print(f"Client {k} - DP clip: {getattr(client_conf, 'dp_clip_norm', 'None')}")
        print(f"Trunk + Head architecture (client-{k}):")
        print(model)

        print(f"Client {k} - Config has dp_noise_std: {hasattr(client_conf, 'dp_noise_std')}")
        if hasattr(client_conf, 'dp_noise_std'):
            print(f"Client {k} - Config dp_noise_std VALUE: {client_conf.dp_noise_std}")
        
        client = Client(model, conf=client_conf, dataset=dataset, dataset_va=dataset_va)
        clients.append(client)
    
    # Evaluate before training
    print("Evaluating initial performance...")
    loss_o_arr = []
    for i, c in enumerate(clients):
        loss, _ = c.eval(on_train=False)
        loss_o_arr.append(loss)
        print(f"Client {group[i]} initial loss: {loss:.4f}")
    loss_o_arr = np.array(loss_o_arr)
    
    # Training loop - SIMPLIFIED (no server object needed!)
    print(f"Starting federated training for {rounds} rounds...")
    for r in tqdm(range(rounds), desc="Training rounds"):
        for i, client in enumerate(clients):
            # Client gets next batch
            batch = client.get_next_batch()
            
            # Client trains on batch (calculates gradients on shared trunk)
            client.train(batch)
            
            # Apply DP to trunk gradients (clip + noise)
            apply_dp_to_gradients(client.model, conf_list[i])
            
            # Client updates ALL weights (trunk + head)
            # Since trunk is shared, this updates it for everyone
            client.update_weights()
            
            # Client zeros gradients
            client.zero_grad()
    
    # Evaluate after training
    print("Evaluating final performance...")
    loss_arr = []
    for i, c in enumerate(clients):
        loss, _ = c.eval(on_train=False)
        loss_arr.append(loss)
        print(f"Client {group[i]} final loss: {loss:.4f}")
    loss_arr = np.array(loss_arr)
    
    return loss_o_arr, loss_arr
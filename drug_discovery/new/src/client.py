import flwr as fl
import torch
import numpy as np
import os
from collections import OrderedDict
import sparsechem as sc

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class DrugDiscoveryClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, conf, loss_fn, privacy_mode, privacy_param, dp_clip, cid, head_dir):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.conf = conf
        self.loss_fn = loss_fn
        self.privacy_mode = privacy_mode
        self.privacy_param = privacy_param
        self.dp_clip = dp_clip
        self.cid = cid
        self.head_dir = head_dir
        
        self.head_path = os.path.join(head_dir, f"head_{cid}.pt")
        self.optim_path = os.path.join(head_dir, f"optim_{cid}.pt") # New Optimizer State Path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if os.path.exists(self.head_path):
            self.model.load_state_dict(torch.load(self.head_path, map_location=self.device), strict=False)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.trunk.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.trunk.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.trunk.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Re-initialize the optimizer. Do NOT load local momentum states for 
        # parameters that have just been aggregated globally by the server.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
            
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.conf.batch_size, shuffle=True, collate_fn=sc.sparse_collate
        )
        
        self.model.train()
        
        # Iterate over the entire dataset (1 local epoch) to ensure gradient steps 
        # strictly correlate with the privacy parameter 'p' (dataset size reduction).
        for batch in train_loader:
            b_x = torch.sparse_coo_tensor(
                batch["x_ind"], batch["x_data"], size=[batch["batch_size"], self.conf.input_size]
            ).to(self.device)
            
            y_ind, y_data = batch["y_ind"].to(self.device), batch["y_data"].to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(b_x)
            
            logits_subset = logits[y_ind[0], y_ind[1]]
            if logits_subset.numel() > 0:
                loss = self.loss_fn(logits_subset, y_data).mean()
                loss.backward()

                # DP Processing: Clip and add noise per batch
                if self.privacy_mode == 'dp' and self.privacy_param > 0.0 and self.dp_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.trunk.parameters(), max_norm=self.dp_clip)
                    for p in self.model.trunk.parameters():
                        if p.grad is not None:
                            noise = torch.normal(mean=0.0, std=self.privacy_param * self.dp_clip, size=p.grad.shape).to(self.device)
                            p.grad += noise

                optimizer.step()

        # Save ONLY the personalized head state. 
        head_state = {k: v for k, v in self.model.state_dict().items() if 'trunk' not in k}
        torch.save(head_state, self.head_path)

        # Free memory to prevent Seed crashing
        torch.cuda.empty_cache()

        # By returning len(self.train_dataset), Flower's FedAvg natively applies 
        # a smaller aggregation weight to clients who suppressed data.
        return self.get_parameters(config={}), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        # Your existing evaluate code remains exactly the same
        self.set_parameters(parameters)
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.conf.batch_size, collate_fn=sc.sparse_collate
        )
        self.model.eval()
        total_loss, samples = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                b_x = torch.sparse_coo_tensor(
                    batch["x_ind"], batch["x_data"], size=[batch["batch_size"], self.conf.input_size]
                ).to(self.device)
                y_ind, y_data = batch["y_ind"].to(self.device), batch["y_data"].to(self.device)
                
                logits = self.model(b_x)
                logits_subset = logits[y_ind[0], y_ind[1]]
                if logits_subset.numel() > 0:
                    loss = self.loss_fn(logits_subset, y_data)
                    total_loss += loss.sum().item()
                    samples += logits_subset.numel()
                
        avg_loss = total_loss / max(samples, 1)
        return float(avg_loss), len(self.test_dataset), {"loss": float(avg_loss)}
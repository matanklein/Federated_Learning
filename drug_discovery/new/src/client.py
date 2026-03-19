import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
import sparsechem as sc

class DrugDiscoveryClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, conf, loss_fn, privacy_mode, privacy_param, dp_clip=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.conf = conf
        self.loss_fn = loss_fn
        self.privacy_mode = privacy_mode
        self.privacy_param = privacy_param
        self.dp_clip = dp_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        # FIX 1: Only extract the Trunk parameters for Server Aggregation
        return [val.cpu().numpy() for _, val in self.model.trunk.state_dict().items()]

    def set_parameters(self, parameters):
        # FIX 1: Only overwrite the local Trunk. The local Head remains isolated.
        params_dict = zip(self.model.trunk.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.trunk.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.conf.batch_size, shuffle=True, collate_fn=sc.sparse_collate
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        
        self.model.train()
        for batch in train_loader:
            b_x = torch.sparse_coo_tensor(
                batch["x_ind"], 
                batch["x_data"], 
                size=[batch["batch_size"], self.conf.input_size]
            ).to(self.device)
            
            y_ind = batch["y_ind"].to(self.device)
            y_data = batch["y_data"].to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(b_x)
            
            # Sparse loss calculation
            logits_subset = logits[y_ind[0], y_ind[1]]
            loss = self.loss_fn(logits_subset, y_data).mean()
            loss.backward()

            # Apply true DP-SGD to the Trunk parameters during backprop
            if self.privacy_mode == 'dp' and self.privacy_param > 0.0 and self.dp_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.trunk.parameters(), max_norm=self.dp_clip)
                for p in self.model.trunk.parameters():
                    if p.grad is not None:
                        noise = torch.normal(
                            mean=0.0, 
                            std=self.privacy_param * self.dp_clip, 
                            size=p.grad.shape
                        ).to(self.device)
                        p.grad += noise

            optimizer.step()

        return self.get_parameters(config={}), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.conf.batch_size, 
            collate_fn=sc.sparse_collate
        )
        
        self.model.eval()
        total_loss = 0.0
        samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                b_x = torch.sparse_coo_tensor(
                    batch["x_ind"], 
                    batch["x_data"], 
                    size=[batch["batch_size"], self.conf.input_size]
                ).to(self.device)
                
                y_ind = batch["y_ind"].to(self.device)
                y_data = batch["y_data"].to(self.device)
                
                logits = self.model(b_x)
                
                logits_subset = logits[y_ind[0], y_ind[1]]
                if logits_subset.numel() > 0:
                    loss = self.loss_fn(logits_subset, y_data)
                    total_loss += loss.sum().item()
                    samples += logits_subset.numel()
                
        avg_loss = total_loss / max(samples, 1)
        return float(avg_loss), len(self.test_dataset), {"loss": float(avg_loss)}
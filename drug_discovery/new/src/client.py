import flwr as fl
import torch
import os
import sparsechem as sc
from collections import OrderedDict

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class DrugDiscoveryClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, conf, loss_fn, privacy_mode, privacy_param, dp_clip, cid, head_dir, share_all_layers, num_workers=0, pin_memory=False):
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
        self.share_all_layers = share_all_layers # New property
        self.num_workers = max(0, int(num_workers))
        self.pin_memory = bool(pin_memory)
        
        self.head_path = os.path.join(head_dir, f"head_{cid}.pt")
        self.optim_path = os.path.join(head_dir, f"optim_{cid}.pt")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Only load a personalized local head if we are NOT sharing all layers
        if not self.share_all_layers and os.path.exists(self.head_path):
            self.model.load_state_dict(torch.load(self.head_path, map_location=self.device), strict=False)

    def _loader_kwargs(self, shuffle=False):
        kwargs = {
            "batch_size": self.conf.batch_size,
            "collate_fn": sc.sparse_collate,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def get_parameters(self, config):
        if self.share_all_layers:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return [val.cpu().numpy() for _, val in self.model.trunk.state_dict().items()]

    def set_parameters(self, parameters):
        if self.share_all_layers:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            params_dict = zip(self.model.trunk.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.trunk.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
            
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, **self._loader_kwargs(shuffle=True)
        )
        
        self.model.train()
        
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

                if self.privacy_mode == 'dp' and self.privacy_param > 0.0 and self.dp_clip is not None:
                    # Choose which parameters to clip based on the sharing strategy
                    target_params = self.model.parameters() if self.share_all_layers else self.model.trunk.parameters()
                    
                    torch.nn.utils.clip_grad_norm_(target_params, max_norm=self.dp_clip)
                    
                    effective_batch_size = max(logits_subset.numel(), 1)
                    
                    for p in target_params:
                        if p.grad is not None:
                            scaled_std = (self.privacy_param * self.dp_clip) / effective_batch_size
                            noise = torch.normal(mean=0.0, std=scaled_std, size=p.grad.shape).to(self.device)
                            p.grad += noise

                optimizer.step()

        # Only save a personalized head state if we are NOT sharing all layers globally
        if not self.share_all_layers:
            head_state = {k: v for k, v in self.model.state_dict().items() if 'trunk' not in k}
            torch.save(head_state, self.head_path)

        return self.get_parameters(config={}), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, **self._loader_kwargs(shuffle=False)
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
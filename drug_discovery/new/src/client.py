import flwr as fl
import torch
import os
import numpy as np
import sparsechem as sc
from collections import OrderedDict
from sklearn.metrics import accuracy_score

def get_parameters(model):
    """
    Extracts ALL network weights (Both the 40-neuron Trunk and the 2808-neuron Head).
    Because all tasks overlap, nothing is kept private.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    """
    Overwrites ALL network weights globally from the FedAvg server.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class DrugDiscoveryClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, conf, loss_fn, privacy_mode, privacy_param, dp_clip, cid, share_all_layers=True):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.conf = conf
        self.loss_fn = loss_fn
        self.privacy_mode = privacy_mode
        self.privacy_param = privacy_param
        self.dp_clip = dp_clip
        self.cid = cid
        
        # We enforce True here to mathematically guarantee 100% parameter sharing
        self.share_all_layers = True 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        if self.privacy_mode == 'dp' and self.privacy_param >= 1.0:
            return self.get_parameters(config={}), len(self.train_dataset), {}

        g = torch.Generator()
        if hasattr(self.conf, 'seed'):
            g.manual_seed(self.conf.seed) 

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.conf.batch_size, shuffle=True, collate_fn=sc.sparse_collate, generator=g
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        
        self.model.train()
        for _ in range(self.conf.epochs):
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

                    # Differential Privacy Injection (Only for p < 1.0)
                    if self.privacy_mode == 'dp' and self.privacy_param > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.dp_clip)
                        
                        sigma = self.privacy_param / (1.0 - self.privacy_param)
                        noise_scale = self.dp_clip * sigma
                        
                        for param in self.model.parameters():
                            if param.grad is not None:
                                noise = torch.normal(
                                    mean=0.0, 
                                    std=noise_scale, 
                                    size=param.grad.size(), 
                                    device=self.device
                                )
                                param.grad.add_(noise)

                    optimizer.step()

        return self.get_parameters(config={}), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.conf.batch_size, shuffle=False, collate_fn=sc.sparse_collate
        )
        self.model.eval()
        total_loss, samples = 0.0, 0
        
        preds_batches, targets_batches = [], []
        output_size = int(self.conf.output_size)

        with torch.no_grad():
            for batch in test_loader:
                batch_size = int(batch["batch_size"])
                b_x = torch.sparse_coo_tensor(
                    batch["x_ind"], batch["x_data"], size=[batch_size, self.conf.input_size]
                ).to(self.device)
                y_ind, y_data = batch["y_ind"].to(self.device), batch["y_data"].to(self.device)
                
                logits = self.model(b_x)
                logits_subset = logits[y_ind[0], y_ind[1]]
                
                if logits_subset.numel() > 0:
                    loss = self.loss_fn(logits_subset, y_data)
                    total_loss += loss.sum().item()
                    samples += logits_subset.numel()

                logits_sig = torch.sigmoid(logits).cpu().numpy()
                dense_targets = np.full((batch_size, output_size), np.nan, dtype=float)
                
                if y_ind.numel() > 0:
                    rows, cols = y_ind[0].cpu().numpy().astype(int), y_ind[1].cpu().numpy().astype(int)
                    dense_targets[rows, cols] = y_data.cpu().numpy()

                preds_batches.append(logits_sig)
                targets_batches.append(dense_targets)

        # Calculate Global Accuracy
        preds_all = np.vstack(preds_batches) if preds_batches else np.zeros((0, output_size))
        targets_all = np.vstack(targets_batches) if targets_batches else np.zeros((0, output_size))

        global_acc = 0.0
        if preds_all.size > 0:
            preds_bin_all = (preds_all > 0.5).astype(int)
            targets_all_zero = np.nan_to_num(targets_all, nan=0.0).astype(int)
            global_acc = accuracy_score(targets_all_zero.flatten(), preds_bin_all.flatten())

        avg_loss = total_loss / max(1, samples)
        return float(avg_loss), samples, {"accuracy": float(global_acc), "loss": float(avg_loss)}
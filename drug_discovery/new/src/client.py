import flwr as fl
import torch
import numpy as np
from collections import OrderedDict

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

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
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.conf.batch_size, shuffle=True, collate_fn=self.train_dataset.collate
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        
        self.model.train()
        for batch in train_loader:
            b_x, b_y = batch["ecfp"].to(self.device), batch["ic50"].to(self.device)
            optimizer.zero_grad()
            logits = self.model(b_x)
            mask = ~torch.isnan(b_y)
            loss = self.loss_fn(logits[mask], b_y[mask])
            loss.mean().backward()
            optimizer.step()

        # Extract updated weights
        updated_parameters = self.get_parameters(config={})

        # Apply Local Differential Privacy if mode is 'dp'
        if self.privacy_mode == 'dp' and self.privacy_param > 0.0:
            noised_parameters = []
            for param in updated_parameters:
                # Add Gaussian noise proportional to privacy_param
                # Modify the scale (std dev) here if your original article used a specific sensitivity bound
                noise = np.random.normal(loc=0.0, scale=self.privacy_param, size=param.shape)
                noised_parameters.append(param + noise)
            updated_parameters = noised_parameters

        return updated_parameters, len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.conf.batch_size, 
            collate_fn=self.test_dataset.collate
        )
        
        self.model.eval()
        total_loss = 0.0
        samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                b_x = batch["ecfp"].to(self.device)
                b_y = batch["ic50"].to(self.device)
                logits = self.model(b_x)
                mask = ~torch.isnan(b_y)
                loss = self.loss_fn(logits[mask], b_y[mask])
                total_loss += loss.sum().item()
                samples += mask.sum().item()
                
        avg_loss = total_loss / max(samples, 1)
        return float(avg_loss), len(self.test_dataset), {"loss": float(avg_loss)}
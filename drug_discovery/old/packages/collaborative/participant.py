import torch
import sparsechem as sc
from torch.utils.data import DataLoader
import itertools as it
from copy import deepcopy
try:
    from opacus import PrivacyEngine
    try:
        import torchcsprng
        _HAS_CSPRNG = True
    except ImportError:
        _HAS_CSPRNG = False
except ImportError:
    PrivacyEngine = None
    _HAS_CSPRNG = False

class Participant:
    def __init__(self, model, conf, dataset, dataset_va=None, sampler=None, loss=None, optimizer=None, dev="cpu"):
        self.model = model.to(dev)
        self.conf = conf
        self.dataset = dataset
        self.dataset_va = dataset_va
        self.dev = dev

        # loaders
        if sampler is not None:
            if dataset:
                self.data_loader = DataLoader(dataset, sampler=sampler, batch_size=conf.batch_size,
                                              num_workers=0, collate_fn=sc.sparse_collate, drop_last=False)
            if dataset_va:
                self.data_loader_va = DataLoader(dataset_va, sampler=sampler, batch_size=conf.batch_size,
                                                 num_workers=0, collate_fn=sc.sparse_collate, drop_last=False)
        else:
            if dataset:
                self.data_loader = DataLoader(dataset, batch_size=conf.batch_size,
                                              num_workers=0, collate_fn=sc.sparse_collate, shuffle=True, drop_last=False)
            if dataset_va:
                self.data_loader_va = DataLoader(dataset_va, batch_size=conf.batch_size,
                                                 num_workers=0, collate_fn=sc.sparse_collate, drop_last=False)
        if dataset:
            self.cyclic_loader = it.cycle(iter(self.data_loader))

        # Opacus requires mean reduction
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean") if loss is None else loss

        # DP scope
        self.dp_delta = float(getattr(conf, "dp_delta", 1e-5))
        self.dp_scope = getattr(conf, "dp_scope", "head")  # "head" | "none"
        self.dp_sigma = float(getattr(conf, "dp_noise_std", 0.0))
        self.dp_clip  = float(getattr(conf, "dp_clip", 1.0))
        self.dp_enabled = (self.dp_sigma > 0.0) and (self.dp_scope == "head")
        self.privacy_engine = None

        # optimizer
        if optimizer is None:
            if conf.optimizer == "SGD":
                params = self.model.parameters() if not self.dp_enabled else self.model.head.parameters()
                optimizer = torch.optim.SGD(params, lr=conf.lr)
            elif conf.optimizer == "ADAM":
                params = self.model.parameters() if not self.dp_enabled else self.model.head.parameters()
                optimizer = torch.optim.Adam(params, lr=conf.lr, weight_decay=conf.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer {conf.optimizer}")
        self.optimizer = optimizer

        # If DP enabled, freeze trunk (sparse layer unsupported by Opacus)
        if self.dp_enabled:
            for p in self.model.trunk.parameters():
                p.requires_grad = False

            if PrivacyEngine is None:
                raise RuntimeError("Opacus not installed but dp_noise_std > 0. Install `opacus`.")
            # Use secure RNG for production-grade noise (slower but proper)
            # self.privacy_engine = PrivacyEngine(secure_mode=True)
            # # Wrap ONLY the head (dense)
            # self.model.head, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            #     module=self.model.head,
            #     optimizer=self.optimizer,
            #     data_loader=self.data_loader,
            #     noise_multiplier=self.dp_sigma,
            #     max_grad_norm=self.dp_clip,
            # )
                # decide secure_mode based on availability & config
            dp_secure = bool(getattr(self.conf, "dp_secure", True))  # prefer secure if available
            secure_mode = dp_secure and _HAS_CSPRNG
            if dp_secure and not _HAS_CSPRNG:
                print("[DP] torchcsprng not found -> using secure_mode=False (fast, non-cryptographic). "
                      "Install `torchcsprng` matching your torch to enable secure RNG.")

            self.privacy_engine = PrivacyEngine(secure_mode=secure_mode)

            # Wrap ONLY the head (dense)
            self.model.head, self.optimizer, self.data_loader = self.privacy_engine.make_private(
                module=self.model.head,
                optimizer=self.optimizer,
                data_loader=self.data_loader,
                noise_multiplier=self.dp_sigma,   # == conf.dp_noise_std
                max_grad_norm=self.dp_clip,      # == conf.dp_clip
            )
            self.cyclic_loader = it.cycle(iter(self.data_loader))


    def get_next_batch(self):
        return next(self.cyclic_loader)

    def _forward_sparse_then_head(self, b):
        X = torch.sparse_coo_tensor(
            b["x_ind"], b["x_data"],
            size=[b["batch_size"], self.conf.input_size],
            device=self.dev
        )
        y_ind = b["y_ind"].to(self.dev)
        y = (b["y_data"].to(self.dev) + 1) / 2.0

        # trunk forward (sparse) – no per-sample grad through here
        with torch.no_grad():
            feats = self.model.trunk(X)
        logits_all = self.model.head(feats)
        logits = logits_all[y_ind[0], y_ind[1]]
        return logits, y

    def _forward_full(self, b):
        X = torch.sparse_coo_tensor(
            b["x_ind"], b["x_data"],
            size=[b["batch_size"], self.conf.input_size],
            device=self.dev
        )
        y_ind = b["y_ind"].to(self.dev)
        y = (b["y_data"].to(self.dev) + 1) / 2.0
        logits_all = self.model(X)
        logits = logits_all[y_ind[0], y_ind[1]]
        return logits, y

    def train(self, b):
        self.model.train()
        if self.dp_enabled:
            logits, y = self._forward_sparse_then_head(b)
        else:
            logits, y = self._forward_full(b)
        loss = self.loss(logits, y)
        loss.backward()
        return float(loss.detach().cpu())

    def eval(self, on_train=True):
        self.model.eval()
        if not self.loss:
            raise RuntimeError("No loss function was given.")
        if on_train:
            results = sc.evaluate_binary(self.model, self.data_loader, self.loss, self.dev)
        else:
            if not hasattr(self, "data_loader_va"):
                raise RuntimeWarning("No validation dataloader.")
            results = sc.evaluate_binary(self.model, self.data_loader_va, self.loss, self.dev)
        aucs = results["aucs"].mean()
        print(f"\tloss={results['logloss']:.5f}\taucs={aucs:.5f}")
        return results["logloss"].numpy().item(), aucs

    def update_weights(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def current_epsilon(self):
        if self.privacy_engine is None:
            return None
        return self.privacy_engine.accountant.get_epsilon(delta=self.dp_delta)


class Server(Participant):
    def __init__(self, model, conf, dataset=None, sampler=None, loss=None):
        # Force NO-DP on server (server.model is a Trunk; no head)
        conf_srv = deepcopy(conf)
        conf_srv.dp_noise_std = 0.0
        conf_srv.dp_scope = "none"
        if conf.optimizer == "SGD":
            opt = torch.optim.SGD(model.parameters(), lr=conf.lr)
        elif conf.optimizer == "ADAM":
            opt = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {conf.optimizer}")
        super().__init__(model=model, conf=conf_srv, dataset=dataset, sampler=sampler, loss=loss, optimizer=opt)


class Client(Participant):
    def __init__(self, model, conf, dataset, dataset_va=None, sampler=None, loss=None):
        if conf.optimizer == "SGD":
            params = model.parameters() if not (float(getattr(conf, "dp_noise_std", 0.0)) > 0.0 and getattr(conf, "dp_scope", "head") == "head") else model.head.parameters()
            opt = torch.optim.SGD(params, lr=conf.lr)
        elif conf.optimizer == "ADAM":
            params = model.parameters() if not (float(getattr(conf, "dp_noise_std", 0.0)) > 0.0 and getattr(conf, "dp_scope", "head") == "head") else model.head.parameters()
            opt = torch.optim.Adam(params, lr=conf.lr, weight_decay=conf.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {conf.optimizer}")
        super().__init__(model=model, conf=conf, dataset=dataset, dataset_va=dataset_va, sampler=sampler, loss=loss, optimizer=opt)

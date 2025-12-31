import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from typing import Dict, List
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from evals.eval_utils import _need_cache, _cleanup_cache_dir, _load_cached

class Evaluator:
    def __init__(self, target_encoder, probe_train_loader, probe_eval_loader, attributes, params, logger) -> None:
        self.params = params
        self.logger = logger
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = target_encoder.to(self.local_rank)
        self.train_loader = probe_train_loader
        self.val_loader = probe_eval_loader
        self.attributes = attributes
        self.ce = nn.CrossEntropyLoss()
    
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger()
        if self.global_rank > 0:
            self.logger.setLevel(logging.ERROR)

        self.epochs_run = 0

    def train_and_eval_probe(self, cache_dir=None, delete_cache: bool = True,) -> Dict[str, float]:

        if cache_dir is None:
            cache_dir = self.params['meta']['cache_dir']
        os.makedirs(cache_dir, exist_ok=True)
        self.model.eval()
        probe = self._train_probe(cache_dir)
        return self._eval_probe(probe, cache_dir, delete_cache=delete_cache)

    def _train_probe(self, cache_dir) -> nn.ModuleDict:
        device = next(self.model.parameters()).device

        # cache + load train split
        if _need_cache(cache_dir, "train"):
            self._extract_and_cache(self.model, self.train_loader, device, cache_dir, "train")
        x_train, a_train = _load_cached(cache_dir, "train", map_location="cpu")

        train_loader_cached = DataLoader(
            TensorDataset(x_train, a_train),
            batch_size=self.params['probe']['batch_size'], shuffle=True, drop_last=False, pin_memory=True,)

        probe = nn.ModuleDict({a.name: nn.Linear(self.model.vit.embed_dim, a.num_classes) for a in self.attributes}).to(device)

        opt = torch.optim.Adam(probe.parameters(), lr=self.params['probe']['lr'])

        for _ in range(self.params['probe']['epochs']):
            probe.train()
            for emb, labels in train_loader_cached:
                emb = emb.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                loss = 0.0
                for spec in self.attributes:
                    logits = probe[spec.name](emb)
                    loss = loss + self.ce(logits, labels)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        return probe


    @torch.no_grad()
    def _eval_probe(self, probe: nn.ModuleDict, cache_dir: str, delete_cache: bool = True,) -> Dict[str, float]:

        device = next(self.model.parameters()).device

        if _need_cache(cache_dir, "test"):
            self._extract_and_cache(self.model, self.val_loader, device, cache_dir, "test")
        x_test, a_test = _load_cached(cache_dir, "test", map_location="cpu")

        test_loader_cached = DataLoader(TensorDataset(x_test, a_test),
                        batch_size=self.params['probe']['batch_size'], shuffle=False, drop_last=False, pin_memory=True,)

        probe.eval()
        correct1 = {a.name: 0 for a in self.attributes}
        correct5 = {a.name: 0 for a in self.attributes}
        counts   = {a.name: 0 for a in self.attributes}

        for emb, labels in test_loader_cached:
            emb = emb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            for spec in self.attributes:
                # same note as train: assumes labels match each head
                scores = probe[spec.name](emb)
                pred1 = scores.argmax(dim=1)
                correct1[spec.name] += int((pred1 == labels).sum().item())

                if self.params['probe']['log_top5']:
                    k = min(5, spec.num_classes)
                    pred5 = scores.topk(k=k, dim=1).indices
                    correct5[spec.name] += int((pred5 == labels[:, None]).any(dim=1).sum().item())

                counts[spec.name] += int(labels.numel())

        if delete_cache:
            _cleanup_cache_dir(cache_dir)

        metrics: Dict[str, float] = {}
        for spec in self.attributes:
            denom = max(1, counts[spec.name])
            accuracy = (correct1[spec.name] / denom)*100
            metrics[f"eval/eval_{spec.name}_top1"] = accuracy
            print( f"epoch {epoch:02d} | {name:<20} " f"top1 {accuracy:6.2f}%  ({correct:>5}/{denom:<5})" )
            if self.params['probe']['log_top5']:
                metrics[f"eval/eval_{spec.name}_top5"] = (correct5[spec.name] / denom)*100

        return metrics

    @torch.no_grad()
    def _extract_and_cache(self, encoder, loader, device, cache_dir, split):
        os.makedirs(cache_dir, exist_ok=True)
        x_chunks, a_chunks = [], []

        encoder.eval()
        for data, attr in loader:
            data = data.to(device, non_blocking=True)
            feats = self.model.encode(data)  # [B,T,C] or [B,C]
            x = feats.mean(dim=1) if feats.dim() == 3 else feats
            x_chunks.append(x.cpu())
            a_chunks.append(attr.cpu())

        x = torch.cat(x_chunks, dim=0)  # [N, D]
        a = torch.cat(a_chunks, dim=0)  # [N, K]
        torch.save(x, os.path.join(cache_dir, f"{split}_X.pt"))
        torch.save(a, os.path.join(cache_dir, f"{split}_A.pt"))



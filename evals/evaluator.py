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
    def __init__(
        self,
            encoder, classifier, optimizer, scaler, scheduler,
            wd_scheduler,  val_loader, train_loader, params, logger
    ) -> None:
        self.params = params
        self.logger = logger
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = encoder.to(self.local_rank)
        self.classifier = classifier.to(self.local_rank)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.attributes = attributes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wd_scheduler = wd_scheduler
        self.scaler = scaler
        self.use_bf16 = self.params['optimization'].get('use_bfloat16', False)
        self.use_fp16 = self.params['optimization'].get('use_float16', False)  # if you ever want real fp16
        self.amp_on = self.use_bf16 or self.use_fp16
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger()
        if self.global_rank > 0:
            self.logger.setLevel(logging.ERROR)

        self.epochs_run = 0
        self.classifier = DDP(self.classifier, device_ids=[self.local_rank], broadcast_buffers=False)


    def train_and_eval_probe(self, encoder: nn.Module, cache_dir=None, delete_cache: bool = True,) -> Dict[str, float]:

        if cache_dir is None:
            cache_dir = "/scratch/u5de/silpa.u5de/sd/cache"
        os.makedirs(cache_dir, exist_ok=True)

        probe = self._train_probe(encoder, cache_dir)
        return self._eval_probe(encoder, probe, cache_dir, delete_cache=delete_cache)

    def _train_probe(
        self,
        encoder: nn.Module,
        cache_dir: str,
    ) -> nn.ModuleDict:
        device = next(encoder.parameters()).device

        # cache + load train split
        if _need_cache(cache_dir, "train"):
            self._extract_and_cache(encoder, self.train_loader, device, cache_dir, "train")
        Xtr, Atr = self._load_cached(cache_dir, "train", map_location="cpu")

        train_loader_cached = DataLoader(
            TensorDataset(Xtr, Atr),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )

        probe = nn.ModuleDict({
            a.name: nn.Linear(encoder.student_backbone.vit.embed_dim, a.num_classes)
            for a in self.attributes
        }).to(device)

        opt = torch.optim.Adam(probe.parameters(), lr=self.cfg.lr)

        for _ in range(self.cfg.epochs):
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
    def _eval_probe(
        self,
        encoder: nn.Module,
        probe: nn.ModuleDict,
        cache_dir: str,
        delete_cache: bool = True,
    ) -> Dict[str, float]:
        device = next(encoder.parameters()).device

        # cache + load test split
        if _need_cache(cache_dir, "test"):
            self._extract_and_cache(encoder, self.probe_eval_loader, device, cache_dir, "test")
        Xte, Ate = _load_cached(cache_dir, "test", map_location="cpu")

        test_loader_cached = DataLoader(
            TensorDataset(Xte, Ate),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

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

                if self.cfg.log_top5:
                    k = min(5, spec.num_classes)
                    pred5 = scores.topk(k=k, dim=1).indices
                    correct5[spec.name] += int((pred5 == labels[:, None]).any(dim=1).sum().item())

                counts[spec.name] += int(labels.numel())

        if delete_cache:
            _cleanup_cache_dir(cache_dir)

        metrics: Dict[str, float] = {}
        for spec in self.attributes:
            denom = max(1, counts[spec.name])
            metrics[f"{self.cfg.log_prefix}/eval_{spec.name}_top1"] = correct1[spec.name] / denom
            if self.cfg.log_top5:
                metrics[f"{self.cfg.log_prefix}/eval_{spec.name}_top5"] = correct5[spec.name] / denom

        return metrics

    @torch.no_grad()
    def _extract_and_cache(self, encoder, loader, device, cache_dir, split):
        os.makedirs(cache_dir, exist_ok=True)
        X_chunks, A_chunks = [], []

        encoder.eval()
        for data, attr in loader:
            data = data.to(device, non_blocking=True)
            feats = encoder.student_backbone.encode(data)  # [B,T,C] or [B,C]
            x = feats.mean(dim=1) if feats.dim() == 3 else feats
            X_chunks.append(x.cpu())
            A_chunks.append(attr.cpu())

        X = torch.cat(X_chunks, dim=0)  # [N, D]
        A = torch.cat(A_chunks, dim=0)  # [N, K]
        torch.save(X, os.path.join(cache_dir, f"{split}_X.pt"))
        torch.save(A, os.path.join(cache_dir, f"{split}_A.pt"))



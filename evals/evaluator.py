import sys
from torch.cuda.amp import autocast
import logging
from typing import Dict, List
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from evals.eval_utils import _need_cache, _cleanup_cache_dir, _load_cached

class Evaluator:
    def __init__(self, target_encoder, probe_train_loader, probe_eval_loader,probe_train_sampler,
                    probe_test_sampler, attributes, params) -> None:
        self.params = params
        self.model = target_encoder
        self.train_loader = probe_train_loader
        self.val_loader = probe_eval_loader
        self.probe_train_sampler = probe_train_sampler
        self.probe_test_sampler = probe_test_sampler
        self.attributes = attributes
        self.ce = nn.CrossEntropyLoss()
    
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger()
        self.epochs_run = 0

    def train_and_eval_probe(self, epoch, cache_dir=None, delete_cache: bool = True,) -> Dict[str, float]:
        if self.probe_train_sampler is not None:
            self.probe_train_sampler.set_epoch(epoch)
        if cache_dir is None:
            base_cache = self.params["meta"]["cache_dir"]
            job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID") or "nojid"
            cache_dir = os.path.join(base_cache, str(job_id))
        os.makedirs(cache_dir, exist_ok=True)
        self.model.eval()
        probe = self._train_probe(cache_dir, epoch)
        return self._eval_probe(epoch, probe, cache_dir, delete_cache=delete_cache)

    def _train_probe(self, cache_dir, epoch) -> nn.ModuleDict:
        device = next(self.model.parameters()).device

        need = _need_cache(cache_dir, "train") if rank0() else None
        if is_dist():
            obj = [need]
            dist.broadcast_object_list(obj, src=0)
            need = obj[0]

        if need:
            self._extract_and_cache(self.train_loader, device, cache_dir, "train")

        x_train, a_train = _load_cached(cache_dir, "train", map_location="cpu")
        gen = torch.Generator()
        gen.manual_seed(int(self.params["meta"]["seed"]) + epoch)
        train_loader_cached = DataLoader(
            TensorDataset(x_train, a_train),
            batch_size=self.params['probe']['batch_size'],
            shuffle=True,
            generator=gen,
            num_workers=0,
            drop_last=False,
            pin_memory=True,)

        probe = nn.ModuleDict({a.name: nn.Linear(self.model.module.embed_dim, a.num_classes) for a in self.attributes}).to(device)

        opt = torch.optim.Adam(probe.parameters(), lr=self.params['probe']['lr'])

        for _ in range(self.params['probe']['epochs']):
            probe.train()
            for emb, labels in train_loader_cached:
                emb = emb.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                loss = torch.zeros((), device=device)
                for spec in self.attributes:
                    logits = probe[spec.name](emb)
                    loss = loss + self.ce(logits, labels[:, spec.index])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        return probe


    @torch.no_grad()
    def _eval_probe(self, epoch, probe: nn.ModuleDict, cache_dir: str, delete_cache: bool = True,) -> Dict[str, float]:

        device = next(self.model.parameters()).device

        need = _need_cache(cache_dir, "test") if rank0() else None
        if is_dist():
            obj = [need]
            dist.broadcast_object_list(obj, src=0)
            need = obj[0]

        if need:
            self._extract_and_cache(self.val_loader, device, cache_dir, "test")

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
                correct1[spec.name] += int((pred1 == labels[:, spec.index]).sum().item())

                if self.params['probe']['log_top5']:
                    k = min(5, spec.num_classes)
                    pred5 = scores.topk(k=k, dim=1).indices
                    correct5[spec.name] += int((pred5 == labels[:, None]).any(dim=1).sum().item())

                counts[spec.name] += int(labels[:,spec.index].numel())

        if delete_cache and rank0():
            _cleanup_cache_dir(cache_dir)
        if is_dist():
            dist.barrier()

        metrics: Dict[str, float] = {}
        acc_sum = 0.0
        n_attr = 0

        for spec in self.attributes:
            name = spec.name
            denom = max(1, counts[name])
            correct = correct1[name]
            accuracy = (correct / denom) * 100.0

            metrics[f"eval/eval_{name}_top1"] = accuracy
            print(f"epoch {epoch:02d} | {name:<20} top1 {accuracy:6.2f}%  ({correct:>5}/{denom:<5})")

            acc_sum += accuracy
            n_attr += 1

        avg_acc = acc_sum / max(1, n_attr)
        metrics["eval/eval_attr_avg_top1"] = avg_acc
        print(f"epoch {epoch:02d} | {'avg':<20} top1 {avg_acc:6.2f}%")
        if self.params['probe']['log_top5']:
            metrics[f"eval/eval_{spec.name}_top5"] = (correct5[spec.name] / denom)*100

        return metrics

    @torch.no_grad()
    def _extract_and_cache(self, loader, device, cache_dir, split):
        os.makedirs(cache_dir, exist_ok=True)
        self.model.eval()

        x_chunks, a_chunks = [], []

        for data, attr in loader:
            data = data.to(device, non_blocking=True)
            with autocast(enabled=True):
                feats = self.model(data)  # [B,T,C] or [B,C]
            x = feats.mean(dim=1) if feats.dim() == 3 else feats
            x_chunks.append(x.float().cpu())  # ✅ store fp32
            a_chunks.append(attr.cpu())

        if len(x_chunks) > 0:
            x_local = torch.cat(x_chunks, dim=0)
            a_local = torch.cat(a_chunks, dim=0)
        else:
            x_local = torch.empty((0, self.model.embedding_dim), dtype=torch.float32)
            a_local = torch.empty((0, len(self.attributes)), dtype=torch.long)

        if is_dist():
            world = dist.get_world_size()
            if rank0():
                gathered_x = [None] * world
                gathered_a = [None] * world
            else:
                gathered_x = None
                gathered_a = None

            dist.gather_object(x_local, gathered_x, dst=0)
            dist.gather_object(a_local, gathered_a, dst=0)

            if rank0():
                x = torch.cat(gathered_x, dim=0)
                a = torch.cat(gathered_a, dim=0)
                torch.save(x, os.path.join(cache_dir, f"{split}_X.pt"))
                torch.save(a, os.path.join(cache_dir, f"{split}_A.pt"))

            dist.barrier()
        else:
            # single process
            torch.save(x_local, os.path.join(cache_dir, f"{split}_X.pt"))
            torch.save(a_local, os.path.join(cache_dir, f"{split}_A.pt"))



def is_dist():
    return dist.is_available() and dist.is_initialized()

def rank0():
    return (not is_dist()) or dist.get_rank() == 0
"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored, supports DDP).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
from contextlib import nullcontext
from itertools import islice
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput, PCVRHyFormer


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    
    Supports DDP multi-GPU training:
    - Only main process (rank 0) saves checkpoints
    - Loss is averaged across GPUs
    - Validation only runs on main process
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping_patience: int = 5,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        contrastive_loss_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        rank: int = 0,
        world_size: int = 1,
        is_main: bool = True,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main
        
        self.schema_path: Optional[str] = schema_path
        self.ns_groups_path: Optional[str] = ns_groups_path

        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        
        model_to_use = model.module if hasattr(model, 'module') else model
        
        if hasattr(model_to_use, 'get_sparse_params'):
            sparse_params = model_to_use.get_sparse_params()
            dense_params = model_to_use.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            if self.is_main:
                logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
                logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model_to_use.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.use_bf16_amp = (
            str(device).startswith('cuda')
            and torch.cuda.is_available()
            and getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
        )
        
        if self.is_main:
            self.early_stopping: EarlyStopping = EarlyStopping(
                checkpoint_path=os.path.join(save_dir, "placeholder", "model.pt"),
                patience=early_stopping_patience,
                label='model',
            )
        else:
            self.early_stopping = None
            
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.contrastive_loss_weight: float = float(contrastive_loss_weight)
        self.contrastive_temperature: float = float(contrastive_temperature)
        
        if self.is_main:
            logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                         f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                         f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                         f"world_size={self.world_size}, "
                         f"bf16_amp={self.use_bf16_amp}")
            if self.contrastive_loss_weight > 0:
                logging.info(
                    f"CIR contrastive loss: weight={self.contrastive_loss_weight:.4f}, "
                    f"temperature={self.contrastive_temperature:.3f}")
        
    def _is_distributed(self) -> bool:
        return (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _autocast_context(self):
        if self.use_bf16_amp:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return nullcontext()

    def _sync_stop_signal(self, should_stop: bool) -> bool:
        if not self._is_distributed():
            return should_stop
        stop_tensor = torch.tensor(
            1 if should_stop else 0, device=self.device, dtype=torch.int32)
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        return bool(stop_tensor.item())

    def _min_steps_per_epoch(self) -> int:
        local_steps = len(self.train_loader)
        if not self._is_distributed():
            return local_steps

        steps_tensor = torch.tensor(
            local_steps, device=self.device, dtype=torch.long)
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.MIN)
        steps = int(steps_tensor.item())
        if steps <= 0:
            raise RuntimeError(
                "At least one DDP rank has zero train batches. "
                "Reduce world_size or use more training Row Groups.")
        return steps

    def _sync_model_state_from_main(self) -> None:
        if not self._is_distributed():
            return
        model_to_sync = self._unwrap_model()
        for param in model_to_sync.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in model_to_sync.buffers():
            dist.broadcast(buffer.data, src=0)

    def _run_validation_and_sync_stop(
        self,
        total_step: int,
        epoch: int,
        prefix: str,
    ) -> bool:
        should_stop = False

        if self.is_main:
            logging.info(f"Evaluating at {prefix} {total_step}")
            val_auc, val_logloss = self.evaluate(epoch=epoch)
            torch.cuda.empty_cache()

            logging.info(
                f"{prefix} {total_step} Validation | "
                f"AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)
            should_stop = (
                self.early_stopping is not None
                and self.early_stopping.early_stop
            )

        should_stop = self._sync_stop_signal(should_stop)
        self.model.train()
        return should_stop

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name."""
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to model.pt."""
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save model.pt plus sidecar files under a global_step sub-dir."""
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale *.best_model directories."""
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in batch to self.device."""
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically."""
        if not self.is_main or self.early_stopping is None:
            return
            
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            self.early_stopping(val_auc, self._unwrap_model(), {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self._unwrap_model(), {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop with DDP support."""
        if self.is_main:
            print("Start training (PCVRHyFormer)")

        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            steps_per_epoch = self._min_steps_per_epoch()
            epoch_iter = islice(enumerate(self.train_loader), steps_per_epoch)

            if self.is_main:
                train_iter = tqdm(
                    epoch_iter,
                    total=steps_per_epoch,
                    dynamic_ncols=True,
                )
            else:
                train_iter = epoch_iter

            loss_sum = 0.0
            num_steps = 0
            should_stop = False

            for step, batch in train_iter:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss
                num_steps += 1

                if self.writer and self.is_main:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                if self.is_main:
                    train_iter.set_postfix({"loss": f"{loss:.4f}"})

                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    should_stop = self._run_validation_and_sync_stop(
                        total_step=total_step,
                        epoch=epoch,
                        prefix="Step",
                    )
                    if should_stop:
                        if self.is_main:
                            logging.info(f"Early stopping at step {total_step}")
                        break

            if should_stop:
                break

            avg_loss = loss_sum / max(num_steps, 1)

            if self.world_size > 1:
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / self.world_size

            if self.is_main:
                logging.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

            should_stop = self._run_validation_and_sync_stop(
                total_step=total_step,
                epoch=epoch,
                prefix=f"Epoch {epoch}",
            )
            if should_stop:
                if self.is_main:
                    logging.info(f"Early stopping at epoch {epoch}")
                break

            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                model_to_use = self.model.module if hasattr(self.model, 'module') else self.model

                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = model_to_use.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = model_to_use.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                if self.is_main:
                    logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                                 f"restored optimizer state for {restored} low-cardinality params")
                self._sync_model_state_from_main()

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ModelInput NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_real_ts: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_real_ts[domain] = device_batch.get(
                f'{domain}_real_ts',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_real_ts=seq_real_ts,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        cir_aux = None
        with self._autocast_context():
            if self.contrastive_loss_weight > 0:
                logits, cir_aux = self.model(model_input, return_cir_aux=True)
            else:
                logits = self.model(model_input)
            logits = logits.squeeze(-1)

        logits_for_loss = logits.float()
        label_for_loss = label.float()

        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(
                logits_for_loss,
                label_for_loss,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits_for_loss, label_for_loss)

        if self.contrastive_loss_weight > 0 and cir_aux is not None:
            contrastive_loss = PCVRHyFormer.cir_contrastive_loss(
                cir_aux, temperature=self.contrastive_temperature)
            loss = loss + self.contrastive_loss_weight * contrastive_loss

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation and return (AUC, logloss)."""
        if not self.is_main:
            return 0.0, 0.0
            
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), dynamic_ncols=True)

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits, _ = self._unwrap_model().predict(model_input)
            logits = logits.squeeze(-1)

        return logits.float(), label

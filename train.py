"""PCVRHyFormer training entry point (self-contained baseline, supports DDP multi-GPU).

Usage:
    Single GPU:  python train.py [--num_epochs 10] [--batch_size 256] ...
    Multi GPU:   torchrun --standalone --nproc_per_node=2 train.py [--num_epochs 10] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import set_seed, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer

def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def build_schema_specs(schema: FeatureSchema) -> List[Tuple[int, int, int]]:
    """Return ``[(fid, offset, length), ...]`` for dense/schema-aware modules."""
    return [(fid, offset, length) for fid, offset, length in schema.entries]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value are skipped instead of receiving '
                             'full vocab-sized embeddings '
                             '(0 = no skipping; all features get full embeddings). '
                             'Useful for saving GPU memory on ultra-high-cardinality '
                             'features that are too sparse to learn reliably.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--use_aligned_pair_tokens', action='store_true', default=True,
                        help='Enable schema-aware tokenizer for aligned user int/dense fields')
    parser.add_argument('--no_aligned_pair_tokens', dest='use_aligned_pair_tokens',
                        action='store_false',
                        help='Disable aligned int-dense pair tokenizer')
    parser.add_argument('--aligned_pair_token_count', type=int, default=4,
                        help='Number of compressed aligned-pair tokens appended to user NS tokens')
    parser.add_argument('--aligned_pair_value_interaction', type=str, default='additive',
                        choices=['additive', 'scale_shift'],
                        help='How aligned dense values interact with int embeddings: '
                             'additive keeps the current best behavior; '
                             'scale_shift additionally learns value-conditioned scaling')
    parser.add_argument('--drop_aligned_pair_from_base', action='store_true', default=True,
                        help='Remove aligned fields from the original user_int/user_dense paths '
                             'after adding aligned-pair tokens')
    parser.add_argument('--keep_aligned_pair_in_base', dest='drop_aligned_pair_from_base',
                        action='store_false',
                        help='Keep aligned fields in the original user_int/user_dense paths too')
    parser.add_argument('--use_item_aware_queries', action='store_true', default=False,
                        help='Add a zero-init item-aware residual to sequence query generation')
    parser.add_argument('--num_interaction_layers', type=int, default=2,
                        help='Number of stacked deep feature interaction layers '
                             'on NS tokens (2 = original best_plus equivalent with gates)')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.0,
                        help='Weight for CIR-based InfoNCE contrastive auxiliary loss '
                             '(0.0 = disabled). Recommended range: 0.05-0.3')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help='Temperature for CIR contrastive loss softmax')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')
    if args.data_dir is None:
        raise ValueError("data_dir is required; pass --data_dir or set TRAIN_DATA_PATH")
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.abspath("checkpoints")
    if args.log_dir is None:
        args.log_dir = os.path.join(args.ckpt_dir, "logs")
    if args.tf_events_dir is None:
        args.tf_events_dir = os.path.join(args.log_dir, "tf_events")

    return args


def setup_distributed(rank: int, world_size: int, local_rank: int) -> None:
    """Initialize distributed training environment."""
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA because the backend is NCCL")
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def worker(
    rank: int,
    world_size: int,
    local_rank: int,
    args: argparse.Namespace,
    distributed: bool,
) -> None:
    """Main worker function for distributed training."""
    if distributed:
        setup_distributed(rank, world_size, local_rank)

    is_main = (rank == 0)
    device = f'cuda:{local_rank}' if distributed else args.device

    # Only rank 0 creates output directories
    if is_main:
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # Use different seed per rank for diversity
    set_seed(args.seed + rank)

    # Only rank 0 creates logger and tensorboard
    writer = None
    if is_main:
        log_file = os.path.join(args.log_dir, f'train.log')
        create_logger(log_file)
        logging.info(f"Args: {vars(args)}")
        logging.info(
            f"Distributed={distributed}, rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}, device={device}")
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tf_events_dir)
    
    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())

    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=max(0, args.num_workers // world_size),
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        rank=rank if distributed else 0,
        world_size=world_size if distributed else 1,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
    else:
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)
    user_int_feature_ids = pcvr_dataset.user_int_schema.feature_ids
    user_dense_feature_specs = build_schema_specs(pcvr_dataset.user_dense_schema)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "user_int_feature_ids": user_int_feature_ids,
        "user_dense_feature_specs": user_dense_feature_specs,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_aligned_pair_tokens": args.use_aligned_pair_tokens,
        "aligned_pair_token_count": args.aligned_pair_token_count,
        "drop_aligned_pair_from_base": args.drop_aligned_pair_from_base,
        "aligned_pair_value_interaction": args.aligned_pair_value_interaction,
        "use_item_aware_queries": args.use_item_aware_queries,
        "num_interaction_layers": args.num_interaction_layers,
        "contrastive_loss_weight": args.contrastive_loss_weight,
    }

    model = PCVRHyFormer(**model_args).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        num_sequences = len(pcvr_dataset.seq_domains)
        model_for_log = model.module if hasattr(model, 'module') else model
        num_ns = model_for_log.num_ns
        T = args.num_queries * num_sequences + num_ns
        logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
        logging.info(f"User NS groups: {user_ns_groups}")
        logging.info(f"Item NS groups: {item_ns_groups}")
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.ckpt_dir,
        early_stopping_patience=args.patience,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        contrastive_loss_weight=args.contrastive_loss_weight,
        contrastive_temperature=args.contrastive_temperature,
        rank=rank,
        world_size=world_size if distributed else 1,
        is_main=is_main,
    )

    trainer.train()
    
    if writer is not None:
        writer.close()
    
    if distributed:
        cleanup_distributed()
    
    if is_main:
        logging.info("Training complete!")


def main() -> None:
    args = parse_args()

    # Detect distributed environment
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Running under torchrun or torch.distributed.launch
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        worker(rank, world_size, local_rank, args, distributed=True)
    else:
        # Single GPU / CPU mode
        args.num_workers = max(1, args.num_workers or 0)
        worker(0, 1, 0, args, distributed=False)


if __name__ == "__main__":
    main()

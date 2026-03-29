"""
Train the FIL-compatible joint encoder on your RF/CFO/Channel dataset.

Usage:
    python baselines/FIL_instance_encoding/fil_train.py \
        --pkl_dataset_path dataset/rf_partition_dict_0.5.pkl \
        --save_path baselines/FIL_instance_encoding/models/fil_joint \
        --epochs 200 \
        --batch_size 512 \
        --gpu_id 0

The trained model is used by fil_eval.py for the comparison.
"""

import argparse
import json
import os
import pickle
import random
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASELINE_DIR, '..', '..'))
_CODE = os.path.join(_ROOT, 'code')
for p in [_CODE, os.path.join(_CODE, 'rep_lr')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rep_lr.py_datasets import TrainDataset
from fil_encoder import FILJointEncoder, compute_mean_trace


# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Train FIL-compatible joint encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--pkl_dataset_path', required=True)
    p.add_argument('--save_path', default='baselines/FIL_instance_encoding/models/fil_joint')
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--batch_size',  type=int,   default=512)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--slice_len',   type=int,   default=1024)
    p.add_argument('--proj_seq_len',    type=int, default=256)
    p.add_argument('--proj_hidden_dim', type=int, default=512)
    p.add_argument('--d2',              type=int, default=128,
                   help='Encoder output dim per channel; latent_dim = 2*d2')
    p.add_argument('--head_hidden_dim', type=int, default=256)
    p.add_argument('--encoder_num_blocks', type=int, default=3)
    p.add_argument('--cfo_head_type', default='simple',
                   choices=['default', 'simple', 'adaptive'])
    p.add_argument('--w_rf',      type=float, default=1.0, help='RF loss weight')
    p.add_argument('--w_cfo',     type=float, default=1.0, help='CFO loss weight')
    p.add_argument('--w_channel', type=float, default=1.0, help='Channel loss weight')
    p.add_argument('--patience',  type=int,   default=20)
    p.add_argument('--gpu_id',    type=int,   default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed',      type=int,   default=42)
    return p.parse_args()


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    _set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    # Save args
    with open(os.path.join(args.save_path, 'fil_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(args.pkl_dataset_path, 'rb') as f:
        content = pickle.load(f)

    train_list = content['train']
    val_list   = content['val']
    max_cfo    = content['max_cfo']
    mean_cfo   = content['mean_cfo']
    std_cfo    = content['std_cfo']

    ID_class_dict = {f'Radio{i}': i for i in range(16)}

    import argparse as _ap
    ds_args = _ap.Namespace(slice_len=args.slice_len)

    train_ds = TrainDataset(train_list, ID_class_dict, ds_args, max_cfo, mean_cfo, std_cfo)
    val_ds   = TrainDataset(val_list,   ID_class_dict, ds_args, max_cfo, mean_cfo, std_cfo,
                            test_mode=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FILJointEncoder(
        slice_len=args.slice_len,
        proj_seq_len=args.proj_seq_len,
        proj_hidden_dim=args.proj_hidden_dim,
        d2=args.d2,
        num_classes=len(ID_class_dict),
        dropout=args.dropout,
        head_hidden_dim=args.head_hidden_dim,
        encoder_num_blocks=args.encoder_num_blocks,
        cfo_head_type=args.cfo_head_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # ── Losses ────────────────────────────────────────────────────────────────
    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    def channel_loss(pred, target):
        return mse_loss(pred.view(pred.size(0), -1), target.view(target.size(0), -1))

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        # ── Training epoch ────────────────────────────────────────────────────
        model.train()
        train_loss_total = 0.0
        train_rf_correct = 0
        n_train = 0

        for batch in tqdm(train_dl, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', leave=False):
            rf_x, rf_y, cfo_x, cfo_y, ch_x, ch_y, _ = batch

            rf_x = rf_x.to(device).float()
            cfo_x = cfo_x.to(device).float()
            ch_x  = ch_x.to(device).float()
            rf_y  = rf_y.to(device).long()
            cfo_y = cfo_y.to(device).float().view(-1, 1)
            ch_y  = ch_y.to(device).float()

            optimizer.zero_grad()
            preds = model(rf_x, cfo_x, ch_x)

            loss_rf  = ce_loss(preds['rf'], rf_y)
            loss_cfo = mse_loss(preds['cfo'], cfo_y)
            loss_ch  = channel_loss(preds['channel'], ch_y)

            loss = args.w_rf * loss_rf + args.w_cfo * loss_cfo + args.w_channel * loss_ch

            if torch.isnan(loss):
                print(f'Warning: NaN loss at epoch {epoch}, skipping batch.')
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_total += loss.item() * rf_x.size(0)
            train_rf_correct += (preds['rf'].argmax(1) == rf_y).sum().item()
            n_train += rf_x.size(0)

        scheduler.step()

        # ── Validation epoch ──────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        val_rf_correct = 0
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f'Epoch {epoch+1}/{args.epochs} [Val]', leave=False):
                rf_x, rf_y, cfo_x, cfo_y, ch_x, ch_y, _ = batch

                rf_x = rf_x.to(device).float()
                cfo_x = cfo_x.to(device).float()
                ch_x  = ch_x.to(device).float()
                rf_y  = rf_y.to(device).long()
                cfo_y = cfo_y.to(device).float().view(-1, 1)
                ch_y  = ch_y.to(device).float()

                preds = model(rf_x, cfo_x, ch_x)

                loss_rf  = ce_loss(preds['rf'], rf_y)
                loss_cfo = mse_loss(preds['cfo'], cfo_y)
                loss_ch  = channel_loss(preds['channel'], ch_y)
                loss = args.w_rf * loss_rf + args.w_cfo * loss_cfo + args.w_channel * loss_ch

                val_loss_total += loss.item() * rf_x.size(0)
                val_rf_correct += (preds['rf'].argmax(1) == rf_y).sum().item()
                n_val += rf_x.size(0)

        val_loss = val_loss_total / n_val
        val_acc  = val_rf_correct / n_val
        train_acc = train_rf_correct / n_train
        print(
            f'Epoch {epoch+1:3d} | '
            f'train_loss={train_loss_total/n_train:.4f} train_rf_acc={train_acc:.3f} | '
            f'val_loss={val_loss:.4f} val_rf_acc={val_acc:.3f}'
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_path, 'fil_best.pt'))
            print(f'  ✓ Saved best model (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}.')
                break

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_path, 'fil_final.pt'))
    print(f'Training complete. Best val_loss: {best_val_loss:.4f}')
    print(f'Model saved to: {args.save_path}')


if __name__ == '__main__':
    train(get_args())

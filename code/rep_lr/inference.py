"""Run a released model on the test split and report its metrics.

Picks a run from the model registry on the Hugging Face Hub, downloads only its
``args.json`` and checkpoint, rebuilds the architecture, and evaluates every task
the model was trained on in a single pass over the data:

    RF fingerprinting   accuracy, macro-F1
    CFO estimation      R², RMSE / MAE / median absolute error (Hz)
    Channel estimation  R² (mean of real and imaginary parts), NMSE (dB)

Examples
--------
    # default: best RF+CFO+Channel model
    python code/rep_lr/inference.py

    # best model of another category, on a reproducible 10% subset
    python code/rep_lr/inference.py --task_type CFO --test_fraction 0.1 --seed 0

    # a specific run, or a local run directory
    python code/rep_lr/inference.py --run_id channel_estimation_20250803_153603__rf_variable
    python code/rep_lr/inference.py --model_dir results/.../rf_fingerprinting_20250101_000000

    # list the candidates of a category
    python code/rep_lr/inference.py --task_type RF+CFO --list

"Best" means the lowest validation loss among runs that have recorded test
metrics; it is selected on validation, not test, data. It optimises the summed
loss, so it is not necessarily the best run on every individual task.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import random

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_io import NUM_CLASSES, build_model, load_weights
from py_datasets import TrainDataset, TrainDatasetRFixed

HF_REPO = 'Aadharsh/RepresentationLearning-models'
TASK_TYPES = ['RF', 'CFO', 'Channel', 'RF+CFO', 'RF+Channel', 'CFO+Channel', 'RF+CFO+Channel']
# task -> registry column holding its recorded test metric
REGISTRY_METRIC = {
    'rf_fingerprinting': 'rf_mean_f1',
    'cfo_estimation': 'cfo_rmse',
    'channel_estimation': 'channel_nmse_db',
}
TASK_SHORT_NAMES = {'rf_fingerprinting': 'RF', 'cfo_estimation': 'CFO', 'channel_estimation': 'Channel'}
RF_SLICE_BATCH = 128  # RF slices per forward pass (a test packet has ~3000 slices)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# ── Model selection ───────────────────────────────────────────────────────────

def read_registry(cache_dir):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, 'registry.csv', repo_type='model', local_dir=cache_dir)
    with open(path) as f:
        return list(csv.DictReader(f))


def tasks_of(row):
    parts = row['task_type'].split('+')
    return [t for t, short in TASK_SHORT_NAMES.items() if short in parts]


def candidates(registry, task_type):
    """Runs of ``task_type`` that trained cleanly and have recorded test metrics."""
    rows = []
    for r in registry:
        if r['task_type'] != task_type or r['error'].strip() or not r['best_val_loss'].strip():
            continue
        if all(r[REGISTRY_METRIC[t]].strip() for t in tasks_of(r)):
            rows.append(r)
    return sorted(rows, key=lambda r: float(r['best_val_loss']))


def download_run(run_id, cache_dir):
    from huggingface_hub import snapshot_download
    # args.json + checkpoint only: the stored prediction dumps are ~200 MB each
    snapshot_download(HF_REPO, repo_type='model', local_dir=cache_dir,
                      allow_patterns=[f'runs/{run_id}/args.json', f'runs/{run_id}/*.pt'])
    return os.path.join(cache_dir, 'runs', run_id)


def find_checkpoint(model_dir):
    ckpts = sorted(glob.glob(os.path.join(model_dir, '*.pt')))
    best = [c for c in ckpts if c.endswith('_best.pt')]
    if not (best or ckpts):
        raise FileNotFoundError(f"no .pt checkpoint in {model_dir}")
    return (best or ckpts)[0]


STALE_CHECKPOINT_HINT = ("Runs trained before 2025-07-17 use an earlier encoder "
                         "that the current models.py cannot load.")


def load_run(model_dir, device):
    """Rebuild and load the run in ``model_dir``; returns (model_dir, train_args, model, ckpt)."""
    with open(os.path.join(model_dir, 'args.json')) as f:
        train_args = argparse.Namespace(**json.load(f))
    ckpt = find_checkpoint(model_dir)
    model = build_model(train_args, NUM_CLASSES).to(device)
    load_weights(model, ckpt, device)
    return model_dir, train_args, model.eval(), ckpt


# ── Forward passes (mirror eval.py) ───────────────────────────────────────────

def fuse(tensors, fusion_type):
    if fusion_type == 'concat':
        return torch.cat(tensors, dim=2)
    if fusion_type == 'depth_concat':
        return torch.cat(tensors, dim=1)
    return torch.sum(torch.stack(tensors), dim=0)


@torch.no_grad()
def predict_sample(model, args, rf_x, cfo_x, channel_x):
    """Predictions for one test packet, for every task the model has.

    rf_x holds all RF slices of the packet (S, 2, L); cfo_x and channel_x are (1, 2, 160).
    Returns {task: prediction}: class id, normalised CFO, or (2, 52) channel.
    """
    out = {}

    if not getattr(args, 'mtl', False):
        task = args.task
        x = {'rf_fingerprinting': rf_x, 'cfo_estimation': cfo_x, 'channel_estimation': channel_x}[task]
        if task == 'rf_fingerprinting':
            logits = torch.cat([model['head'](model['encoder'](model['projection'](x[i:i + RF_SLICE_BATCH])))
                                for i in range(0, x.size(0), RF_SLICE_BATCH)])
            out[task] = torch.mode(logits.argmax(1)).values.item()
        else:
            h = model['projection'](x)
            if task == 'cfo_estimation' and getattr(args, 'direct_cfo', False):
                y = model['head'](h)  # direct CFO bypasses the encoder
            else:
                y = model['head'](model['encoder'](h))
            out[task] = y.item() if task == 'cfo_estimation' else y[0].cpu().numpy()
        return out

    fusion = getattr(args, 'fusion_type', 'sum')
    proj = model['projections']
    raw = {'cfo_estimation': cfo_x, 'channel_estimation': channel_x}
    projected = {t: proj[t](raw[t]) for t in args.task if t != 'rf_fingerprinting'}

    if 'rf_fingerprinting' in args.task:
        rf_proj = torch.cat([proj['rf_fingerprinting'](rf_x[i:i + RF_SLICE_BATCH])
                             for i in range(0, rf_x.size(0), RF_SLICE_BATCH)])

        # RF: every slice fused with the packet's other-task projections, majority vote
        others = [projected[t] for t in args.task if t != 'rf_fingerprinting']
        votes = []
        for i in range(0, rf_proj.size(0), RF_SLICE_BATCH):
            p = rf_proj[i:i + RF_SLICE_BATCH]
            if fusion in ('concat', 'depth_concat'):
                fused = fuse([p] + [o.repeat(p.shape[0], 1, 1) for o in others], fusion)
            else:
                fused = p + (torch.sum(torch.stack(others), dim=0) if others else 0)
            votes.append(model['heads']['rf_fingerprinting'](model['encoder'](fused)).argmax(1))
        out['rf_fingerprinting'] = torch.mode(torch.cat(votes)).values.item()

        # CFO / channel see the RF projection averaged over the packet's slices
        projected['rf_fingerprinting'] = rf_proj.mean(dim=0, keepdim=True)

    encoded = None
    for task in ('cfo_estimation', 'channel_estimation'):
        if task not in args.task:
            continue
        if task == 'cfo_estimation' and getattr(args, 'direct_cfo', False):
            y = model['heads'][task](projected[task])
        else:
            if encoded is None:
                encoded = model['encoder'](fuse([projected[t] for t in args.task], fusion))
            y = model['heads'][task](encoded)
        out[task] = y.item() if task == 'cfo_estimation' else y[0].cpu().numpy()
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(task, y_true, y_pred, mean_cfo, std_cfo):
    if task == 'rf_fingerprinting':
        return {'accuracy': accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro')}

    if task == 'cfo_estimation':
        t = np.asarray(y_true) * std_cfo + mean_cfo
        p = np.asarray(y_pred) * std_cfo + mean_cfo
        return {'r2': r2_score(t, p),
                'rmse_hz': float(np.sqrt(mean_squared_error(t, p))),
                'mae_hz': mean_absolute_error(t, p),
                'median_ae_hz': float(np.median(np.abs(t - p)))}

    t, p = np.stack(y_true), np.stack(y_pred)  # (N, 2, 52): real, imaginary
    t_c, p_c = t[:, 0] + 1j * t[:, 1], p[:, 0] + 1j * p[:, 1]
    nmse = np.mean(np.abs(t_c - p_c) ** 2) / np.mean(np.abs(t_c) ** 2)
    return {'r2': (r2_score(t[:, 0], p[:, 0]) + r2_score(t[:, 1], p[:, 1])) / 2,
            'nmse_db': float(10 * np.log10(nmse))}


def print_report(results, registry_row, n_files):
    names = {'rf_fingerprinting': 'RF fingerprinting', 'cfo_estimation': 'CFO estimation',
             'channel_estimation': 'Channel estimation'}
    labels = {'accuracy': 'Accuracy', 'macro_f1': 'Macro-F1', 'r2': 'R²', 'rmse_hz': 'RMSE (Hz)',
              'mae_hz': 'MAE (Hz)', 'median_ae_hz': 'Median AE (Hz)', 'nmse_db': 'NMSE (dB)'}
    # registry column -> metric key it corresponds to
    recorded = {'rf_mean_f1': 'macro_f1', 'cfo_rmse': 'rmse_hz', 'channel_nmse_db': 'nmse_db'}

    print(f"\nResults on {n_files} test packets")
    print(f"{'task':20} {'metric':16} {'value':>10}   {'registry':>10}")
    print('-' * 62)
    for task, metrics in results.items():
        for i, (k, v) in enumerate(metrics.items()):
            ref = ''
            col = REGISTRY_METRIC[task]
            if registry_row and recorded[col] == k and registry_row[col].strip():
                ref = f"{float(registry_row[col]):10.4f}"
            print(f"{names[task] if i == 0 else '':20} {labels[k]:16} {v:10.4f}   {ref:>10}")
    if registry_row:
        print("\nRegistry values were computed on the full test split; expect them to match "
              "only when --test_fraction is 1.0. CFO RMSE in particular is dominated by a "
              "few outlier packets and varies strongly between subsets.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(os.path.join(REPO_ROOT, '.env'))

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = parser.add_argument_group('model selection')
    sel.add_argument('--task_type', default='RF+CFO+Channel', choices=TASK_TYPES,
                     help='Registry category; its lowest-validation-loss run is used.')
    sel.add_argument('--run_id', default=None, help='Use this registry run instead of the best one.')
    sel.add_argument('--model_dir', default=None,
                     help='Local run directory (args.json + .pt); skips the registry entirely.')
    sel.add_argument('--list', action='store_true', help='List the candidates for --task_type and exit.')
    sel.add_argument('--cache_dir', default=os.path.join(REPO_ROOT, 'models'),
                     help='Where registry.csv and downloaded runs are stored.')

    data = parser.add_argument_group('data')
    data.add_argument('--pkl_dataset_path', default=None,
                      help='Partition pickle. Defaults to $PKL_FILE_PATH/rf_partition_dict_0.5.pkl.')
    data.add_argument('--data_root', default=None, help='Directory of the .mat files. Defaults to $DATA_BASE_PATH.')
    data.add_argument('--test_fraction', type=float, default=1.0, help='Fraction of the test split to evaluate.')
    data.add_argument('--seed', type=int, default=0, help='Seed for choosing the --test_fraction subset.')
    data.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output_json', default=None, help='Also write the metrics to this file.')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Resolve which run to evaluate, and load it
    registry_row = None
    if args.model_dir:
        model_dir, train_args, model, ckpt = load_run(args.model_dir, device)
    else:
        registry = read_registry(args.cache_dir)
        if args.run_id:
            rows = [r for r in registry if r['run_id'] == args.run_id]
            if not rows:
                parser.error(f"run_id '{args.run_id}' is not in the registry")
        else:
            rows = candidates(registry, args.task_type)
            if args.list:
                print(f"{'run_id':84} {'val_loss':>8} {'rf_f1':>7} {'cfo_rmse':>9} {'nmse_db':>8}")
                for r in rows:
                    print(f"{r['run_id']:84} {float(r['best_val_loss']):8.4f} {r['rf_mean_f1'] or '-':>7.7} "
                          f"{r['cfo_rmse'] or '-':>9.9} {r['channel_nmse_db'] or '-':>8.8}")
                print("\nRuns trained before 2025-07-17 use an earlier encoder and cannot be "
                      "loaded; automatic selection skips them.")
                return
            if not rows:
                parser.error(f"no usable runs for task type {args.task_type}")

        # Best first; skip checkpoints the current models.py cannot load
        for registry_row in rows:
            print(f"Run: {registry_row['run_id']} ({registry_row['task_type']})")
            try:
                model_dir, train_args, model, ckpt = load_run(
                    download_run(registry_row['run_id'], args.cache_dir), device)
                break
            except (RuntimeError, KeyError) as e:
                reason = str(e).splitlines()[0]
                if args.run_id:
                    raise SystemExit(f"cannot load {args.run_id}: {reason}\n{STALE_CHECKPOINT_HINT}")
                print(f"  skipped, checkpoint does not load ({reason}). {STALE_CHECKPOINT_HINT}")
        else:
            raise SystemExit(f"no loadable run for task type {args.task_type}")

    tasks = train_args.task if getattr(train_args, 'mtl', False) else [train_args.task]

    # Data. The pickle path inside args.json is from the training machine; never use it.
    pkl_path = args.pkl_dataset_path or os.path.join(os.getenv('PKL_FILE_PATH', ''), 'rf_partition_dict_0.5.pkl')
    if not os.path.isfile(pkl_path):
        parser.error(f"partition pickle not found at '{pkl_path}'; pass --pkl_dataset_path or set PKL_FILE_PATH")
    if args.data_root:
        os.environ['DATA_BASE_PATH'] = args.data_root  # read by py_datasets, also in workers
    with open(pkl_path, 'rb') as f:
        content = pickle.load(f)

    test_list = list(content['test'])
    random.Random(args.seed).shuffle(test_list)
    test_list = test_list[:int(len(test_list) * args.test_fraction)]
    mean_cfo, std_cfo = content['mean_cfo'], content['std_cfo']

    dataset_cls = TrainDatasetRFixed if getattr(train_args, 'rf_fixed', False) else TrainDataset
    dataset = dataset_cls(test_list, None, train_args, content['max_cfo'], mean_cfo, std_cfo, test_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Checkpoint: {ckpt}\nTasks: {', '.join(tasks)} | device: {device}")

    y_true = {t: [] for t in tasks}
    y_pred = {t: [] for t in tasks}
    for rf_x, rf_y, cfo_x, cfo_y, ch_x, ch_y, _ in tqdm(loader, desc='Inference'):
        preds = predict_sample(model, train_args, rf_x.squeeze(0).to(device).float(),
                               cfo_x.to(device).float(), ch_x.to(device).float())
        truth = {'rf_fingerprinting': int(rf_y), 'cfo_estimation': float(cfo_y),
                 'channel_estimation': ch_y[0].numpy()}
        for t in tasks:
            y_true[t].append(truth[t])
            y_pred[t].append(preds[t])

    results = {t: compute_metrics(t, y_true[t], y_pred[t], mean_cfo, std_cfo) for t in tasks}
    print_report(results, registry_row, len(test_list))

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({'run_id': registry_row['run_id'] if registry_row else None, 'model_dir': model_dir,
                       'test_files': len(test_list), 'test_fraction': args.test_fraction,
                       'seed': args.seed, 'metrics': results}, f, indent=2, default=float)
        print(f"\nMetrics written to {args.output_json}")


if __name__ == '__main__':
    main()

"""IQ Signal Datasets for NeuraCrypt benchmarking.

Wraps the RF fingerprinting / CFO / channel estimation tasks from the
Oracle dataset.  Each sample is a 2-channel IQ tensor that is first
reshaped to a 2-D grid and then bilinearly up-sampled to (2, 256, 256)
so that it can be fed into NeuraCrypt's PrivateEncoder unchanged
(except for the in_channels 1 → 2 change).

Reshape logic
─────────────
  RF       (2, 1024) → (2, 32, 32)  [1024 = 32²]   → (2, 256, 256)
  CFO      (2, 160)  → (2, 13, 13)  [pad 9→169=13²] → (2, 256, 256)
  Channel  (2, 160)  → same as CFO                  → (2, 256, 256)
"""

import os
import random
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from scipy.io import loadmat

from sandstone.datasets.factory import RegisterDataset

IMG_SIZE = 256
RF_SLICE_LEN = 1024

# Channel output is (2, 52) = 104 elements
CHANNEL_OUTPUT_DIM = 104


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers (reuse read_file logic from code/rep_lr/py_datasets.py)
# ──────────────────────────────────────────────────────────────────────────────

def _rms_normalize(x: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(torch.real(x) ** 2 + torch.imag(x) ** 2))
    return x / (rms + 1e-9)


def _to_iq(x: torch.Tensor) -> torch.Tensor:
    """Complex 1-D tensor → (2, L) float tensor (I row, Q row)."""
    out = torch.zeros(2, x.shape[0])
    out[0] = torch.real(x)
    out[1] = torch.imag(x)
    return out


def _read_file(file_path: str, mean_cfo: float, std_cfo: float):
    """Mirror of code/rep_lr/py_datasets.py::read_file.

    Returns
    -------
    RF_X     : (2, total_len)   float  – raw IQ, not yet sliced
    RF_y     : int              – device class 0…15
    CFO_X    : (2, 160)         float
    CFO_y    : float scalar     – standardised CFO
    Channel_X: (2, 160)         float
    Channel_y: (2, 52)          float
    """
    content = loadmat(file_path)
    RF_input = torch.from_numpy(content['Packet'])[:, 0]
    RF_output = content['Radio'][0]

    parts = file_path.split('/')
    # drop leading empty string from absolute path
    parts = [p for p in parts if p]
    filename = parts[-1]
    dir_parts = parts[:-1]
    suffix = filename.lstrip('RFfingerprinting')

    def sibling_path(prefix):
        return '/' + '/'.join(dir_parts) + '/' + prefix + suffix

    content_cfo = loadmat(sibling_path('CFOEstimation'))
    CFO_input = torch.from_numpy(content_cfo['LSTF'])[:, 0]
    CFO_output = torch.from_numpy(content_cfo['CFO']).float().squeeze()

    content_ch = loadmat(sibling_path('ChannelEstimation'))
    Channel_input = torch.from_numpy(content_ch['LLTF'])[:, 0]
    Channel_output = torch.from_numpy(content_ch['EstChannnel'])[:, 0]

    RF_input    = _rms_normalize(RF_input)
    Channel_input  = _rms_normalize(Channel_input)
    Channel_output = _rms_normalize(Channel_output)
    CFO_input   = _rms_normalize(CFO_input)

    RF_X     = _to_iq(RF_input)
    RF_y     = int(RF_output.lstrip('Radio'))
    CFO_X    = _to_iq(CFO_input)
    CFO_y    = ((CFO_output - mean_cfo) / std_cfo).float()
    Channel_X = _to_iq(Channel_input)
    Channel_y = _to_iq(Channel_output).contiguous()  # (2, 52)

    return RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y


def _iq_to_image(x: torch.Tensor, target_size: int = IMG_SIZE) -> torch.Tensor:
    """Reshape (2, L) IQ tensor → (2, target_size, target_size).

    Uses nearest-square padding then bilinear upsampling so that all
    information is preserved without any lossy transform.
    """
    C, L = x.shape
    side = int(np.ceil(np.sqrt(L)))
    pad = side * side - L
    if pad > 0:
        x = F.pad(x, (0, pad))
    x = x.view(C, side, side)
    x = F.interpolate(
        x.unsqueeze(0).float(),
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Base IQ dataset (not using Abstract_Dataset to avoid CXR metadata machinery)
# ──────────────────────────────────────────────────────────────────────────────

class _BaseIQDataset(data.Dataset):
    """Base class for IQ-signal NeuraCrypt datasets.

    Sub-classes must implement ``_get_xy(file_path)`` which returns
    ``(x_image, y)`` where ``x_image`` is already (2, 256, 256) and
    ``y`` is a scalar / tensor label.
    """

    def __init__(self, args, augmentations, split_group: str):
        super().__init__()
        self.args = args
        self.split_group = split_group

        pkl = pickle.load(open(args.pkl_dataset_path, 'rb'))
        self.mean_cfo = pkl['mean_cfo']
        self.std_cfo  = pkl['std_cfo']

        key_map = {'train': 'train', 'dev': 'val', 'test': 'test'}
        self.file_list = pkl[key_map[split_group]]

        # Build per-sample weights (used by WeightedRandomSampler for class_bal)
        labels = [self._get_label_for_weight(fp) for fp in self.file_list]
        label_counts = Counter(labels)
        weight_per_label = 1.0 / len(label_counts)
        label_weights = {
            lbl: weight_per_label / cnt for lbl, cnt in label_counts.items()
        }
        self.weights = [label_weights[lbl] for lbl in labels]

        if not hasattr(args, 'exam_to_year_dict'):
            args.exam_to_year_dict = {}

    def _get_label_for_weight(self, file_path: str):
        """Return a hashable label used only for balanced sampling weights.

        For regression tasks return 0 (uniform weights).
        """
        return 0

    def _get_xy(self, file_path: str):
        raise NotImplementedError

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            x, y = self._get_xy(file_path)
        except Exception as e:
            # Fall back to first sample on read error
            x, y = self._get_xy(self.file_list[0])

        return {
            'x':      x,
            'y':      y,
            'source': torch.tensor(0),   # single source → always use encoder_0
            'exam':   file_path,
        }

    @staticmethod
    def set_args(args):
        args.num_chan = 2
        args.img_size = [IMG_SIZE, IMG_SIZE]
        if not hasattr(args, 'exam_to_year_dict'):
            args.exam_to_year_dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Task-specific datasets
# ──────────────────────────────────────────────────────────────────────────────

@RegisterDataset('iq_rf_fingerprinting')
class IQRFDataset(_BaseIQDataset):
    """RF fingerprinting (classification, 16 classes)."""

    def _get_label_for_weight(self, file_path: str):
        content = loadmat(file_path)
        return int(content['Radio'][0].lstrip('Radio'))

    def _get_xy(self, file_path: str):
        RF_X, RF_y, *_ = _read_file(file_path, 0.0, 1.0)
        # Random slice to RF_SLICE_LEN (matches rep_lr training)
        if self.split_group == 'train':
            max_start = RF_X.shape[1] - RF_SLICE_LEN
            start = random.randint(0, max(0, max_start))
            RF_X = RF_X[:, start:start + RF_SLICE_LEN]
        else:
            RF_X = RF_X[:, :RF_SLICE_LEN]
        # Pad if shorter than slice len
        if RF_X.shape[1] < RF_SLICE_LEN:
            RF_X = F.pad(RF_X, (0, RF_SLICE_LEN - RF_X.shape[1]))

        x = _iq_to_image(RF_X)
        y = torch.tensor(RF_y, dtype=torch.long)
        return x, y

    @staticmethod
    def set_args(args):
        _BaseIQDataset.set_args(args)
        args.num_classes = 16
        args.regression  = False
        args.tuning_metric = 'accuracy'


@RegisterDataset('iq_cfo_estimation')
class IQCFODataset(_BaseIQDataset):
    """CFO estimation (scalar regression)."""

    def _get_xy(self, file_path: str):
        _, _, CFO_X, CFO_y, *_ = _read_file(file_path, self.mean_cfo, self.std_cfo)
        x = _iq_to_image(CFO_X)
        y = CFO_y.view(1).float()
        return x, y

    @staticmethod
    def set_args(args):
        _BaseIQDataset.set_args(args)
        args.num_classes = 1
        args.regression  = True
        args.tuning_metric = 'loss'


@RegisterDataset('iq_channel_estimation')
class IQChannelDataset(_BaseIQDataset):
    """Channel estimation (vector regression, output dim = 2×52 = 104)."""

    def _get_xy(self, file_path: str):
        *_, Channel_X, Channel_y = _read_file(file_path, 0.0, 1.0)
        x = _iq_to_image(Channel_X)
        y = Channel_y.view(104).float()   # flatten (2,52) → 104 for MSE loss
        return x, y

    @staticmethod
    def set_args(args):
        _BaseIQDataset.set_args(args)
        args.num_classes = 104
        args.regression  = True
        args.tuning_metric = 'loss'


# ──────────────────────────────────────────────────────────────────────────────
# Multi-task dataset (mirrors rep_lr MTL: one .mat file → all 3 task labels)
# ──────────────────────────────────────────────────────────────────────────────

class IQMTLDataset(data.Dataset):
    """Multi-task IQ dataset.

    Each sample is one .mat file, yielding separate (2,256,256) images and
    labels for all three tasks simultaneously.  This is the direct analog of
    rep_lr's MTL setup where all tasks share a single input packet.

    Returns a dict with keys:
        rf_x      : (2, 256, 256) float
        rf_y      : int tensor  — device class 0…15
        cfo_x     : (2, 256, 256) float
        cfo_y     : (1,)  float  — standardised CFO
        channel_x : (2, 256, 256) float
        channel_y : (104,) float — flattened (2,52) channel output
    """

    def __init__(self, args, augmentations, split_group: str):
        super().__init__()
        self.args        = args
        self.split_group = split_group

        pkl = pickle.load(open(args.pkl_dataset_path, 'rb'))
        self.mean_cfo = pkl['mean_cfo']
        self.std_cfo  = pkl['std_cfo']

        key_map = {'train': 'train', 'dev': 'val', 'test': 'test'}
        self.file_list = pkl[key_map[split_group]]

        # uniform weights (MTL has no single label to balance on)
        self.weights = [1.0] * len(self.file_list)

        if not hasattr(args, 'exam_to_year_dict'):
            args.exam_to_year_dict = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            return self._load(file_path)
        except Exception:
            return self._load(self.file_list[0])

    def _load(self, file_path: str) -> dict:
        RF_X, RF_y, CFO_X, CFO_y, Channel_X, Channel_y = _read_file(
            file_path, self.mean_cfo, self.std_cfo)

        # Slice RF to fixed length
        if self.split_group == 'train':
            max_start = RF_X.shape[1] - RF_SLICE_LEN
            start = random.randint(0, max(0, max_start))
            RF_X = RF_X[:, start:start + RF_SLICE_LEN]
        else:
            RF_X = RF_X[:, :RF_SLICE_LEN]
        if RF_X.shape[1] < RF_SLICE_LEN:
            RF_X = F.pad(RF_X, (0, RF_SLICE_LEN - RF_X.shape[1]))

        return {
            'rf_x':      _iq_to_image(RF_X),
            'rf_y':      torch.tensor(RF_y, dtype=torch.long),
            'cfo_x':     _iq_to_image(CFO_X),
            'cfo_y':     CFO_y.view(1).float(),
            'channel_x': _iq_to_image(Channel_X),
            'channel_y': Channel_y.view(CHANNEL_OUTPUT_DIM).float(),
        }

    @staticmethod
    def set_args(args):
        args.num_chan  = 2
        args.img_size  = [IMG_SIZE, IMG_SIZE]
        args.num_classes = 16   # RF head is the primary task for checkpoint metric
        args.regression  = False
        args.tuning_metric = 'accuracy'
        if not hasattr(args, 'exam_to_year_dict'):
            args.exam_to_year_dict = {}

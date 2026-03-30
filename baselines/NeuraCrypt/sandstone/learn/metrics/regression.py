"""Regression metrics (R²) for NeuraCrypt IQ benchmarking."""

from collections import OrderedDict

import numpy as np
from sklearn.metrics import r2_score

from sandstone.learn.metrics.factory import RegisterMetric


@RegisterMetric("r2_score")
def get_r2_metrics(logging_dict, args):
    """Compute R² between accumulated predictions and ground-truth targets.

    Works for both scalar (CFO, shape B×1) and vector (Channel, shape B×104)
    regression outputs.  Flattens to 1-D for scalar case, keeps as 2-D for
    multi-output (sklearn averages over outputs with 'uniform_average').
    """
    stats_dict = OrderedDict()

    pred  = np.array(logging_dict['pred'])   # (N, out_dim) or (N,)
    golds = np.array(logging_dict['golds'])  # same shape

    pred  = pred.reshape(len(pred),  -1)
    golds = golds.reshape(len(golds), -1)

    if pred.shape[1] == 1:
        # Scalar regression (CFO)
        r2 = r2_score(golds.ravel(), pred.ravel())
    else:
        # Vector regression (Channel): average R² across output dimensions
        r2 = r2_score(golds, pred, multioutput='uniform_average')

    stats_dict['r2'] = float(r2)
    return stats_dict

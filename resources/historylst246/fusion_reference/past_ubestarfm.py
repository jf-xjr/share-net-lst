"""Past-only, nine-source-pool adapter of pinned official ubESTARFM.

No labels, metadata, city identity, fitting objective, or file I/O enters
predict_scene. Input arrays follow the released six-input scene interface.
Upstream algorithm files remain byte-for-byte unchanged. See protocol.json.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
from ubestarfm.api import train_arrays, predict_arrays
from historylst.metrics import repair

def repair_numpy(prediction, coarse, support):
    return np.nan_to_num(repair(prediction, coarse, support), nan=0.0)

UPSTREAM_COMMIT = '171cfbdc92a7e63ef8769face326f00cb650061a'
WINDOW_RADIUS = 25
PATCH_SIZE = 200
VALUE_RANGE = (240.0, 380.0)
MIN_REFERENCE_SUPPORT = 12
DEFAULT_WORKERS = 4


def _scene(array, ndim, name):
    array = np.asarray(array)
    if array.ndim == ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != ndim:
        raise ValueError(f'{name} must represent exactly one scene')
    return array


def _historical_references(history):
    _, _, rows, cols = history.shape
    if rows % 4 or cols % 4:
        raise ValueError('fine grid must be divisible by the released factor four')
    references = {}
    for slot in range(history.shape[0]):
        coverage = np.asarray(history[slot, 2], np.float64)
        valid = coverage > 0
        if not valid.any():
            continue
        age = np.asarray(history[slot, 8], np.float64)[valid] * 3652.5
        if not np.isfinite(age).all() or np.min(age) <= 0:
            raise ValueError('each active history slot must be strictly in the past')
        temperature = np.asarray(history[slot, 0], np.float64) * 20.0 + 300.0
        if not np.isfinite(temperature[valid]).all():
            raise ValueError('valid historical temperature must be finite')
        block_shape = (rows // 4, 4, cols // 4, 4)
        counts = valid.reshape(block_shape).sum(axis=(1, 3))
        sums = np.where(valid, temperature, 0.0).reshape(block_shape).sum(axis=(1, 3))
        coarse = np.where(counts >= MIN_REFERENCE_SUPPORT,
                          sums / np.maximum(counts, 1), np.nan)
        coarse = coarse.repeat(4, 0).repeat(4, 1)
        fine = np.where(valid, temperature, np.nan)
        references[slot] = (fine, coarse, coverage, float(np.median(age)))
    return references


def predict_scene(fine, coarse, support, context, emissivity, history,
                  *, mode='pool', workers=DEFAULT_WORKERS, return_details=False):
    """Return projected float64 [1,1,H,W] prediction from six arrays only.

    `history` can have seven/nine slots or zeroed withheld slots. Pool mode
    applies every unordered pair with replacement and averages the finite
    predictions using geometric-mean observed clear coverage. Nearest-two
    mode is an explicitly secondary native-pair adaptation. context and
    emissivity are accepted, but are not features used by ubESTARFM.
    """
    fine = _scene(fine, 3, 'fine')
    coarse = _scene(coarse, 3, 'coarse')
    support = _scene(support, 3, 'support').astype(bool)
    history = _scene(history, 4, 'history')
    rows, cols = fine.shape[-2:]
    if coarse.shape != (1, rows // 4, cols // 4) or support.shape != (1, rows, cols):
        raise ValueError('coarse or support shape is incompatible with fine')
    if history.shape[1:] != (9, rows, cols):
        raise ValueError('history must have nine encoded channels per slot')
    if mode not in ('pool', 'nearest_two'):
        raise ValueError('unknown frozen reference-combination mode')
    references = _historical_references(history)
    slots = sorted(references)
    if mode == 'pool':
        pairs = list(itertools.combinations_with_replacement(slots, 2))
    elif slots:
        chosen = sorted(slots, key=lambda k: (references[k][3], k))[:2]
        pairs = [(chosen[0], chosen[-1])]
    else:
        pairs = []
    target_coarse = coarse[0].astype(np.float64).repeat(4, 0).repeat(4, 1)

    def run_pair(pair):
        left, right = (references[k] for k in pair)
        joint = np.isfinite(left[0]) & np.isfinite(right[0]) & \
                np.isfinite(left[1]) & np.isfinite(right[1]) & np.isfinite(target_coarse)
        if not joint.any():
            return None, None
        model = train_arrays(left[0], right[0], left[1], right[1],
                             window_radius=WINDOW_RADIUS, patch_size=PATCH_SIZE,
                             method='zero_bias', workers=1)
        prediction = predict_arrays(model, [target_coarse], VALUE_RANGE, workers=1)[0]
        weight = np.where(np.isfinite(prediction), np.sqrt(left[2] * right[2]), 0.0)
        return np.nan_to_num(prediction, nan=0.0), weight

    weighted = np.zeros((rows, cols), np.float64)
    denominator = np.zeros_like(weighted)
    successful = 0
    # First pair serializes numba's first-use compilation; subsequent kernels release GIL.
    first = [run_pair(pairs[0])] if pairs else []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        rest = executor.map(run_pair, pairs[1:])
        for prediction, weight in itertools.chain(first, rest):
            if prediction is None:
                continue
            weighted += prediction * weight
            denominator += weight
            successful += 1
    fallback = fine[0].astype(np.float64)
    if not np.isfinite(fallback).all():
        raise ValueError('released coarse interpolation fallback must be finite')
    proposal = np.where(denominator > 0, weighted / np.maximum(denominator, 1e-300), fallback)
    prediction = repair_numpy(proposal[None, None], coarse[None], support[None])
    if not np.isfinite(prediction).all():
        raise ValueError('final prediction contains nonfinite values')
    if not return_details:
        return prediction
    supported = support[0]
    return prediction, {
        'mode': mode, 'active_slots': slots, 'reference_pairs': pairs,
        'pair_count': len(pairs), 'pairs_with_joint_observations': successful,
        'supported_fusion_fraction': float(np.mean((denominator > 0)[supported])) if supported.any() else 0.,
        'supported_fallback_fraction': float(np.mean((denominator == 0)[supported])) if supported.any() else 0.,
    }


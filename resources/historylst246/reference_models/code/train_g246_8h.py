#!/usr/bin/env python3
"""Full-scene G246 experiment with strict RMSE selection and bounded runtime.

Uses only the registered Fit603 supervision and development Validation45.
The frozen old campaign is not modified. Cache metadata never enters forward.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import signal
import shutil
import time

import numpy as np
import torch
from torch import nn

from train_g246_metric import atomic_json, atomic_torch_save, scene_metrics

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/g246_8h_20260905'
KEYS = ('fine', 'coarse', 'support', 'context', 'target', 'formal', 'valid')
METRICS = ('rmse_k', 'mae_k', 'auprc_q90', 'iou_q90', 'true_hotspot_mae_q90_k')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class Cache:
    def __init__(self, root, role, device, preload, detail_root=None, hourly_root=None,
                 emissivity_root=None, historical_root=None):
        self.root = Path(root) / role
        meta = json.loads((self.root / 'metadata.json').read_text())
        self.records = meta['scenes']
        self.arrays = {k: np.load(self.root / (k + '.npy'), mmap_mode='r') for k in KEYS}
        self.device = device
        self.tensors = {}
        self.detail = None
        self.hourly = None
        self.emissivity = None
        self.history = None
        if detail_root is not None:
            from build_g246_8h_optical_detail import OpticalDetailCache
            self.detail = OpticalDetailCache(detail_root, role,
                                            expected_scene_ids=[r['scene_id'] for r in self.records])
        if hourly_root is not None:
            from build_g246_8h_hourly_cache import HourlyCache
            self.hourly = HourlyCache(hourly_root, role,
                                     expected_scene_ids=[r['scene_id'] for r in self.records])
        if emissivity_root is not None:
            from build_g246_8h_emissivity_cache import EmissivityCache
            self.emissivity = EmissivityCache(emissivity_root, role,
                                            expected_scene_ids=[r['scene_id'] for r in self.records])
        if historical_root is not None:
            history_schema = json.loads((Path(historical_root) / 'manifest.json').read_text()).get('schema')
            if history_schema == 'g246-8h-historical-nine-cache-v1':
                from build_g246_8h_historical_nine_cache import HistoricalNineCache as HistoryCache
            elif history_schema == 'g246-8h-historical-seven-cache-v1':
                from build_g246_8h_historical_seven_cache import HistoricalSevenCache as HistoryCache
            elif history_schema == 'g246-8h-historical-six-cache-v1':
                from build_g246_8h_historical_six_cache import HistoricalSixCache as HistoryCache
            else:
                from build_g246_8h_historical_cache import HistoricalCache as HistoryCache
            self.history = HistoryCache(historical_root, role,
                                           expected_scene_ids=[r['scene_id'] for r in self.records])
        n = len(self.records)
        assert all(a.shape[0] == n for a in self.arrays.values())
        if preload:
            for key, a in self.arrays.items():
                dtype = torch.bool if a.dtype == np.bool_ else torch.float32
                result = torch.empty(a.shape, dtype=dtype, device=device)
                for start in range(0, n, 16):
                    result[start:start+16].copy_(torch.from_numpy(np.array(a[start:start+16])))
                self.tensors[key] = result
        region_cities = defaultdict(set)
        city_scenes = defaultdict(int)
        for row in self.records:
            region_cities[row['region']].add(row['city'])
            city_scenes[row['city']] += 1
        self.probabilities = np.array([
            1 / (len(region_cities) * len(region_cities[r['region']]) * city_scenes[r['city']])
            for r in self.records
        ], dtype=np.float64)
        self.probabilities /= self.probabilities.sum()
        self.city_indices = defaultdict(list)
        for i, row in enumerate(self.records):
            self.city_indices[row['city']].append(i)

    def batch(self, indices):
        if self.tensors:
            index = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            result = {k: t.index_select(0, index) for k, t in self.tensors.items()}
        else:
            result = {k: torch.from_numpy(np.array(a[indices])).to(self.device)
                      for k, a in self.arrays.items()}
        if self.detail is not None:
            result['detail'] = torch.from_numpy(np.array(self.detail.array[indices])).to(self.device)
        if self.hourly is not None:
            result['hourly'] = torch.from_numpy(np.array(self.hourly.array[indices])).to(self.device)
        if self.emissivity is not None:
            result['emissivity'] = torch.from_numpy(np.array(self.emissivity.array[indices])).to(self.device)
        if self.history is not None:
            result['history'] = torch.from_numpy(np.array(self.history.array[indices])).to(self.device)
        return result

    def temporal_batch(self, indices):
        result = self.batch(indices)
        joined = []
        for i in indices:
            siblings = sorted(self.city_indices[self.records[i]['city']],
                              key=lambda j: self.records[j]['scene_id'])
            if len(siblings) != 3:
                raise ValueError('temporal input requires exactly three dates per city')
            joined.append([int(i)] + [j for j in siblings if j != i])
        flat = np.array(joined).reshape(-1)
        for key in ('fine', 'coarse', 'support', 'context'):
            if self.tensors:
                value = self.tensors[key].index_select(0, torch.as_tensor(flat, device=self.device))
            else:
                value = torch.from_numpy(np.array(self.arrays[key][flat])).to(self.device)
            result[key] = value.reshape(len(indices), 3, *value.shape[1:])
        return result


class LegacyModel(nn.Module):
    def __init__(self, family):
        super().__init__()
        from train_g246_r2 import build_model
        self.family = family
        kw = dict(width=48, fine_channels=52, context_dim=19)
        if family == 'r6a':
            self.net = build_model('calibrated_q', **kw, calibrated_no_geo_core=True,
                                   calibrated_activation_checkpointing=False)
        else:
            self.net = build_model('ipmr_q', **kw, ipmr_activation_checkpointing=False)

    def forward(self, fine, coarse, support, context):
        b = fine.shape[0]
        context19 = torch.cat((context[:, :5], context.new_zeros(b, 4), context[:, 5:]), dim=1)
        return self.net(fine[:, None], coarse[:, None], support[:, None],
                        context19[:, None], torch.ones(b, 1, device=fine.device, dtype=torch.bool),
                        torch.zeros(b, device=fine.device, dtype=torch.long)).float()


def create_model(family, width):
    if family in ('historical_r6a', 'historical_innovation_r6a', 'historical_innovation_emissivity_r6a',
                  'historical_innovation_emissivity_r6a_six', 'historical_attended_innovation_emissivity_r6a',
                  'historical_multiscale_innovation_emissivity_r6a',
                  'historical_multiscale_innovation_emissivity_r6a_six',
                  'historical_recent_innovation_emissivity_r6a_seven',
                  'historical_recent_refinement_emissivity_r6a_seven',
                  'historical_recent_innovation_emissivity_r6a_nine',
                  'historical_recent_refinement_emissivity_r6a_nine'):
        return HistoricalModel(width, innovation='innovation' in family, emissivity='emissivity' in family,
                               source_count=9 if family.endswith('_nine') else 7 if family.endswith('_seven') else 6 if family.endswith('_six') else 3,
                               attended='attended' in family, multiscale='multiscale' in family,
                               recent='recent' in family, recent_refinement='recent_refinement' in family)
    if family == 'dino_r6a':
        from g246_8h_dino import DinoR6Net
        return DinoR6Net(LegacyModel('r6a'), width=width)
    if family == 'resnet18':
        from g246_8h_resnet import G246EightHourResNet18
        return G246EightHourResNet18(width=width)
    if family in ('r6a', 'r9'):
        return LegacyModel(family)
    if family in ('temporal', 'temporal_r6a'):
        return TemporalModel(width, 'r6a' if family == 'temporal_r6a' else 'multiscale')
    if family == 'optical_native_r6a':
        return OpticalModel('native_r6a', width)
    if family == 'optical_emissivity_r6a':
        return OpticalEmissivityModel(width)
    if family.startswith('optical_'):
        return OpticalModel(family.removeprefix('optical_'), width)
    if family == 'hourly_r6a':
        return HourlyModel(width)
    if family == 'emissivity_r6a':
        return EmissivityModel(width)
    if family != 'multiscale':
        raise ValueError(f'unsupported model family: {family}')
    from g246_8h_network import G2468HNetwork
    widths = tuple(max(8, round(c * width / 48 / 8) * 8) for c in (48, 80, 128, 192, 256))
    return G2468HNetwork(widths=widths)


class TemporalModel(nn.Module):
    def __init__(self, width, query_family='multiscale'):
        super().__init__()
        from g246_8h_temporal import G246EightHourTemporalNet
        self.net = G246EightHourTemporalNet(width=width)
        if query_family == 'r6a':
            self.net.query_network = LegacyModel('r6a')

    def forward(self, fine, coarse, support, context):
        return self.net(fine, coarse, support, context,
                        torch.zeros(fine.shape[0], device=fine.device, dtype=torch.long))


class OpticalModel(nn.Module):
    def __init__(self, base_family, width):
        super().__init__()
        if base_family == 'native_r6a':
            from g246_8h_native_optical import NativeOpticalNet
            self.net = NativeOpticalNet(LegacyModel('r6a'))
        else:
            from g246_8h_optical_network import OpticalDetailNet
            self.net = OpticalDetailNet(create_model(base_family, width))

    def forward(self, fine, coarse, support, context, detail):
        return self.net(fine, coarse, support, context, detail)


class HourlyModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        from g246_8h_hourly_network import HourlyConditionedNet
        self.net = HourlyConditionedNet(LegacyModel('r6a'))

    def forward(self, fine, coarse, support, context, hourly):
        return self.net(fine, coarse, support, context, hourly)


class EmissivityModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        from g246_8h_emissivity_network import EmissivityNet
        self.net = EmissivityNet(LegacyModel('r6a'))

    def forward(self, fine, coarse, support, context, emissivity):
        return self.net(fine, coarse, support, context, emissivity)


class OpticalEmissivityModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        from g246_8h_optical_emissivity import OpticalEmissivityNet
        self.net = OpticalEmissivityNet(OpticalModel('r6a', width))

    @property
    def model_config(self):
        config = dict(self.net.model_config)
        config['emissivity_width'] = config.pop('width')
        return config

    def forward(self, fine, coarse, support, context, detail, emissivity):
        return self.net(fine, coarse, support, context, detail, emissivity)


class HistoricalModel(nn.Module):
    def __init__(self, width, *, innovation=False, emissivity=False, source_count=3, attended=False,
                 multiscale=False, recent=False, recent_refinement=False):
        super().__init__()
        self.innovation, self.emissivity = innovation, emissivity
        self.source_count, self.attended = source_count, attended
        self.multiscale = multiscale
        self.recent = recent
        self.recent_refinement = recent_refinement
        backbone = (HistoricalModel(width, innovation=True, emissivity=True, source_count=6)
                    if recent_refinement else EmissivityModel(width) if emissivity else LegacyModel('r6a'))
        if recent_refinement:
            if multiscale or attended or source_count not in (7, 9) or not emissivity:
                raise ValueError('recent refinement requires its registered seven- or nine-source emissivity contract')
            if source_count == 9:
                from g246_8h_historical_recent_refinement_multi import HistoricalRecentMultiRefinementNet
                self.net = HistoricalRecentMultiRefinementNet(backbone, width=width, modality_dropout=0.0)
            else:
                from g246_8h_historical_recent_refinement import HistoricalRecentRefinementNet
                self.net = HistoricalRecentRefinementNet(backbone, width=width, modality_dropout=0.0)
        elif recent:
            if multiscale or attended or source_count not in (7, 9):
                raise ValueError('recent input requires a separately registered seven- or nine-source family')
            if source_count == 9:
                from g246_8h_historical_recent_multi import HistoricalRecentMultiInnovationNet as RecentNet
            else:
                from g246_8h_historical_recent import HistoricalRecentInnovationNet as RecentNet
            self.net = RecentNet(backbone, width=width, source_count=source_count,
                initial_gain=.28, initial_replacement_gain=.22, modality_dropout=.25)
        elif multiscale:
            if attended:
                raise ValueError('multiscale and attended are separately registered families')
            from g246_8h_historical_multiscale import HistoricalMultiscaleInnovationNet
            self.net = HistoricalMultiscaleInnovationNet(backbone, width=width, source_count=source_count,
                initial_gain=.28, initial_replacement_gain=.22, modality_dropout=.25)
        elif attended:
            from g246_8h_historical_attended import HistoricalAttendedInnovationNet
            self.net = HistoricalAttendedInnovationNet(backbone, width=width, source_count=source_count,
                initial_gain=.28, initial_replacement_gain=.22, modality_dropout=.25,
                initial_attention_gain=0.0)
        elif source_count == 6:
            from g246_8h_historical_multisource import HistoricalMultiSourceNet
            self.net = HistoricalMultiSourceNet(backbone, width=width, source_count=6,
                initial_gain=.28, initial_replacement_gain=.22, modality_dropout=.25)
        elif innovation:
            from g246_8h_historical_innovation_network import HistoricalInnovationNet
            self.net = HistoricalInnovationNet(backbone, width=width,
                initial_gain=.28 if emissivity else .33,
                initial_replacement_gain=.22 if emissivity else .26, modality_dropout=.25)
        else:
            from g246_8h_historical_network import HistoricalGuideNet
            self.net = HistoricalGuideNet(backbone, width=width, initial_gain=.1, modality_dropout=.25)

    @property
    def model_config(self):
        return {**self.net.model_config,
                'backbone_family': 'historical_innovation_emissivity_r6a_six' if self.recent_refinement
                                   else 'emissivity_r6a' if self.emissivity else 'r6a'}

    def forward(self, fine, coarse, support, context, *extra):
        if self.emissivity:
            if len(extra) != 2:
                raise ValueError('historical emissivity model requires emissivity then history')
            emissivity, history = extra
            return self.net(fine, coarse, support, context, history, emissivity)
        if len(extra) != 1:
            raise ValueError('historical model requires one history tensor')
        return self.net(fine, coarse, support, context, extra[0])


def forward_batch(model, batch):
    inputs = [batch[k] for k in ('fine', 'coarse', 'support', 'context')]
    if isinstance(model, (OpticalModel, OpticalEmissivityModel)):
        inputs.append(batch['detail'])
    if isinstance(model, HourlyModel):
        inputs.append(batch['hourly'])
    if isinstance(model, (EmissivityModel, OpticalEmissivityModel)) or \
            (isinstance(model, HistoricalModel) and model.emissivity):
        inputs.append(batch['emissivity'])
    if isinstance(model, HistoricalModel):
        inputs.append(batch['history'])
    return model(*inputs)


def initialize(model, path, source, snapshot_path=None):
    from train_g246_r2 import _selected_initialization_state
    from g246_8h_dino import DinoR6Net
    checkpoint_bytes = Path(path).read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    if checkpoint.get('locked_test_opened') is not False:
        raise ValueError('initialization must explicitly be locked-test closed')
    if checkpoint.get('schema') == 'g246-8h-deploy-v1':
        state = checkpoint['state_dict']
        if isinstance(model, HistoricalModel) and model.recent_refinement and \
                checkpoint['model_spec']['family'] == 'historical_innovation_emissivity_r6a_six':
            model.net.load_six_deploy(checkpoint)
        elif isinstance(model, HistoricalModel) and model.recent_refinement and model.source_count == 9 and \
                checkpoint['model_spec']['family'] == 'historical_recent_refinement_emissivity_r6a_seven':
            model.net.load_seven_deploy(checkpoint)
        elif isinstance(model, HistoricalModel) and model.multiscale and \
                checkpoint['model_spec']['family'] in (
                    'historical_innovation_emissivity_r6a', 'historical_innovation_emissivity_r6a_six'):
            model.net.load_historical_deploy(checkpoint)
        elif isinstance(model, HistoricalModel) and model.attended and \
                checkpoint['model_spec']['family'] == 'historical_innovation_emissivity_r6a':
            model.net.load_historical_deploy(checkpoint)
        elif isinstance(model, HistoricalModel) and checkpoint['model_spec']['family'] == \
                ('emissivity_r6a' if model.emissivity else 'r6a'):
            model.net.backbone.load_state_dict(state, strict=True)
        elif isinstance(model, OpticalEmissivityModel) and checkpoint['model_spec']['family'] == 'optical_r6a':
            model.net.load_optical_deploy(checkpoint)
        elif isinstance(model, TemporalModel) and checkpoint['model_spec']['family'] in ('multiscale', 'r6a'):
            model.net.initialize_query_weights(state)
        elif isinstance(model, DinoR6Net) and checkpoint['model_spec']['family'] == 'r6a':
            model.backbone.load_state_dict(state, strict=True)
        elif isinstance(model, (OpticalModel, HourlyModel, EmissivityModel)) and checkpoint['model_spec']['family'] in ('r6a', 'multiscale'):
            model.net.backbone.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
        key = 'state_dict'
    elif checkpoint.get('schema') == 'g246-8h-checkpoint-v1':
        key = 'ema_state_dict' if source == 'ema' else 'model_state_dict'
        state = checkpoint[key]
        model.load_state_dict(state, strict=True)
    else:
        state, key, _ = _selected_initialization_state(checkpoint, source)
        if isinstance(model, HistoricalModel):
            if model.emissivity:
                raise ValueError('historical/emissivity model requires a complete emissivity deployment')
            model.net.backbone.net.load_state_dict(state, strict=True)
        elif isinstance(model, OpticalEmissivityModel):
            raise ValueError('optical/emissivity combination requires a complete optical deployment initialization')
        elif isinstance(model, DinoR6Net):
            model.backbone.net.load_state_dict(state, strict=True)
        elif isinstance(model, (OpticalModel, HourlyModel, EmissivityModel)):
            model.net.backbone.net.load_state_dict(state, strict=True)
        elif isinstance(model, TemporalModel):
            model.net.query_network.net.load_state_dict(state, strict=True)
        else:
            model.net.load_state_dict(state, strict=True)
    receipt = {'path': str(Path(path).resolve()), 'sha256': checkpoint_sha256, 'state_key': key}
    if snapshot_path is not None:
        snapshot_path = Path(snapshot_path)
        temporary = snapshot_path.with_suffix(snapshot_path.suffix + '.partial')
        temporary.write_bytes(checkpoint_bytes)
        temporary.replace(snapshot_path)
        receipt['immutable_checkpoint_snapshot'] = str(snapshot_path.resolve())
    return receipt


def repair_numpy(prediction, coarse, support):
    value = np.asarray(prediction, np.float64).copy()
    s = np.asarray(support, bool)
    n, _, h, w = value.shape
    block_shape = (n, 1, h//4, 4, w//4, 4)
    count = s.reshape(block_shape).sum(axis=(3, 5))
    sums = np.where(s, value, 0).reshape(block_shape).sum(axis=(3, 5))
    mean = sums / np.maximum(count, 1)
    correction = np.where(np.isfinite(coarse) & (count > 0), coarse - mean, 0)
    value += np.repeat(np.repeat(correction, 4, axis=-2), 4, axis=-1)
    return np.where(s, value, 0)


def score(predictions, data):
    rows = []
    for i, meta in enumerate(data.records):
        rows.append({**meta, **scene_metrics(predictions[i, 0], data.arrays['target'][i, 0],
                                           data.arrays['formal'][i, 0])})
    cities = defaultdict(list)
    for row in rows:
        cities[row['city']].append(row)
    avg = lambda records: {k: float(np.mean([r[k] for r in records])) for k in METRICS}
    per_city = {city: {'region': rr[0]['region'], **avg(rr)} for city, rr in cities.items()}
    regions = sorted({r['region'] for r in rows})
    per_region = {region: avg([r for r in per_city.values() if r['region'] == region]) for region in regions}
    return {'aggregation': 'scene_to_city_to_region_equal_region',
            'equal_region': avg(list(per_region.values())), 'per_region': per_region,
            'per_city': per_city, 'per_scene': rows}


@torch.inference_mode()
def evaluate(model, data, batch_size, amp):
    model.eval()
    predictions = []
    for start in range(0, len(data.records), batch_size):
        batcher = data.temporal_batch if isinstance(model, TemporalModel) else data.batch
        b = batcher(np.arange(start, min(start+batch_size, len(data.records))))
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            pred = forward_batch(model, b)
        predictions.append(pred.float().cpu().numpy())
    prediction = repair_numpy(np.concatenate(predictions), data.arrays['coarse'], data.arrays['support'])
    return score(prediction, data), prediction


def objective(pred, batch, auxiliary):
    target = torch.nan_to_num(batch['target'].float())
    squared = (pred.float() - target).square()
    mask = batch['formal'].bool()
    dims = (1, 2, 3)
    mse = torch.where(mask, squared, 0).sum(dims) / mask.sum(dims).clamp_min(1)
    loss = (mse + 1e-6).sqrt()
    if auxiliary:
        mask = batch['valid'].bool()
        other = torch.where(mask, squared, 0).sum(dims) / mask.sum(dims).clamp_min(1)
        loss = (1-auxiliary) * loss + auxiliary * (other + 1e-6).sqrt()
    return loss.mean()


def train(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('existing training run requires --resume')
    manifest_path = args.cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('locked_test_opened') is not False:
        raise ValueError('cache must be explicitly locked-test closed')
    if args.family.startswith('optical_') != bool(args.detail_root):
        raise ValueError('optical models require --detail-root; other models do not consume it')
    if (args.family == 'hourly_r6a') != bool(args.hourly_root):
        raise ValueError('hourly models require --hourly-root; other models do not consume it')
    if (args.family in ('emissivity_r6a', 'optical_emissivity_r6a', 'historical_innovation_emissivity_r6a',
                        'historical_innovation_emissivity_r6a_six',
                        'historical_attended_innovation_emissivity_r6a',
                        'historical_multiscale_innovation_emissivity_r6a',
                        'historical_multiscale_innovation_emissivity_r6a_six',
                        'historical_recent_innovation_emissivity_r6a_seven',
                        'historical_recent_refinement_emissivity_r6a_seven',
                        'historical_recent_innovation_emissivity_r6a_nine',
                        'historical_recent_refinement_emissivity_r6a_nine')) != bool(args.emissivity_root):
        raise ValueError('emissivity models require --emissivity-root')
    if args.family.startswith('historical_') != bool(args.historical_root):
        raise ValueError('historical models require --historical-root')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')
    rng = np.random.default_rng(args.seed)
    model = create_model(args.family, args.width)
    nparams = sum(p.numel() for p in model.parameters())
    if nparams >= 20_000_000:
        raise ValueError(f'parameter budget exceeded: {nparams}')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    snapshot = out / 'source'
    snapshot.mkdir(exist_ok=True)
    source_dependencies = {}
    code_hashes = {}
    for path in (ROOT / 'code').glob('*.py'):
        if not args.resume:
            shutil.copy2(path, snapshot / path.name)
        code_hashes[path.name] = sha(path)
    config.update(parameter_count=nparams, cache_manifest_sha256=sha(manifest_path),
                  locked_test_opened=False, training_scope='Fit603 only',
                  validation_scope='development Validation45, adaptively used',
                  objective='equal-region / city / scene smoothed full-scene RMSE',
                  selection='strict minimum unrounded validation RMSE across raw and EMA',
                  input_contract='query Fine52, physical Context15, coarse and support only',
                  geometric_augmentation=args.augment, scene_sampling='equal region then city then date',
                  source_code_sha256=code_hashes)
    if args.detail_root:
        config['optical_detail_manifest_sha256'] = sha(args.detail_root / 'manifest.json')
        config['input_contract'] += '; aligned native 30m optical detail and optical QA only'
    if args.hourly_root:
        config['hourly_manifest_sha256'] = sha(args.hourly_root / 'manifest.json')
        config['input_contract'] += '; causal 48h physical forcing tokens'
    if args.emissivity_root:
        config['emissivity_manifest_sha256'] = sha(args.emissivity_root / 'manifest.json')
        config['input_contract'] += '; joint-masked upstream emissivity mean, variation and availability'
        config['emissivity_prior_initialization'] = {
            'gain': 0.5, 'scope': 'Fit12 scalar pilot, all-Fit anchor; not whole-stack OOF',
            'pilot_receipt_sha256': sha(CAMPAIGN / 'emissivity_pilot_scalar_screen.json')}
    if args.historical_root:
        historical_manifest = json.loads((args.historical_root / 'manifest.json').read_text())
        if historical_manifest.get('status') != 'complete' or historical_manifest.get('locked_test_opened') is not False:
            raise ValueError('historical feature cache must be complete and locked-test closed')
        config['historical_manifest_sha256'] = sha(args.historical_root / 'manifest.json')
        config['input_contract'] += (f'; six frozen pre-2021 historical slots plus {model.source_count - 6} fully excluded '
                                     '8--64 day pre-query observations with historical-only QA'
                                     if model.recent else
                                     f'; {model.source_count} registered pre-2021 historical thermal slots with historical-only QA')
        config['historical_prior_initialization'] = {
            'gain': model.net.initial_gain,
            'replacement_gain': getattr(model.net, 'initial_replacement_gain', 0.),
            'scope': 'exploratory Fit12 correction leave-city-out; all-Fit neural anchor; not whole-stack OOF',
            'pilot_receipt_sha256': sha(CAMPAIGN / ('historical_fit12_innovation_followup/result.json'
                if model.innovation else 'historical_fit12_conditional_screen/result.json')),
            'training_supervision': 'unchanged Fit603',
            'source_time_contract': '2026 present-day historical replay; no claim of at-query public availability'}
        if model.attended:
            config['historical_prior_initialization']['attention_gain'] = 0.0
            config['historical_prior_initialization']['attention_scope'] = 'zero preserves full trained historical initialization; weights learned on Fit603'
        if model.recent_refinement:
            config['historical_prior_initialization'] = {
                'gain': 0.0, 'replacement_gain': 0.0,
                'scope': 'all new outputs zero; complete trained six-source backbone reused; no new coefficient fitting',
                'training_supervision': 'unchanged Fit603', 'recent_modality_dropout': 0.0}
    if isinstance(model, TemporalModel):
        config['input_contract'] = ('query and two same-city auxiliary dates: Fine52, physical '
                                    'Context15, coarse and support only; no auxiliary fine targets')
        alignment_path = CAMPAIGN / 'temporal_alignment.json'
        alignment = json.loads(alignment_path.read_text())
        if alignment.get('locked_test_opened') is not False or not all(
            alignment['roles'][role]['all_three_date_grids_identical'] for role in ('fit', 'validation')
        ):
            raise ValueError('three-date canonical grid audit must pass')
        config['temporal_alignment_sha256'] = sha(alignment_path)
    if args.init_checkpoint and not args.resume:
        config['initialization'] = initialize(model, args.init_checkpoint, args.init_source,
                                              snapshot_path=out / 'initialization.pt')
        if isinstance(model, HistoricalModel):
            config['historical_prior_initialization']['actual_loaded_gain'] = float(model.net.physical_gain.detach())
            config['historical_prior_initialization']['actual_loaded_replacement_gain'] = float(
                getattr(model.net, 'replacement_gain', torch.tensor(0.)).detach())
            if model.attended:
                config['historical_prior_initialization']['actual_loaded_attention_gain'] = float(
                    model.net.physical_attention_gain.detach())
    elif args.family == 'resnet18' and not args.resume:
        config['initialization'] = model.load_pretrained()
        config['pretraining_scope'] = 'existing local ImageNet ResNet18 weights; G246 targets Fit603 only'
    if args.family == 'dino_r6a' and not args.resume:
        if not args.init_checkpoint:
            raise ValueError('DINO residual model requires a trained r6a initialization')
        config['encoder_initialization'] = model.load_pretrained()
        config['pretraining_scope'] = 'official external DINOv2 self-supervised visual prior; G246 targets Fit603 only'
        from g246_8h_deployment_sources import capture_dino_source, source_summary, materialize_dino_source
        source_dependencies['dinov2'] = capture_dino_source()
        config['source_dependencies'] = {'dinov2': source_summary(source_dependencies['dinov2'])}
        materialize_dino_source(source_dependencies['dinov2'], snapshot)
    config['runtime'] = {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
                         'gpu': torch.cuda.get_device_name(0)}
    model.to(device)
    if args.family in ('resnet18', 'dino_r6a'):
        encoder = [p for name, p in model.named_parameters() if name.startswith('encoder.')]
        decoder = [p for name, p in model.named_parameters() if not name.startswith('encoder.')]
        optimizer = torch.optim.AdamW([
            {'params': encoder, 'lr_scale': args.encoder_lr_multiplier},
            {'params': decoder, 'lr_scale': 1.0},
        ], lr=args.lr, weight_decay=args.weight_decay)
    elif isinstance(model, HistoricalModel) and model.recent_refinement:
        gains = [model.net.physical_gain, model.net.replacement_gain]
        gain_ids = {id(p) for p in gains}
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.parameters() if id(p) not in gain_ids], 'lr_scale': 1.0},
            {'params': gains, 'lr_scale': args.recent_prior_lr_multiplier},
        ], lr=args.lr, weight_decay=args.weight_decay)
    elif isinstance(model, HistoricalModel) and model.attended:
        attention = model.net.physical_attention_gain
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.parameters() if p is not attention], 'lr_scale': 1.0},
            {'params': [attention], 'lr_scale': args.attention_lr_multiplier},
        ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    update = 0
    history = []
    best = float('inf')
    if args.resume:
        cp = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        saved = cp['config']
        source_dependencies = cp.get('source_dependencies', {})
        if args.family == 'dino_r6a':
            from g246_8h_deployment_sources import source_summary, materialize_dino_source
            bundle = source_dependencies['dinov2']
            if source_summary(bundle) != saved['source_dependencies']['dinov2']:
                raise ValueError('resume DINO architecture source binding differs')
            materialize_dino_source(bundle, snapshot)
        for k in ('family', 'width', 'lr', 'max_updates', 'seed', 'auxiliary', 'cache_manifest_sha256',
                  'batch_size', 'warmup', 'weight_decay', 'ema_decay', 'amp', 'augment',
                  'encoder_lr_multiplier', 'optical_detail_manifest_sha256', 'hourly_manifest_sha256',
                  'emissivity_manifest_sha256', 'historical_manifest_sha256', 'attention_lr_multiplier',
                  'recent_prior_lr_multiplier'):
            default = 'none' if k == 'augment' else 1.0 if k in (
                'encoder_lr_multiplier', 'attention_lr_multiplier', 'recent_prior_lr_multiplier') else None
            if config.get(k, default) != saved.get(k, default):
                raise ValueError(f'resume scientific config differs: {k}')
        model.load_state_dict(cp['model_state_dict'])
        ema.load_state_dict(cp['ema_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        scaler.load_state_dict(cp['scaler_state_dict'])
        update, history, best = cp['update'], cp['history'], cp['best_rmse_k']
        rng.bit_generator.state = cp['numpy_rng']
        torch.set_rng_state(cp['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in cp['cuda_rng']])
        config = saved
    atomic_json(out / 'config.json', config)
    print(json.dumps({'event': 'model_ready', 'parameter_count': nparams, 'update': update}), flush=True)
    fit = Cache(args.cache, 'fit', device, args.preload_gpu, args.detail_root, args.hourly_root,
                args.emissivity_root, args.historical_root)
    validation = Cache(args.cache, 'validation', device, args.preload_gpu, args.detail_root, args.hourly_root,
                       args.emissivity_root, args.historical_root)
    if len(fit.records) != 603 or len(validation.records) != 45:
        raise ValueError('requires exactly current Fit603 and Validation45')
    if isinstance(model, TemporalModel):
        for role, data in [('fit', fit), ('validation', validation)]:
            audited = {r['scene_id'] for rows in alignment['roles'][role]['cities'].values() for r in rows}
            if audited != {r['scene_id'] for r in data.records}:
                raise ValueError('temporal grid audit scene membership differs')
    print(json.dumps({'event': 'cache_ready', 'gpu_memory_bytes': torch.cuda.memory_allocated()}), flush=True)
    start = time.time()
    deadline = min(float(json.loads((CAMPAIGN / 'campaign.json').read_text())['deadline_unix']) - 120,
                   start + args.max_minutes * 60)
    pause = {'requested': False}
    def stop_signal(signum, frame):
        pause['requested'] = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def cpu_state(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def save_last():
        atomic_torch_save(out / 'last.pt', {
            'schema': 'g246-8h-checkpoint-v1', 'config': config, 'update': update,
            'source_dependencies': source_dependencies,
            'model_state_dict': cpu_state(model), 'ema_state_dict': cpu_state(ema),
            'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'best_rmse_k': best, 'history': history, 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'locked_test_opened': False})

    def eval_and_save():
        nonlocal best
        record = {'update': update, 'wall_seconds': time.time()-start}
        for label, module in [('raw', model), ('ema', ema)]:
            metrics, predictions = evaluate(module, validation, args.eval_batch_size, args.amp)
            record[label] = metrics
            rmse = metrics['equal_region']['rmse_k']
            if rmse < best:
                best = rmse
                deploy = {'schema': 'g246-8h-deploy-v1', 'state_dict': cpu_state(module),
                          'source_dependencies': source_dependencies,
                          'model_spec': {'family': args.family, 'width': args.width,
                                         **getattr(module, 'model_config', {})},
                          'parameter_count': nparams,
                          'named_parameter_shapes': {k: list(p.shape) for k, p in module.named_parameters()},
                          'config': config, 'selected_update': update, 'selected_weights': label,
                          'validation_rmse_k': rmse, 'cache_manifest_sha256': sha(manifest_path),
                          'locked_test_opened': False}
                atomic_torch_save(out / 'deploy.pt', deploy)
                np.save(out / 'validation_predictions.npy', predictions, allow_pickle=False)
                atomic_json(out / 'best_metrics.json', {'update': update, 'weights': label, **metrics})
                print(json.dumps({'event': 'best', 'update': update, 'weights': label, 'rmse_k': rmse}), flush=True)
        history.append(record)
        atomic_json(out / 'history.json', {'config': config, 'records': history, 'best_rmse_k': best})
        save_last()
        print(json.dumps({'event': 'evaluation', 'update': update,
                          'raw_rmse_k': record['raw']['equal_region']['rmse_k'],
                          'ema_rmse_k': record['ema']['equal_region']['rmse_k'],
                          'best_rmse_k': best, 'wall_seconds': time.time()-start}), flush=True)

    if not history:
        eval_and_save()
    losses = []
    logged_at = time.time()
    while update < args.max_updates and time.time() < deadline and not pause['requested']:
        if (out / 'STOP').exists():
            break
        ids = rng.choice(len(fit.records), args.batch_size, p=fit.probabilities)
        batch = fit.temporal_batch(ids) if isinstance(model, TemporalModel) else fit.batch(ids)
        if args.augment == 'd4':
            from g246_8h_augment import transform_batch
            batch = transform_batch(batch, int(rng.integers(8)))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step = update + 1
        if step <= args.warmup:
            lr = args.lr * step / max(args.warmup, 1)
        else:
            ratio = min(1., (step-args.warmup) / max(1, args.max_updates-args.warmup))
            lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * ratio)))
        for group in optimizer.param_groups:
            group['lr'] = lr * group.get('lr_scale', 1.0)
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
            pred = forward_batch(model, batch)
            loss = objective(pred, batch, args.auxiliary)
        if not bool(torch.isfinite(loss)):
            save_last()
            raise FloatingPointError('non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        update += 1
        decay = min(args.ema_decay, (1+update)/(10+update))
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, 1-decay)
            for eb, b in zip(ema.buffers(), model.buffers()):
                eb.copy_(b)
        losses.append(float(loss.detach()))
        if update % 50 == 0:
            status = {'event': 'train', 'update': update, 'mean_loss_k': float(np.mean(losses)),
                      'lr': lr, 'seconds_per_update': (time.time()-logged_at)/len(losses),
                      'best_rmse_k': best, 'elapsed_seconds': time.time()-start,
                      'deadline_unix': deadline, 'pid': __import__('os').getpid(),
                      'status': 'running', 'locked_test_opened': False}
            atomic_json(out / 'state.json', status)
            print(json.dumps(status), flush=True)
            losses.clear()
            logged_at = time.time()
        if update % args.eval_interval == 0:
            eval_and_save()
            logged_at = time.time()
    if not history or history[-1]['update'] != update:
        eval_and_save()
    else:
        save_last()
    atomic_json(out / 'state.json', {'status': 'completed' if update >= args.max_updates else 'paused',
                                     'update': update, 'best_rmse_k': best,
                                     'parameter_count': nparams, 'wall_seconds': time.time()-start,
                                     'locked_test_opened': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT / 'artifacts/g246_8h/cache_v1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--family', choices=('multiscale', 'r6a', 'r9', 'resnet18', 'dino_r6a', 'temporal', 'temporal_r6a',
                                        'optical_r6a', 'optical_multiscale', 'optical_native_r6a',
                                        'hourly_r6a', 'emissivity_r6a', 'optical_emissivity_r6a',
                                        'historical_r6a', 'historical_innovation_r6a',
                                        'historical_innovation_emissivity_r6a',
                                        'historical_innovation_emissivity_r6a_six',
                                        'historical_attended_innovation_emissivity_r6a',
                                        'historical_multiscale_innovation_emissivity_r6a',
                                        'historical_multiscale_innovation_emissivity_r6a_six',
                                        'historical_recent_innovation_emissivity_r6a_seven',
                                        'historical_recent_refinement_emissivity_r6a_seven',
                                        'historical_recent_innovation_emissivity_r6a_nine',
                                        'historical_recent_refinement_emissivity_r6a_nine'), default='multiscale')
    p.add_argument('--width', type=int, default=48)
    p.add_argument('--init-checkpoint', type=Path)
    p.add_argument('--detail-root', type=Path)
    p.add_argument('--hourly-root', type=Path)
    p.add_argument('--emissivity-root', type=Path)
    p.add_argument('--historical-root', type=Path)
    p.add_argument('--init-source', choices=('selected', 'raw', 'ema'), default='selected')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--encoder-lr-multiplier', type=float, default=1.0)
    p.add_argument('--attention-lr-multiplier', type=float, default=1.0)
    p.add_argument('--recent-prior-lr-multiplier', type=float, default=1.0)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--max-updates', type=int, default=6000)
    p.add_argument('--eval-interval', type=int, default=250)
    p.add_argument('--max-minutes', type=float, default=100)
    p.add_argument('--auxiliary', type=float, default=0.0)
    p.add_argument('--ema-decay', type=float, default=0.995)
    p.add_argument('--augment', choices=('none', 'd4'), default='none')
    p.add_argument('--preload-gpu', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--threads', type=int, default=6)
    train(p.parse_args())


if __name__ == '__main__':
    main()

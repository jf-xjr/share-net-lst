"""Portable single-view inference for the three retained compact checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'resources/historylst246'), str(ROOT / 'research/sub04_20260913')]
import numpy as np
import torch
from historylst.data import Dataset, INPUTS
from historylst.metrics import repair
from compact_query_product.model import build_model, PARAMETERS


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[20260905, 20260912, 20260913], default=20260905)
    parser.add_argument('--checkpoint', type=Path, help='Optional relocated copy of the selected checkpoint')
    parser.add_argument('--data', type=Path, default=ROOT / 'resources/historylst246')
    parser.add_argument('--split', choices=['fit', 'validation', 'test'], default='test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='Check checkpoint identity and strict loading; no data or predictions')
    args = parser.parse_args()
    relative = f'research/sub04_20260913/compact_query_product/runs/compact_{args.seed}/best.pt'
    checkpoint = args.checkpoint or ROOT / relative
    manifest = json.loads((ROOT / 'release-assets.json').read_text())
    record, = [item for group in manifest['groups'] for item in group['files'] if item['path'] == relative]
    if sha(checkpoint) != record['sha256']:
        raise ValueError('Checkpoint differs from the selected release weights')
    torch.set_num_threads(args.threads)
    model = build_model(history_dropout=0., emissivity_dropout=0.)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    if args.check:
        print(json.dumps(dict(seed=args.seed, parameters=PARAMETERS, strict_loading='passed', predictions_generated=False)))
        return
    if args.output is None:
        parser.error('--output is required for inference')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    if args.output.suffix != '.npy':
        parser.error('--output must end in .npy')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Dataset(args.data, args.split, labels=False)
    model = model.eval().to(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    predictions = np.lib.format.open_memmap(temporary, mode='w+', dtype='float64', shape=(len(data), 1, 160, 160))
    with torch.inference_mode():
        for index in range(len(data)):
            batch = data.batch([index])
            values = {key: torch.from_numpy(batch[key]).to(args.device) for key in INPUTS}
            predicted = model(**values).cpu().numpy()
            predictions[index:index + 1] = repair(predicted, batch['coarse'], batch['support'])
            if (index + 1) % 10 == 0:
                print(f'{index + 1}/{len(data)} scenes', flush=True)
    predictions.flush()
    del predictions
    temporary.rename(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(dict(seed=args.seed, split=args.split,
        checkpoint_sha256=record['sha256'], prediction_sha256=sha(args.output), scene_count=len(data),
        labels_opened=False, dtype='float64', model_precision='float32', tf32=False, views=1,
        scene_ids=[r['scene_id'] for r in data.records]), indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Freeze and confirm the L23 local decline on disjoint images.

Plan is read-only and imports no ML libraries. Freeze samples IDs only; extract
requires a frozen protocol. Training and statistics reuse the audited harness.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import phenomenon as p

SPLITS = ('train', 'val', 'test')
ARTIFACTS = ('run.json', 'checkpoint.pt', 'selection.json', 'history.json',
             'metrics.json', 'per_image.json', 'test_predictions.npz')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_splits(splits):
    if set(splits) != set(SPLITS):
        raise ValueError('Expected train/val/test image lists')
    seen = set()
    for name in SPLITS:
        ids = splits[name]
        if not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('Image IDs must be nonempty nonnegative integer lists')
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping split IDs')
        seen.update(ids)
    return seen


def exclusions(s):
    used, hashes = set(), {}
    for name in s['exclusion_split_files']:
        file = Path(name).resolve()
        used.update(validate_splits(read_json(file)))
        hashes[str(file)] = digest(file)
    return used, hashes


def select_fresh(pools, used, c):
    import numpy as np
    validate_splits(pools)
    rng = np.random.default_rng(c['split_seed'])
    result = {}
    for name in SPLITS:
        candidates = sorted(set(pools[name]) - used)
        n = c['max_' + name]
        if len(candidates) < n:
            raise ValueError(f'Insufficient unseen {name} images: {len(candidates)} < {n}')
        result[name] = sorted(rng.choice(candidates, n, replace=False).tolist())
    validate_splits(result)
    return result


def check_settings(c, s):
    if c['model'] != 'facebook/dinov2-large' or c['layers'] != [23] or c['channels'] != ['post_attention', 'residual']:
        raise ValueError('Confirmation is limited to DINOv2-L L23, post_attention/residual')
    if not c['model_revision'] or any(type(c['max_' + k]) is not int or c['max_' + k] <= 0 for k in SPLITS):
        raise ValueError('Pin model revision and positive split sizes')
    if not s['exclusion_split_files']:
        raise ValueError('List every previous exploratory split file')
    if c['max_tasks'] < 4:
        raise ValueError('Confirmation requires an explicit budget of four binding tasks')
    expected = [dict(name='full_seed0', kind='full', seed=0, weight_decay=0.0),
                dict(name='factor64', kind='factor', factor_width=64, seed=0, weight_decay=0.0)]
    if s['variants'] != expected or s['primary_variant'] != 'full_seed0' or s['max_binding_tasks'] != 4:
        raise ValueError('Use the fixed full/factor64 four-task panel')
    for key in ('sampling_seed', 'bootstrap_seed'):
        if type(s[key]) is not int or s[key] < 0:
            raise ValueError(f'Invalid {key}')
    for key in ('bootstrap_repeats', 'bootstrap_bins', 'verify_images'):
        if type(s[key]) is not int or s[key] < (2 if key != 'verify_images' else 1):
            raise ValueError(f'Invalid {key}')
    for key in ('minimum_effect', 'bootstrap_max_auc_error'):
        if not isinstance(s[key], (float, int)) or not math.isfinite(s[key]) or not 0 < s[key] < 1:
            raise ValueError(f'Invalid {key}')
    out = Path(c['output']).resolve()
    for file in s['exclusion_split_files']:
        old = Path(file).resolve().parent
        if out == old or out.is_relative_to(old) or old.is_relative_to(out):
            raise ValueError('Confirmation output must be separate from exploratory runs')
    p.enforce_budget(c)


def protocol(c, s):
    used, inputs = exclusions(s)
    files = [p.HERE / name for name in ('confirm_l23.py', 'phenomenon.py',
             'run_server.py', 'investigate_decline.py', 'validate_decline.py', 'local_readout.py')]
    files += [p.HERE.parent / 'vit-object-binding' / name for name in
              ('src/data_extractor.py', 'src/utils/models.py')]
    if c.get('dataset_factory'):
        module = c['dataset_factory'].split(':', 1)[0]
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin or not Path(spec.origin).is_file():
            raise ValueError('Cannot freeze custom dataset factory source')
        files.append(Path(spec.origin))
    # Numeric IDs are meaningful only for the same dataset index. Custom factories
    # must supply an equivalent stable index file in the config.
    index = c.get('confirmation_index_file')
    if not index:
        root = Path(c['dataset_root'])
        index = root / 'index_ade20k.pkl'
        if not index.exists():
            index = root / 'ADE20K_2021_17_01/index_ade20k.pkl'
    files.append(Path(index))
    versions = {}
    for package in ('torch', 'torchvision', 'transformers', 'numpy', 'pandas', 'scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(config=c, checks=s, exclusion_sha256=inputs,
                implementation_and_index_sha256={str(f.resolve()): digest(f) for f in files},
                package_versions=versions,
                excluded_images=len(used),
                estimand='pooled hard-instance AUROC, residual minus post_attention',
                primary_rule=f"full_seed0 image-cluster paired 95% CI upper bound < -{s['minimum_effect']}",
                scope='Conditional on fitted probes; no information-loss or model-use claim')


def freeze(c, s, core):
    out = Path(c['output'])
    current = protocol(c, s)
    if (out / 'frozen_protocol.json').exists():
        verify_frozen(c, s)
        print('Protocol already frozen; unchanged.')
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('Freeze requires a new empty output directory')
    # Request the original loader\'s full eligible pools before exclusion/sampling.
    base = core.Dataset(dict(c, **{'max_' + k: 0 for k in SPLITS}))
    used, _ = exclusions(s)
    splits = select_fresh(base.splits, used, c)
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'splits.json', splits)
    current['splits_sha256'] = digest(out / 'splits.json')
    write(out / 'requested_config.json', c)
    write(out / 'frozen_protocol.json', current)
    print(f'Frozen disjoint split and protocol: {out}')


def verify_frozen(c, s):
    out = Path(c['output'])
    recorded = read_json(out / 'frozen_protocol.json')
    current = protocol(c, s)
    current['splits_sha256'] = digest(out / 'splits.json')
    if current != recorded:
        raise ValueError('Frozen code/config/index/exclusions/splits changed; use a new experiment')
    splits = read_json(out / 'splits.json')
    used, _ = exclusions(s)
    if validate_splits(splits) & used:
        raise ValueError('Confirmation images overlap exploratory images')
    if any(len(splits[k]) != c['max_' + k] for k in SPLITS):
        raise ValueError('Frozen split counts differ')
    return splits


def attach_frozen_dataset(c, s, core):
    splits = verify_frozen(c, s)
    original = core.Dataset

    def factory(cfg):
        base = original(dict(cfg, **{'max_' + k: 0 for k in SPLITS}))
        validate_splits(base.splits)
        if any(not set(splits[k]).issubset(base.splits[k]) for k in SPLITS):
            raise ValueError('Frozen IDs not in original loader eligible pools')
        base.splits = splits
        return base

    core.Dataset = factory


def task_folder(c, name, channel):
    return Path(c['output']) / 'confirmation' / name / 'binding' / channel / 'layer_23'


def verify_completed(dest):
    missing = [str(dest / f) for f in ARTIFACTS if not (dest / f).is_file()]
    if missing:
        raise ValueError('Incomplete probe artifacts: ' + ', '.join(missing))


def train(c, s, core):
    import investigate_decline as inv
    import validate_decline as v
    inv.validate_caches(c)
    for variant in s['variants']:
        out = Path(c['output']) / 'confirmation' / variant['name']
        for ch in c['channels']:
            dest = task_folder(c, variant['name'], ch)
            identity = dict(config=c, variant=variant, channel=ch, layer=23,
                            sampling_seed=s['sampling_seed'], selection='validation hard-instance AUROC')
            if (dest / 'run.json').exists() and read_json(dest / 'run.json') != identity:
                raise ValueError('Probe identity changed')
            if (dest / 'metrics.json').exists():
                verify_completed(dest)
            else:
                write(dest / 'run.json', identity)
        proxy = SimpleNamespace(cached=core.cached, seed_all=core.seed_all,
            batch_logits=core.batch_logits, eval_probe=core.eval_probe,
            probe=lambda cfg, mode, variant=variant: v.make_readout(cfg, variant, core))
        settings = dict(baseline_layers=[], probe_seed=variant['seed'],
                        sampling_seed=s['sampling_seed'], probe_weight_decay=variant['weight_decay'])
        inv.train_binding(c, settings, out, core=proxy)


def gate(ci, epsilon):
    if ci[1] < -epsilon:
        return 'meaningful_decline_confirmed'
    if ci[1] < 0:
        return 'negative_direction_only'
    if ci[0] >= -epsilon and ci[1] <= epsilon:
        return 'small_difference_supported'
    if ci[0] > 0:
        return 'opposite_direction'
    return 'inconclusive'


def report(c, s):
    import numpy as np
    import investigate_decline as inv
    import validate_decline as v
    import local_readout as lr
    inv.validate_caches(c)
    rows = []
    meta, _ = inv.load_meta(c, 'test', 'residual')
    for variant in s['variants']:
        folders = [task_folder(c, variant['name'], ch) for ch in c['channels']]
        for dest in folders:
            verify_completed(dest)
        a, b = [v.predictions(dest / 'test_predictions.npz') for dest in folders]
        for prediction in (a, b):
            lr.check_eval_sample(c, prediction, meta)
        binding = v.paired_bootstrap(a, b, 'instance', s['bootstrap_repeats'],
            s['bootstrap_bins'], s['bootstrap_seed'], s['bootstrap_max_auc_error'])
        category = v.paired_bootstrap(a, b, 'category', s['bootstrap_repeats'],
            s['bootstrap_bins'], s['bootstrap_seed'], s['bootstrap_max_auc_error'])
        crop = meta['position'][np.arange(len(meta['image_id']))[:, None], a['patches']] // ((c['crop_size']//14)**2)
        i, j = np.triu_indices(c['eval_patches'], 1)
        ranking = inv.anchored_rankings(a, b, crop[:, i] != crop[:, j])
        support = {}
        for task in ('instance', 'category'):
            mask, y = v.task_arrays(a, task)
            support[task] = dict(positive=int((mask & y).sum()), negative=int((mask & ~y).sum()),
                images_with_pairs=int(mask.any(axis=1).sum()),
                images_with_both_labels=int(((mask & y).any(axis=1) & (mask & ~y).any(axis=1)).sum()))
        rows.append(dict(variant=variant['name'], binding=binding, support=support,
                         binding_probe_category_score=category, anchored_rankings=ranking,
                         assessment=gate(binding['delta_ci95'], s['minimum_effect'])))
    primary = next(r for r in rows if r['variant'] == s['primary_variant'])
    result = dict(primary_variant=s['primary_variant'], primary_assessment=primary['assessment'],
        minimum_effect=s['minimum_effect'], results=rows,
        robustness_note='factor64 and anchor rankings are secondary; cannot rescue a failed primary endpoint',
        uncertainty='Image sampling only, conditional on fitted probes; no training-seed CI',
        category_note='Category score from a binding probe is not an independent category probe',
        frozen_protocol_sha256=digest(Path(c['output']) / 'frozen_protocol.json'))
    write(Path(c['output']) / 'confirmation/report.json', result)
    print(json.dumps({k: result[k] for k in ('primary_variant', 'primary_assessment')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(p.HERE / 'config-confirmation.json'))
    parser.add_argument('--checks', default=str(p.HERE / 'config-confirmation-checks.json'))
    parser.add_argument('--dataset-root')
    parser.add_argument('--output')
    parser.add_argument('--stage', choices=('plan', 'freeze', 'extract', 'verify-positions', 'train', 'report', 'all'), default='plan')
    args = parser.parse_args()
    c = p.read_config(args.config, args.dataset_root, args.output)
    s = read_json(args.checks)
    check_settings(c, s)
    if args.stage == 'plan':
        plan = p.resource_plan(c)
        plan.update(training_tasks=4, optimizer_steps=4*c['epochs']*math.ceil(c['max_train']/c['batch_size']),
                    primary_variant=s['primary_variant'], minimum_effect=s['minimum_effect'],
                    exclusion_files=s['exclusion_split_files'],
                    missing_exclusion_files=[f for f in s['exclusion_split_files'] if not Path(f).is_file()],
                    next_stage='freeze', note='Only this entry point enforces disjoint frozen images.')
        print(json.dumps(plan, indent=2))
        return
    if args.stage == 'freeze':
        core = p.core_module(c)
        freeze(c, s, core)
        return
    # All is intentionally extraction onward: it never silently freezes a protocol.
    verify_frozen(c, s)
    stages = ['extract', 'verify-positions', 'train', 'report'] if args.stage == 'all' else [args.stage]
    core = None
    for stage in stages:
        if stage != 'report' and core is None:
            core = p.core_module(c)
            attach_frozen_dataset(c, s, core)
        if stage == 'extract':
            p.extract(c, core)
        elif stage == 'verify-positions':
            import validate_decline as v
            v.verify_positions(c, dict(layers=[23], channels=c['channels'], verify_images=s['verify_images']),
                               Path(c['output']) / 'confirmation', core=core)
        elif stage == 'train':
            audit = read_json(Path(c['output']) / 'confirmation/runtime_position_audit.json')
            if not audit['allclose']:
                raise ValueError('Computation-position audit failed')
            train(c, s, core)
        elif stage == 'report':
            report(c, s)


if __name__ == '__main__':
    main()

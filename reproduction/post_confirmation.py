"""Cached L23 cross-readout evaluation and two independent semantic probes.

Default plan is read-only and needs no ML dependencies. No backbone load or
feature extraction. Source confirmation files are never written. This follow-up
uses an already inspected test split and is exploratory, not a new confirmation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import confirm_l23 as source
import phenomenon as p

CHANNELS = ('post_attention', 'residual')
SEMANTIC_FILES = ('run.json', 'checkpoint.pt', 'history.json', 'selection.json',
                  'metrics.json', 'per_image.json')


def digest(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_settings(path):
    s = source.read_json(path)
    for key in ('probe_seed', 'sampling_seed', 'bootstrap_seed', 'semantic_bootstrap_seed'):
        if type(s[key]) is not int or s[key] < 0:
            raise ValueError(f'Invalid {key}')
    for key in ('semantic_num_classes', 'semantic_epochs', 'max_semantic_tasks', 'bootstrap_repeats', 'bootstrap_bins'):
        if type(s[key]) is not int or s[key] < 2:
            raise ValueError(f'Invalid {key}')
    if s['max_semantic_tasks'] != 2:
        raise ValueError('This panel permits exactly two independent semantic probes')
    for key in ('semantic_learning_rate', 'bootstrap_max_auc_error', 'score_atol', 'score_rtol'):
        if type(s[key]) not in (int, float) or not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f'Invalid {key}')
    return s


def output_path(root, requested=None):
    out = Path(requested).resolve() if requested else root / 'post_confirmation_v1'
    if (out == root or not out.is_relative_to(root) or out.is_relative_to(root / 'channels')
            or out.is_relative_to(root / 'confirmation')):
        raise ValueError('Use a separate output directory inside the source run, outside caches/confirmation')
    return out


def context(root, device=None):
    frozen = source.read_json(root / 'frozen_protocol.json')
    c = dict(frozen['config'], output=str(root))
    if device:
        c['device'] = device
    if c['layers'] != [23] or c['channels'] != list(CHANNELS):
        raise ValueError('Expected the L23 two-position confirmation run')
    return c, frozen


def inventory(root, frozen):
    paths = [root / name for name in ('frozen_protocol.json', 'splits.json', 'requested_config.json',
              'confirmation/report.json', 'confirmation/runtime_position_audit.json')]
    for channel in CHANNELS:
        for split in source.SPLITS:
            folder = root / 'channels' / channel / 'cache' / split
            paths.extend(folder / name for name in ('manifest.json', 'labels.npz', 'layer_23.npy'))
        for variant in frozen['checks']['variants']:
            folder = source.task_folder(dict(output=str(root)), variant['name'], channel)
            paths.extend(folder / name for name in source.ARTIFACTS)
    return paths


def require_gate(report, frozen):
    name = frozen['checks']['primary_variant']
    row = next(r for r in report['results'] if r['variant'] == name)
    effect = frozen['checks']['minimum_effect']
    if (report['primary_variant'] != name or report['minimum_effect'] != effect
            or report['primary_assessment'] != 'meaningful_decline_confirmed'
            or source.gate(row['binding']['delta_ci95'], effect) != 'meaningful_decline_confirmed'):
        raise ValueError('Source confirmation did not pass its frozen primary gate')


def verify_source(root, c, frozen):
    absent = [str(f) for f in inventory(root, frozen) if not f.is_file()]
    if absent:
        raise ValueError('Missing source artifacts; no extraction/retraining fallback: ' + ', '.join(absent))
    report = source.read_json(root / 'confirmation/report.json')
    if digest(root / 'frozen_protocol.json') != report['frozen_protocol_sha256']:
        raise ValueError('Source protocol hash differs from report')
    if digest(root / 'splits.json') != frozen['splits_sha256']:
        raise ValueError('Source frozen split changed')
    if source.read_json(root / 'requested_config.json') != frozen['config']:
        raise ValueError('Source requested config differs from frozen config')
    require_gate(report, frozen)
    splits = source.read_json(root / 'splits.json')
    source.validate_splits(splits)
    if any(len(splits[k]) != c['max_' + k] for k in source.SPLITS):
        raise ValueError('Source split counts differ')
    audit = source.read_json(root / 'confirmation/runtime_position_audit.json')
    if not audit['allclose'] or not all(x['allclose'] for x in audit['comparisons']):
        raise ValueError('Source position audit failed')
    # Check reused code against the recorded implementation without invoking
    # verify_frozen(), which would require original images/index for cache-only work.
    for original, expected in frozen['implementation_and_index_sha256'].items():
        normalized = original.replace('\\', '/')
        file = None
        for marker, base in (('/reproduction/', p.HERE), ('/vit-object-binding/', p.HERE.parent / 'vit-object-binding')):
            if marker in normalized:
                file = base / normalized.split(marker, 1)[1]
                break
        if file is not None and digest(file) != expected:
            raise ValueError(f'Reused implementation differs from frozen source: {file}')
    return report


def prepare(root, c, frozen, s, out):
    import numpy as np
    import investigate_decline as inv
    import validate_decline as v
    import local_readout as lr
    report = verify_source(root, c, frozen)
    inv.validate_caches(c)
    source_files = [f for f in inventory(root, frozen) if f.suffix != '.npy']
    features = []
    for f in inventory(root, frozen):
        if f.suffix == '.npy':
            stat = f.stat()
            features.append(dict(path=str(f), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    for variant in frozen['checks']['variants']:
        reference = None
        row = next(x for x in report['results'] if x['variant'] == variant['name'])
        for ch, endpoint in zip(CHANNELS, ('first_auc_exact', 'second_auc_exact')):
            dest = source.task_folder(c, variant['name'], ch)
            run = source.read_json(dest / 'run.json')
            expected = dict(config=frozen['config'], variant=variant, channel=ch, layer=23,
                sampling_seed=frozen['checks']['sampling_seed'], selection='validation hard-instance AUROC')
            if run != expected:
                raise ValueError('Source probe identity differs')
            history = source.read_json(dest / 'history.json')
            selected = source.read_json(dest / 'selection.json')
            best = max(history, key=lambda e: e['val_hard_instance_auroc'])
            if best['epoch'] != selected['epoch'] or best['val_hard_instance_auroc'] != selected['val_hard_instance_auroc']:
                raise ValueError('Source validation selection differs')
            pred = v.predictions(dest / 'test_predictions.npz')
            meta, _ = inv.load_meta(c, 'test', ch)
            lr.check_eval_sample(c, pred, meta)
            if reference is not None:
                v.aligned(reference, pred)
            reference = pred
            mask, y = v.task_arrays(pred, 'instance')
            measured = v.auc(y[mask], pred['score'][mask])
            saved = source.read_json(dest / 'metrics.json')['instance_vs_same_class']['auroc']
            if not np.isclose(measured, saved, rtol=0, atol=1e-12) or not np.isclose(measured, row['binding'][endpoint], rtol=0, atol=1e-12):
                raise ValueError('Source predictions disagree with metrics/report')
    code_files = [p.HERE / name for name in ('post_confirmation.py', 'semantic_uncertainty.py',
                  'investigate_decline.py', 'run_server.py', 'validate_decline.py', 'local_readout.py')]
    versions = {}
    for package in ('torch', 'numpy', 'transformers', 'scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    identity = dict(config=c, settings=s, source_run=str(root),
        source_sha256={str(f): digest(f) for f in source_files}, feature_files=features,
        implementation_sha256={str(f): digest(f) for f in code_files}, package_versions=versions,
        scope='Exploratory follow-up on the inspected confirmation split; fitted-probe conditional CIs',
        feature_integrity='Cache identity/header/alignment and size/mtime; no full multi-GiB feature hashes')
    file = out / 'requested_settings.json'
    if file.exists() and source.read_json(file) != identity:
        raise ValueError('Follow-up code/settings/source changed; choose a new --output')
    source.write(file, identity)


def cross_folder(out, variant, fitted, evaluated):
    return out / 'cross_readouts' / variant / f'{fitted}_to_{evaluated}'


def cross_readouts(c, frozen, s, out, core):
    import numpy as np
    import torch
    import validate_decline as v
    import local_readout as lr
    for variant in frozen['checks']['variants']:
        for fitted in CHANNELS:
            src = source.task_folder(c, variant['name'], fitted)
            core.seed_all(variant['seed'])
            model = v.make_readout(c, variant, core)
            model.load_state_dict(torch.load(src / 'checkpoint.pt', map_location=c['device'], weights_only=True))
            model.eval()
            # Always verify the diagonal before generating off-diagonal results.
            metrics, records, detail = core.eval_probe(p.channel_config(c, fitted), model, 'test', 23, True)
            actual = dict(score=np.stack([x[0] for x in detail]), label=np.stack([x[1] for x in detail]),
                patches=np.stack([x[2] for x in detail]), condition=np.stack([x[3] for x in detail]),
                image_id=core.cached(p.channel_config(c, fitted), 'test', 23)[1]['image_id'])
            baseline = v.predictions(src / 'test_predictions.npz')
            v.aligned(actual, baseline)
            if not np.allclose(actual['score'], baseline['score'], atol=s['score_atol'], rtol=s['score_rtol']):
                raise ValueError('Loaded checkpoint does not reproduce source diagonal scores')
            for evaluated in CHANNELS:
                dest = cross_folder(out, variant['name'], fitted, evaluated)
                required = ('metrics.json', 'per_image.json', 'test_predictions.npz')
                if (dest / 'metrics.json').exists():
                    if any(not (dest / f).is_file() for f in required):
                        raise ValueError('Incomplete cross-readout artifacts')
                    continue
                if evaluated == fitted:
                    result, per_image, predictions = metrics, records, detail
                else:
                    result, per_image, predictions = core.eval_probe(p.channel_config(c, evaluated), model, 'test', 23, True)
                meta = core.cached(p.channel_config(c, evaluated), 'test', 23)[1]
                dest.mkdir(parents=True, exist_ok=True)
                v.save_predictions(dest / 'test_predictions.npz', predictions, meta['image_id'])
                lr.check_eval_sample(c, v.predictions(dest / 'test_predictions.npz'), meta)
                source.write(dest / 'per_image.json', per_image)
                source.write(dest / 'metrics.json', result)


def semantic(c, s, out, core):
    import investigate_decline as inv
    for ch in CHANNELS:
        dest = out / 'semantic' / ch / 'layer_23'
        expected = dict(config=c, settings=s, channel=ch, layer=23,
                        selection='validation macro recall; first epoch wins ties')
        if (dest / 'run.json').exists() and source.read_json(dest / 'run.json') != expected:
            raise ValueError('Semantic probe settings changed')
        if (dest / 'metrics.json').exists():
            if any(not (dest / f).is_file() for f in SEMANTIC_FILES):
                raise ValueError('Incomplete semantic probe artifacts')
        else:
            source.write(dest / 'run.json', expected)
    inv.train_semantic(c, s, out, core=core)


def paired(a, b, task, s):
    import validate_decline as v
    return v.paired_bootstrap(a, b, task, s['bootstrap_repeats'], s['bootstrap_bins'],
                             s['bootstrap_seed'], s['bootstrap_max_auc_error'])


def make_report(c, frozen, s, out):
    import numpy as np
    import investigate_decline as inv
    import validate_decline as v
    import local_readout as lr
    import semantic_uncertainty as u
    matrices, comparisons = [], []
    meta, _ = inv.load_meta(c, 'test', 'residual')
    i, j = np.triu_indices(c['eval_patches'], 1)
    predictions = {}
    for variant in frozen['checks']['variants']:
        name = variant['name']
        for fitted in CHANNELS:
            for evaluated in CHANNELS:
                dest = cross_folder(out, name, fitted, evaluated)
                if not (dest / 'metrics.json').exists():
                    continue
                pred = v.predictions(dest / 'test_predictions.npz')
                lr.check_eval_sample(c, pred, meta)
                predictions[name, fitted, evaluated] = pred
                metrics = source.read_json(dest / 'metrics.json')
                row = dict(variant=name, fitted_at=fitted, evaluated_at=evaluated)
                for task, metric in (('instance', 'instance_vs_same_class'), ('category', 'class_grouping_cross_instance')):
                    mask, y = v.task_arrays(pred, task)
                    exact = v.auc(y[mask], pred['score'][mask])
                    if exact != metrics[metric]['auroc']:
                        raise ValueError('Cross-readout metrics disagree with predictions')
                    row[task + '_auroc'] = exact
                    row[task + '_support'] = dict(positive=int((mask & y).sum()), negative=int((mask & ~y).sum()))
                matrices.append(row)
        # Two fixed-probe changes plus two target-fixed changes, and the
        # refitted diagonal contrast. All use paired images; none is a recovery claim.
        z, h = CHANNELS
        edges = [('fixed_post_attention_probe', (z, z), (z, h)),
                 ('fixed_residual_probe', (h, z), (h, h)),
                 ('probe_change_on_post_attention', (z, z), (h, z)),
                 ('probe_change_on_residual', (z, h), (h, h)),
                 ('refitted_diagonal', (z, z), (h, h))]
        for label, first, second in edges:
            keys = [(name, *first), (name, *second)]
            if any(k not in predictions for k in keys):
                continue
            a, b = [predictions[k] for k in keys]
            crop = meta['position'][np.arange(len(meta['image_id']))[:, None], a['patches']] // ((c['crop_size']//14)**2)
            comparisons.append(dict(variant=name, comparison=label, first=list(first), second=list(second),
                binding=paired(a, b, 'instance', s), binding_probe_category_score=paired(a, b, 'category', s),
                anchored_rankings=inv.anchored_rankings(a, b, crop[:, i] != crop[:, j])))
    semantic_folders = [out / 'semantic' / ch / 'layer_23' for ch in CHANNELS]
    semantic_done = [(d / 'metrics.json').exists() for d in semantic_folders]
    semantic_comparison = None
    if all(semantic_done):
        for dest in semantic_folders:
            if any(not (dest / f).is_file() for f in SEMANTIC_FILES):
                raise ValueError('Incomplete semantic probe artifacts')
            records = source.read_json(dest / 'per_image.json')
            if [r['image_id'] for r in records] != meta['image_id'].tolist() or any(not r.get('evaluation_fingerprint') for r in records):
                raise ValueError('Semantic evaluation IDs/fingerprints missing or changed')
            _, support, _ = u.record_arrays(records)
            expected_support = []
            for row, image_id in enumerate(meta['image_id']):
                picks = np.random.default_rng(100000+c['split_seed']+int(image_id)).choice(c['saved_patches'], c['eval_patches'], replace=False)
                labels = meta['semantic'][row, picks].astype(int)-1
                valid = (~meta['background'][row, picks].astype(bool)) & (labels >= 0)
                if (labels[valid] >= s['semantic_num_classes']).any():
                    raise ValueError('Semantic class exceeds configured taxonomy')
                sample = [int(image_id), picks.tolist(), meta['position'][row, picks].tolist(), labels.tolist(), valid.tolist()]
                fingerprint = hashlib.sha256(json.dumps(sample, separators=(',', ':')).encode()).hexdigest()
                if records[row]['evaluation_fingerprint'] != fingerprint:
                    raise ValueError('Semantic fingerprint disagrees with cached evaluation patches/labels')
                expected_support.append(np.bincount(labels[valid], minlength=s['semantic_num_classes']))
            if not np.array_equal(support, expected_support):
                raise ValueError('Semantic class support disagrees with cached evaluation labels')
        semantic_comparison = u.compare_folders(*semantic_folders, s['bootstrap_repeats'], s['semantic_bootstrap_seed'])
    result = dict(scope='Exploratory follow-up, conditional on fitted probes; no multiplicity correction',
        cross_matrix=matrices, cross_comparisons=comparisons,
        independent_semantic=semantic_comparison,
        status=dict(cross_cells_completed=len(matrices), cross_cells_expected=8,
                    semantic_tasks_completed=sum(semantic_done), semantic_tasks_expected=2,
                    complete=len(matrices)==8 and all(semantic_done)),
        notes=['Cross-readout sensitivity cannot establish information loss or model use.',
               'Binding-probe category scores are separate from independent semantic accuracy/macro recall.',
               'Anchor rankings have point estimates and per-image counts, no independent-triple CI.'],
        requested_settings_sha256=digest(out / 'requested_settings.json'))
    source.write(out / 'report.json', result)
    # Compact matrices can be inspected without opening the large per-image report.
    import csv
    with (out / 'cross_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ('variant', 'fitted_at', 'evaluated_at', 'instance_auroc', 'category_auroc')
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(matrices)
    if semantic_comparison is not None:
        with (out / 'semantic_summary.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = ('metric', 'post_attention', 'residual', 'delta', 'ci95_low', 'ci95_high')
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, values in semantic_comparison['metrics'].items():
                writer.writerow(dict(metric=name, post_attention=values['first_exact'], residual=values['second_exact'],
                    delta=values['delta_exact'], ci95_low=values['delta_ci95'][0], ci95_high=values['delta_ci95'][1]))
    print(json.dumps(result['status'], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default='runs/confirmation_l23_v1')
    parser.add_argument('--settings', default=str(p.HERE / 'config-post-confirmation.json'))
    parser.add_argument('--output')
    parser.add_argument('--device', help='Probe device override; source frozen config is retained')
    parser.add_argument('--stage', choices=('plan', 'audit', 'cross-readouts', 'semantic', 'report', 'all'), default='plan')
    args = parser.parse_args()
    root = Path(args.source_run).resolve()
    out = output_path(root, args.output)
    c, frozen = context(root, args.device)
    s = read_settings(args.settings)
    if args.stage == 'plan':
        print(json.dumps(dict(source_run=str(root), output=str(out), layer=23,
            cross_readout_cells=8, new_binding_tasks=0, new_semantic_tasks=2,
            semantic_epochs=s['semantic_epochs'], backbone_forward_images=0, new_feature_cache_gib=0,
            missing_source_files=[str(f) for f in inventory(root, frozen) if not f.is_file()],
            all='audit -> cross-readouts -> semantic -> report', scope='Exploratory follow-up'), indent=2))
        return
    prepare(root, c, frozen, s, out)
    stages = ['cross-readouts', 'semantic', 'report'] if args.stage == 'all' else [args.stage]
    core = None
    for stage in stages:
        if stage in ('cross-readouts', 'semantic') and core is None:
            core = p.core_module(c)
        if stage == 'cross-readouts':
            cross_readouts(c, frozen, s, out, core)
        elif stage == 'semantic':
            semantic(c, s, out, core)
        elif stage == 'report':
            make_report(c, frozen, s, out)
        elif stage == 'audit':
            print('Source artifacts, provenance, predictions and cache alignment verified.')


if __name__ == '__main__':
    main()

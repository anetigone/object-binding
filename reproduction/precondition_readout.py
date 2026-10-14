"""Train-only invertible linear preconditioning of both L23 MLP endpoints.

Two full probes, one fixed ridge, no centering/truncation or backbone forward.
Default plan is read-only and imports no ML libraries.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, json, math
import phenomenon as p
import investigate_decline as inv
import local_readout as local

CHANNELS = ('post_attention', 'residual')
RAW_FILES = ('checkpoint.pt', 'metrics.json', 'selection.json', 'history.json', 'test_predictions.npz')
FIT_FILES = ('transform.npz', 'audit.json')


def read_settings(path, c):
    s = json.loads(Path(path).read_text(encoding='utf-8'))
    if type(s['layer']) is not int or not 0 <= s['layer'] < c['num_layers']:
        raise ValueError('Invalid preconditioning block index')
    source = Path(s['source_investigation'])
    if source.is_absolute() or '..' in source.parts or not source.parts:
        raise ValueError('source_investigation must stay within the existing run')
    for key in ('probe_seed', 'sampling_seed', 'statistics_seed', 'bootstrap_seed'):
        if type(s[key]) is not int or s[key] < 0: raise ValueError(f'Invalid {key}')
    for key in ('max_statistics_images', 'statistics_patches_per_image', 'statistics_batch_images'):
        if type(s[key]) is not int or s[key] < 1: raise ValueError(f'Invalid {key}')
    if s['statistics_patches_per_image'] > c['saved_patches']:
        raise ValueError('Too many statistics patches per image')
    if type(s['max_new_binding_tasks']) is not int or s['max_new_binding_tasks'] != 2:
        raise ValueError('This experiment permits exactly two new binding tasks')
    if type(s['probe_weight_decay']) not in (int, float) or s['probe_weight_decay'] != 0:
        raise ValueError('Use unregularized full readouts; preconditioning changes parameter penalties')
    for key in ('ridge_relative', 'minimum_effect', 'bootstrap_max_auc_error'):
        if type(s[key]) not in (int, float) or not math.isfinite(s[key]) or not 0 < s[key] < 1:
            raise ValueError(f'Invalid {key}')
    for key in ('bootstrap_repeats', 'bootstrap_bins'):
        if type(s[key]) is not int or s[key] < 2: raise ValueError(f'Invalid {key}')
    return s


def source_root(c, s):
    run = Path(c['output']).resolve()
    root = (run / s['source_investigation']).resolve()
    if not root.is_relative_to(run): raise ValueError('Source escapes run directory')
    return root


def raw_sources(c, s):
    import mlp_update as mlp
    root = source_root(c, s)
    training = mlp.source_settings(c, root, 'steps')
    if s['layer'] not in training['step_layers'] or s['layer'] not in training['baseline_layers']:
        raise ValueError('Selected layer absent from the completed source steps/baseline')
    for key in ('probe_seed', 'sampling_seed', 'probe_weight_decay'):
        if training[key] != s[key]: raise ValueError(f'Source full readout {key} differs')
    residual = inv.files(c, training, 'residual', s['layer'])
    inv.verify_baseline_run(c, training, residual, s['layer'])
    return dict(post_attention=root / 'steps/binding/post_attention' / f"layer_{s['layer']}",
                residual=residual)


def fit_folder(out, channel):
    return out / 'preconditioners' / channel


def new_folder(out, channel, layer):
    return out / 'binding' / channel / f'layer_{layer}'


def prepare(c, s, out):
    import validate_decline as v
    sources = raw_sources(c, s)
    for folder in sources.values():
        absent = local.missing(folder, RAW_FILES)
        if absent: raise ValueError('Missing raw full-readout results: ' + ', '.join(absent))
    inv.validate_caches(dict(c, channels=list(CHANNELS), layers=[s['layer']]))
    meta, _ = inv.load_meta(c, 'test', 'residual')
    predictions = [v.predictions(sources[channel] / 'test_predictions.npz') for channel in CHANNELS]
    v.aligned(*predictions)
    for prediction in predictions: local.check_eval_sample(c, prediction, meta)
    cache_hashes = {}
    for channel in CHANNELS:
        for split in ('train', 'val', 'test'):
            folder = Path(p.channel_config(c, channel)['output']) / 'cache' / split
            for name in ('manifest.json', 'labels.npz'):
                cache_hashes[f'{channel}/{split}/{name}'] = local.digest(folder / name)
    identity = dict(base_config=c, settings=s, cache_metadata_sha256=cache_hashes,
        source_settings_sha256=local.digest(source_root(c, s) / 'steps/requested_settings.json'),
        raw_sources_sha256={channel: {name: local.digest(folder / name) for name in RAW_FILES}
            for channel, folder in sources.items()},
        residual_run_sha256=local.digest(sources['residual'] / 'run.json'),
        implementation_sha256={name: local.digest(p.HERE / name) for name in
            ('precondition_readout.py', 'local_readout.py', 'mlp_update.py', 'investigate_decline.py',
             'validate_decline.py', 'run_server.py')},
        probe_source_sha256=local.digest(p.HERE.parent / 'vit-object-binding/src/utils/models.py'))
    file = out / 'requested_settings.json'
    if file.exists() and json.loads(file.read_text(encoding='utf-8')) != identity:
        raise ValueError('Preconditioning inputs/code/settings changed; choose a new --experiment-output')
    if not file.exists() and out.exists() and any(out.iterdir()):
        raise ValueError('Nonempty output without provenance; choose a new directory')
    inv.write(file, identity)
    return sources


def matrix_from_moment(moment, ridge_relative):
    """No mean subtraction. Preserve training RMS via one positive scalar."""
    import numpy as np
    moment = np.asarray(moment, dtype=np.float64)
    if moment.ndim != 2 or moment.shape[0] != moment.shape[1] or not np.isfinite(moment).all():
        raise ValueError('Invalid second moment')
    moment = (moment + moment.T) / 2
    eigenvalues, vectors = np.linalg.eigh(moment)
    scale = float(np.trace(moment) / len(moment))
    if not math.isfinite(scale) or scale <= 0 or eigenvalues[0] < -1e-8 * scale:
        raise ValueError('Second moment is zero or not positive semidefinite')
    if not math.isfinite(ridge_relative) or ridge_relative <= 0:
        raise ValueError('Ridge must be positive')
    eigenvalues = np.maximum(eigenvalues, 0.)  # roundoff only, never drop directions
    ridge = ridge_relative * scale
    rms_gain = math.sqrt(float(eigenvalues.sum() / (eigenvalues / (eigenvalues + ridge)).sum()))
    t64 = (vectors * (rms_gain / np.sqrt(eigenvalues + ridge))) @ vectors.T
    transform = t64.astype(np.float32)
    if not np.isfinite(transform).all(): raise ValueError('Non-finite preconditioner')
    applied = transform.astype(np.float64)
    te = np.linalg.eigvalsh(applied)
    if te[0] <= 0 or te[-1] / te[0] > 1e4:
        raise ValueError('Applied float32 transform is not safely invertible')
    transformed = applied @ moment @ applied.T
    expected_rms = float(np.trace(transformed) / np.trace(moment))
    relative_error = float(np.linalg.norm(applied - t64) / np.linalg.norm(t64))
    if abs(expected_rms - 1) > 1e-4 or relative_error > 1e-4:
        raise ValueError('Float32 transform failed numerical/RMS audit')
    audit = dict(ridge_absolute=ridge, ridge_relative=ridge_relative, rms_gain=rms_gain,
        original_second_moment_eigenvalues_min_max=[float(eigenvalues[0]), float(eigenvalues[-1])],
        transformed_second_moment_eigenvalues_min_max=np.linalg.eigvalsh(transformed)[[0, -1]].tolist(),
        transform_eigenvalues_min_max=[float(te[0]), float(te[-1])],
        transform_condition_number=float(te[-1] / te[0]), applied_trace_ratio=expected_rms,
        float32_relative_matrix_error=relative_error, centered=False, truncated_dimensions=0)
    return transform, audit


def fit(c, s, out):
    import hashlib
    import numpy as np
    prepare(c, s, out)
    meta, _ = inv.load_meta(c, 'train', 'residual')
    # The same images/patches in both endpoints; no labels select the sample.
    rows = np.sort(np.random.default_rng(s['statistics_seed']).choice(len(meta['image_id']),
        min(len(meta['image_id']), s['max_statistics_images']), replace=False))
    picks = np.stack([np.random.default_rng(200000 + s['statistics_seed'] + int(meta['image_id'][row])).choice(
        c['saved_patches'], s['statistics_patches_per_image'], replace=False) for row in rows])
    fingerprint = hashlib.sha256(meta['image_id'][rows].tobytes() + picks.tobytes()).hexdigest()
    for channel in CHANNELS:
        folder = fit_folder(out, channel)
        if (folder / 'audit.json').exists():
            load_transform(c, s, out, channel); continue
        file = Path(p.channel_config(c, channel)['output']) / 'cache/train' / f"layer_{s['layer']}.npy"
        features = np.load(file, mmap_mode='r')
        try:
            moment = np.zeros((c['embed_dim'], c['embed_dim']), dtype=np.float64)
            count = 0
            for start in range(0, len(rows), s['statistics_batch_images']):
                subset = rows[start:start + s['statistics_batch_images']]
                patch = picks[start:start + len(subset)]
                batch = np.asarray(features[subset[:, None], patch], dtype=np.float64).reshape(-1, c['embed_dim'])
                if not np.isfinite(batch).all(): raise ValueError('Non-finite training statistics features')
                moment += batch.T @ batch; count += len(batch)
            moment /= count
            transform, audit = matrix_from_moment(moment, s['ridge_relative'])
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / 'transform.npz', transform=transform)
            audit.update(fit_split='train', images=len(rows), tokens=count,
                selected_training_image_ids=meta['image_id'][rows].tolist(), sample_sha256=fingerprint,
                formula='T = g * (E_train[x x^T] + ridge_relative * trace(S)/d * I)^(-1/2); '
                        'g preserves sampled training mean squared L2 norm',
                transform_sha256=local.digest(folder / 'transform.npz'))
            inv.write(folder / 'audit.json', audit)  # fit completion marker last
            print(f'{channel}: fit train-only transform on {count} tokens', flush=True)
        finally:
            mapping = getattr(features, '_mmap', None)
            if mapping is not None: mapping.close()


def load_transform(c, s, out, channel):
    import numpy as np
    folder = fit_folder(out, channel)
    absent = local.missing(folder, FIT_FILES)
    if absent: raise ValueError('Missing fitted preconditioner; run --stage fit: ' + ', '.join(absent))
    audit = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
    if (audit.get('fit_split') != 'train' or audit.get('centered') is not False
        or audit.get('truncated_dimensions') != 0 or audit.get('ridge_relative') != s['ridge_relative']
        or audit.get('transform_sha256') != local.digest(folder / 'transform.npz')):
        raise ValueError('Preconditioner audit/hash differs')
    with np.load(folder / 'transform.npz') as bank: transform = bank['transform'].copy()
    if transform.shape != (c['embed_dim'], c['embed_dim']) or transform.dtype != np.float32 or not np.isfinite(transform).all():
        raise ValueError('Invalid applied preconditioner')
    return transform, audit


def wrap_probe(probe, transform):
    import torch
    class Preconditioned(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = probe
            self.register_buffer('transform', torch.tensor(transform, dtype=torch.float32))
        def forward_pairwise(self, x, y):
            return self.readout.forward_pairwise(x @ self.transform.T, y @ self.transform.T)
    return Preconditioned()


def new_identity(c, s, out, channel):
    return dict(config=c, settings=s, channel=channel, layer=s['layer'], kind='full',
        selection=local.SELECTION,
        transform_sha256=local.digest(fit_folder(out, channel) / 'transform.npz'))


def verify_new(c, s, out, channel):
    folder = new_folder(out, channel, s['layer'])
    absent = local.missing(folder, local.NEW_FILES)
    if absent: raise ValueError('Missing completed preconditioned readout: ' + ', '.join(absent))
    if json.loads((folder / 'run.json').read_text(encoding='utf-8')) != new_identity(c, s, out, channel):
        raise ValueError('Preconditioned readout identity differs')


def train(c, s, out):
    prepare(c, s, out)
    transforms = {channel: load_transform(c, s, out, channel)[0] for channel in CHANNELS}
    pending = []
    for channel in CHANNELS:
        if (new_folder(out, channel, s['layer']) / 'metrics.json').exists():
            verify_new(c, s, out, channel)
        else: pending.append(channel)
    if not pending:
        print('Both preconditioned full probes complete; skipping training.', flush=True); return
    core = p.core_module(c)
    settings = dict(s, baseline_layers=[])
    for channel in pending:
        transform = transforms[channel]
        proxy = SimpleNamespace(cached=core.cached, seed_all=core.seed_all,
            batch_logits=core.batch_logits, eval_probe=core.eval_probe,
            probe=lambda cfg, mode: wrap_probe(core.probe(cfg, mode), transform).to(cfg['device']))
        inv.write(new_folder(out, channel, s['layer']) / 'run.json', new_identity(c, s, out, channel))
        inv.train_binding(dict(c, channels=[channel], layers=[s['layer']]), settings, out, core=proxy)


def joint_bootstrap(data, s):
    """Four endpoints share every sampled image multiplicity, including gap change."""
    import numpy as np
    import validate_decline as v
    for other in data[1:]: v.aligned(data[0], other)
    hist = [v.image_histograms(item, 'instance', s['bootstrap_bins']) for item in data]
    if any(abs(h[2] - h[3]) > s['bootstrap_max_auc_error'] for h in hist):
        raise ValueError('Binned AUROC approximation error exceeds tolerance; increase bootstrap_bins')
    n = len(data[0]['image_id']); rng = np.random.default_rng(s['bootstrap_seed']); batches = []
    for start in range(0, s['bootstrap_repeats'], 64):
        w = rng.multinomial(n, np.full(n, 1/n), size=min(64, s['bootstrap_repeats'] - start)).astype(float)
        batches.append(np.stack([v.auc_histogram(w @ h[0], w @ h[1]) for h in hist], axis=1))
    values = np.concatenate(batches); valid = np.isfinite(values).all(axis=1)
    if valid.sum() < .95 * len(values): raise ValueError('Too many resamples lack both classes')
    values = values[valid]; exact = np.array([h[2] for h in hist])
    def contrast(weights):
        return dict(delta_exact=float(exact @ weights),
                    delta_ci95=np.quantile(values @ weights, [.025, .975]).tolist())
    return dict(estimand='pooled hard-instance AUROC; shared image-cluster bootstrap across four endpoints',
        images=n, replicates=s['bootstrap_repeats'], valid_replicates=int(valid.sum()),
        order=['raw_post_attention', 'raw_residual', 'preconditioned_post_attention', 'preconditioned_residual'],
        auc_exact=exact.tolist(), auc_ci95=np.quantile(values, [.025, .975], axis=0).T.tolist(),
        raw_mlp_delta=contrast([-1, 1, 0, 0]), preconditioned_mlp_delta=contrast([0, 0, -1, 1]),
        post_attention_improvement=contrast([-1, 0, 1, 0]), residual_improvement=contrast([0, -1, 0, 1]),
        gap_narrowing=contrast([1, -1, -1, 1]),
        gap_narrowing_definition='(raw z AUROC - raw h AUROC) - (preconditioned z AUROC - preconditioned h AUROC); positive=narrower',
        histogram_point_error=[h[3] - h[2] for h in hist],
        approximation_note='Points use exact ranks; CIs use fixed quantile histograms, not exact-rank bootstrap.')


def report(c, s, out):
    import validate_decline as v
    sources = prepare(c, s, out)
    audits = {channel: load_transform(c, s, out, channel)[1] for channel in CHANNELS}
    for channel in CHANNELS: verify_new(c, s, out, channel)
    folders = [sources[channel] for channel in CHANNELS] + [new_folder(out, channel, s['layer']) for channel in CHANNELS]
    data = [v.predictions(folder / 'test_predictions.npz') for folder in folders]
    meta, _ = inv.load_meta(c, 'test', 'residual')
    for prediction in data: local.check_eval_sample(c, prediction, meta)
    paired = joint_bootstrap(data, s)
    measurements = [dict(condition=name, source=str(folder), hard_instance_auroc=auc, ci95=ci)
        for name, folder, auc, ci in zip(paired['order'], folders, paired['auc_exact'], paired['auc_ci95'])]
    gap = paired['gap_narrowing']
    low, high = gap['delta_ci95']; epsilon = s['minimum_effect']
    conclusion = ('meaningful_gap_narrowing_supported' if low > epsilon else
                  'meaningful_gap_widening_supported' if high < -epsilon else
                  'ci_within_small_gap_change_band' if low >= -epsilon and high <= epsilon else
                  'inconclusive_at_prespecified_effect')
    result = dict(layer=s['layer'], settings=s, measurements=measurements, paired_comparisons=paired,
        assessment=conclusion, preconditioner_audits=audits, new_binding_tasks=2,
        backbone_forward_images=0, new_feature_cache_gib=0,
        interpretation='Invertible linear coordinates preserve the full symmetric bilinear function family. '
            'Gap narrowing is consistent with optimization/finite-sample parameterization effects, not proof of '
            'a unique cause, complete information preservation, or causal downstream use. '
            'The fixed split has been used for exploration; independent confirmation remains necessary.')
    inv.write(out / 'report.json', result)
    with (out / 'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['condition', 'source', 'hard_instance_auroc', 'ci95'])
        writer.writeheader(); writer.writerows(measurements)
    print(json.dumps(dict(assessment=conclusion, gap_narrowing=gap, measurements=measurements), indent=2))
    return result


def plan(c, s, out):
    # Source paths and missing files only; no numpy/torch import or mutation.
    root = source_root(c, s)
    provenance = root / 'steps/requested_settings.json'
    sources = {'post_attention': root / 'steps/binding/post_attention' / f"layer_{s['layer']}"}
    if provenance.exists():
        training = json.loads(provenance.read_text(encoding='utf-8'))['settings']
        sources['residual'] = inv.files(c, training, 'residual', s['layer'])
    else:
        sources['residual'] = Path(c['output']) / 'validation/readouts/residual/full_seed0' / f"layer_{s['layer']}"
    absent = [str(provenance)] if not provenance.is_file() else []
    for channel, folder in sources.items():
        absent += local.missing(folder, RAW_FILES + (('run.json',) if channel == 'residual' else ()))
    for channel in CHANNELS:
        for split in ('train', 'val', 'test'):
            folder = Path(p.channel_config(c, channel)['output']) / 'cache' / split
            absent += local.missing(folder, ('manifest.json', 'labels.npz', f"layer_{s['layer']}.npy"))
    return dict(layer=s['layer'], experiment_output=str(out), max_new_binding_tasks=2,
        pending_binding_tasks=sum(bool(local.missing(new_folder(out, channel, s['layer']), local.NEW_FILES)) for channel in CHANNELS),
        new_semantic_tasks=0, backbone_forward_images=0, new_feature_cache_gib=0,
        train_statistics_images_upper_bound=min(c['max_train'], s['max_statistics_images']) if c['max_train'] else s['max_statistics_images'],
        statistics_patches_per_image=s['statistics_patches_per_image'], ridge_relative=s['ridge_relative'],
        preconditioner_storage_mib=2 * c['embed_dim']**2 * 4 / 2**20,
        missing_source_files=absent, minimum_effect=s['minimum_effect'],
        note='One fixed ridge for both endpoints; no centering, PCA truncation, tuning panel, extraction or backbone. '
             'Read-only inventory; fit/train/report validate sources and provenance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(p.HERE / 'config-overnight.json'))
    parser.add_argument('--settings', default=str(p.HERE / 'config-precondition-readout.json'))
    parser.add_argument('--output', help='Existing base run')
    parser.add_argument('--experiment-output', help='Separate new directory within the existing run')
    parser.add_argument('--stage', choices=('plan', 'fit', 'train', 'report', 'all'), default='plan')
    args = parser.parse_args(); c = p.read_config(args.config, output=args.output); s = read_settings(args.settings, c)
    run = Path(c['output']).resolve(); root = source_root(c, s)
    out = Path(args.experiment_output).resolve() if args.experiment_output else (root / 'precondition_readout').resolve()
    protected = (run / 'channels', run / 'validation', root / 'steps', root / 'depth', root / 'local_readout', root / 'mlp_update', root / 'norm_control')
    if (out == run or not out.is_relative_to(run) or out == root or root.is_relative_to(out)
        or any(out.is_relative_to(folder.resolve()) or folder.resolve().is_relative_to(out) for folder in protected)):
        raise ValueError('Use a separate experiment directory inside the run, outside existing caches/results')
    if args.stage == 'plan': print(json.dumps(plan(c, s, out), indent=2)); return
    for stage in (('fit', 'train', 'report') if args.stage == 'all' else (args.stage,)):
        {'fit': fit, 'train': train, 'report': report}[stage](c, s, out)


if __name__ == '__main__': main()

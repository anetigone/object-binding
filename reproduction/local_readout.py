"""One matched factor readout before L23 MLP; reuse the completed after endpoint.

Default plan is read-only and needs no ML dependencies. Never extracts features,
loads a backbone, trains semantic probes, or expands the readout panel.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, math
import phenomenon as p
import investigate_decline as inv

SELECTION = 'validation instance-vs-same-class AUROC, first epoch wins ties'
CONFIG_FIELDS = ('model', 'model_revision', 'family', 'embed_dim', 'crop_size',
    'saved_patches', 'sampled_patches', 'eval_patches', 'split_seed', 'dataset_name',
    'dataset_factory', 'max_train', 'max_val', 'max_test', 'epochs', 'batch_size',
    'learning_rate', 'scheduler_step', 'scheduler_gamma')
SOURCE_FILES = ('run.json', 'checkpoint.pt', 'metrics.json', 'selection.json',
    'history.json', 'test_predictions.npz')
NEW_FILES = SOURCE_FILES + ('per_image.json',)


def read_settings(path, c):
    s = json.loads(Path(path).read_text(encoding='utf-8'))
    if type(s['layer']) is not int or not 0 <= s['layer'] < c['num_layers']:
        raise ValueError('Invalid local-readout layer')
    source = Path(s['residual_source'])
    if source.is_absolute() or '..' in source.parts or not source.parts:
        raise ValueError('residual_source must stay within the existing run')
    variant = s['variant']
    if (variant.get('kind') != 'factor' or variant.get('shared_final_layernorm', False)
        or type(variant.get('factor_width')) is not int or variant['factor_width'] <= 0
        or type(variant.get('seed')) is not int or variant['seed'] < 0
        or not isinstance(variant.get('name'), str) or not variant['name'].replace('_', '').isalnum()
        or type(variant.get('weight_decay')) not in (int, float)
        or not math.isfinite(variant['weight_decay']) or variant['weight_decay'] < 0):
        raise ValueError('Use one unnormalized factor readout with valid width/seed/weight decay')
    for name in ('sampling_seed', 'bootstrap_seed'):
        if type(s[name]) is not int or s[name] < 0: raise ValueError(f'Invalid {name}')
    if type(s['max_new_binding_tasks']) is not int or s['max_new_binding_tasks'] != 1:
        raise ValueError('This experiment permits exactly one new binding task')
    for name in ('bootstrap_repeats', 'bootstrap_bins'):
        if type(s[name]) is not int or s[name] < 2: raise ValueError(f'Invalid {name}')
    for name in ('minimum_effect', 'bootstrap_max_auc_error'):
        if type(s[name]) not in (int, float) or not math.isfinite(s[name]) or not 0 < s[name] < 1:
            raise ValueError(f'Invalid {name}')
    return s


def source_folder(c, s):
    run = Path(c['output']).resolve()
    source = (run / s['residual_source']).resolve()
    if not source.is_relative_to(run): raise ValueError('Residual source escapes run directory')
    return source


def new_folder(s, out):
    return out / 'binding/post_attention' / f"layer_{s['layer']}"


def new_identity(c, s):
    return dict(config=c, variant=s['variant'], layer=s['layer'],
        sampling_seed=s['sampling_seed'], selection=SELECTION, channel='post_attention')


def verify_new(c, s, out):
    dest = new_folder(s, out)
    absent = missing(dest, NEW_FILES)
    if absent: raise ValueError('Missing completed post-attention readout: ' + ', '.join(absent))
    if json.loads((dest / 'run.json').read_text(encoding='utf-8')) != new_identity(c, s):
        raise ValueError('Post-attention readout identity differs')


def missing(folder, names):
    return [str(folder / name) for name in names if not (folder / name).is_file()]


def verify_source(c, s):
    source = source_folder(c, s)
    absent = missing(source, SOURCE_FILES)
    if absent: raise ValueError('Missing completed residual readout files: ' + ', '.join(absent))
    run = json.loads((source / 'run.json').read_text(encoding='utf-8'))
    if (run.get('variant') != s['variant'] or run.get('layer') != s['layer']
        or run.get('sampling_seed') != s['sampling_seed'] or run.get('selection') != SELECTION):
        raise ValueError('Residual readout variant/layer/sampling/selection differs')
    recorded = run['config']
    if any(recorded.get(key) != c.get(key) for key in CONFIG_FIELDS):
        raise ValueError('Residual training/data config differs; use the actual completed server config')
    for key in ('output', 'dataset_root'):
        if Path(recorded[key]).resolve() != Path(c[key]).resolve():
            raise ValueError(f'Residual {key} differs from the requested run')
    return source


def scope_config(c, s):
    return dict(c, channels=['post_attention', 'residual'], layers=[s['layer']])


def digest(file):
    value = hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): value.update(chunk)
    return value.hexdigest()


def check_eval_sample(c, prediction, meta):
    import numpy as np
    inv.check_prediction_meta(prediction, meta)
    expected = np.stack([np.random.default_rng(100000 + c['split_seed'] + int(image_id)).choice(
        c['saved_patches'], c['eval_patches'], replace=False) for image_id in meta['image_id']])
    if not np.array_equal(prediction['patches'], expected):
        raise ValueError('Evaluation patch sampler differs')


def prepare(c, s, out):
    import validate_decline as v
    source = verify_source(c, s)
    inv.validate_caches(scope_config(c, s))
    meta, _ = inv.load_meta(c, 'test', 'residual')
    check_eval_sample(c, v.predictions(source / 'test_predictions.npz'), meta)
    identity = dict(base_config=c, settings=s,
        residual_source_sha256={name: digest(source / name) for name in SOURCE_FILES},
        implementation_sha256={name: digest(p.HERE / name) for name in
            ('local_readout.py', 'investigate_decline.py', 'validate_decline.py', 'run_server.py')},
        probe_source_sha256=digest(p.HERE.parent / 'vit-object-binding/src/utils/models.py'))
    file = out / 'requested_settings.json'
    if file.exists() and json.loads(file.read_text(encoding='utf-8')) != identity:
        raise ValueError('Local readout inputs/code/settings changed; choose a new --experiment-output')
    if not file.exists() and out.exists() and any(out.iterdir()):
        raise ValueError('Experiment output is nonempty without provenance; choose a new directory')
    inv.write(file, identity)
    return source


def plan(c, s, out):
    dest = new_folder(s, out)
    source = source_folder(c, s)
    feature_files = [Path(p.channel_config(c, channel)['output']) / 'cache' / split / name
        for channel in ('post_attention', 'residual') for split in ('train', 'val', 'test')
        for name in ('manifest.json', 'labels.npz', f"layer_{s['layer']}.npy")]
    return dict(layer=s['layer'], variant=s['variant'], residual_source=str(source),
        experiment_output=str(out), max_new_binding_tasks=1,
        new_binding_tasks=0 if not missing(dest, NEW_FILES) else 1,
        new_semantic_tasks=0, backbone_forward_images=0, new_feature_cache_gib=0,
        missing_source_files=missing(source, SOURCE_FILES),
        missing_cache_files=[str(file) for file in feature_files if not file.is_file()],
        minimum_effect=s['minimum_effect'],
        note='Read-only file inventory; train/report perform config, cache and prediction alignment checks. '
             'No extraction fallback. Does not start the preconditioning experiment.')


def train(c, s, out):
    import validate_decline as v
    prepare(c, s, out)
    dest = new_folder(s, out)
    if (dest / 'metrics.json').exists():
        verify_new(c, s, out)
        print('Matched post-attention readout complete; skipping training.', flush=True)
        return
    cfg = dict(c, channels=['post_attention'], layers=[s['layer']])
    settings = dict(baseline_layers=[], probe_seed=s['variant']['seed'],
        sampling_seed=s['sampling_seed'], probe_weight_decay=s['variant']['weight_decay'])
    core = p.core_module(c)
    # Use exactly the validation panel's factor architecture and initialization.
    proxy = SimpleNamespace(cached=core.cached, seed_all=core.seed_all,
        batch_logits=core.batch_logits, eval_probe=core.eval_probe,
        probe=lambda config, mode: v.make_readout(config, s['variant'], core))
    inv.write(dest / 'run.json', new_identity(c, s))
    inv.train_binding(cfg, settings, out, core=proxy)


def assessment(ci, epsilon):
    low, high = ci
    if high < -epsilon: return 'meaningful_decline_supported'
    if low > epsilon: return 'meaningful_increase_supported'
    if low >= -epsilon and high <= epsilon: return 'ci_within_small_difference_band'
    return 'inconclusive_at_prespecified_effect'


def report(c, s, out):
    import numpy as np
    import validate_decline as v
    source = prepare(c, s, out)
    dest = new_folder(s, out)
    verify_new(c, s, out)
    a, b = [v.predictions(folder / 'test_predictions.npz') for folder in (dest, source)]
    v.aligned(a, b)
    meta, _ = inv.load_meta(c, 'test', 'residual')
    for prediction in (a, b): check_eval_sample(c, prediction, meta)
    paired = v.paired_bootstrap(a, b, 'instance', s['bootstrap_repeats'], s['bootstrap_bins'],
        s['bootstrap_seed'], s['bootstrap_max_auc_error'])
    rows = np.arange(len(meta['image_id']))[:, None]
    crops = meta['position'][rows, a['patches']] // ((c['crop_size'] // 14) ** 2)
    i, j = np.triu_indices(c['eval_patches'], 1)
    ranking = inv.anchored_rankings(a, b, crops[:, i] != crops[:, j])
    result = dict(layer=s['layer'], variant=s['variant'], first='post_attention', second='residual',
        delta_definition='residual minus post_attention', first_source=str(dest), second_source=str(source),
        binding=paired, anchored_rankings=ranking, minimum_effect=s['minimum_effect'],
        assessment=assessment(paired['delta_ci95'], s['minimum_effect']),
        backbone_forward_images=0, new_feature_cache_gib=0,
        note='Exploratory, conditional on fitted probes and the fixed split; not information loss or '
             'causal downstream use. CI crossing zero is not evidence of equivalence.')
    inv.write(out / 'report.json', result)
    print(json.dumps({key: result[key] for key in ('binding', 'minimum_effect', 'assessment')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(p.HERE / 'config-overnight.json'))
    parser.add_argument('--settings', default=str(p.HERE / 'config-local-readout.json'))
    parser.add_argument('--output', help='Existing base run (not the experiment subdirectory)')
    parser.add_argument('--experiment-output', help='New experiment directory inside the existing run')
    parser.add_argument('--stage', choices=('plan', 'train', 'report', 'all'), default='plan')
    args = parser.parse_args()
    c = p.read_config(args.config, output=args.output)
    s = read_settings(args.settings, c)
    run = Path(c['output']).resolve()
    out = Path(args.experiment_output).resolve() if args.experiment_output else run / 'investigation/local_readout'
    source = source_folder(c, s)
    if (out == run or not out.is_relative_to(run) or out.is_relative_to(source)
        or source.is_relative_to(out) or out.is_relative_to(run / 'channels')
        or out.is_relative_to(run / 'validation')):
        raise ValueError('Experiment output must be a separate directory inside the run, outside sources/caches/validation')
    if args.stage == 'plan':
        print(json.dumps(plan(c, s, out), indent=2)); return
    if args.stage in ('train', 'all'): train(c, s, out)
    if args.stage in ('report', 'all'): report(c, s, out)


if __name__ == '__main__': main()

"""Raw-artifact audit followed by one train-only h-to-z ridge alignment.

No backbone, no probe training, no parameter search. Plan and review are usable
with JSON/CSV downloads. Formal stages fail before fitting if raw assets are absent.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import hashlib
import json
import math
import post_confirmation as post
import confirm_l23 as src
import phenomenon as p


def settings(path, c):
    s = src.read_json(path)
    for key in ('max_statistics_images', 'statistics_patches_per_image', 'cpu_threads'):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f'Invalid {key}')
    if s['statistics_patches_per_image'] > c['saved_patches']:
        raise ValueError('Statistics patch count exceeds cache')
    for key in ('statistics_seed', 'bootstrap_seed', 'semantic_bootstrap_seed'):
        if type(s[key]) is not int or s[key] < 0:
            raise ValueError(f'Invalid {key}')
    for key in ('bootstrap_repeats', 'bootstrap_bins'):
        if type(s[key]) is not int or s[key] < 2:
            raise ValueError(f'Invalid {key}')
    for key in ('ridge_relative', 'minimum_improvement', 'bootstrap_max_auc_error'):
        if type(s[key]) not in (int, float) or not math.isfinite(s[key]) or not 0 < s[key] < 1:
            raise ValueError(f'Invalid {key}')
    return s


def inventory(root, frozen, followup):
    paths = post.inventory(root, frozen)
    paths += [followup / name for name in ('requested_settings.json', 'report.json')]
    for variant in frozen['checks']['variants']:
        for fitted in post.CHANNELS:
            for evaluated in post.CHANNELS:
                d = post.cross_folder(followup, variant['name'], fitted, evaluated)
                paths.extend(d / f for f in ('metrics.json', 'per_image.json', 'test_predictions.npz'))
    for channel in post.CHANNELS:
        d = followup / 'semantic' / channel / 'layer_23'
        paths.extend(d / f for f in post.SEMANTIC_FILES)
    return list(dict.fromkeys(paths))


def recorded_path(name, root=None, source_prefix=None):
    normalized = name.replace('\\', '/')
    if root is not None and source_prefix:
        prefix=source_prefix.replace('\\','/').rstrip('/')+'/'
        if normalized.startswith(prefix):
            return root/normalized[len(prefix):]
    for marker, base in (('/reproduction/', p.HERE), ('/vit-object-binding/', p.HERE.parent / 'vit-object-binding'),
                         ('/runs/', p.HERE.parent / 'runs')):
        if marker in normalized:
            return base / normalized.split(marker, 1)[1]
    return Path(name)


def review(root, frozen, followup, file):
    """Metadata/semantic-count audit; explicitly does not certify missing binaries."""
    import numpy as np
    import semantic_uncertainty as u
    from threadpoolctl import threadpool_limits
    original = src.read_json(root / 'confirmation/report.json')
    report = src.read_json(followup / 'report.json')
    identity = src.read_json(followup / 'requested_settings.json')
    if post.digest(followup / 'requested_settings.json') != report['requested_settings_sha256']:
        raise ValueError('Follow-up provenance hash mismatch')
    if post.digest(root / 'frozen_protocol.json') != original['frozen_protocol_sha256']:
        raise ValueError('Confirmation protocol hash mismatch')
    if post.digest(root / 'splits.json') != frozen['splits_sha256']:
        raise ValueError('Frozen split hash mismatch')
    verified, unavailable = [], []
    for name, expected in identity['source_sha256'].items():
        f = recorded_path(name,root,identity['source_run'])
        if f.is_file():
            if post.digest(f) != expected:
                raise ValueError(f'Source metadata/binary hash mismatch: {f}')
            verified.append(str(f))
        else:
            unavailable.append(str(f))
    for name, expected in identity['implementation_sha256'].items():
        f = recorded_path(name)
        if post.digest(f) != expected:
            raise ValueError(f'Follow-up implementation hash mismatch: {f}')
    splits = src.read_json(root / 'splits.json')
    src.validate_splits(splits)
    for channel in post.CHANNELS:
        d = followup / 'semantic' / channel / 'layer_23'
        history = src.read_json(d / 'history.json'); selection = src.read_json(d / 'selection.json')
        best = max(history, key=lambda e: e['val_macro_recall'])
        if best['epoch'] != selection['epoch'] or best['val_macro_recall'] != selection['val_macro_recall']:
            raise ValueError('Semantic validation selection mismatch')
        if [r['image_id'] for r in src.read_json(d / 'per_image.json')] != splits['test']:
            raise ValueError('Semantic image IDs differ from split')
    with threadpool_limits(limits=4):
        recomputed = u.compare_folders(followup / 'semantic/post_attention/layer_23',
            followup / 'semantic/residual/layer_23', identity['settings']['bootstrap_repeats'],
            identity['settings']['semantic_bootstrap_seed'])
    for metric, values in recomputed['metrics'].items():
        for key in ('first_exact', 'second_exact', 'delta_exact', 'delta_ci95'):
            np.testing.assert_allclose(values[key], report['independent_semantic']['metrics'][metric][key], rtol=0, atol=1e-12)
    absent = [str(f) for f in inventory(root, frozen, followup) if not f.is_file()]
    result = dict(scope='Local metadata and per-image semantic count review; not a raw binding audit',
        status=report['status'], verified_source_hashes=verified, unavailable_source_hashes=unavailable,
        missing_raw_artifacts=absent, raw_binding_recomputed=False,
        semantic_bootstrap_recomputed=recomputed, source_protocol_sha256=post.digest(root / 'frozen_protocol.json'),
        source_followup_report_sha256=post.digest(followup / 'report.json'),
        next_step='Raw audit before fitting a single train-only linear h-to-z map')
    src.write(file, result)
    print(json.dumps(dict(semantic_ci_recomputed=True, missing_raw_artifacts=len(absent), raw_binding_recomputed=False), indent=2))
    return result


def prepare(root, c, frozen, followup, s, out):
    missing = [str(f) for f in inventory(root, frozen, followup) if not f.is_file()]
    if missing:
        raise ValueError('Raw audit prerequisite missing; no fit/extraction/retraining fallback: ' + ', '.join(missing))
    identity = src.read_json(followup / 'requested_settings.json')
    report = src.read_json(followup / 'report.json')
    if not report['status']['complete'] or post.digest(followup / 'requested_settings.json') != report['requested_settings_sha256']:
        raise ValueError('Follow-up incomplete or provenance changed')
    if identity['config'] != dict(c, output=identity['config']['output'], device=identity['config']['device']):
        raise ValueError('Follow-up dataset/model config differs')
    post.prepare(root, c, frozen, identity['settings'], out / 'source_provenance')
    files = [f for f in inventory(root, frozen, followup) if f.suffix != '.npy']
    for name, expected in identity['source_sha256'].items():
        # Formal execution uses the server's actual paths. Do not infer content
        # equivalence by filenames when the provided bundle has been relocated.
        f = Path(name)
        if not f.is_file() or post.digest(f) != expected:
            raise ValueError(f'Follow-up source changed: {f}')
    code = [p.HERE / n for n in ('linear_alignment.py', 'post_confirmation.py', 'semantic_uncertainty.py',
                                 'run_server.py', 'investigate_decline.py', 'validate_decline.py')]
    requested = dict(config=c, settings=s, source_followup=str(followup),
        input_sha256={str(f): post.digest(f) for f in files}, implementation_sha256={str(f): post.digest(f) for f in code},
        source_provenance_sha256=post.digest(out / 'source_provenance/requested_settings.json'),
        scope='Exploratory alignment after inspecting this split; no new independent confirmation')
    dest = out / 'requested_settings.json'
    if dest.exists() and src.read_json(dest) != requested:
        raise ValueError('Alignment code/settings/source changed; use a new --output')
    src.write(dest, requested)


def prediction_from_detail(detail, ids):
    import numpy as np
    return dict(score=np.stack([r[0] for r in detail]), label=np.stack([r[1] for r in detail]),
                patches=np.stack([r[2] for r in detail]), condition=np.stack([r[3] for r in detail]), image_id=ids)


def load_probe(c, variant, core, file):
    import torch
    import validate_decline as v
    core.seed_all(variant['seed'])
    model = v.make_readout(c, variant, core)
    model.load_state_dict(torch.load(file, map_location=c['device'], weights_only=True))
    return model.eval()


def raw_audit(c, frozen, followup, out, core):
    import numpy as np
    import torch
    import validate_decline as v
    import investigate_decline as inv
    import local_readout as lr
    import semantic_uncertainty as u
    fs = src.read_json(followup / 'requested_settings.json')['settings']
    original = src.read_json(Path(c['output']) / 'confirmation/report.json')
    cross = src.read_json(followup / 'report.json')
    checks = frozen['checks']
    endpoints = []
    for variant in checks['variants']:
        data = []
        for channel in post.CHANNELS:
            folder = src.task_folder(c, variant['name'], channel)
            baseline = v.predictions(folder / 'test_predictions.npz')
            model = load_probe(c, variant, core, folder / 'checkpoint.pt')
            _, _, detail = core.eval_probe(p.channel_config(c, channel), model, 'test', 23, True)
            actual = prediction_from_detail(detail, baseline['image_id'])
            v.aligned(actual, baseline)
            if not np.allclose(actual['score'], baseline['score'], atol=fs['score_atol'], rtol=fs['score_rtol']):
                raise ValueError('Source checkpoint fails diagonal prediction reproduction')
            data.append(baseline)
        computed = v.paired_bootstrap(*data, 'instance', checks['bootstrap_repeats'], checks['bootstrap_bins'],
                                     checks['bootstrap_seed'], checks['bootstrap_max_auc_error'])
        saved = next(r for r in original['results'] if r['variant'] == variant['name'])['binding']
        for key in ('first_auc_exact', 'second_auc_exact', 'delta_exact', 'delta_ci95'):
            np.testing.assert_allclose(computed[key], saved[key], rtol=0, atol=1e-12)
        endpoints.append(dict(variant=variant['name'], binding_ci_recomputed=computed, checkpoint_diagonals_reproduced=True))
    for row in cross['cross_matrix']:
        file = post.cross_folder(followup, row['variant'], row['fitted_at'], row['evaluated_at']) / 'test_predictions.npz'
        pred = v.predictions(file)
        meta, _ = inv.load_meta(c, 'test', row['evaluated_at'])
        lr.check_eval_sample(c, pred, meta)
        for task in ('instance', 'category'):
            mask, y = v.task_arrays(pred, task)
            if not np.isclose(v.auc(y[mask], pred['score'][mask]), row[task+'_auroc'], rtol=0, atol=1e-12):
                raise ValueError('Raw cross predictions disagree with report')
    semantic_diagonals = []
    classes = fs['semantic_num_classes']
    for channel in post.CHANNELS:
        d = followup / 'semantic' / channel / 'layer_23'
        core.seed_all(fs['probe_seed'])
        model = torch.nn.Linear(c['embed_dim'], classes).to(c['device'])
        model.load_state_dict(torch.load(d / 'checkpoint.pt', map_location=c['device'], weights_only=True)); model.eval()
        metrics, records = inv.evaluate_semantic(c, core, model, channel, 23, 'test', classes)
        if records != src.read_json(d / 'per_image.json') or metrics != src.read_json(d / 'metrics.json'):
            raise ValueError('Semantic checkpoint fails exact per-image prediction reproduction')
        semantic_diagonals.append(channel)
    semantic_ci = u.compare_folders(followup / 'semantic/post_attention/layer_23', followup / 'semantic/residual/layer_23',
                                   fs['bootstrap_repeats'], fs['semantic_bootstrap_seed'])
    for metric, values in semantic_ci['metrics'].items():
        np.testing.assert_allclose(values['delta_ci95'], cross['independent_semantic']['metrics'][metric]['delta_ci95'], rtol=0, atol=1e-12)
    result = dict(raw_audit_passed=True, source_request_sha256=post.digest(out / 'requested_settings.json'),
        binding=endpoints, cross_cells_auroc_recomputed=len(cross['cross_matrix']), cross_ci_recomputed=False,
        semantic_checkpoint_diagonals_reproduced=semantic_diagonals, semantic_bootstrap_recomputed=semantic_ci)
    src.write(out / 'raw_audit.json', result)
    print('Raw predictions, binding CIs and checkpoint diagonals verified.')
    return result


def require_audit(out):
    audit = src.read_json(out / 'raw_audit.json')
    if not audit['raw_audit_passed'] or audit['source_request_sha256'] != post.digest(out / 'requested_settings.json'):
        raise ValueError('A matching successful raw audit is required before fitting/evaluation')


class MappedFeatures:
    def __init__(self, bank, matrix):
        self.bank, self.matrix = bank, matrix
        self.shape, self.dtype = bank.shape, bank.dtype

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, index):
        import numpy as np
        return np.asarray(self.bank[index] @ self.matrix, dtype=np.float32)


def ridge_map(h, z, regularization):
    """Row-vector map, no centering/intercept. Small-array reference for tests."""
    import numpy as np
    gram = h.T @ h / len(h); cross = h.T @ z / len(h)
    ridge = regularization * np.trace(gram) / gram.shape[0]
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Degenerate training second moment')
    return np.linalg.solve(gram + ridge*np.eye(gram.shape[0]), cross)


def fit(c, s, out, core):
    import numpy as np
    require_audit(out)
    file = out / 'alignment.npz'
    if file.exists():
        load_map(c, out)
        print('Alignment complete; skipping fit.')
        return
    h, meta = core.cached(p.channel_config(c, 'residual'), 'train', 23)
    z, zmeta = core.cached(p.channel_config(c, 'post_attention'), 'train', 23)
    if any(not np.array_equal(meta[k], zmeta[k]) for k in meta):
        raise ValueError('Training map pairs are not aligned')
    rng = np.random.default_rng(s['statistics_seed'])
    rows = np.sort(rng.choice(len(h), min(len(h), s['max_statistics_images']), replace=False))
    d = c['embed_dim']; gram = np.zeros((d,d)); cross = np.zeros((d,d)); mean_z = np.zeros(d)
    n = 0; samples = []; squared_z = 0.
    train_ids = set(src.read_json(Path(c['output']) / 'splits.json')['train'])
    for row in rows:
        image_id = int(meta['image_id'][row])
        if image_id not in train_ids:
            raise ValueError('Alignment statistics include a non-train image')
        picks = rng.choice(c['saved_patches'], s['statistics_patches_per_image'], replace=False)
        a = np.asarray(h[row,picks], dtype=np.float64); b = np.asarray(z[row,picks], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Non-finite training features')
        gram += a.T@a; cross += a.T@b; mean_z += b.sum(0); squared_z += float((b*b).sum()); n += len(picks)
        samples.append([image_id, picks.tolist(), meta['position'][row,picks].tolist()])
    gram /= n; cross /= n; mean_z /= n
    ridge = s['ridge_relative']*np.trace(gram)/d
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Degenerate training second moment')
    system = gram + ridge*np.eye(d)
    matrix64 = np.linalg.solve(system, cross)
    matrix = matrix64.astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite fitted map')
    residual = float(np.linalg.norm(system@matrix64-cross)/max(np.linalg.norm(cross), 1e-30))
    if residual > 1e-8:
        raise ValueError('Ridge solve residual too large')
    np.savez_compressed(file, matrix=matrix, mean_z=mean_z)
    audit = dict(images=len(rows), tokens=n, image_ids=[int(meta['image_id'][r]) for r in rows], train_only=True,
        sample_sha256=hashlib.sha256(json.dumps(samples, separators=(',',':')).encode()).hexdigest(),
        ridge_relative=s['ridge_relative'], ridge_absolute=float(ridge), no_centering=True, no_intercept=True,
        relative_solve_residual=residual, train_target_mean_square=squared_z/(n*d),
        alignment_sha256=post.digest(file), requested_settings_sha256=post.digest(out/'requested_settings.json'))
    src.write(out / 'fit_audit.json', audit)
    print(f'Fitted one h-to-z map from {n} paired train tokens.')


def load_map(c, out):
    import numpy as np
    audit = src.read_json(out / 'fit_audit.json')
    file = out / 'alignment.npz'
    if audit['alignment_sha256'] != post.digest(file) or audit['requested_settings_sha256'] != post.digest(out / 'requested_settings.json'):
        raise ValueError('Alignment artifact/provenance changed')
    with np.load(file) as values:
        matrix = values['matrix'].copy(); mean = values['mean_z'].copy()
    if matrix.shape != (c['embed_dim'],c['embed_dim']) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError('Invalid alignment matrix')
    return matrix, mean


def reconstruction(c, matrix, mean, core):
    import numpy as np
    results = []
    for split in ('val', 'test'):
        h, meta = core.cached(p.channel_config(c,'residual'),split,23)
        z, _ = core.cached(p.channel_config(c,'post_attention'),split,23)
        total = dict(identity_sse=0., aligned_sse=0., target_sse=0., train_mean_sse=0.); tokens=0
        for row, image_id in enumerate(meta['image_id']):
            picks=np.random.default_rng(100000+c['split_seed']+int(image_id)).choice(c['saved_patches'],c['eval_patches'],replace=False)
            a=np.asarray(h[row,picks],dtype=np.float64); b=np.asarray(z[row,picks],dtype=np.float64)
            mapped=np.asarray(h[row,picks]@matrix,dtype=np.float64)
            for key, diff in [('identity_sse',a-b),('aligned_sse',mapped-b),('target_sse',b),('train_mean_sse',b-mean)]:
                total[key]+=float((diff*diff).sum())
            tokens+=len(picks)
        results.append(dict(split=split, images=len(h), tokens=tokens,
            identity_mse=total['identity_sse']/(tokens*c['embed_dim']), aligned_mse=total['aligned_sse']/(tokens*c['embed_dim']),
            aligned_relative_squared_error=total['aligned_sse']/max(total['target_sse'],1e-30),
            train_mean_baseline_mse=total['train_mean_sse']/(tokens*c['embed_dim'])))
    return results


def evaluate(c, frozen, followup, s, out, core):
    import numpy as np
    import torch
    import validate_decline as v
    import investigate_decline as inv
    require_audit(out)
    matrix, mean = load_map(c,out)
    bank, meta = core.cached(p.channel_config(c,'residual'),'test',23)
    mapped = MappedFeatures(bank,matrix)
    def cached(cfg,split,layer):
        if split!='test' or layer!=23:
            raise ValueError('Mapped evaluation only accepts test/L23')
        return mapped,meta
    for variant in frozen['checks']['variants']:
        dest=out/'binding'/variant['name']
        if (dest/'metrics.json').exists():
            if not (dest/'test_predictions.npz').exists() or not (dest/'per_image.json').exists():
                raise ValueError('Incomplete aligned binding evaluation')
            continue
        model=load_probe(c,variant,core,src.task_folder(c,variant['name'],'post_attention')/'checkpoint.pt')
        metrics,records,detail=core.eval_probe(p.channel_config(c,'residual'),model,'test',23,True,cached_fn=cached)
        dest.mkdir(parents=True,exist_ok=True); v.save_predictions(dest/'test_predictions.npz',detail,meta['image_id'])
        src.write(dest/'per_image.json',records); src.write(dest/'metrics.json',metrics)
    fs=src.read_json(followup/'requested_settings.json')['settings']
    core.seed_all(fs['probe_seed']); model=torch.nn.Linear(c['embed_dim'],fs['semantic_num_classes']).to(c['device'])
    model.load_state_dict(torch.load(followup/'semantic/post_attention/layer_23/checkpoint.pt',map_location=c['device'],weights_only=True)); model.eval()
    for name, provider in [('identity',core),('aligned',SimpleNamespace(cached=cached))]:
        dest=out/'semantic'/name
        if (dest/'metrics.json').exists():
            if not (dest/'per_image.json').exists():
                raise ValueError('Incomplete aligned semantic evaluation')
            continue
        metrics,records=inv.evaluate_semantic(c,provider,model,'residual',23,'test',fs['semantic_num_classes'])
        src.write(dest/'per_image.json',records); src.write(dest/'metrics.json',metrics)
    src.write(out/'reconstruction.json',reconstruction(c,matrix,mean,core))


def report(c, frozen, followup, s, out):
    import numpy as np
    import validate_decline as v
    import investigate_decline as inv
    import local_readout as lr
    import semantic_uncertainty as u
    require_audit(out); load_map(c,out)
    rows=[]; comparisons=[]; meta,_=inv.load_meta(c,'test','residual')
    for variant in frozen['checks']['variants']:
        name=variant['name']; folders=[src.task_folder(c,name,'post_attention'),
            post.cross_folder(followup,name,'post_attention','residual'),out/'binding'/name,
            src.task_folder(c,name,'residual')]
        data=[v.predictions(d/'test_predictions.npz') for d in folders]
        for prediction in data:
            lr.check_eval_sample(c,prediction,meta)
        contrasts={}
        for label,first,second in [('migration_improvement',1,2),('remaining_z_gap',0,2),('versus_refitted_h_probe',3,2)]:
            contrasts[label]=post.paired(data[first],data[second],'instance',s)
        improvement=contrasts['migration_improvement']
        assessment=('meaningful_migration_improvement' if improvement['delta_ci95'][0]>s['minimum_improvement'] else 'inconclusive_at_fixed_improvement')
        i,j=np.triu_indices(c['eval_patches'],1)
        crops=meta['position'][np.arange(len(meta['image_id']))[:,None],data[0]['patches']]//((c['crop_size']//14)**2)
        comparisons.append(dict(variant=name,assessment=assessment,binding=contrasts,
            category_score_improvement=post.paired(data[1],data[2],'category',s),
            anchored_rankings=inv.anchored_rankings(data[1],data[2],crops[:,i]!=crops[:,j])))
        for condition,pred in zip(('z_reference','identity_h_frozen_z_probe','aligned_h_frozen_z_probe','refitted_h_probe'),data):
            mask,y=v.task_arrays(pred,'instance')
            rows.append(dict(variant=name,condition=condition,hard_instance_auroc=v.auc(y[mask],pred['score'][mask])))
    semantic=u.compare_folders(out/'semantic/identity',out/'semantic/aligned',s['bootstrap_repeats'],s['semantic_bootstrap_seed'])
    semantic_zgap=u.compare_folders(followup/'semantic/post_attention/layer_23',out/'semantic/aligned',s['bootstrap_repeats'],s['semantic_bootstrap_seed'])
    primary=next(x for x in comparisons if x['variant']==frozen['checks']['primary_variant'])
    result=dict(scope='Exploratory on an inspected split; conditional CIs without multiplicity correction',
        primary_variant=frozen['checks']['primary_variant'],primary_assessment=primary['assessment'],
        measurements=rows,comparisons=comparisons,independent_semantic_improvement=semantic,
        remaining_semantic_z_gap=semantic_zgap,reconstruction=src.read_json(out/'reconstruction.json'),
        fit=src.read_json(out/'fit_audit.json'),raw_audit_passed=True,
        notes=['Improvement is of a frozen external decoder, not original-model behavior.',
               'A linear map does not expand the full bilinear function family; it need not be invertible.',
               'Failure cannot establish absent information. MSE recovery alone is not relation recovery.'])
    src.write(out/'report.json',result)
    with (out/'measurements.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=('variant','condition','hard_instance_auroc')); writer.writeheader();writer.writerows(rows)
    print(json.dumps(comparisons[0]['binding']['migration_improvement'],indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',default='runs/confirmation_l23_v1')
    parser.add_argument('--source-followup')
    parser.add_argument('--settings',default=str(p.HERE/'config-linear-alignment.json'))
    parser.add_argument('--output')
    parser.add_argument('--review-output',default='research_notes/2026-10-14-l23-review.json')
    parser.add_argument('--device')
    parser.add_argument('--stage',choices=('plan','review','audit','fit','evaluate','report','all'),default='plan')
    args=parser.parse_args();root=Path(args.source_run).resolve()
    followup=Path(args.source_followup).resolve() if args.source_followup else root/'post_confirmation_v1'
    out=post.output_path(root,args.output or str(root/'linear_alignment_v1'))
    if out==followup or out.is_relative_to(followup) or followup.is_relative_to(out):
        raise ValueError('Alignment output must be separate from source follow-up')
    c,frozen=post.context(root,args.device);s=settings(args.settings,c)
    if args.stage=='plan':
        print(json.dumps(dict(source_run=str(root),output=str(out),map='h @ B -> z, no intercept/centering',
            ridge_relative=s['ridge_relative'],maximum_train_tokens=min(c['max_train'],s['max_statistics_images'])*s['statistics_patches_per_image'],
            new_probe_training_tasks=0,backbone_forward_images=0,new_feature_cache_gib=0,
            missing_raw_artifacts=[str(f) for f in inventory(root,frozen,followup) if not f.is_file()],
            all='raw audit -> train-only fit -> frozen probe evaluation -> report'),indent=2));return
    if args.stage=='review':
        review_file=Path(args.review_output).resolve()
        if review_file.is_relative_to(root) or review_file.is_relative_to(followup) or review_file.is_relative_to(p.HERE):
            raise ValueError('Review output must be outside source runs and experiment code')
        review(root,frozen,followup,review_file);return
    prepare(root,c,frozen,followup,s,out)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=s['cpu_threads']):
        core=None
        for stage in (['audit','fit','evaluate','report'] if args.stage=='all' else [args.stage]):
            if stage!='report' and core is None:core=p.core_module(c)
            if stage=='audit':raw_audit(c,frozen,followup,out,core)
            elif stage=='fit':fit(c,s,out,core)
            elif stage=='evaluate':evaluate(c,frozen,followup,s,out,core)
            elif stage=='report':report(c,frozen,followup,s,out)


if __name__=='__main__':main()

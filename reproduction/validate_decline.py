"""Validate a cached layer decline before expanding models or datasets.

Default plan is read-only. Bootstrap/audit/geometry do not run a backbone.
Readouts fit a small fixed sensitivity panel; final-LN control loads weights only.
"""
from pathlib import Path
import argparse, copy, csv, hashlib, json, math
import importlib.metadata
import numpy as np
from sklearn.metrics import roc_auc_score
import phenomenon as p

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def read_settings(path):
    s = json.loads(Path(path).read_text(encoding='utf-8'))
    if len(s['layers'])!=2 or s['layers'][0]>=s['layers'][1]:
        raise ValueError('Choose exactly two increasing block indices')
    if set(s['channels'])-set(p.CHANNELS): raise ValueError('Unknown channel')
    names = [v['name'] for v in s['variants']]
    if len(names)!=len(set(names)) or any(not n.replace('_','').isalnum() for n in names):
        raise ValueError('Variant names must be distinct safe path components')
    if any(v['kind'] not in ('full','factor') or v['weight_decay']<0 or
        (v['kind']=='factor' and v.get('factor_width',0)<=0) for v in s['variants']):
        raise ValueError('Invalid readout specification')
    return s

def folder(c,s,channel,layer):
    return Path(p.channel_config(c,channel)['output'])/'probes'/s['baseline_probe']/f'layer_{layer}'/f"seed_{s['baseline_seed']}"

def predictions(file):
    with np.load(file) as z:
        required = {'score','label','patches','condition','image_id'}
        if required-set(z.files): raise ValueError(f'Missing alignment metadata in {file}; regenerate evaluation with current code.')
        data = {k:z[k].copy() for k in required}
    if data['score'].shape!=data['condition'].shape or data['score'].shape!=data['label'].shape:
        raise ValueError('Prediction shape mismatch')
    if not np.isfinite(data['score']).all(): raise ValueError('Non-finite scores')
    if len(np.unique(data['image_id']))!=len(data['image_id']): raise ValueError('Duplicate image IDs')
    return data

def aligned(a,b):
    for key in ('image_id','patches','condition','label'):
        if not np.array_equal(a[key],b[key]): raise ValueError(f'Layer comparison is not paired: {key} differs')

def task_arrays(data,task):
    code = data['condition']
    if task=='instance': return (code==1)|(code==2),code==1
    if task=='category': return (code==2)|(code==3),code==2
    raise ValueError(task)

def auc(y,score):
    return float(roc_auc_score(y,score)) if len(y) and y.any() and (~y).any() else None

def auc_histogram(pos,neg):
    """Weighted pooled AUROC with half credit for within-bin ties."""
    denom = pos.sum(axis=-1)*neg.sum(axis=-1)
    numerator = (pos*(np.cumsum(neg,axis=-1)-.5*neg)).sum(axis=-1)
    return np.divide(numerator,denom,out=np.full_like(numerator,np.nan,dtype=float),where=denom>0)

def image_histograms(data,task,bins):
    mask,y = task_arrays(data,task)
    scores = data['score']
    exact = auc(y[mask],scores[mask])
    if exact is None: raise ValueError('Task lacks positive or negative examples')
    edges = np.unique(np.quantile(scores[mask],np.linspace(0,1,bins+1)[1:-1]))
    nb = len(edges)+1
    pos = np.zeros((len(scores),nb),dtype=float); neg = np.zeros_like(pos)
    for row in range(len(scores)):
        indices = np.searchsorted(edges,scores[row,mask[row]],side='right')
        target = y[row,mask[row]]
        pos[row] = np.bincount(indices[target],minlength=nb)
        neg[row] = np.bincount(indices[~target],minlength=nb)
    approx = float(auc_histogram(pos.sum(0),neg.sum(0)))
    return pos,neg,exact,approx

def paired_bootstrap(a,b,task,repeats=10000,bins=2048,seed=31415,max_error=.0005):
    aligned(a,b)
    ha = image_histograms(a,task,bins); hb = image_histograms(b,task,bins)
    if any(abs(h[2]-h[3])>max_error for h in (ha,hb)):
        raise ValueError('Binned AUROC approximation error exceeds tolerance; increase bootstrap_bins.')
    n = len(a['image_id']); rng = np.random.default_rng(seed)
    first,second = [],[]
    # The same image multiplicities are used at both layers. All pairs of each
    # sampled image remain together; pairs are never sampled independently.
    for start in range(0,repeats,64):
        weights = rng.multinomial(n,np.full(n,1/n),size=min(64,repeats-start)).astype(float)
        first.append(auc_histogram(weights@ha[0],weights@ha[1]))
        second.append(auc_histogram(weights@hb[0],weights@hb[1]))
    first,second = np.concatenate(first),np.concatenate(second)
    valid = np.isfinite(first)&np.isfinite(second)
    if valid.sum()<.95*repeats: raise ValueError('Too many resamples lack both classes')
    delta = second[valid]-first[valid]
    return {'estimand':'pooled pair AUROC; image-cluster paired bootstrap',
        'images':n,'replicates':repeats,'valid_replicates':int(valid.sum()),'bins_requested':bins,
        'first_auc_exact':ha[2],'second_auc_exact':hb[2],'delta_exact':hb[2]-ha[2],
        'first_ci95':np.quantile(first[valid],[.025,.975]).tolist(),
        'second_ci95':np.quantile(second[valid],[.025,.975]).tolist(),
        'delta_ci95':np.quantile(delta,[.025,.975]).tolist(),
        'histogram_point_error':[ha[3]-ha[2],hb[3]-hb[2]],
        'approximation_note':'CIs use fixed quantile bins, point estimates use exact ranking; this is not an exact-rank bootstrap.'}

def bootstrap(c,s,out):
    rows = []
    for channel in s['channels']:
        a,b = [predictions(folder(c,s,channel,l)/'test_predictions.npz') for l in s['layers']]
        for task in ('instance','category'):
            result = paired_bootstrap(a,b,task,s['bootstrap_repeats'],s['bootstrap_bins'],s['bootstrap_seed'],s['bootstrap_max_auc_error'])
            rows.append(dict(channel=channel,task=task,layers=s['layers'],**result))
    write(out/'paired_bootstrap.json',rows)

def audit(c,s,out):
    rows = []
    for channel in s['channels']:
        manifests = []
        labels = []
        for layer in s['layers']:
            base = Path(p.channel_config(c,channel)['output'])/'cache'/'test'
            manifest = json.loads((base/'manifest.json').read_text())
            if layer not in manifest['layers'] or not manifest['complete']:
                raise ValueError('Requested layer cache incomplete')
            expected = p.channel_location(channel)
            if manifest.get('location')!=expected or manifest.get('channel')!=channel:
                raise ValueError('Missing or inconsistent computation-location provenance')
            if manifest['shape'][1:]!=[c['saved_patches'],c['embed_dim']]:
                raise ValueError('Cached dimensions do not match the requested model/patch configuration')
            manifests.append(manifest)
            labels.append(predictions(folder(c,s,channel,layer)/'test_predictions.npz'))
        aligned(*labels)
        rows.append({'channel':channel,'layers':s['layers'],'locations':[m['location'] for m in manifests],
            'same_patch_pairs':True,'same_signature':manifests[0]['signature']==manifests[1]['signature'],
            'final_layernorm_in_residual_cache':None,
            'final_layernorm_excluded_by_declared_location':channel=='residual',
            'scope':'Manifest and prediction alignment audit; cannot certify a changed server extractor from provenance alone.'})
    write(out/'position_audit.json',rows)

def verify_positions(c,s,out,core=None):
    """Recompute only a few images to check cached values against actual hooks."""
    import torch
    if core is None: core = p.core_module(c)
    # Match extract(): RNG state, TF32, cuDNN benchmark and determinism must be
    # configured before loading the model or processing any image.
    core.seed_all(c.get('split_seed',0))
    ds,proc = core.Dataset(c),core.processor(c)
    count = s.get('verify_images',3)
    if count<1: raise ValueError('verify_images must be positive for runtime verification')
    arrays = {}; reference = None
    for ch in s['channels']:
        for layer in s['layers']:
            acts,meta = core.cached(p.channel_config(c,ch),'test',layer)
            if reference is None: reference = meta
            elif any(not np.array_equal(meta[k],reference[k]) for k in reference):
                raise ValueError('Runtime audit caches are not aligned')
            arrays[ch,layer] = acts
    model = core.load_backbone(c); capture = p.Capture(model,s['layers'],s['channels'])
    rows = []
    try:
        for row in range(min(count,len(reference['image_id']))):
            image_id = int(reference['image_id'][row]); rgb,_,_,_ = ds.read(image_id)
            pixels = proc.forward(rgb).permute(0,3,1,2)
            positions = reference['position'][row]; grid = c['crop_size']//14
            for crop in range(len(pixels)):
                selected = np.flatnonzero(positions//(grid*grid)==crop)
                if not len(selected): continue
                capture.indices = positions[selected]%(grid*grid); capture.values.clear()
                with torch.inference_mode(): model(pixel_values=pixels[crop:crop+1].to(c['device']))
                for ch,layer in arrays:
                    actual = capture.values[ch,layer]; saved = arrays[ch,layer][row,selected]
                    difference = actual-saved
                    denominator = float(np.linalg.norm(saved))
                    norms = np.linalg.norm(actual,axis=-1)*np.linalg.norm(saved,axis=-1)
                    valid = norms>0
                    cosine = np.sum(actual*saved,axis=-1)[valid]/norms[valid]
                    rows.append({'image_id':image_id,'crop':crop,'channel':ch,'layer':layer,
                        'max_absolute_error':float(np.max(np.abs(difference))),
                        'rms_error':float(np.sqrt(np.mean(difference**2))),
                        'relative_l2_error':float(np.linalg.norm(difference)/denominator) if denominator else None,
                        'mean_patch_cosine':float(cosine.mean()) if len(cosine) else None,
                        'cached_abs_max':float(np.abs(saved).max()),
                        'allclose':bool(np.allclose(actual,saved,rtol=1e-4,atol=1e-4))})
    finally: capture.close()
    write(out/'runtime_position_audit.json',{'images':min(count,len(reference['image_id'])),
        'atol':1e-4,'rtol':1e-4,'allclose':all(r['allclose'] for r in rows),'comparisons':rows,
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,
            'transformers':importlib.metadata.version('transformers'),
            'model_revision_requested':c.get('model_revision'),
            'model_dtype':str(next(model.parameters(),torch.empty(0)).dtype),
            'split_seed':c.get('split_seed',0),
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_benchmark':torch.backends.cudnn.benchmark},
        'meaning':'Compare cached vectors with current actual post-block/projection hooks, not final normalized outputs.'})
    if not all(r['allclose'] for r in rows):
        raise ValueError('Cached features differ from current computation; inspect runtime_position_audit.json before interpreting a decline.')

def make_readout(c,v,core,normalizer=None):
    from types import SimpleNamespace as NS
    import torch
    if v['kind']=='full': model = core.probe(c,'quadratic')
    else:
        cls = core.load_author_classes(core.UPSTREAM/'src/utils/models.py',['QuadraticFixedRankProbe'])['QuadraticFixedRankProbe']
        model = cls(NS(probe=NS(in_dim=c['embed_dim'],rank=v['factor_width'])))
    if normalizer is not None:
        class Normalized(torch.nn.Module):
            def __init__(self,readout,norm):
                super().__init__(); self.readout = readout; self.norm = copy.deepcopy(norm).requires_grad_(False)
            def forward_pairwise(self,x,y): return self.readout.forward_pairwise(self.norm(x),self.norm(y))
        model = Normalized(model,normalizer)
    return model.to(c['device'])

def exact_task_score(detail,task='instance'):
    data = {'score':np.stack([v[0] for v in detail]),'condition':np.stack([v[3] for v in detail])}
    mask,y = task_arrays(data,task)
    return auc(y[mask],data['score'][mask])

def save_predictions(file,detail,image_ids):
    np.savez_compressed(file,score=np.stack([v[0] for v in detail]),label=np.stack([v[1] for v in detail]),
        patches=np.stack([v[2] for v in detail]),condition=np.stack([v[3] for v in detail]),image_id=image_ids)

def readouts(c,s,out):
    import torch
    core = p.core_module(c)
    # Validate requested caches/geometry before loading any weights.
    audit(c,s,out)
    normalizer = None
    if any(v.get('shared_final_layernorm') for v in s['variants']):
        # One weight load, no image forward and no new feature cache.
        cpu_cfg = dict(c,device='cpu')
        backbone = core.load_backbone(cpu_cfg)
        normalizer = copy.deepcopy(backbone.layernorm).cpu().requires_grad_(False)
        torch.save(normalizer.state_dict(),out/'shared_final_layernorm.pt')
        del backbone
    for channel in s['channels']:
        cfg = p.channel_config(c,channel)
        meta_test = core.cached(cfg,'test',s['layers'][0])[1]
        for variant in s['variants']:
            if variant.get('shared_final_layernorm') and channel!='residual':
                raise ValueError('Final LayerNorm control is defined only for residual')
            for layer in s['layers']:
                dest = out/'readouts'/channel/variant['name']/f'layer_{layer}'
                run_identity = {'config':c,'variant':variant,'layer':layer,'sampling_seed':s['sampling_seed'],
                    'selection':'validation instance-vs-same-class AUROC, first epoch wins ties'}
                if (dest/'run.json').exists() and json.loads((dest/'run.json').read_text())!=run_identity:
                    raise ValueError('Readout settings changed; choose a new --validation-output directory')
                if (dest/'metrics.json').exists(): continue
                dest.mkdir(parents=True,exist_ok=True); write(dest/'run.json',run_identity)
                core.seed_all(variant['seed'])
                model = make_readout(c,variant,core,normalizer if variant.get('shared_final_layernorm') else None)
                optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=c['learning_rate'],weight_decay=variant['weight_decay'])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,c['scheduler_step'],c['scheduler_gamma'])
                acts,meta = core.cached(cfg,'train',layer)
                rng = np.random.default_rng(s['sampling_seed']) # same batches/patches across all variants and layers
                history,best = [],-1.
                for epoch in range(c['epochs']):
                    model.train(); order = rng.permutation(len(acts)); losses = []
                    for start in range(0,len(order),c['batch_size']):
                        rows = order[start:start+c['batch_size']]
                        picks = [rng.choice(c['saved_patches'],c['sampled_patches'],replace=False) for _ in rows]
                        x = torch.tensor(np.stack([acts[r,q] for r,q in zip(rows,picks)]),device=c['device'])
                        ids = torch.tensor(np.stack([meta['instance'][r,q] for r,q in zip(rows,picks)]),device=c['device'])
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(core.batch_logits(model,x),(ids[:,:,None]==ids[:,None,:]).float())
                        loss.backward(); optimizer.step(); losses.append((loss.item(),len(rows)))
                    scheduler.step(); model.eval()
                    val,_,detail = core.eval_probe(cfg,model,'val',layer,True)
                    score = exact_task_score(detail)
                    if score is None: raise ValueError('Validation lacks hard-instance positives/negatives')
                    history.append({'epoch':epoch+1,'train_loss':sum(a*n for a,n in losses)/sum(n for _,n in losses),'val_hard_instance_auroc':score})
                    if score>best:
                        best = score; torch.save(model.state_dict(),dest/'checkpoint.pt')
                        write(dest/'selection.json',{'epoch':epoch+1,'val_hard_instance_auroc':score,'criterion':run_identity['selection']})
                    print(f"{channel} {variant['name']} layer={layer} epoch={epoch+1} val hard AUROC={score:.4f}",flush=True)
                model.load_state_dict(torch.load(dest/'checkpoint.pt',map_location=c['device'],weights_only=True)); model.eval()
                result,records,detail = core.eval_probe(cfg,model,'test',layer,True)
                write(dest/'history.json',history); write(dest/'per_image.json',records); write(dest/'metrics.json',result)
                save_predictions(dest/'test_predictions.npz',detail,meta_test['image_id'])

def boundary_fraction(mask,patch=14):
    boundary = np.zeros_like(mask,dtype=bool)
    changes = mask[1:]!=mask[:-1]; boundary[1:]|=changes; boundary[:-1]|=changes
    changes = mask[:,1:]!=mask[:,:-1]; boundary[:,1:]|=changes; boundary[:,:-1]|=changes
    h,w = mask.shape
    return boundary.reshape(h//patch,patch,w//patch,patch).mean(axis=(1,3))

def patch_geometry(c,s,core,pred):
    import torch
    ds,proc = core.Dataset(c),core.processor(c)
    cfg = p.channel_config(c,s['channels'][0]); meta = core.cached(cfg,'test',s['layers'][0])[1]
    if not np.array_equal(meta['image_id'],pred['image_id']): raise ValueError('Geometry image IDs mismatch')
    geometry = {key:[] for key in ('distance','area_mean','boundary_max','cross_crop')}
    i,j = np.triu_indices(c['eval_patches'],1); grid = c['crop_size']//14; per_crop = grid*grid
    for row,image_id in enumerate(pred['image_id']):
        _,_,inst,_ = ds.read(int(image_id))
        resized = proc.resize_image(torch.from_numpy(inst[:,:,None]),'nearest')
        crops = [a[:,:,0].numpy() for a in proc.crop_image(resized)]
        modes = proc.forward_patch(inst[:,:,None]).numpy().reshape(-1)
        picks = pred['patches'][row]; positions = meta['position'][row,picks].astype(int)
        if not np.array_equal(modes[positions],meta['instance'][row,picks]): raise ValueError('Geometry annotations do not match cached instance IDs')
        crop_id,local = positions//per_crop,positions%per_crop
        areas = []
        area_maps = []
        for crop in crops:
            values,counts = np.unique(crop,return_counts=True)
            area_maps.append(dict(zip(values,(counts/crop.size).tolist())))
        for crop,patch_id,instance in zip(crop_id,local,meta['instance'][row,picks]):
            areas.append(area_maps[crop][instance])
            # Build each crop boundary map once below; avoid computing per patch.
        bmaps = [boundary_fraction(a).reshape(-1) for a in crops]
        boundaries = np.array([bmaps[crop][patch_id] for crop,patch_id in zip(crop_id,local)])
        areas = np.array(areas)
        x,y = local%grid,local//grid
        geometry['distance'].append(np.hypot(x[i]-x[j],y[i]-y[j])/(math.sqrt(2)*(grid-1)))
        geometry['area_mean'].append((areas[i]+areas[j])/2)
        geometry['boundary_max'].append(np.maximum(boundaries[i],boundaries[j]))
        geometry['cross_crop'].append(crop_id[i]!=crop_id[j])
        if row%25==0: print(f'annotation geometry {row+1}/{len(pred["image_id"])}',flush=True)
    return {k:np.asarray(v) for k,v in geometry.items()}

def stratified_auc(a,b,geometry):
    aligned(a,b)
    mask,y = task_arrays(a,'instance'); mask &= ~geometry['cross_crop']
    specs = {'distance':[0,.1,.25,.5,1.001],'area_mean':[0,.01,.05,.15,.35,1.001],
        'boundary_max':[0,.01,.05,.2,1.001]}
    strata = np.zeros_like(a['condition'],dtype=int); multiplier = 1; rows = []
    for key,edges in specs.items():
        values = geometry[key]
        codes = np.searchsorted(np.array(edges[1:-1]),values,side='right')
        strata += codes*multiplier; multiplier *= len(edges)-1
        positive,negative = values[mask&y],values[mask&~y]
        rows.append({'variable':key,'positive_quantiles':np.quantile(positive,[.1,.5,.9]).tolist() if len(positive) else None,
            'negative_quantiles':np.quantile(negative,[.1,.5,.9]).tolist() if len(negative) else None})
    results = []; included = np.zeros_like(mask); total_weight = 0.; first = second = 0.
    for cell in np.unique(strata[mask]):
        selected = mask&(strata==cell); pos=int((selected&y).sum()); neg=int((selected&~y).sum())
        if min(pos,neg)<20: continue
        aa,bb = auc(y[selected],a['score'][selected]),auc(y[selected],b['score'][selected])
        weight = min(pos,neg); total_weight += weight; first += weight*aa; second += weight*bb
        included |= selected
        results.append({'cell':int(cell),'positive':pos,'negative':neg,'first_auc':aa,'second_auc':bb,'delta':bb-aa,'weight':weight})
    full_mask,_ = task_arrays(a,'instance')
    return {'scope':'exploratory within-crop coarse geometry strata, not a causal adjustment',
        'area_definition':'mean of endpoint instance pixel-area fractions within respective resized crops; not sampled patch coverage',
        'boundary_definition':'max endpoint fraction of pixels adjacent to an instance-label change (4-neighbour)',
        'bin_edges':specs,'minimum_examples_per_class_per_cell':20,
        'cross_crop_pairs_excluded':int((full_mask&geometry['cross_crop']).sum()),
        'eligible_pairs':int(mask.sum()),'included_positive':int((included&y).sum()),'included_negative':int((included&~y).sum()),
        'eligible_positive':int((mask&y).sum()),'eligible_negative':int((mask&~y).sum()),
        'summaries':rows,'strata':results,
        'overlap_weighted_first_auc':first/total_weight if total_weight else None,
        'overlap_weighted_second_auc':second/total_weight if total_weight else None,
        'overlap_weighted_delta':(second-first)/total_weight if total_weight else None,
        'weight_definition':'min(positive count,negative count) per retained cell; same weights for both layers'}

def geometry(c,s,out):
    core = p.core_module(c)
    a,b = [predictions(folder(c,s,s['channels'][0],l)/'test_predictions.npz') for l in s['layers']]
    aligned(a,b)
    geo_file = out/'test_geometry.npz'
    if geo_file.exists():
        with np.load(geo_file) as z:
            if not np.array_equal(z['image_id'],a['image_id']) or not np.array_equal(z['patches'],a['patches']):
                raise ValueError('Existing geometry uses different patches')
            geo = {k:z[k].copy() for k in ('distance','area_mean','boundary_max','cross_crop')}
    else:
        geo = patch_geometry(c,s,core,a)
        np.savez_compressed(geo_file,image_id=a['image_id'],patches=a['patches'],**geo)
    results = []
    for channel in s['channels']:
        aa,bb = [predictions(folder(c,s,channel,l)/'test_predictions.npz') for l in s['layers']]
        aligned(a,aa); aligned(aa,bb)
        results.append(dict(channel=channel,layers=s['layers'],**stratified_auc(aa,bb,geo)))
    write(out/'geometry_checks.json',results)

def report(c,s,out):
    rows = []; completed = 0
    for channel in s['channels']:
        for v in s['variants']:
            results = []
            for l in s['layers']:
                file = out/'readouts'/channel/v['name']/f'layer_{l}'/'metrics.json'
                if file.exists():
                    results.append(json.loads(file.read_text())); completed += 1
            if len(results)!=2: continue
            aa,bb = [r['instance_vs_same_class']['auroc'] for r in results]
            rows.append({'channel':channel,'variant':v['name'],'first_layer':s['layers'][0],'second_layer':s['layers'][1],
                'first_hard_auc':aa,'second_hard_auc':bb,'delta':bb-aa if aa is not None and bb is not None else None,
                'seed':v['seed'],'weight_decay':v['weight_decay'],'factor_width':v.get('factor_width'),
                'effective_rank_upper_bound':2*v['factor_width'] if v['kind']=='factor' else c['embed_dim'],
                'shared_final_layernorm':v.get('shared_final_layernorm',False)})
    if rows:
        with (out/'readout_sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    expected = len(s['channels'])*len(s['layers'])*len(s['variants'])
    write(out/'status.json',{'completed_training_tasks':completed,'expected_training_tasks':expected,'complete':completed==expected,
        'paired_variants':len(rows),
        'negative_delta_variants':sum(r['delta'] is not None and r['delta']<0 for r in rows),
        'nonnegative_delta_variants':sum(r['delta'] is not None and r['delta']>=0 for r in rows),
        'bootstrap_done':(out/'paired_bootstrap.json').exists(),'position_audit_done':(out/'position_audit.json').exists(),
        'runtime_position_audit_done':(out/'runtime_position_audit.json').exists(),
        'geometry_done':(out/'geometry_checks.json').exists(),
        'note':'Readouts use a new common validation selector; no test-set variant tuning.'})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(p.HERE/'config-overnight.json'))
    parser.add_argument('--checks',default=str(p.HERE/'config-validation.json'))
    parser.add_argument('--dataset-root'); parser.add_argument('--validation-output')
    parser.add_argument('--stage',choices=['plan','audit','verify-positions','bootstrap','readouts','geometry','report','all'],default='plan')
    args = parser.parse_args(); c = p.read_config(args.config,args.dataset_root); s = read_settings(args.checks)
    if set(s['layers'])-set(c['layers']): raise ValueError('Validation layers not in cached-run config')
    tasks = len(s['variants'])*len(s['layers'])*len(s['channels'])
    if tasks>s['max_training_tasks']: raise ValueError('Readout sensitivity panel exceeds explicit task limit')
    if args.stage=='plan':
        print(json.dumps({'layers':s['layers'],'channels':s['channels'],'new_feature_cache_gib':0,
            'training_tasks':tasks,'bootstrap_repeats':s['bootstrap_repeats'],
            'selection':'validation hard-instance AUROC for every new readout',
            'backbone_forward_images':s.get('verify_images',3),'new_feature_extraction':False,'geometry_reads':'test masks only, CPU',
            'final_ln_control':'loads frozen model weights once, no forward',
            'rank_note':'Author factor width r gives symmetric W rank <= 2r, not <= r.'},indent=2)); return
    out = Path(args.validation_output or str(Path(c['output'])/'validation')); out.mkdir(parents=True,exist_ok=True)
    provenance = {'config':c,'checks':s}
    file = out/'requested_checks.json'
    if file.exists() and json.loads(file.read_text())!=provenance:
        raise ValueError('Validation settings changed; use a new --validation-output directory')
    write(file,provenance)
    for stage in (['audit','verify-positions','bootstrap','readouts','geometry','report'] if args.stage=='all' else [args.stage]):
        globals()[stage.replace('-','_')](c,s,out)

if __name__=='__main__': main()

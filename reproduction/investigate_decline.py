"""Investigate a validated decline without expanding models or datasets.

Default plan performs no model loading. Conditions reuse validation predictions;
depth/steps extract only missing cache entries. Semantic probes are opt-in.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, hashlib, json
import phenomenon as p


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_settings(path, c):
    s = json.loads(Path(path).read_text(encoding='utf-8'))
    for name in ('baseline_layers', 'depth_layers', 'step_layers'):
        layers = s[name]
        if not layers or layers != sorted(set(layers)) or any(type(l) is not int or not 0 <= l < c['num_layers'] for l in layers):
            raise ValueError(f'Invalid {name}')
    if len(s['baseline_layers']) != 2: raise ValueError('Exactly two baseline layers required')
    if not 0 < s['near_distance'] < 1 or not 0 <= s['boundary_threshold'] <= 1:
        raise ValueError('Invalid condition thresholds')
    for key in ('minimum_examples_per_class','semantic_num_classes','semantic_epochs','max_binding_tasks','max_semantic_tasks'):
        if type(s[key]) is not int or s[key] <= 0: raise ValueError(f'Invalid {key}')
    if s['semantic_learning_rate'] <= 0 or s['probe_weight_decay'] < 0: raise ValueError('Invalid optimizer settings')
    source = Path(s['baseline_source'])
    if source.is_absolute() or '..' in source.parts: raise ValueError('baseline_source must stay within the existing run')
    return s


def scope_config(c, s, scope, out):
    cfg = dict(c, channels=['residual'] if scope == 'depth' else ['block_input','post_attention','residual'],
        layers=s['depth_layers'] if scope == 'depth' else s['step_layers'], probes=['quadratic'], seeds=[s['probe_seed']],
        max_tasks=s['max_binding_tasks'], provenance_output=str(out))
    p.enforce_budget(cfg)
    return cfg


def files(c, s, channel, layer):
    return Path(c['output']) / s['baseline_source'] / f'layer_{layer}'


def baseline(c, s):
    import validate_decline as v
    a, b = [v.predictions(files(c,s,'residual',l)/'test_predictions.npz') for l in s['baseline_layers']]
    v.aligned(a,b)
    for layer in s['baseline_layers']: verify_baseline_run(c,s,files(c,s,'residual',layer),layer)
    return a,b


def verify_baseline_run(c, s, folder, layer):
    run = json.loads((folder/'run.json').read_text())
    variant = run.get('variant',{})
    expected = {'kind':'full','seed':s['probe_seed'],'weight_decay':s['probe_weight_decay']}
    if any(variant.get(k)!=value for k,value in expected.items()) or variant.get('shared_final_layernorm',False):
        raise ValueError('Baseline readout differs from the configured full readout')
    keys = ('model','model_revision','embed_dim','crop_size','saved_patches','eval_patches','sampled_patches',
        'split_seed','dataset_name','dataset_factory','max_train','max_val','max_test','epochs','batch_size',
        'learning_rate','scheduler_step','scheduler_gamma')
    if any(run['config'].get(k)!=c.get(k) for k in keys) or run['layer']!=layer or run['sampling_seed']!=s['sampling_seed']:
        raise ValueError('Baseline training/data settings differ; use the actual completed server config')
    if run.get('selection')!='validation instance-vs-same-class AUROC, first epoch wins ties':
        raise ValueError('Baseline must use the common validation hard-instance selector')


def load_meta(c, split, channel='residual'):
    import numpy as np
    folder = Path(p.channel_config(c,channel)['output'])/'cache'/split
    manifest = json.loads((folder/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('schema')!=2 or manifest.get('location')!=p.channel_location(channel):
        raise ValueError(f'Incomplete or incorrect-location cache: {folder}')
    if manifest.get('channel') != channel: raise ValueError('Cache channel mismatch')
    splits = json.loads((Path(c['output'])/'splits.json').read_text())
    source_hash = hashlib.sha256((p.HERE.parent/'vit-object-binding/src/data_extractor.py').read_bytes()).hexdigest()
    if manifest['signature']!=p.identity(c,splits,source_hash): raise ValueError('Cache identity differs from requested run')
    with np.load(folder/'labels.npz') as z: meta = {k:z[k].copy() for k in z.files}
    if manifest['shape']!=[len(meta['image_id']),c['saved_patches'],c['embed_dim']]: raise ValueError('Cache dimensions differ')
    if not np.array_equal(meta['image_id'],splits[split]): raise ValueError('Cache split differs')
    return meta,manifest


def check_prediction_meta(pred, meta):
    import numpy as np
    if not np.array_equal(pred['image_id'],meta['image_id']): raise ValueError('Prediction/cache image alignment differs')
    patches = pred['patches']; n,k = patches.shape
    if not np.issubdtype(patches.dtype,np.integer): raise ValueError('Patch indices must be integers')
    if (patches<0).any() or (patches>=meta['instance'].shape[1]).any() or any(len(set(row))!=k for row in patches):
        raise ValueError('Invalid prediction patch indices')
    i,j = np.triu_indices(k,1); rows = np.arange(n)[:,None]
    ids,sem,bg = [meta[key][rows,patches] for key in ('instance','semantic','background')]
    y = ids[:,i]==ids[:,j]; fg = (~bg[:,i].astype(bool)) & (~bg[:,j].astype(bool)); known = (sem[:,i]>0)&(sem[:,j]>0)
    code = np.zeros(y.shape,dtype=np.uint8); code[fg]=4; code[fg&known&y]=1
    code[fg&known&~y&(sem[:,i]==sem[:,j])]=2; code[fg&known&~y&(sem[:,i]!=sem[:,j])]=3
    if not np.array_equal(pred['label'],y) or not np.array_equal(pred['condition'],code):
        raise ValueError('Prediction pair labels differ from cached annotations')


def sampled_competition(meta, patches, per_crop):
    """Count sampled same-class instances within each crop; a lower bound only."""
    import numpy as np
    result = np.zeros(patches.shape,dtype=np.int32)
    for row,picks in enumerate(patches):
        crop = meta['position'][row]//per_crop
        sem,ids = meta['semantic'][row],meta['instance'][row]
        valid = (~meta['background'][row].astype(bool))&(sem>0)
        counts = {(int(cr),int(cl)):len(np.unique(ids[valid&(crop==cr)&(sem==cl)]))
            for cr,cl in set(zip(crop[valid].tolist(),sem[valid].tolist()))}
        result[row] = [counts.get((int(crop[q]),int(sem[q])),0) for q in picks]
    return result


def condition_masks(a, geo, competition, s):
    import numpy as np
    code = a['condition']; hard = (code==1)|(code==2)
    same_crop = ~geo['cross_crop']; i,j = np.triu_indices(a['patches'].shape[1],1)
    count = np.minimum(competition[:,i],competition[:,j])
    boundary = geo['boundary_max']>s['boundary_threshold']
    near = geo['distance']<=s['near_distance']
    return {'hard_instance':hard,'foreground_binding':(code>=1)&(code<=3),
        'instance_vs_cross_class':(code==1)|(code==3),
        'within_crop':hard&same_crop,'cross_crop':hard&~same_crop,
        'near':hard&same_crop&near,'far':hard&same_crop&~near,
        'interior':hard&same_crop&~boundary,'boundary':hard&same_crop&boundary,
        'sampled_two_instances':hard&same_crop&(count==2),
        'sampled_three_plus_instances':hard&same_crop&(count>=3)}


def conditional_metrics(a,b,masks,minimum):
    import numpy as np
    import validate_decline as v
    v.aligned(a,b); rows=[]
    y = a['condition']==1
    for name,mask in masks.items():
        pos,neg = int((mask&y).sum()),int((mask&~y).sum())
        aa,bb = (v.auc(y[mask],d['score'][mask]) for d in (a,b)) if min(pos,neg)>=minimum else (None,None)
        per_image = [v.auc(y[r,m],a['score'][r,m]) for r,m in enumerate(mask)]
        image_deltas = [v.auc(y[r,m],b['score'][r,m])-first for r,(m,first) in enumerate(zip(mask,per_image)) if first is not None]
        rows.append({'condition':name,'positive':pos,'negative':neg,'images_with_pairs':int(mask.any(1).sum()),
            'images_with_both_labels':len(image_deltas),'first_auc':aa,'second_auc':bb,
            'delta':bb-aa if aa is not None else None,
            'macro_image_delta':float(np.mean(image_deltas)) if image_deltas else None,
            'raw_positive_score_median':[float(np.median(d['score'][mask&y])) if pos else None for d in (a,b)],
            'raw_negative_score_median':[float(np.median(d['score'][mask&~y])) if neg else None for d in (a,b)],
            'raw_score_note':'Uncalibrated logits; cross-layer medians do not establish attraction or repulsion.'})
    return rows


def anchored_rankings(a,b,cross_crop):
    """Exact shared-anchor ordering, with half credit for ties and fixed triples."""
    import numpy as np
    import validate_decline as v
    v.aligned(a,b); k=a['patches'].shape[1]; i,j=np.triu_indices(k,1)
    totals = dict(triples=0,first_credit=0.,second_credit=0.,strict_win_to_loss=0,strict_loss_to_win=0)
    per_image=[]
    for row,image_id in enumerate(a['image_id']):
        code=np.zeros((k,k),dtype=np.uint8); code[i,j]=a['condition'][row]; code[j,i]=code[i,j]
        eligible=np.zeros((k,k),dtype=bool); eligible[i,j]=~cross_crop[row]; eligible[j,i]=eligible[i,j]
        scores=[]
        for d in (a,b):
            m=np.zeros((k,k)); m[i,j]=d['score'][row]; m[j,i]=m[i,j]; scores.append(m)
        local=dict(triples=0,first_credit=0.,second_credit=0.,strict_win_to_loss=0,strict_loss_to_win=0)
        for anchor in range(k):
            pos=(code[anchor]==1)&eligible[anchor]; neg=(code[anchor]==2)&eligible[anchor]
            if not pos.any() or not neg.any(): continue
            diffs=[m[anchor,pos,None]-m[anchor,neg][None,:] for m in scores]
            local['triples']+=diffs[0].size
            for key,diff in zip(('first_credit','second_credit'),diffs): local[key]+=float((diff>0).sum()+.5*(diff==0).sum())
            local['strict_win_to_loss']+=int(((diffs[0]>0)&(diffs[1]<0)).sum())
            local['strict_loss_to_win']+=int(((diffs[0]<0)&(diffs[1]>0)).sum())
        for key in totals: totals[key]+=local[key]
        n=local['triples']
        per_image.append(dict(image_id=int(image_id),**local,delta=(local['second_credit']-local['first_credit'])/n if n else None))
    n=totals['triples']
    return {'scope':'Within-crop anchor/positive/negative triples; positive=same instance, negative=same class different instance',
        'weighting':'pooled triples; triples sharing anchors/images are dependent, no independent-triple significance test',
        **totals,'first_ordering':totals['first_credit']/n if n else None,'second_ordering':totals['second_credit']/n if n else None,
        'delta':(totals['second_credit']-totals['first_credit'])/n if n else None,'per_image':per_image}


def conditions(c,s,out,ci=False):
    import numpy as np
    import validate_decline as v
    a,b=baseline(c,s); meta,_=load_meta(c,'test'); check_prediction_meta(a,meta)
    geo_file=Path(c['output'])/'validation/test_geometry.npz'
    with np.load(geo_file) as z:
        if not np.array_equal(z['image_id'],a['image_id']) or not np.array_equal(z['patches'],a['patches']):
            raise ValueError('Validation geometry uses different patch pairs')
        geo={key:z[key].copy() for key in ('distance','area_mean','boundary_max','cross_crop')}
    if any(value.shape!=a['score'].shape for value in geo.values()): raise ValueError('Geometry dimensions differ')
    selected=meta['position'][np.arange(len(a['image_id']))[:,None],a['patches']]
    crop=selected//((c['crop_size']//14)**2); i,j=np.triu_indices(a['patches'].shape[1],1)
    if not np.array_equal(geo['cross_crop'],crop[:,i]!=crop[:,j]): raise ValueError('Geometry crop labels differ from cache')
    if any(not np.isfinite(geo[key]).all() for key in ('distance','area_mean','boundary_max')):
        raise ValueError('Non-finite geometry')
    competition=sampled_competition(meta,a['patches'],(c['crop_size']//14)**2)
    masks=condition_masks(a,geo,competition,s)
    rows=conditional_metrics(a,b,masks,s['minimum_examples_per_class'])
    if ci:
        for row in rows:
            if row['first_auc'] is None: continue
            mask=masks[row['condition']]; keep=mask.any(1)
            datasets=[]
            for source in (a,b):
                d={key:value[keep].copy() for key,value in source.items()}
                d['condition']=np.where(mask[keep],np.where(a['condition'][keep]==1,1,2),0).astype(np.uint8)
                datasets.append(d)
            try:
                row['bootstrap']=v.paired_bootstrap(*datasets,'instance',s['bootstrap_repeats'],s['bootstrap_bins'],
                    s['bootstrap_seed'],s['bootstrap_max_auc_error'])
            except ValueError as error:
                row['bootstrap_unavailable']=str(error)
    write(out/('condition_ci.json' if ci else 'conditions.json'),{'layers':s['baseline_layers'],'source':s['baseline_source'],
        'exploratory':True,'minimum_examples_per_class':s['minimum_examples_per_class'],
        'competition_definition':'Same-class instances observed among saved patches in the same crop; lower bound, not full-scene count',
        'not_measured':['appearance similarity','occlusion continuity'], 'conditions':rows})
    if not ci: write(out/'anchored_rankings.json',anchored_rankings(a,b,geo['cross_crop']))
    fields=['condition','positive','negative','images_with_pairs','images_with_both_labels','first_auc','second_auc','delta','macro_image_delta']
    with (out/('condition_ci.csv' if ci else 'conditions.csv')).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} condition summaries to {out}',flush=True)


def cache_jobs(c):
    jobs=[]
    for channel in c['channels']:
        for layer in c['layers']: jobs.append((channel,layer))
    return jobs


def validate_caches(c):
    import numpy as np
    reference={}
    for channel in c['channels']:
        for split in ('train','val','test'):
            meta,manifest=load_meta(c,split,channel)
            if set(c['layers'])-set(manifest['layers']): raise ValueError('Missing layers; run --stage extract first')
            if split in reference and any(not np.array_equal(meta[key],reference[split][key]) for key in meta):
                raise ValueError('Computation-point caches are not aligned')
            reference[split]=meta
            for layer in c['layers']:
                a=np.load(Path(p.channel_config(c,channel)['output'])/'cache'/split/f'layer_{layer}.npy',mmap_mode='r')
                if list(a.shape)!=manifest['shape'] or a.dtype!=np.float32: raise ValueError('Invalid feature array')


def reusable_binding(c,s,channel,layer):
    if channel!='residual' or layer not in s['baseline_layers']: return None
    src=files(c,s,channel,layer)
    if not (src/'metrics.json').exists(): return None
    verify_baseline_run(c,s,src,layer)
    return src


def train_binding(c,s,out):
    import numpy as np
    import torch
    import validate_decline as v
    validate_caches(c); core=p.core_module(c)
    for channel,layer in cache_jobs(c):
        src=reusable_binding(c,s,channel,layer)
        dest=out/'binding'/channel/f'layer_{layer}'
        if src:
            write(dest/'reused.json',{'source':str(src),'selection':'validation hard-instance AUROC'})
            print(f'Reusing validated {channel} L{layer}',flush=True); continue
        if (dest/'metrics.json').exists(): continue
        dest.mkdir(parents=True,exist_ok=True); cfg=p.channel_config(c,channel)
        core.seed_all(s['probe_seed']); model=core.probe(c,'quadratic').to(c['device'])
        optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=s['probe_weight_decay'])
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,c['scheduler_step'],c['scheduler_gamma'])
        acts,meta=core.cached(cfg,'train',layer); rng=np.random.default_rng(s['sampling_seed']); best=-1.; history=[]
        for epoch in range(c['epochs']):
            model.train(); order=rng.permutation(len(acts)); loss_sum=0.; count=0
            for start in range(0,len(order),c['batch_size']):
                rows=order[start:start+c['batch_size']]
                picks=[rng.choice(c['saved_patches'],c['sampled_patches'],replace=False) for _ in rows]
                x=torch.tensor(np.stack([acts[r,q] for r,q in zip(rows,picks)]),device=c['device'])
                ids=torch.tensor(np.stack([meta['instance'][r,q] for r,q in zip(rows,picks)]),device=c['device'])
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(core.batch_logits(model,x),(ids[:,:,None]==ids[:,None,:]).float())
                loss.backward(); optimizer.step(); loss_sum+=loss.item()*len(rows); count+=len(rows)
            scheduler.step(); model.eval(); _,_,detail=core.eval_probe(cfg,model,'val',layer,True)
            score=v.exact_task_score(detail)
            if score is None: raise ValueError('Validation lacks hard-instance labels')
            history.append(dict(epoch=epoch+1,train_loss=loss_sum/count,val_hard_instance_auroc=score))
            if score>best:
                best=score; torch.save(model.state_dict(),dest/'checkpoint.pt')
                write(dest/'selection.json',dict(epoch=epoch+1,val_hard_instance_auroc=score,criterion='validation hard-instance AUROC; first epoch wins ties'))
            print(f'{channel} L{layer} binding epoch {epoch+1}: val={score:.4f}',flush=True)
        model.load_state_dict(torch.load(dest/'checkpoint.pt',map_location=c['device'],weights_only=True)); model.eval()
        metrics,records,detail=core.eval_probe(cfg,model,'test',layer,True)
        write(dest/'history.json',history); write(dest/'per_image.json',records)
        v.save_predictions(dest/'test_predictions.npz',detail,core.cached(cfg,'test',layer)[1]['image_id'])
        write(dest/'metrics.json',metrics) # completion marker written last


def semantic_metrics(target,predicted,classes):
    import numpy as np
    target=np.asarray(target); predicted=np.asarray(predicted)
    total=np.bincount(target,minlength=classes); correct=np.bincount(target[target==predicted],minlength=classes)
    supported=total>0
    return {'n':int(len(target)),'accuracy':float((target==predicted).mean()) if len(target) else None,
        'macro_recall':float((correct[supported]/total[supported]).mean()) if supported.any() else None,
        'supported_classes':int(supported.sum()),'class_support':total.tolist(),'class_correct':correct.tolist()}


def evaluate_semantic(c,core,model,channel,layer,split,classes):
    import numpy as np
    import torch
    acts,meta=core.cached(p.channel_config(c,channel),split,layer); targets=[]; predicted=[]; details=[]
    with torch.inference_mode():
        for row,image_id in enumerate(meta['image_id']):
            picks=np.random.default_rng(100000+c['split_seed']+int(image_id)).choice(c['saved_patches'],c['eval_patches'],replace=False)
            y=meta['semantic'][row,picks].astype(np.int64)-1
            valid=(~meta['background'][row,picks].astype(bool))&(y>=0)
            if (y[valid]>=classes).any(): raise ValueError('Semantic class exceeds configured taxonomy')
            x=torch.tensor(np.asarray(acts[row,picks]),device=c['device'])
            pred=model(x).argmax(-1).cpu().numpy(); targets.append(y[valid]); predicted.append(pred[valid])
            details.append(dict(image_id=int(image_id),**semantic_metrics(y[valid],pred[valid],classes)))
    return semantic_metrics(np.concatenate(targets),np.concatenate(predicted),classes),details


def train_semantic(c,s,out):
    import numpy as np
    import torch
    validate_caches(c)
    if len(cache_jobs(c))>s['max_semantic_tasks']: raise ValueError('Semantic task limit exceeded')
    core=p.core_module(c); classes=s['semantic_num_classes']
    for channel,layer in cache_jobs(c):
        dest=out/'semantic'/channel/f'layer_{layer}'
        if (dest/'metrics.json').exists(): continue
        dest.mkdir(parents=True,exist_ok=True); core.seed_all(s['probe_seed'])
        model=torch.nn.Linear(c['embed_dim'],classes).to(c['device'])
        optimizer=torch.optim.Adam(model.parameters(),lr=s['semantic_learning_rate'])
        acts,meta=core.cached(p.channel_config(c,channel),'train',layer)
        rng=np.random.default_rng(s['sampling_seed']); best=-1.; history=[]
        for epoch in range(s['semantic_epochs']):
            model.train(); order=rng.permutation(len(acts)); loss_sum=0.; count=0
            for start in range(0,len(order),c['batch_size']):
                rows=order[start:start+c['batch_size']]; picks=[rng.choice(c['saved_patches'],c['sampled_patches'],replace=False) for _ in rows]
                x=np.stack([acts[r,q] for r,q in zip(rows,picks)]).reshape(-1,c['embed_dim'])
                y=np.stack([meta['semantic'][r,q] for r,q in zip(rows,picks)]).reshape(-1).astype(np.int64)-1
                bg=np.stack([meta['background'][r,q] for r,q in zip(rows,picks)]).reshape(-1).astype(bool)
                valid=(~bg)&(y>=0)
                if not valid.any(): continue
                if (y[valid]>=classes).any(): raise ValueError('Semantic class exceeds configured taxonomy')
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(torch.tensor(x[valid],device=c['device'])),torch.tensor(y[valid],device=c['device']))
                loss.backward(); optimizer.step(); loss_sum+=loss.item()*int(valid.sum()); count+=int(valid.sum())
            if not count: raise ValueError('No known foreground training labels')
            model.eval(); val,_=evaluate_semantic(c,core,model,channel,layer,'val',classes); score=val['macro_recall']
            if score is None: raise ValueError('No known foreground validation labels')
            history.append(dict(epoch=epoch+1,train_loss=loss_sum/count,val_macro_recall=score))
            if score>best:
                best=score; torch.save(model.state_dict(),dest/'checkpoint.pt')
                write(dest/'selection.json',dict(epoch=epoch+1,val_macro_recall=score,criterion='validation macro recall; first epoch wins ties'))
            print(f'{channel} L{layer} semantic epoch {epoch+1}: val macro recall={score:.4f}',flush=True)
        model.load_state_dict(torch.load(dest/'checkpoint.pt',map_location=c['device'],weights_only=True)); model.eval()
        metrics,details=evaluate_semantic(c,core,model,channel,layer,'test',classes)
        write(dest/'history.json',history); write(dest/'per_image.json',details); write(dest/'metrics.json',metrics)


def report(c,s,out):
    import validate_decline as v
    rows=[]; comparisons=[]; expected=cache_jobs(c)
    for channel,layer in expected:
        dest=out/'binding'/channel/f'layer_{layer}'; src=reusable_binding(c,s,channel,layer) or dest
        row=dict(channel=channel,layer=layer,location=p.channel_location(channel),binding_source=str(src),binding_complete=False,semantic_complete=False)
        if (src/'metrics.json').exists():
            m=json.loads((src/'metrics.json').read_text()); row.update(binding_complete=True,
                hard_instance_auroc=m['instance_vs_same_class']['auroc'],binding_category_auroc=m['class_grouping_cross_instance']['auroc'])
        sem=out/'semantic'/channel/f'layer_{layer}'/'metrics.json'
        if sem.exists():
            m=json.loads(sem.read_text()); row.update(semantic_complete=True,semantic_accuracy=m['accuracy'],semantic_macro_recall=m['macro_recall'])
        rows.append(row)
    edges=[]
    if c['channels']==['residual']:
        edges=[(('residual',a),('residual',b)) for a,b in zip(c['layers'],c['layers'][1:])]
    else:
        edges=[((a,layer),(b,layer)) for layer in c['layers'] for a,b in [('block_input','post_attention'),('post_attention','residual')]]
    bykey={(r['channel'],r['layer']):r for r in rows}
    for first,second in edges:
        aa,bb=bykey[first],bykey[second]
        if not aa['binding_complete'] or not bb['binding_complete']: continue
        a,b=[v.predictions(Path(r['binding_source'])/'test_predictions.npz') for r in (aa,bb)]
        v.aligned(a,b)
        comparisons.append({'first':list(first),'second':list(second),
            **v.paired_bootstrap(a,b,'instance',s['bootstrap_repeats'],s['bootstrap_bins'],s['bootstrap_seed'],s['bootstrap_max_auc_error'])})
    write(out/'report.json',{'measurements':rows,'paired_comparisons':comparisons,
        'binding_complete':sum(r['binding_complete'] for r in rows),'binding_expected':len(expected),
        'semantic_complete':sum(r['semantic_complete'] for r in rows),'semantic_expected_if_requested':len(expected),
        'note':'Independent point-specific probes measure decodability, not causal model use; semantic classification is separate from binding-category AUROC.'})
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (out/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    plot_measurements(c,rows,out)


def plot_measurements(c,rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    specs=[('hard_instance_auroc','Instance binding AUROC'),('binding_category_auroc','Category AUROC of binding scores'),
        ('semantic_macro_recall','Independent semantic probe macro recall')]
    for ax,(key,title) in zip(axes,specs):
        if c['channels']==['residual']:
            selected=[r for r in rows if r.get(key) is not None]
            ax.plot([r['layer'] for r in selected],[r[key] for r in selected],marker='o')
            ax.set_xlabel('Block index (0-based)')
        else:
            names=['block_input','post_attention','residual']
            for layer in c['layers']:
                selected={r['channel']:r for r in rows if r['layer']==layer}
                ax.plot(range(3),[selected[name].get(key,float('nan')) for name in names],marker='o',label=f'L{layer}')
            ax.set_xticks(range(3),['Block input','After attention','After MLP']); ax.legend()
        if not any(r.get(key) is not None for r in rows): ax.text(.5,.5,'Not run',ha='center',transform=ax.transAxes)
        ax.set_title(title); ax.set_ylim(0,1); ax.grid(alpha=.2)
    fig.savefig(out/'measurements.png',dpi=160); plt.close(fig)


def plan(c,s,cfg,out):
    additional=0; missing=[]; existing=sum(f.stat().st_size for f in Path(c['output']).glob('channels/*/cache/*/layer_*.npy'))
    for channel,layer in cache_jobs(cfg):
        for split in ('train','val','test'):
            folder=Path(p.channel_config(cfg,channel)['output'])/'cache'/split
            file=folder/'manifest.json'; manifest=json.loads(file.read_text()) if file.exists() else {}
            if manifest.get('complete') and layer in manifest.get('layers',[]) and (folder/f'layer_{layer}.npy').exists(): continue
            n=manifest.get('shape',[c['max_'+split]])[0]
            if not n:
                n=p.resource_plan(cfg)['counts'][split]
            additional+=n*c['saved_patches']*c['embed_dim']*4; missing.append([channel,layer,split])
    reused=sum(reusable_binding(c,s,ch,layer) is not None for ch,layer in cache_jobs(cfg))
    return {'output':str(out),'layers':cfg['layers'],'channels':cfg['channels'],'default_stage':'plan',
        'missing_cache_entries':missing,'additional_cache_gib_upper_estimate':additional/2**30,
        'retained_cache_gib':existing/2**30,'retained_plus_additional_gib':(existing+additional)/2**30,
        'cache_budget_gib':cfg['cache_budget_gib'],'within_retained_cache_budget':(existing+additional)/2**30<=cfg['cache_budget_gib'],
        'binding_tasks':len(cache_jobs(cfg)),'reusable_validated_binding_tasks':reused,
        'new_binding_tasks':len(cache_jobs(cfg))-reused,'semantic_tasks_if_requested':len(cache_jobs(cfg)),
        'conditions':'existing validation predictions + geometry + label cache only; no backbone or training',
        'semantic':'explicit --stage semantic; not part of all',
        'all':'extract -> train binding -> report for the selected scope only',
        'limitations':'Conditions are exploratory; no appearance or occlusion labels; no causal intervention in this stage.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(p.HERE/'config-overnight.json'))
    parser.add_argument('--settings',default=str(p.HERE/'config-investigation.json'))
    parser.add_argument('--dataset-root'); parser.add_argument('--investigation-output')
    parser.add_argument('--scope',choices=['depth','steps'],default='depth')
    parser.add_argument('--stage',choices=['plan','conditions','condition-ci','extract','train','semantic','report','all'],default='plan')
    args=parser.parse_args(); c=p.read_config(args.config,args.dataset_root); s=read_settings(args.settings,c)
    root=Path(args.investigation_output or str(Path(c['output'])/'investigation'))
    out=root/('conditions' if args.stage in ('conditions','condition-ci') else args.scope)
    cfg=scope_config(c,s,args.scope,out)
    if args.stage=='plan': print(json.dumps(plan(c,s,cfg,out),indent=2)); return
    # Keep investigation settings and extraction provenance outside the original run files.
    provenance={'base_config':c,'settings':s,'scope':args.scope if out.name!='conditions' else 'conditions'}
    identity_file=out/'requested_settings.json'
    if identity_file.exists() and json.loads(identity_file.read_text())!=provenance:
        raise ValueError('Investigation settings changed; use a new --investigation-output')
    write(identity_file,provenance)
    if args.stage in ('conditions','condition-ci'):
        conditions(c,s,out,ci=args.stage=='condition-ci'); return
    for stage in (['extract','train','report'] if args.stage=='all' else [args.stage]):
        if stage=='extract': p.extract(cfg,p.core_module(cfg))
        elif stage=='train': train_binding(cfg,s,out)
        elif stage=='semantic': train_semantic(cfg,s,out)
        elif stage=='report': report(cfg,s,out)


if __name__=='__main__': main()

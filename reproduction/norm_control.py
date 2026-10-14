"""Match post-block token L2 norms to post-attention token norms using caches.

One new binding/semantic condition; both raw endpoints reference old results.
No backbone, full feature-cache writes, or changes to the alpha experiment.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, math
import phenomenon as p
import investigate_decline as inv
import mlp_update as mlp
import semantic_uncertainty as uncertainty

CHANNEL='norm_matched_residual'


def read_settings(path,c):
    s=json.loads(Path(path).read_text(encoding='utf-8'))
    if type(s['layer']) is not int or not 0<=s['layer']<c['num_layers']: raise ValueError('Invalid norm-control block index')
    if type(s['epsilon']) not in (int,float) or not math.isfinite(s['epsilon']) or s['epsilon']<=0:
        raise ValueError('epsilon must be finite and positive')
    for name in ('max_new_binding_tasks','max_new_semantic_tasks','semantic_bootstrap_repeats'):
        if type(s[name]) is not int or s[name]<1: raise ValueError(f'Invalid {name}')
    if s['semantic_bootstrap_repeats']<2: raise ValueError('At least two semantic bootstrap replicates required')
    return s


def norm_match(z,h,epsilon):
    """Positive scalar per token preserves h direction; no centering/LayerNorm."""
    import numpy as np
    if z.shape!=h.shape or z.dtype!=h.dtype or z.ndim<1 or not np.issubdtype(h.dtype,np.floating):
        raise ValueError('Norm-control inputs must have matching floating shapes/dtypes')
    zd,hd=z.astype(np.float64),h.astype(np.float64)
    nz=np.linalg.norm(zd,axis=-1,keepdims=True); nh=np.linalg.norm(hd,axis=-1,keepdims=True)
    if not np.isfinite(nz).all() or not np.isfinite(nh).all(): raise ValueError('Non-finite norm-control inputs')
    if ((nh<=epsilon)&(nz>epsilon)).any():
        raise ValueError('Near-zero h cannot preserve direction while matching a nonzero z norm')
    result=(hd*(nz/np.maximum(nh,epsilon))).astype(h.dtype)
    if not np.isfinite(result).all(): raise ValueError('Non-finite norm-matched features')
    return result


class NormMatchedFeatures:
    def __init__(self,z,h,epsilon):
        if z.shape!=h.shape or z.dtype!=h.dtype: raise ValueError('Norm-control feature shapes/dtypes differ')
        self.z,self.h,self.epsilon=z,h,epsilon; self.shape,self.dtype=z.shape,z.dtype

    def __len__(self): return len(self.z)

    def __getitem__(self,index):
        if isinstance(index,tuple) and len(index)>2: raise ValueError('Norm matching requires the full feature dimension')
        return norm_match(self.z[index],self.h[index],self.epsilon)


class NormProvider(mlp.BlendProvider):
    def __init__(self,c,settings):
        super().__init__(c,dict(layer=settings['layer'],alphas=[]))
        self.channels={CHANNEL:None}; self.epsilon=settings['epsilon']

    def cached(self,c,split,layer):
        if layer!=self.layer or c.get('channel')!=CHANNEL: raise ValueError('Unexpected norm-control cache request')
        z,h=self.banks[split]
        return NormMatchedFeatures(z,h,self.epsilon),self.metadata[split]


def virtual_config(c,settings):
    return dict(c,layers=[settings['layer']],channels=[CHANNEL],probes=['quadratic'])


def train(c,training,settings,root,out,kind):
    mlp.check_endpoints(c,training,settings,root)
    provider=NormProvider(c,settings)
    try:
        original=p.core_module(c)
        core=SimpleNamespace(cached=provider.cached,probe=original.probe,seed_all=original.seed_all,
            batch_logits=original.batch_logits,eval_probe=original.eval_probe)
        cfg=virtual_config(c,settings)
        if kind=='binding': inv.train_binding(cfg,training,out,core=core,cache_validator=provider.validate)
        else:
            semantic_settings=dict(training,max_semantic_tasks=settings['max_new_semantic_tasks'])
            inv.train_semantic(cfg,semantic_settings,out,core=core,cache_validator=provider.validate)
    finally: provider.close()


def summary(values):
    import numpy as np
    values=np.asarray(values)
    return {'n':int(values.size),'mean':float(values.mean()) if values.size else None,
        'quantile_probabilities':[0,.01,.1,.5,.9,.99,1],
        'quantiles':np.quantile(values,[0,.01,.1,.5,.9,.99,1]).tolist() if values.size else None}


def audit(c,settings,out):
    """Use the fixed test eval-patch sample; do not scan all training features."""
    import numpy as np
    provider=NormProvider(c,settings)
    collected={name:[] for name in ('z_norm','h_norm','matched_norm','h_over_z','relative_norm_error','h_cosine_matched','known_foreground')}
    max_absolute_error=0.; near_zero_z=0; near_zero_h=0
    try:
        zbank,hbank=provider.banks['test']; meta=provider.metadata['test']
        for row,image_id in enumerate(meta['image_id']):
            picks=np.random.default_rng(100000+c['split_seed']+int(image_id)).choice(c['saved_patches'],c['eval_patches'],replace=False)
            z,h=np.asarray(zbank[row,picks]),np.asarray(hbank[row,picks]); matched=norm_match(z,h,settings['epsilon'])
            nz,nh,nm=[np.linalg.norm(a.astype(np.float64),axis=-1) for a in (z,h,matched)]
            difference=np.abs(nm-nz); max_absolute_error=max(max_absolute_error,float(difference.max()))
            near_zero_z+=int((nz<=settings['epsilon']).sum()); near_zero_h+=int((nh<=settings['epsilon']).sum())
            ratio=np.divide(nh,nz,out=np.full_like(nh,np.nan),where=nz>settings['epsilon'])
            cosine=np.divide((h.astype(np.float64)*matched).sum(-1),nh*nm,out=np.full_like(nh,np.nan),where=(nh*nm)>0)
            values={'z_norm':nz,'h_norm':nh,'matched_norm':nm,'h_over_z':ratio,
                'relative_norm_error':difference/np.maximum(nz,settings['epsilon']), 'h_cosine_matched':cosine,
                'known_foreground':(~meta['background'][row,picks].astype(bool))&(meta['semantic'][row,picks]>0)}
            for name,value in values.items(): collected[name].append(value)
        arrays={key:np.concatenate(value) for key,value in collected.items()}
        groups={}
        for name,mask in [('all',np.ones(len(arrays['z_norm']),bool)),('known_foreground',arrays['known_foreground'])]:
            groups[name]={key:summary(value[mask&np.isfinite(value)]) for key,value in arrays.items() if key!='known_foreground'}
        if not np.allclose(arrays['matched_norm'],arrays['z_norm'],rtol=2e-6,atol=2*settings['epsilon']):
            raise ValueError('Norm-matched output failed L2 norm tolerance')
        inv.write(out/'norm_audit.json',{'layer':settings['layer'],'scope':'fixed test eval patches, same sampler as both probes',
            'test_images':len(meta['image_id']),'eval_patches_per_image':c['eval_patches'],'groups':groups,
            'max_absolute_norm_error':max_absolute_error,'near_zero_z_tokens':near_zero_z,'near_zero_h_tokens':near_zero_h,
            'norm_tolerance':{'rtol':2e-6,'atol':2*settings['epsilon']},
            'formula':'h * ||z||_2 / max(||h||_2, epsilon), independently per patch',
            'precision':'L2 norm and scaling in float64, returned features float32',
            'note':'All and known-foreground summaries use the same fixed test sample; neither tuning nor evidence of downstream causal use.'})
    finally: provider.close()


def result_sources(c,training,settings,root,out):
    first=mlp.endpoint_folders(c,training,root,settings['layer'],0)
    last=mlp.endpoint_folders(c,training,root,settings['layer'],1)
    matched=(out/'binding'/CHANNEL/f"layer_{settings['layer']}",out/'semantic'/CHANNEL/f"layer_{settings['layer']}")
    return {'post_attention':first,'residual':last,CHANNEL:matched}


def report(c,training,settings,root,out):
    import validate_decline as v
    sources=result_sources(c,training,settings,root,out); rows=[]; comparisons=[]
    for condition,(binding,semantic) in sources.items():
        row=dict(condition=condition,layer=settings['layer'],endpoint_reused=condition!=CHANNEL,
            binding_source=str(binding),semantic_source=str(semantic),binding_complete=False,semantic_complete=False)
        if (binding/'metrics.json').exists():
            mlp.require_results(binding,'binding'); data=json.loads((binding/'metrics.json').read_text())
            row.update(binding_complete=True,hard_instance_auroc=data['instance_vs_same_class']['auroc'],binding_category_auroc=data['class_grouping_cross_instance']['auroc'])
        if (semantic/'metrics.json').exists():
            mlp.require_results(semantic,'semantic'); mlp.validate_semantic_cohort(c,semantic,training['semantic_num_classes'])
            data=json.loads((semantic/'metrics.json').read_text())
            row.update(semantic_complete=True,semantic_accuracy=data['accuracy'],semantic_macro_recall=data['macro_recall'])
        rows.append(row)
    byname={row['condition']:row for row in rows}
    for first,second in [('post_attention','residual'),('residual',CHANNEL),('post_attention',CHANNEL)]:
        a,b=byname[first],byname[second]; comparison=dict(first=first,second=second)
        if a['binding_complete'] and b['binding_complete']:
            predictions=[v.predictions(sources[name][0]/'test_predictions.npz') for name in (first,second)]
            meta,_=inv.load_meta(c,'test','residual')
            for prediction in predictions: inv.check_prediction_meta(prediction,meta)
            comparison['binding']=v.paired_bootstrap(*predictions,'instance',training['bootstrap_repeats'],training['bootstrap_bins'],
                training['bootstrap_seed'],training['bootstrap_max_auc_error'])
        if a['semantic_complete'] and b['semantic_complete']:
            comparison['semantic']=uncertainty.compare_folders(sources[first][1],sources[second][1],
                settings['semantic_bootstrap_repeats'],settings['semantic_bootstrap_seed'])
        comparisons.append(comparison)
    inv.write(out/'report.json',{'layer':settings['layer'],'formula':'h * ||z||_2 / max(||h||_2, epsilon), independently per patch',
        'measurements':rows,'paired_comparisons':comparisons,'expected_conditions':3,
        'binding_complete':sum(row['binding_complete'] for row in rows),'semantic_complete':sum(row['semantic_complete'] for row in rows),
        'new_feature_cache_gib':0,'backbone_forward_images':0,'norm_audit_available':(out/'norm_audit.json').exists(),
        'interpretation':'Replacing h norms with z norms may also transfer scalar information from z; recovery does not alone prove a pure optimization artifact or a unique cause.',
        'note':'Positive per-token scaling preserves h directions and approximately restores z L2 norms (checked in audit); both probes refit. This is not LayerNorm or a downstream model intervention.'})
    mlp.save_csv(out/'measurements.csv',rows); plot(rows,out)


def plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    labels=['After attention','After MLP','MLP norm matched']
    for ax,(key,title) in zip(axes,[('hard_instance_auroc','Instance binding AUROC'),('semantic_accuracy','Semantic accuracy'),('semantic_macro_recall','Semantic macro recall')]):
        ax.plot(range(3),[row.get(key,float('nan')) for row in rows],marker='o')
        ax.set_xticks(range(3),labels); ax.set(title=title,ylim=(0,1)); ax.grid(alpha=.2)
    fig.savefig(out/'measurements.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(p.HERE/'config-overnight.json'))
    parser.add_argument('--settings',default=str(p.HERE/'config-norm-control.json'))
    parser.add_argument('--dataset-root'); parser.add_argument('--source-investigation'); parser.add_argument('--output')
    parser.add_argument('--stage',choices=['plan','audit','binding','semantic','report','all'],default='plan')
    args=parser.parse_args(); c=p.read_config(args.config,args.dataset_root); settings=read_settings(args.settings,c)
    root=Path(args.source_investigation or str(Path(c['output'])/'investigation')).resolve()
    out=Path(args.output or str(root/'norm_control')).resolve()
    if out in (Path(c['output']).resolve(),root,root/'steps',root/'depth',root/'mlp_update'):
        raise ValueError('Use a separate norm-control output directory')
    if args.stage=='plan':
        pending={kind:not (out/kind/CHANNEL/f"layer_{settings['layer']}"/'metrics.json').exists() for kind in ('binding','semantic')}
        print(json.dumps({'layer':settings['layer'],'output':str(out),'epsilon':settings['epsilon'],
            'formula':'h * ||z||_2 / max(||h||_2, epsilon), independently per patch',
            'new_binding_tasks':1,'new_semantic_tasks':1,'pending_binding_tasks':int(pending['binding']),
            'pending_semantic_tasks':int(pending['semantic']),'raw_endpoints_reused':True,
            'new_feature_cache_gib':0,'backbone_forward_images':0,
            'audit_scope':'fixed test eval patches only; no training or source annotation reads',
            'missing_source_cache_or_settings':mlp.source_inventory(c,settings,root,'steps'),
            'all':'audit -> norm-matched binding -> norm-matched semantic -> report'},indent=2)); return
    training=mlp.source_settings(c,root,'steps')
    if settings['layer'] not in training['step_layers']: raise ValueError('Selected block absent from completed steps scope')
    mlp.check_endpoints(c,training,settings,root)
    requested=mlp.provenance(c,training,settings,root)
    requested['control_settings']=requested.pop('update_settings')
    file=out/'requested_settings.json'
    if file.exists() and json.loads(file.read_text())!=requested: raise ValueError('Norm-control source/settings changed; choose a new --output')
    code={'code_commit':p.git_commit(),'files_sha256':{name:hashlib.sha256((p.HERE/name).read_bytes()).hexdigest() for name in
        ('norm_control.py','mlp_update.py','semantic_uncertainty.py','investigate_decline.py','run_server.py')}}
    code_file=out/'code_provenance.json'
    if code_file.exists():
        recorded=json.loads(code_file.read_text())
        if args.stage in ('binding','semantic','all') and recorded['files_sha256']!=code['files_sha256']:
            raise ValueError('Training implementation changed; choose a new --output')
    inv.write(file,requested)
    if not code_file.exists(): inv.write(code_file,code)
    if args.stage in ('report','all'): inv.write(out/'report_code_provenance.json',code)
    for stage in (['audit','binding','semantic','report'] if args.stage=='all' else [args.stage]):
        if stage=='audit': audit(c,settings,out)
        elif stage in ('binding','semantic'): train(c,training,settings,root,out,stage)
        else: report(c,training,settings,root,out)


if __name__=='__main__': main()

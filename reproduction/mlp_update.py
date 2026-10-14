"""Cached MLP-update interpolation and paired semantic uncertainty.

No backbone forward or full interpolated feature cache. Default plan is read-only.
Endpoints reference completed steps/validation results; only interior alphas train.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, hashlib, json, math
import phenomenon as p
import investigate_decline as inv
import semantic_uncertainty as uncertainty

BASE_FIELDS=('model','model_revision','family','embed_dim','crop_size','saved_patches','sampled_patches',
    'eval_patches','split_seed','dataset_name','dataset_factory','max_train','max_val','max_test',
    'epochs','batch_size','learning_rate','scheduler_step','scheduler_gamma')


def read_settings(path,c):
    s=json.loads(Path(path).read_text(encoding='utf-8'))
    layer=s['layer']; alphas=s['alphas']
    if type(layer) is not int or not 0<=layer<c['num_layers']: raise ValueError('Invalid MLP block index')
    if not isinstance(alphas,list) or any(type(a) not in (float,int) or not math.isfinite(a) or not 0<=a<=1 for a in alphas):
        raise ValueError('Alphas must be finite real numbers in [0,1]')
    if alphas!=sorted(set(alphas)) or not alphas or alphas[0]!=0 or alphas[-1]!=1:
        raise ValueError('Use distinct sorted alphas including 0 and 1')
    if len(set(alpha_channel(a) for a in alphas))!=len(alphas): raise ValueError('Alpha output names collide')
    for key in ('max_new_binding_tasks','max_new_semantic_tasks','semantic_bootstrap_repeats'):
        if type(s[key]) is not int or s[key]<1: raise ValueError(f'Invalid {key}')
    if s['semantic_bootstrap_repeats']<2: raise ValueError('At least two semantic bootstrap replicates required')
    interior=sum(0<a<1 for a in alphas)
    if interior>min(s['max_new_binding_tasks'],s['max_new_semantic_tasks']): raise ValueError('Interior-alpha task budget exceeded')
    return s


def alpha_channel(alpha):
    return 'mlp_alpha_'+format(alpha,'.12g').replace('.','p')


def source_settings(c,root,scope):
    file=root/scope/'requested_settings.json'
    provenance=json.loads(file.read_text(encoding='utf-8'))
    recorded=provenance['base_config']
    if any(recorded.get(key)!=c.get(key) for key in BASE_FIELDS):
        raise ValueError(f'Source {scope} training/data config differs; pass the actual completed server config')
    for key in ('output','dataset_root'):
        if Path(recorded[key]).resolve()!=Path(c[key]).resolve(): raise ValueError(f'Source {scope} {key} differs')
    if provenance.get('scope')!=scope: raise ValueError('Source investigation scope differs')
    return provenance['settings']


def endpoint_folders(c,training,root,layer,alpha):
    if alpha==0:
        binding=root/'steps/binding/post_attention'/f'layer_{layer}'
        channel='post_attention'
    elif alpha==1:
        binding=inv.reusable_binding(c,training,'residual',layer) or root/'steps/binding/residual'/f'layer_{layer}'
        channel='residual'
    else: raise ValueError('Endpoint alpha must be 0 or 1')
    semantic=root/'steps/semantic'/channel/f'layer_{layer}'
    return binding,semantic


def require_results(folder,kind):
    names=['metrics.json','per_image.json','selection.json']
    if kind=='binding': names+=['test_predictions.npz']
    missing=[str(folder/name) for name in names if not (folder/name).exists()]
    if missing: raise ValueError('Missing completed source results: '+', '.join(missing))


def validate_semantic_cohort(c,folder,classes):
    """Reconstruct exact label support using the established evaluation sampler."""
    import numpy as np
    records=json.loads((folder/'per_image.json').read_text()); ids,support,_=uncertainty.record_arrays(records)
    meta,_=inv.load_meta(c,'test','residual')
    if not np.array_equal(ids,meta['image_id']) or support.shape[1]!=classes:
        raise ValueError('Semantic source image order or taxonomy differs from cache')
    for row,image_id in enumerate(ids):
        picks=np.random.default_rng(100000+c['split_seed']+int(image_id)).choice(c['saved_patches'],c['eval_patches'],replace=False)
        target=meta['semantic'][row,picks].astype(np.int64)-1
        valid=(~meta['background'][row,picks].astype(bool))&(target>=0)
        if (target[valid]>=classes).any(): raise ValueError('Semantic cache label exceeds taxonomy')
        expected=np.bincount(target[valid],minlength=classes)
        if not np.array_equal(support[row],expected): raise ValueError('Semantic source class support differs from current evaluation patches')
        if records[row].get('evaluation_fingerprint'):
            sample=[int(image_id),picks.tolist(),meta['position'][row,picks].tolist(),target.tolist(),valid.tolist()]
            fingerprint=hashlib.sha256(json.dumps(sample,separators=(',',':')).encode()).hexdigest()
            if fingerprint!=records[row]['evaluation_fingerprint']: raise ValueError('Semantic source sample fingerprint differs from cache')


class BlendedFeatures:
    """Only materialize the indexed rows/patches, never the entire feature bank."""
    def __init__(self,z,h,alpha):
        if z.shape!=h.shape or z.dtype!=h.dtype: raise ValueError('Interpolation feature shapes/dtypes differ')
        self.z,self.h,self.alpha=z,h,float(alpha)
        self.shape,self.dtype=z.shape,z.dtype

    def __len__(self): return len(self.z)

    def __getitem__(self,index):
        if self.alpha==0: return self.z[index]
        if self.alpha==1: return self.h[index]
        z,h=self.z[index],self.h[index]
        return z+self.alpha*(h-z)


class BlendProvider:
    def __init__(self,c,settings):
        import numpy as np
        self.c,self.layer=c,settings['layer']; self.banks={}; self.metadata={}
        self.channels={alpha_channel(a):a for a in settings['alphas'] if 0<a<1}
        cfg=dict(c,layers=[self.layer],channels=['post_attention','residual'])
        inv.validate_caches(cfg)
        for split in ('train','val','test'):
            matrices=[]
            for channel in cfg['channels']:
                meta,_=inv.load_meta(c,split,channel)
                folder=Path(p.channel_config(c,channel)['output'])/'cache'/split
                matrices.append(np.load(folder/f'layer_{self.layer}.npy',mmap_mode='r'))
                if split in self.metadata and any(not np.array_equal(meta[k],self.metadata[split][k]) for k in meta):
                    raise ValueError('MLP endpoint cache patches/labels are not aligned')
                self.metadata[split]=meta
            self.banks[split]=matrices

    def validate(self,c):
        if c['layers']!=[self.layer] or set(c['channels'])-set(self.channels): raise ValueError('Unexpected interpolation task')
        if any(c.get(k)!=self.c.get(k) for k in BASE_FIELDS): raise ValueError('Interpolation training config changed')

    def cached(self,c,split,layer):
        if layer!=self.layer or c.get('channel') not in self.channels: raise ValueError('Unexpected interpolation cache request')
        z,h=self.banks[split]
        return BlendedFeatures(z,h,self.channels[c['channel']]),self.metadata[split]

    def close(self):
        for matrices in self.banks.values():
            for matrix in matrices:
                mapping=getattr(matrix,'_mmap',None)
                if mapping is not None: mapping.close()
        self.banks.clear()


def virtual_config(c,settings):
    return dict(c,layers=[settings['layer']],channels=[alpha_channel(a) for a in settings['alphas'] if 0<a<1],
        probes=['quadratic'])


def check_endpoints(c,training,settings,root):
    import validate_decline as v
    for alpha in (0,1):
        binding,semantic=endpoint_folders(c,training,root,settings['layer'],alpha)
        require_results(binding,'binding'); require_results(semantic,'semantic')
        channel='post_attention' if alpha==0 else 'residual'
        meta,_=inv.load_meta(c,'test',channel)
        inv.check_prediction_meta(v.predictions(binding/'test_predictions.npz'),meta)
        validate_semantic_cohort(c,semantic,training['semantic_num_classes'])


def train(c,training,settings,root,out,kind):
    check_endpoints(c,training,settings,root)
    cfg=virtual_config(c,settings)
    if not cfg['channels']: return
    provider=BlendProvider(c,settings)
    try:
        original=p.core_module(c)
        core=SimpleNamespace(cached=provider.cached,probe=original.probe,seed_all=original.seed_all,
            batch_logits=original.batch_logits,eval_probe=original.eval_probe)
        if kind=='binding': inv.train_binding(cfg,training,out,core=core,cache_validator=provider.validate)
        else:
            cfg_training=dict(training,max_semantic_tasks=settings['max_new_semantic_tasks'])
            inv.train_semantic(cfg,cfg_training,out,core=core,cache_validator=provider.validate)
    finally: provider.close()


def semantic_ci(c,settings,root,out,scope):
    results=[]
    scopes=['depth','steps'] if scope=='both' else [scope]
    for name in scopes:
        training=source_settings(c,root,name)
        if name=='depth':
            locations=[('residual',layer) for layer in training['depth_layers']]
        else:
            locations=[(channel,settings['layer']) for channel in ('block_input','post_attention','residual')]
            if settings['layer'] not in training['step_layers']: raise ValueError('Selected block absent from source steps settings')
        folders=[root/name/'semantic'/channel/f'layer_{layer}' for channel,layer in locations]
        for folder in folders:
            require_results(folder,'semantic'); validate_semantic_cohort(c,folder,training['semantic_num_classes'])
        for i in range(len(folders)-1):
            result=uncertainty.compare_folders(folders[i],folders[i+1],settings['semantic_bootstrap_repeats'],settings['semantic_bootstrap_seed'])
            results.append(dict(scope=name,first=list(locations[i]),second=list(locations[i+1]),**result))
        if name=='depth' and len(folders)>2:
            result=uncertainty.compare_folders(folders[0],folders[-1],settings['semantic_bootstrap_repeats'],settings['semantic_bootstrap_seed'])
            results.append(dict(scope=name,first=list(locations[0]),second=list(locations[-1]),**result))
    inv.write(out/f'semantic_ci_{scope}.json',{'comparisons':results,'no_training':True,'no_backbone_forward':True})
    rows=[]
    for r in results:
        for metric,m in r['metrics'].items():
            rows.append(dict(scope=r['scope'],first=str(r['first']),second=str(r['second']),metric=metric,
                first_exact=m['first_exact'],second_exact=m['second_exact'],delta=m['delta_exact'],
                delta_ci95_low=m['delta_ci95'][0],delta_ci95_high=m['delta_ci95'][1]))
    save_csv(out/f'semantic_ci_{scope}.csv',rows)


def save_csv(file,rows):
    if not rows: return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with file.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def report(c,training,settings,root,out):
    import validate_decline as v
    rows=[]; pairs=[]; sources={}
    for alpha in settings['alphas']:
        if alpha in (0,1): binding,semantic=endpoint_folders(c,training,root,settings['layer'],alpha)
        else:
            channel=alpha_channel(alpha); binding=out/'binding'/channel/f"layer_{settings['layer']}"
            semantic=out/'semantic'/channel/f"layer_{settings['layer']}"
        row=dict(alpha=alpha,layer=settings['layer'],binding_source=str(binding),semantic_source=str(semantic),
            endpoint_reused=alpha in (0,1),binding_complete=False,semantic_complete=False)
        if (binding/'metrics.json').exists():
            require_results(binding,'binding'); m=json.loads((binding/'metrics.json').read_text())
            row.update(binding_complete=True,hard_instance_auroc=m['instance_vs_same_class']['auroc'],binding_category_auroc=m['class_grouping_cross_instance']['auroc'])
        if (semantic/'metrics.json').exists():
            require_results(semantic,'semantic'); validate_semantic_cohort(c,semantic,training['semantic_num_classes'])
            m=json.loads((semantic/'metrics.json').read_text()); row.update(semantic_complete=True,semantic_accuracy=m['accuracy'],semantic_macro_recall=m['macro_recall'])
        sources[alpha]=(binding,semantic); rows.append(row)
    comparison_alphas=list(zip(settings['alphas'],settings['alphas'][1:]))
    if len(settings['alphas'])>2: comparison_alphas.append((0.,1.))
    byalpha={r['alpha']:r for r in rows}
    for first,second in comparison_alphas:
        a,b=byalpha[first],byalpha[second]; entry=dict(first_alpha=first,second_alpha=second)
        if a['binding_complete'] and b['binding_complete']:
            predictions=[v.predictions(sources[alpha][0]/'test_predictions.npz') for alpha in (first,second)]
            meta,_=inv.load_meta(c,'test','residual')
            for prediction in predictions: inv.check_prediction_meta(prediction,meta)
            entry['binding']=v.paired_bootstrap(*predictions,'instance',training['bootstrap_repeats'],training['bootstrap_bins'],
                training['bootstrap_seed'],training['bootstrap_max_auc_error'])
        if a['semantic_complete'] and b['semantic_complete']:
            entry['semantic']=uncertainty.compare_folders(sources[first][1],sources[second][1],
                settings['semantic_bootstrap_repeats'],settings['semantic_bootstrap_seed'])
        pairs.append(entry)
    inv.write(out/'report.json',{'layer':settings['layer'],'formula':'z + alpha * (h - z); z=post_attention, h=post_block',
        'measurements':rows,'paired_comparisons':pairs,
        'binding_complete':sum(r['binding_complete'] for r in rows),'semantic_complete':sum(r['semantic_complete'] for r in rows),
        'expected_points':len(rows),'new_feature_cache_gib':0,'backbone_forward_images':0,
        'note':'Each interior point refits both probes with source training/validation rules; measures cached decodability, not causal downstream model behavior.'})
    save_csv(out/'measurements.csv',rows)
    plot(rows,out)


def plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,(key,label) in zip(axes,[('hard_instance_auroc','Instance binding AUROC'),('semantic_accuracy','Semantic accuracy'),('semantic_macro_recall','Semantic macro recall')]):
        selected=[r for r in rows if r.get(key) is not None]
        ax.plot([r['alpha'] for r in selected],[r[key] for r in selected],marker='o')
        ax.set(title=label,xlabel='MLP residual update strength alpha',xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.2)
    fig.savefig(out/'measurements.png',dpi=160); plt.close(fig)


def source_inventory(c,settings,root,semantic_scope):
    required=[root/'steps/requested_settings.json']
    for channel in ('post_attention','residual'):
        for split in ('train','val','test'):
            folder=Path(p.channel_config(c,channel)['output'])/'cache'/split
            required.extend([folder/'manifest.json',folder/'labels.npz',folder/f"layer_{settings['layer']}.npy"])
    if semantic_scope in ('depth','both'): required.append(root/'depth/requested_settings.json')
    if (root/'steps/requested_settings.json').exists():
        training=source_settings(c,root,'steps')
        for alpha in (0,1):
            binding,semantic=endpoint_folders(c,training,root,settings['layer'],alpha)
            for folder in (binding,semantic): required.extend(folder/name for name in ('metrics.json','per_image.json','selection.json'))
            required.append(binding/'test_predictions.npz')
    return [str(path) for path in required if not path.exists()]


def provenance(c,training,settings,root):
    files=[root/'steps/requested_settings.json']
    for alpha in (0,1):
        binding,semantic=endpoint_folders(c,training,root,settings['layer'],alpha)
        for folder in (binding,semantic):
            files.extend(folder/name for name in ('metrics.json','selection.json','per_image.json'))
            files.extend(folder/name for name in ('checkpoint.pt','history.json') if (folder/name).exists())
        files.append(binding/'test_predictions.npz')
    for channel in ('post_attention','residual'):
        for split in ('train','val','test'):
            folder=Path(p.channel_config(c,channel)['output'])/'cache'/split
            files.extend([folder/'manifest.json',folder/'labels.npz'])
    digests={str(file):hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    features=[]
    for channel in ('post_attention','residual'):
        for split in ('train','val','test'):
            file=Path(p.channel_config(c,channel)['output'])/'cache'/split/f"layer_{settings['layer']}.npy"
            stat=file.stat(); features.append(dict(path=str(file),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return {'base_config':c,'update_settings':settings,'source_investigation':str(root),'training_settings':training,
        'source_metadata_sha256':digests,'feature_files':features,
        'feature_integrity_note':'Feature files checked by signature/header/alignment and recorded size/mtime, not full multi-GiB content hashes.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(p.HERE/'config-overnight.json'))
    parser.add_argument('--settings',default=str(p.HERE/'config-mlp-update.json'))
    parser.add_argument('--dataset-root'); parser.add_argument('--source-investigation'); parser.add_argument('--output')
    parser.add_argument('--semantic-scope',choices=['depth','steps','both'],default='steps')
    parser.add_argument('--stage',choices=['plan','semantic-ci','binding','semantic','report','all'],default='plan')
    args=parser.parse_args(); c=p.read_config(args.config,args.dataset_root); settings=read_settings(args.settings,c)
    root=Path(args.source_investigation or str(Path(c['output'])/'investigation')).resolve()
    out=Path(args.output or str(root/'mlp_update')).resolve()
    if out==Path(c['output']).resolve() or out in (root,root/'steps',root/'depth'):
        raise ValueError('Use a separate MLP-update output directory')
    if args.stage=='plan':
        interior=sum(0<a<1 for a in settings['alphas'])
        pending={kind:sum(not (out/kind/alpha_channel(a)/f"layer_{settings['layer']}"/'metrics.json').exists()
            for a in settings['alphas'] if 0<a<1) for kind in ('binding','semantic')}
        print(json.dumps({'layer':settings['layer'],'alphas':settings['alphas'],'output':str(out),
            'new_binding_tasks':interior,'new_semantic_tasks':interior,'endpoints_reused':True,
            'pending_binding_tasks':pending['binding'],'pending_semantic_tasks':pending['semantic'],
            'new_feature_cache_gib':0,'backbone_forward_images':0,'semantic_ci_scope':args.semantic_scope,
            'missing_source_cache_or_settings':source_inventory(c,settings,root,args.semantic_scope),
            'training_rules':'Inherited from completed steps/requested_settings.json, validation-only selection',
            'all':'semantic-ci -> interior binding -> interior semantic -> report; no extraction'},indent=2)); return
    training=source_settings(c,root,'steps')
    if settings['layer'] not in training['step_layers']: raise ValueError('Selected block absent from completed steps scope')
    if args.stage=='semantic-ci':
        # A separate read-only source analysis; no mutation of source reports/provenance.
        out.mkdir(parents=True,exist_ok=True)
        semantic_ci(c,settings,root,out,args.semantic_scope); return
    check_endpoints(c,training,settings,root)
    requested=provenance(c,training,settings,root); file=out/'requested_settings.json'
    if file.exists() and json.loads(file.read_text())!=requested: raise ValueError('MLP-update source/settings changed; choose a new --output')
    inv.write(file,requested)
    code={'code_commit':p.git_commit(),'files_sha256':{name:hashlib.sha256((p.HERE/name).read_bytes()).hexdigest() for name in
        ('mlp_update.py','semantic_uncertainty.py','investigate_decline.py','run_server.py')}}
    code_file=out/'code_provenance.json'
    if code_file.exists():
        recorded=json.loads(code_file.read_text())
        if args.stage in ('binding','semantic','all') and recorded['files_sha256']!=code['files_sha256']:
            raise ValueError('Training implementation changed; use a new --output instead of mixing completed probes')
    else: inv.write(code_file,code)
    if args.stage in ('report','all'): inv.write(out/'report_code_provenance.json',code)
    for stage in (['semantic-ci','binding','semantic','report'] if args.stage=='all' else [args.stage]):
        if stage=='semantic-ci': semantic_ci(c,settings,root,out,args.semantic_scope)
        elif stage in ('binding','semantic'): train(c,training,settings,root,out,stage)
        else: report(c,training,settings,root,out)


if __name__=='__main__': main()

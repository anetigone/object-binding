"""Resource-bounded phenomenon screening. --stage plan needs no ML dependencies."""
from pathlib import Path
import argparse, copy, hashlib, importlib, json, math, shutil, subprocess

HERE = Path(__file__).resolve().parent
MODELS = {
    'facebook/dinov2-small': {'family':'dinov2','embed_dim':384,'num_layers':12,'num_heads':6},
    'facebook/dinov2-base': {'family':'dinov2','embed_dim':768,'num_layers':12,'num_heads':12},
    'facebook/dinov2-large': {'family':'dinov2','embed_dim':1024,'num_layers':24,'num_heads':16},
}
CHANNELS = ('residual','q','k','v')

def read_config(path, dataset_root=None, output=None):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    spec = MODELS.get(c['model'])
    if spec is None:
        raise ValueError('Model adapter not implemented. Register a model and its extraction adapter first.')
    c.update(spec)
    if dataset_root: c['dataset_root'] = dataset_root
    if output: c['output'] = output
    c['output'] = str(Path(c['output']).resolve())
    c.setdefault('dataset_factory',None)
    c.setdefault('dataset_name','ade20k')
    c.setdefault('channels',['residual'])
    c.setdefault('cache_budget_gib',24)
    c.setdefault('max_tasks',5)
    c.setdefault('model_revision',None)
    c.setdefault('crop_size',518)
    if 'layers' not in c:
        c['layers'] = sorted(set(round(t*(spec['num_layers']-1)) for t in c.pop('depth_fractions',[0,.25,.5,.75,1])))
    if not c['layers'] or len(set(c['layers']))!=len(c['layers']) or any(not isinstance(l,int) or l<0 or l>=spec['num_layers'] for l in c['layers']):
        raise ValueError('Invalid block indices for this model')
    if not c['channels'] or len(set(c['channels']))!=len(c['channels']) or set(c['channels'])-set(CHANNELS):
        raise ValueError('Invalid channels')
    if c['crop_size']!=518:
        raise ValueError('Current DINOv2 adapter preserves the existing 518/14 preprocessing.')
    if not 2<=c['eval_patches']<=c['saved_patches']<=1369 or not 0<c['sampled_patches']<=c['saved_patches']:
        raise ValueError('Invalid patch counts')
    if not c['seeds'] or not c['probes'] or set(c['probes'])-{'quadratic','linear','diag_quadratic'}:
        raise ValueError('Invalid seeds/probes')
    return c

def channel_config(c, channel):
    cfg = copy.deepcopy(c)
    cfg['channel'] = channel
    cfg['output'] = str(Path(c['output'])/'channels'/channel)
    return cfg

def identity(c, splits, source_hash):
    fields = ('model','model_revision','family','embed_dim','crop_size','saved_patches','split_seed','dataset_name','dataset_factory')
    payload = {k:c[k] for k in fields}
    payload.update(dataset_root=str(Path(c['dataset_root']).resolve()),splits=splits,preprocessor_sha256=source_hash,cache_schema=2)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def resource_plan(c, counts=None):
    if counts is None:
        # No dataset access/download in the plan stage. Limits bound actual split sizes.
        if any(c['max_'+s]==0 for s in ('train','val','test')):
            if c['dataset_name']!='ade20k':
                raise ValueError('Unlimited custom dataset requires an actual count before extraction.')
            totals = {'train':20448,'val':5113,'test':1999}
            counts = {s:c['max_'+s] or totals[s] for s in totals}
        else:
            counts = {s:c['max_'+s] for s in ('train','val','test')}
    if any(n<=0 for n in counts.values()):
        raise ValueError('Every split must be non-empty')
    gib = sum(counts.values())*len(c['layers'])*len(c['channels'])*c['saved_patches']*c['embed_dim']*4/2**30
    tasks = len(c['layers'])*len(c['channels'])*len(c['probes'])*len(c['seeds'])
    return {'dataset':c['dataset_name'],'model':c['model'],'counts':counts,'counts_are_estimates':True,
        'layers':c['layers'],'channels':c['channels'],'fp32_cache_gib':gib,'training_tasks':tasks,
        'optimizer_steps':tasks*c['epochs']*math.ceil(counts['train']/c['batch_size']),
        'one_backbone_pass_per_crop':True,'budget_gib':c['cache_budget_gib'],'task_limit':c['max_tasks'],
        'within_budget':gib<=c['cache_budget_gib'] and tasks<=c['max_tasks'],
        'note':'Training costs scale by channels/seeds/probes; plan does not estimate wall time.'}

def enforce_budget(c, counts=None):
    plan = resource_plan(c,counts)
    if not plan['within_budget']:
        raise ValueError(f"Plan exceeds configured budget: {plan['fp32_cache_gib']:.2f} GiB, {plan['training_tasks']} tasks. Reduce scope or explicitly edit the budget.")
    return plan

def core_module(c):
    import run_server as core
    original_dataset = core.Dataset
    if c['dataset_factory']:
        module, name = c['dataset_factory'].split(':',1)
        factory = getattr(importlib.import_module(module),name)
    else:
        factory = original_dataset
    core.Dataset = factory
    source = core.UPSTREAM/'src/data_extractor.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    core.cache_signature = lambda cfg, ds: identity(cfg,ds.splits,source_hash)
    return core

class Capture:
    """Sample actual block outputs and Q/K/V projection outputs during one forward.

    residual[l]: post-block l, before the model's final LayerNorm.
    q/k/v[l]: projections of norm1(block-l input), before attention/head reshape.
    They describe the same block, not the same point in its computation.
    """
    def __init__(self, model, layers, channels):
        self.handles, self.values, self.indices = [], {}, None
        for layer in layers:
            block = model.encoder.layer[layer]
            for channel in channels:
                module = block if channel=='residual' else getattr(block.attention.attention,{'q':'query','k':'key','v':'value'}[channel])
                self.handles.append(module.register_forward_hook(self.hook(channel,layer)))

    def hook(self,channel,layer):
        def capture(module, inputs, output):
            if isinstance(output,tuple): output = output[0]
            if self.indices is None: raise RuntimeError('Set patch indices before forwarding')
            self.values[(channel,layer)] = output[0,self.indices+1].detach().float().cpu().numpy()
        return capture

    def close(self):
        for handle in self.handles: handle.remove()

def extract(c, core):
    import numpy as np
    import torch
    core.seed_all(c['split_seed'])
    ds = core.Dataset(c)
    counts = {s:len(ds.splits[s]) for s in ('train','val','test')}
    plan = enforce_budget(c,counts)
    plan['counts_are_estimates'] = False
    root = Path(c['output']); root.mkdir(parents=True,exist_ok=True)
    # Freeze splits: every channel/model adapter must consume these exact IDs for this run.
    split_file = root/'splits.json'
    if split_file.exists() and json.loads(split_file.read_text())!=ds.splits:
        raise RuntimeError('Splits changed; choose a new output directory')
    core.save_json(split_file,ds.splits)
    core.save_json(root/'resource_plan.json',plan)
    signature = core.cache_signature(c,ds)
    pending = {}
    additional = 0
    for channel in c['channels']:
        cfg = channel_config(c,channel)
        for split,ids in ds.splits.items():
            folder = Path(cfg['output'])/'cache'/split
            manifest_path = folder/'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest['signature']!=signature or manifest.get('schema')!=2:
                    raise RuntimeError(f'Incompatible cache: {folder}. Use a new output directory.')
                existing = set(manifest['layers'])
                for l in existing:
                    if not (folder/f'layer_{l}.npy').exists(): raise RuntimeError('Cache manifest references a missing layer')
            else: existing = set()
            missing = [l for l in c['layers'] if l not in existing]
            if missing:
                pending[channel,split] = (folder,missing,existing)
                additional += len(ids)*len(missing)*c['saved_patches']*c['embed_dim']*4
    if not pending:
        print('All requested channels/layers cached; no extraction.',flush=True); return
    existing_bytes = sum(p.stat().st_size for p in root.glob('channels/*/cache/*/layer_*.npy'))
    if (existing_bytes+additional)/2**30 > c['cache_budget_gib']:
        raise RuntimeError('Retained caches plus requested additions exceed cache_budget_gib. Reduce channels/layers or explicitly raise the budget.')
    # Conservative overhead; protect disk before allocation. Existing unrelated caches remain untouched.
    if additional*1.05 > shutil.disk_usage(root).free:
        raise RuntimeError(f'Not enough free disk for {additional/2**30:.2f} GiB of new features')
    model = core.load_backbone(c)
    if model.config.hidden_size!=c['embed_dim'] or model.config.num_hidden_layers!=c['num_layers']:
        raise RuntimeError('Loaded backbone differs from adapter dimensions')
    proc = core.processor(c)
    for split, ids in ds.splits.items():
        jobs = {ch:pending[ch,split] for ch in c['channels'] if (ch,split) in pending}
        if not jobs: continue
        layers = sorted({l for _,ls,_ in jobs.values() for l in ls})
        capture = Capture(model,layers,list(jobs))
        maps = {}
        for channel,(folder,missing,_) in jobs.items():
            folder.mkdir(parents=True,exist_ok=True)
            for l in missing:
                maps[channel,l] = np.lib.format.open_memmap(folder/f'layer_{l}.npy',mode='w+',dtype=np.float32,
                    shape=(len(ids),c['saved_patches'],c['embed_dim']))
        labels = {key:[] for key in ('instance','semantic','background','position')}
        try:
            for row,image_id in enumerate(ids):
                rgb,sem,inst,bg = ds.read(image_id)
                pixel = proc.forward(rgb).permute(0,3,1,2)
                masks = [proc.forward_patch(a[:,:,None]).reshape(len(pixel),-1).numpy() for a in (inst,sem,bg)]
                ncrops,patches = masks[0].shape
                rng = np.random.default_rng(c['split_seed']+image_id)
                n = c['saved_patches']//ncrops
                if n==0: raise ValueError('More crops than saved patches')
                picks = [rng.choice(patches,n,replace=False) for _ in range(ncrops)]
                picks[-1] = rng.choice(patches,c['saved_patches']-n*(ncrops-1),replace=False)
                chunks = {key:[] for key in maps}
                with torch.inference_mode():
                    for crop,p in enumerate(picks):
                        capture.indices = p; capture.values.clear()
                        model(pixel_values=pixel[crop:crop+1].to(c['device']))
                        for key in maps: chunks[key].append(capture.values[key])
                for key,mm in maps.items(): mm[row] = np.concatenate(chunks[key])
                positions = np.concatenate([p+crop*patches for crop,p in enumerate(picks)])
                for key,mask in zip(('instance','semantic','background'),masks):
                    labels[key].append(mask.reshape(-1)[positions])
                labels['position'].append(positions)
                if row%10==0 or row+1==len(ids): print(f'extract {split}: {row+1}/{len(ids)} channels={list(jobs)}',flush=True)
            for mm in maps.values(): mm.flush()
            for channel,(folder,missing,existing) in jobs.items():
                new = {k:np.asarray(v) for k,v in labels.items()}; new['image_id'] = np.asarray(ids)
                label_file = folder/'labels.npz'
                if label_file.exists():
                    with np.load(label_file) as old:
                        if any(not np.array_equal(old[k],v) for k,v in new.items()): raise RuntimeError('Cached patch selection differs')
                else: np.savez_compressed(label_file,**new)
                core.save_json(folder/'manifest.json',{'schema':2,'signature':signature,'complete':True,
                    'shape':[len(ids),c['saved_patches'],c['embed_dim']],'layers':sorted(existing|set(missing)),
                    'channel':channel,'precision':'float32','num_heads':c['num_heads'],
                    'location':'post_block' if channel=='residual' else 'norm1_input_projection'})
        finally:
            capture.close()
            maps.clear()
            # A loop variable can otherwise keep the last mmap open on Windows.
            if 'mm' in locals(): del mm
    core.save_json(root/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'model_commit':getattr(model.config,'_commit_hash',None),'model':c['model'],
        'code_commit':git_commit(),'config':c})

def git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    except (OSError,subprocess.CalledProcessError): return None

def report(c,core):
    import pandas as pd
    import matplotlib.pyplot as plt
    rows = []
    groups = ('foreground','same_instance_foreground','same_class_different_instance','different_class',
              'instance_vs_same_class','class_grouping_cross_instance')
    for channel in c['channels']:
        cfg = channel_config(c,channel)
        for mode in c['probes']:
            for layer in c['layers']:
                for seed in c['seeds']:
                    folder = Path(cfg['output'])/'probes'/mode/f'layer_{layer}'/f'seed_{seed}'
                    file = folder/'metrics.json'
                    if not file.exists(): continue
                    data = json.loads(file.read_text())
                    row = dict(dataset=c['dataset_name'],model=c['model'],channel=channel,probe=mode,layer=layer,
                        block_fraction=layer/(c['num_layers']-1),seed=seed)
                    for group in groups:
                        for metric in ('n','balanced_accuracy','auroc','recall','false_positive_rate'):
                            row[group+'_'+metric] = data.get(group,{}).get(metric)
                    rows.append(row)
    if not rows: raise RuntimeError('No completed runs')
    root = Path(c['output']); df = pd.DataFrame(rows); df.to_csv(root/'phenomenon_metrics.csv',index=False)
    expected = len(c['channels'])*len(c['layers'])*len(c['seeds'])*len(c['probes'])
    core.save_json(root/'report_status.json',{'completed':len(rows),'expected':expected,'complete':len(rows)==expected})
    specs = [('instance_vs_same_class_auroc','Same instance vs same-class different instance AUROC'),
        ('class_grouping_cross_instance_auroc','Same class vs different class (different instances) AUROC'),
        ('foreground_balanced_accuracy','Foreground balanced accuracy'),
        ('same_instance_foreground_recall','Same-instance recall'),
        ('same_class_different_instance_false_positive_rate','Same-class different-instance FPR'),
        ('different_class_false_positive_rate','Different-class FPR')]
    fig,axes = plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for ax,(metric,title) in zip(axes.flat,specs):
        for (channel,mode),frame in df.groupby(['channel','probe']):
            stats = frame.groupby('layer')[metric].agg(['mean','std'])
            ax.plot(stats.index,stats['mean'],marker='.',label=f'{channel}/{mode}')
            sd = stats['std'].fillna(0)
            ax.fill_between(stats.index,stats['mean']-sd,stats['mean']+sd,alpha=.12)
        ax.set(title=title,xlabel='Block index (0-based)',ylim=(0,1)); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.savefig(root/'phenomenon_curves.png',dpi=180); plt.close(fig)
    (root/'observations.md').write_text(
        '# 现象确认\n\n先比较 instance_vs_same_class AUROC 的层级变化，再检查跨实例的类别分组 AUROC。\n'
        'FPR/召回使用固定0阈值，只作补充。seed标准差不是图像置信区间。\n'
        f'已完成 {len(rows)}/{expected} 个任务。中层峰值及深层类别分组都不是预设结论。\n'
        'Q/K/V是block输入归一化后的投影，residual是block输出；通道probe差异不能证明信息迁移或因果作用。\n',encoding='utf-8')

def qk_scores(q,k,heads):
    """Symmetric pre-softmax per-head interaction; not attention probability."""
    import numpy as np
    n,d = q.shape
    q = q.reshape(n,heads,d//heads); k = k.reshape(n,heads,d//heads)
    score = np.einsum('ihd,jhd->hij',q,k)/math.sqrt(d//heads)
    return (score+score.transpose(0,2,1))/2

def interactions(c,core):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    root = Path(c['output'])
    summaries = []
    for layer in c['layers']:
        results = {}
        for split in ('val','test'):
            q,meta = core.cached(channel_config(c,'q'),split,layer)
            k,kmeta = core.cached(channel_config(c,'k'),split,layer)
            if any(not np.array_equal(meta[key],kmeta[key]) for key in meta): raise RuntimeError('Q/K patch alignment mismatch')
            ys,ss = [],[]
            i,j = np.triu_indices(c['eval_patches'],1)
            for row,image_id in enumerate(meta['image_id']):
                p = np.random.default_rng(100000+c['split_seed']+int(image_id)).choice(c['saved_patches'],c['eval_patches'],replace=False)
                ids,sem,bg = meta['instance'][row,p],meta['semantic'][row,p],meta['background'][row,p].astype(bool)
                y = ids[i]==ids[j]
                mask = (~bg[i])&(~bg[j])&(sem[i]>0)&(sem[j]>0)&(y|(sem[i]==sem[j]))
                ys.append(y[mask]); ss.append(qk_scores(np.asarray(q[row,p]),np.asarray(k[row,p]),c['num_heads'])[:,i,j][:,mask])
            y,s = np.concatenate(ys),np.concatenate(ss,axis=1)
            aucs = [float(roc_auc_score(y,a)) if y.any() and (~y).any() else None for a in s]
            results[split] = {'auroc_per_head':aucs,'pairs':len(y),'positive':int(y.sum())}
        valid = [(v,h) for h,v in enumerate(results['val']['auroc_per_head']) if v is not None]
        chosen = max(valid)[1] if valid else None
        summaries.append({'layer':layer,'head_selected_on_val':chosen,
            'selected_test_auroc':results['test']['auroc_per_head'][chosen] if chosen is not None else None,**results})
    core.save_json(root/'qk_interactions.json',{'score':'symmetric pre-softmax QK, no learned probe',
        'task':'same-instance vs same-class different-instance','selection':'head chosen only by validation AUROC','layers':summaries})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(HERE/'config.json'))
    parser.add_argument('--dataset-root'); parser.add_argument('--output')
    parser.add_argument('--stage',choices=['plan','extract','train','report','visualize','interactions','all'],default='plan')
    args = parser.parse_args()
    c = read_config(args.config,args.dataset_root,args.output)
    if args.stage=='plan':
        print(json.dumps(resource_plan(c),indent=2)); return
    enforce_budget(c)
    core = core_module(c)
    root = Path(c['output']); root.mkdir(parents=True,exist_ok=True)
    core.save_json(root/'requested_config.json',c)
    for stage in (['extract','train','report','visualize'] if args.stage=='all' else [args.stage]):
        if stage=='extract': extract(c,core)
        elif stage=='train':
            for ch in c['channels']: core.train(channel_config(c,ch))
        elif stage=='report': report(c,core)
        elif stage=='visualize':
            if 'residual' in c['channels']: core.visualize(channel_config(c,'residual'))
        elif stage=='interactions': interactions(c,core)

if __name__=='__main__': main()

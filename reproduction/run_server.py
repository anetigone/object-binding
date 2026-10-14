"""NeurIPS 2025 IsSameObject reproduction with explicit phenomenon measurements.

Reuses the pinned authors' ImageProcessor and probe classes through AST loading,
avoiding their hard-coded workstation paths and missing external ADE20K module.
Stages: extract, train, report, visualize, all. No stage runs on import.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json, math, pickle, platform, sys, time
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / 'vit-object-binding'
PIN = '014c66b45ea262f9b6eec83ff388a1e1c10dfcaa'

def load_author_classes(file, names):
    tree = ast.parse(file.read_text(encoding='utf-8'))
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in body} != set(names):
        raise RuntimeError(f'Upstream classes missing: {names}')
    namespace = dict(torch=torch, nn=nn, F=F, np=np, math=math, torchvision=torchvision, TF=TF)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(file), 'exec'), namespace)
    return {name: namespace[name] for name in names}

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def setup(config_path, dataset_root=None, output=None):
    c = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if dataset_root:
        c['dataset_root'] = str(Path(dataset_root).resolve())
    if output:
        c['output'] = output
    c['output'] = str(Path(c['output']).resolve())
    if c['model'] != 'facebook/dinov2-large' or c['crop_size'] != 518:
        raise ValueError('This first reproduction is scoped to DINOv2-L at 518x518.')
    if not 0 < c['sampled_patches'] <= c['saved_patches'] <= 1369:
        raise ValueError('Invalid patch counts')
    if not 2 <= c['eval_patches'] <= c['saved_patches']:
        raise ValueError('Invalid eval_patches')
    if not c['layers'] or any(l < 0 or l > 23 for l in c['layers']):
        raise ValueError('Layers must be zero-based indices 0..23')
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'requested_config.json', c)
    return c

class Dataset:
    def __init__(self, c):
        root = Path(c['dataset_root'])
        self.root = root if (root/'index_ade20k.pkl').exists() else root/'ADE20K_2021_17_01'
        with (self.root/'index_ade20k.pkl').open('rb') as f:
            self.index = pickle.load(f)
        # Exactly the authors' 150-class mapping, including their two special cases.
        csv_path = self.root/'objectInfo150.csv'
        if not csv_path.exists():
            csv_path = UPSTREAM/'objectInfo150.csv'
        df = pd.read_csv(csv_path)
        name_to_id = dict(zip(df['Name'], df['Idx']))
        mapping = [0]
        for name in self.index['objectnames']:
            name = name.replace(', ', ';').replace(' ', ';')
            if name in name_to_id:
                oid = int(name_to_id[name])
            elif name in ('door', 'double;door'):
                oid = 15
            elif name == 'television;receiver;television;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box':
                oid = 90
            elif name == 'screen;crt;screen':
                oid = 142
            else:
                oid = 0
            mapping.append(oid)
        self.mapping = np.array(mapping, dtype=np.int32)
        train = [i for i, folder in enumerate(self.index['folder']) if 'training' in folder]
        test = [i for i, folder in enumerate(self.index['folder']) if 'validation' in folder]
        boundary = int(.8*len(train))
        corrupt = {7689, 8904, 8905, 20433, 20543, 21029, 22979, 23011, 23048, 24103, 24794, 27278}
        # These exclusions are positions in the training subset, as in official code.
        splits = {'train': [v for k,v in enumerate(train) if k < boundary and k not in corrupt],
                  'val': [v for k,v in enumerate(train) if k >= boundary and k not in corrupt],
                  'test': [v for k,v in enumerate(test) if k != 20]}
        rng = np.random.default_rng(c['split_seed'])
        self.splits = {}
        for split, ids in splits.items():
            limit = c['max_'+split]
            if limit and limit < len(ids):
                ids = sorted(rng.choice(ids, limit, replace=False).tolist())
            self.splits[split] = ids
        sets = [set(v) for v in self.splits.values()]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def image_path(self, image_id):
        folder = str(self.index['folder'][image_id]).replace('\\', '/')
        marker = 'ADE20K_2021_17_01/'
        rel = folder.split(marker, 1)[-1] if marker in folder else folder
        if not rel.startswith('images/'):
            raise ValueError(f'Unexpected index folder: {folder}')
        path = (self.root/rel/str(self.index['filename'][image_id])).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError('Dataset path escapes dataset root')
        return path

    def read(self, image_id):
        path = self.image_path(image_id)
        with Image.open(path) as img:
            rgb = np.asarray(img.convert('RGB')).copy()
        with Image.open(path.with_name(path.stem+'_seg.png')) as img:
            raw = np.asarray(img.convert('RGB')).copy()
        cls = (raw[:,:,0].astype(np.int32)//10)*256 + raw[:,:,1].astype(np.int32)
        if cls.max() >= len(self.mapping):
            raise ValueError(f'Class id out of range in {path}')
        # Preserve original IsSameObject equivalence: instance ID comes from B.
        inst = np.unique(raw[:,:,2], return_inverse=True)[1].reshape(cls.shape).astype(np.int32)
        return rgb, self.mapping[cls], inst, (cls == 0).astype(np.int32)

def processor(c):
    cls = load_author_classes(UPSTREAM/'src/data_extractor.py', ['ImageProcessor'])['ImageProcessor']
    cfg = NS(model=NS(patch_size=14), data_extractor=NS(image_processor=NS(
        mean=[.485,.456,.406], std=[.229,.224,.225], shortest_edge=518,
        crop_size=[518,518], normalize=True)))
    return cls(cfg)

def probe(c, mode):
    names = {'quadratic':'QuadraticProbe', 'linear':'LinearProbe', 'diag_quadratic':'DiagonalQuadraticProbe'}
    cls = load_author_classes(UPSTREAM/'src/utils/models.py', [names[mode]])[names[mode]]
    return cls(NS(probe=NS(in_dim=c.get('embed_dim',1024))))

def load_backbone(c):
    model = AutoModel.from_pretrained(c['model'], revision=c['model_revision'],
        cache_dir=str(Path(c.get('hf_cache',HERE/'hf_cache')).expanduser()),
        local_files_only=c.get('offline',False),
        attn_implementation='sdpa').to(c['device'])
    model.eval().requires_grad_(False)
    return model

def cache_signature(c, ds):
    payload = {k:c[k] for k in ('model','model_revision','crop_size','saved_patches','layers','split_seed')}
    payload['splits'] = ds.splits
    payload['sources'] = {str(p.relative_to(UPSTREAM)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (UPSTREAM/'src/data_extractor.py', UPSTREAM/'src/utils/models.py')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def extract(c):
    seed_all(c['split_seed'])
    ds = Dataset(c)
    out = Path(c['output'])
    signature = cache_signature(c, ds)
    save_json(out/'splits.json', ds.splits)
    manifests = {}
    for split, ids in ds.splits.items():
        path = out/'cache'/split/'manifest.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['signature'] != signature:
                raise RuntimeError('Cache configuration changed; use a new output directory.')
            manifests[split] = saved
    if len(manifests) == 3:
        print('All feature caches complete; skipping extraction.', flush=True)
        return
    gib = sum(len(ids) for ids in ds.splits.values())*len(c['layers'])*c['saved_patches']*1024*4/2**30
    print(f'Feature cache estimate: {gib:.1f} GiB; FP32; layers are block-output indices 0..23.', flush=True)
    model, proc = load_backbone(c), processor(c)
    for split, ids in ds.splits.items():
        if split in manifests:
            continue
        folder = out/'cache'/split
        folder.mkdir(parents=True, exist_ok=True)
        shape = (len(ids), c['saved_patches'], 1024)
        maps = {l:np.lib.format.open_memmap(folder/f'layer_{l}.npy', mode='w+', dtype=np.float32, shape=shape) for l in c['layers']}
        insts, classes, backgrounds, positions = [], [], [], []
        started = time.monotonic()
        for row, image_id in enumerate(ids):
            rgb, sem, inst, bg = ds.read(image_id)
            pixel = proc.forward(rgb).permute(0,3,1,2)
            masks = [proc.forward_patch(a[:,:,None]).reshape(pixel.shape[0], -1).numpy() for a in (inst,sem,bg)]
            ncrops, patches = masks[0].shape
            rng = np.random.default_rng(c['split_seed']+image_id)
            n_each = c['saved_patches']//ncrops
            if n_each == 0:
                raise ValueError('More crops than saved patches')
            picks = [rng.choice(patches, n_each, replace=False) for _ in range(ncrops)]
            remainder = c['saved_patches']-n_each*(ncrops-1)
            picks[-1] = rng.choice(patches, remainder, replace=False)
            # Crop microbatch 1 limits GPU memory; all author crops are retained.
            chunks = {l:[] for l in c['layers']}
            with torch.inference_mode():
                for crop in range(ncrops):
                    result = model(pixel_values=pixel[crop:crop+1].to(c['device']), output_hidden_states=True)
                    for l in c['layers']:
                        chunks[l].append(result.hidden_states[l+1][0,1:][picks[crop]].float().cpu().numpy())
                    del result
            for l in c['layers']:
                maps[l][row] = np.concatenate(chunks[l])
            flat_idx = np.concatenate([p+crop*patches for crop,p in enumerate(picks)])
            for dest, mask in zip((insts,classes,backgrounds), masks):
                dest.append(mask.reshape(-1)[flat_idx])
            positions.append(flat_idx)
            if row%10 == 0 or row+1 == len(ids):
                print(f'extract {split} {row+1}/{len(ids)} {time.monotonic()-started:.1f}s', flush=True)
        for mm in maps.values():
            mm.flush()
        np.savez_compressed(folder/'labels.npz', image_id=ids, instance=np.asarray(insts),
            semantic=np.asarray(classes), background=np.asarray(backgrounds), position=np.asarray(positions))
        save_json(folder/'manifest.json', {'signature':signature, 'shape':shape, 'layers':c['layers'], 'complete':True})
    save_json(out/'environment.json', {'python':sys.version, 'platform':platform.platform(),
        'torch':torch.__version__, 'torchvision':torchvision.__version__, 'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'upstream_expected_commit':PIN, 'model_revision':c['model_revision'], 'precision':'float32'})

def cached(c, split, layer):
    folder = Path(c['output'])/'cache'/split
    manifest = json.loads((folder/'manifest.json').read_text())
    if layer not in manifest['layers'] or not manifest['complete']:
        raise RuntimeError('Feature cache not complete')
    return np.load(folder/f'layer_{layer}.npy', mmap_mode='r'), dict(np.load(folder/'labels.npz'))

def batch_logits(model, x):
    # Official forward_pairwise returns logits; loop also handles linear/diagonal probes.
    return torch.stack([model.forward_pairwise(a,a) for a in x])

def metrics(y, score, compute_auc=True):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    if not len(y):
        return {'n':0, 'accuracy':None, 'balanced_accuracy':None, 'auroc':None, 'positive_rate':None, 'predicted_positive_rate':None, 'recall':None, 'false_positive_rate':None}
    pred = score > 0
    pos, neg = y.sum(), (~y).sum()
    recall = float(pred[y].mean()) if pos else None
    fpr = float(pred[~y].mean()) if neg else None
    return {'n':int(len(y)), 'accuracy':float((pred==y).mean()),
        'balanced_accuracy':(recall+1-fpr)/2 if pos and neg else None,
        'auroc':float(roc_auc_score(y,score)) if pos and neg and compute_auc else None,
        'positive_rate':float(y.mean()), 'predicted_positive_rate':float(pred.mean()),
        'recall':recall, 'false_positive_rate':fpr}

def eval_probe(c, model, split, layer, return_detail=False):
    acts, meta = cached(c, split, layer)
    groups = {k:([],[]) for k in ('all','foreground','same_class_different_instance','different_class','same_instance_foreground','background', 'instance_vs_same_class', 'class_grouping_cross_instance')}
    records, detail = [], []
    upper = np.triu_indices(c['eval_patches'], k=1)
    with torch.inference_mode():
        for row in range(len(acts)):
            # Fixed independent samples for val/test, shared by every layer/probe/seed.
            rng = np.random.default_rng(100000+c['split_seed']+int(meta['image_id'][row]))
            p = rng.choice(c['saved_patches'],c['eval_patches'],replace=False)
            x = torch.tensor(np.asarray(acts[row,p]), device=c['device'])
            s = model.forward_pairwise(x,x).cpu().numpy()[upper]
            ids, sem, bg = meta['instance'][row,p], meta['semantic'][row,p], meta['background'][row,p].astype(bool)
            i,j = upper
            y = ids[i] == ids[j]
            fg = ~bg[i]&~bg[j]
            known = (sem[i]>0)&(sem[j]>0)
            conditions = {'all':np.ones_like(y), 'foreground':fg,
                'same_class_different_instance':fg&known&(~y)&(sem[i]==sem[j]),
                'different_class':fg&known&(~y)&(sem[i]!=sem[j]),
                'same_instance_foreground':fg&y, 'background':bg[i]|bg[j]}
            conditions['instance_vs_same_class'] = fg&known&(y|(sem[i]==sem[j]))
            conditions['class_grouping_cross_instance'] = fg&known&(~y)
            for name, mask in conditions.items():
                target = (sem[i]==sem[j]) if name=='class_grouping_cross_instance' else y
                groups[name][0].append(target[mask]); groups[name][1].append(s[mask])
            record = {'image_id':int(meta['image_id'][row]),
                'positive':int(y.sum()), 'negative':int((~y).sum()),
                'true_positive':int(((s>0)&y).sum()), 'true_negative':int(((s<=0)&(~y)).sum()),
                'same_class_negative':int(conditions['same_class_different_instance'].sum()),
                'same_class_false_positive':int(((s>0)&conditions['same_class_different_instance']).sum())}
            for name,mask in conditions.items():
                target = (sem[i]==sem[j]) if name=='class_grouping_cross_instance' else y
                record[name] = {'positive':int((mask&target).sum()), 'negative':int((mask&~target).sum()),
                    'true_positive':int((mask&target&(s>0)).sum()), 'false_positive':int((mask&~target&(s>0)).sum())}
            records.append(record)
            if return_detail:
                code = np.zeros(len(y),dtype=np.uint8)
                code[fg] = 4
                code[fg&known&y] = 1
                code[conditions['same_class_different_instance']] = 2
                code[conditions['different_class']] = 3
                detail.append((s.astype(np.float32), y, p, code))
    result = {k:metrics(np.concatenate(v[0]),np.concatenate(v[1]),compute_auc=split=='test') for k,v in groups.items()}
    result['always_different_accuracy'] = 1-result['all']['positive_rate']
    if split == 'test':
        result['semantic_equality_baseline'] = semantic_baseline(c, meta)
    return result, records, detail

def semantic_baseline(c, meta):
    labels, scores = [], []
    upper = np.triu_indices(c['eval_patches'],1)
    for row in range(len(meta['image_id'])):
        rng = np.random.default_rng(100000+c['split_seed']+int(meta['image_id'][row]))
        p = rng.choice(c['saved_patches'],c['eval_patches'],replace=False)
        ids, sem = meta['instance'][row,p],meta['semantic'][row,p]
        labels.append((ids[:,None]==ids[None,:])[upper])
        scores.append(np.where((sem[:,None]==sem[None,:])[upper],1.,-1.))
    return metrics(np.concatenate(labels),np.concatenate(scores))

def train(c):
    ds = Dataset(c)
    signature = cache_signature(c,ds)
    for split in ('train','val','test'):
        manifest = json.loads((Path(c['output'])/'cache'/split/'manifest.json').read_text())
        if manifest['signature'] != signature:
            raise RuntimeError('Feature cache differs from requested dataset/model configuration.')
    # Adding probe types or seeds reuses the same cached features and completed runs.
    training_settings = {k:c[k] for k in ('epochs','batch_size','learning_rate',
        'weight_decay','scheduler_step','scheduler_gamma','sampled_patches','eval_patches')}
    training_settings['cache_signature'] = signature
    provenance = Path(c['output'])/'training_config.json'
    if provenance.exists() and json.loads(provenance.read_text()) != training_settings:
        raise RuntimeError('Training configuration changed; use a new output directory.')
    save_json(provenance,training_settings)
    for mode in c['probes']:
        for layer in c['layers']:
            train_acts, meta = cached(c,'train',layer)
            for seed in c['seeds']:
                folder = Path(c['output'])/'probes'/mode/f'layer_{layer}'/f'seed_{seed}'
                if (folder/'metrics.json').exists():
                    print('skip completed', folder, flush=True); continue
                folder.mkdir(parents=True,exist_ok=True)
                seed_all(seed)
                model = probe(c,mode).to(c['device'])
                optimizer = torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,c['scheduler_step'],c['scheduler_gamma'])
                rng = np.random.default_rng(seed)
                best, history = -1., []
                for epoch in range(c['epochs']):
                    model.train()
                    order = rng.permutation(len(train_acts))
                    losses = []
                    for start in range(0,len(order),c['batch_size']):
                        rows = order[start:start+c['batch_size']]
                        patches = np.stack([rng.choice(c['saved_patches'],c['sampled_patches'],replace=False) for _ in rows])
                        x = torch.tensor(np.stack([train_acts[r,p] for r,p in zip(rows,patches)]),device=c['device'])
                        ids = torch.tensor(np.stack([meta['instance'][r,p] for r,p in zip(rows,patches)]),device=c['device'])
                        target = (ids[:,:,None] == ids[:,None,:]).float()
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.binary_cross_entropy_with_logits(batch_logits(model,x),target)
                        loss.backward(); optimizer.step()
                        losses.append((loss.item(),len(rows)))
                    scheduler.step()
                    model.eval()
                    val, _, _ = eval_probe(c,model,'val',layer)
                    acc = val['all']['accuracy']
                    history.append({'epoch':epoch+1,'train_loss':sum(a*n for a,n in losses)/sum(n for _,n in losses), 'val_accuracy':acc, 'val_balanced_accuracy':val['all']['balanced_accuracy']})
                    if acc > best:
                        best = acc
                        torch.save(model.state_dict(),folder/'checkpoint.pt')
                        save_json(folder/'selection.json', {'epoch':epoch+1,'val_accuracy':best,'criterion':'all-pairs validation accuracy'})
                    print(f'{mode} layer={layer} seed={seed} epoch={epoch+1} val={acc:.4f}',flush=True)
                model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=c['device'],weights_only=True))
                model.eval()
                result, records, detail = eval_probe(c,model,'test',layer,True)
                save_json(folder/'history.json',history)
                save_json(folder/'per_image.json',records)
                np.savez_compressed(folder/'test_predictions.npz',score=np.stack([r[0] for r in detail]),
                    label=np.stack([r[1] for r in detail]),patches=np.stack([r[2] for r in detail]),
                    condition=np.stack([r[3] for r in detail]),image_id=cached(c,'test',layer)[1]['image_id'])
                save_json(folder/'metrics.json',result)

def report(c):
    out = Path(c['output'])
    rows = []
    for mode in c['probes']:
        for layer in c['layers']:
            for seed in c['seeds']:
                path = out/'probes'/mode/f'layer_{layer}'/f'seed_{seed}'/'metrics.json'
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                row = {'probe':mode,'layer':layer,'seed':seed,'always_different':data['always_different_accuracy']}
                for group in ('all','foreground','same_class_different_instance','different_class','same_instance_foreground'):
                    for key in ('n','accuracy','balanced_accuracy','auroc','recall','false_positive_rate'):
                        row[group+'_'+key] = data[group][key]
                rows.append(row)
    if not rows:
        raise RuntimeError('No completed runs to report')
    df = pd.DataFrame(rows)
    df.to_csv(out/'metrics.csv',index=False)
    expected = len(c['probes'])*len(c['layers'])*len(c['seeds'])
    save_json(out/'report_status.json',{'completed':len(rows),'expected':expected,'complete':len(rows)==expected})
    specs = [('all_accuracy','Accuracy'),('all_balanced_accuracy','Balanced accuracy'),
        ('all_auroc','AUROC'),('foreground_balanced_accuracy','Foreground balanced accuracy'),
        ('same_instance_foreground_recall','Same-instance foreground recall'),
        ('same_class_different_instance_false_positive_rate','Same-class different-instance false positive rate')]
    fig, axes = plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for ax,(key,label) in zip(axes.flat,specs):
        for mode in c['probes']:
            frame = df[df.probe==mode].groupby('layer')[key].agg(['mean','std'])
            if frame.empty: continue
            x,y = frame.index.to_numpy(),frame['mean'].to_numpy()
            ax.plot(x,y,marker='.',label=mode)
            deviation = frame['std'].fillna(0).to_numpy()
            ax.fill_between(x,y-deviation,y+deviation,alpha=.12)
        ax.set(title=label,xlabel='Block output index (0-based)',ylim=(0,1))
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.savefig(out/'layer_curves.png',dpi=180); plt.close(fig)
    q = df[df.probe=='quadratic'].groupby('layer').mean(numeric_only=True)
    if not q.empty:
        fg = q.foreground_balanced_accuracy.dropna()
        fg_peak = str(int(fg.idxmax())) if len(fg) else '无有效正负配对'
        lines = ['# 实际运行后的现象记录','', '曲线阴影是训练随机种子的标准差，不是图像置信区间。', '',
            f'完成 {len(rows)}/{expected} 个配置任务；不完整结果需谨慎解读。', '',
            f"二次 probe 的总体准确率峰值层：{int(q.all_accuracy.idxmax())}。",
            f"前景平衡准确率峰值层：{fg_peak}。", '',
            '下列数值只描述本次数据与 probe；不能单独证明原模型依赖实例绑定。', '',
            '| 层 | 前景平衡准确率 | 同实例召回 | 同类不同实例误绑定率 | 跨类别误绑定率 |',
            '|---|---|---|---|---|']
        for layer,row in q.iterrows():
            lines.append(f'| {layer} | {row.foreground_balanced_accuracy:.4f} | {row.same_instance_foreground_recall:.4f} | {row.same_class_different_instance_false_positive_rate:.4f} | {row.different_class_false_positive_rate:.4f} |')
        lines += ['', '必须结合样本数、各 seed、训练收敛曲线和可视化检查；中层峰值及类别聚合都不是预设结论。']
        (out/'observations.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote metrics.csv, layer_curves.png, observations.md',flush=True)

def visualize(c):
    if 'quadratic' not in c['probes'] or c['visualize_images'] == 0:
        return
    ds, proc = Dataset(c), processor(c)
    out = Path(c['output'])
    model = None
    count = 0
    for image_id in ds.splits['test']:
        rgb,sem,inst,bg = ds.read(image_id)
        pixel = proc.forward(rgb).permute(0,3,1,2)
        imask = proc.forward_patch(inst[:,:,None])[0,:,:,0].numpy().reshape(-1)
        smask = proc.forward_patch(sem[:,:,None])[0,:,:,0].numpy().reshape(-1)
        bmask = proc.forward_patch(bg[:,:,None])[0,:,:,0].numpy().reshape(-1).astype(bool)
        candidates = []
        for category in np.unique(smask[(smask>0)&(~bmask)]):
            ids, counts = np.unique(imask[(smask==category)&(~bmask)],return_counts=True)
            valid = ids[counts>=5]
            if len(valid)>=2:
                candidates.append((int(category),int(valid[0]),int(valid[1])))
        if not candidates: continue
        if model is None: model = load_backbone(c)
        category, instance, other = candidates[0]
        coords = np.flatnonzero((imask==instance)&(smask==category)&(~bmask))
        query = int(coords[len(coords)//2])
        folder = out/'visualizations'/str(image_id)
        folder.mkdir(parents=True,exist_ok=True)
        maps = []
        # Capture block outputs directly: hidden_states tuple conventions can differ
        # between Transformers versions, especially around final LayerNorm.
        from phenomenon import Capture
        capture = Capture(model,c['layers'],['residual'])
        capture.indices = np.arange(37*37)
        try:
            with torch.inference_mode():
                model(pixel_values=pixel[:1].to(c['device']))
            hidden = dict(capture.values)
        finally:
            capture.close()
        with torch.inference_mode():
            for layer in c['layers']:
                trained = probe(c,'quadratic').to(c['device'])
                checkpoint = out/'probes'/'quadratic'/f'layer_{layer}'/f"seed_{c['seeds'][0]}"/'checkpoint.pt'
                trained.load_state_dict(torch.load(checkpoint,map_location=c['device'],weights_only=True))
                x = torch.tensor(hidden['residual',layer],device=c['device'])
                score = trained.forward(x[query:query+1].expand_as(x),x).sigmoid().cpu().numpy()
                maps.append(score.reshape(37,37))
        image_cfg = NS(model=NS(patch_size=14),data_extractor=NS(image_processor=NS(mean=[.485,.456,.406],std=[.229,.224,.225],shortest_edge=518,crop_size=[518,518],normalize=False)))
        pcls = load_author_classes(UPSTREAM/'src/data_extractor.py',['ImageProcessor'])['ImageProcessor']
        cropped_rgb = pcls(image_cfg).forward(rgb)[0].numpy().clip(0,1)
        display_layers = [l for l in (0,3,6,9,12,15,18,21,23) if l in c['layers']]
        fig,axes = plt.subplots(2,math.ceil((len(display_layers)+2)/2),figsize=(18,7),constrained_layout=True)
        for ax in axes.flat: ax.axis('off')
        axes.flat[0].imshow(cropped_rgb)
        axes.flat[0].scatter((query%37)*14+7,(query//37)*14+7,c='red',s=35)
        axes.flat[0].set_title('Query patch (red)')
        target = np.where(imask==instance,1,np.where(imask==other,2,0)).reshape(37,37)
        axes.flat[1].imshow(target,vmin=0,vmax=2,cmap='viridis'); axes.flat[1].set_title('Two ground-truth instances')
        for ax,layer in zip(list(axes.flat)[2:],display_layers):
            ax.imshow(maps[c['layers'].index(layer)],vmin=0,vmax=1,cmap='magma')
            ax.set_title(f'Layer {layer}: P(same instance)')
        fig.savefig(folder/'query_maps.png',dpi=180); plt.close(fig)
        np.savez_compressed(folder/'query_maps.npz',scores=np.stack(maps),layers=c['layers'],
            query=query,instance=imask.reshape(37,37),semantic=smask.reshape(37,37))
        save_json(folder/'case.json',{'image_id':image_id,'query':query,'category':category,
            'query_instance':instance,'other_instance':other,'crop':0,'seed':c['seeds'][0],
            'selection':'first test images with >=2 same-category instances of >=5 patches; not selected by probe result'})
        count += 1
        if count >= c['visualize_images']: break
    print(f'Visualized {count} cases with same-category multiple instances.',flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(HERE/'config.json'))
    parser.add_argument('--dataset-root')
    parser.add_argument('--output')
    parser.add_argument('--stage',choices=['extract','train','report','visualize','all'],default='all')
    args = parser.parse_args()
    c = setup(args.config,args.dataset_root,args.output)
    stages = ['extract','train','report','visualize'] if args.stage=='all' else [args.stage]
    for stage in stages:
        globals()[stage](c)

if __name__ == '__main__':
    main()

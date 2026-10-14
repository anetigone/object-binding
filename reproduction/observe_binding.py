"""Observe L23 relations with distance controls and a paired training-objective test.

Independent run: extraction is supported, old checkpoints are not prerequisites.
Only plan is dependency-free. Other stages use the existing backbone adapter.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import phenomenon as p


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(c):
    s = c['observation']
    if c['layers'] != [23] or c['channels'] != ['post_attention', 'residual']:
        raise ValueError('This observation is scoped to the two L23 endpoints')
    if not math.isfinite(s['distance_bin_patches']) or s['distance_bin_patches'] <= 0:
        raise ValueError('Distance bin width must be finite and positive')
    if set(s['objectives']) != {'all_pairs', 'matched_anchor'} or len(s['objectives']) != 2:
        raise ValueError('Both training objectives are required exactly once')
    names = [v['name'] for v in s['variants']]
    if not names or len(set(names)) != len(names):
        raise ValueError('Variant names must be unique')
    for v in s['variants']:
        if not v['name'].replace('_', '').isalnum() or v['kind'] not in ('full', 'factor'):
            raise ValueError('Invalid variant')
        if v['kind'] == 'factor' and not 0 < v['factor_width'] <= c['embed_dim']:
            raise ValueError('Invalid factor width')
    if c['epochs'] <= 0 or c['batch_size'] <= 0 or s['bootstrap_repeats'] <= 0:
        raise ValueError('Training and bootstrap budgets must be positive')
    if s['cases_per_group'] < 0 or s['case_min_triples'] < 1:
        raise ValueError('Invalid case budget')
    plan = p.resource_plan(c)
    plan['training_tasks'] = 2*len(s['variants'])*len(s['objectives'])
    plan['optimizer_steps_upper_bound'] = plan['training_tasks']*c['epochs']*math.ceil(c['max_train']/c['batch_size'])
    plan.pop('optimizer_steps', None)
    plan['within_budget'] = plan['fp32_cache_gib'] <= c['cache_budget_gib'] and plan['training_tasks'] <= c['max_tasks']
    if not plan['within_budget']:
        raise ValueError('Observation exceeds cache/task budget')
    return plan


def prepare(c):
    """Freeze the independent run before extraction; never adopt an old run."""
    root = Path(c['output'])
    protocol = root/'observation_protocol.json'
    sources = [Path(__file__), p.HERE/'phenomenon.py', p.HERE/'run_server.py',
               p.HERE/'validate_decline.py', p.HERE.parent/'vit-object-binding/src/data_extractor.py',
               p.HERE.parent/'vit-object-binding/src/utils/models.py']
    payload = dict(config=c, source_hashes={str(f.relative_to(p.HERE.parent)): digest(f) for f in sources})
    if c.get('observation_splits_file'):
        payload['split_file_sha256'] = digest(c['observation_splits_file'])
    if protocol.exists():
        if read(protocol) != payload:
            raise ValueError('Observation protocol changed; use a new --output')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('Independent output required; refusing to adopt or overwrite an existing run')
        write(protocol, payload)
        write(root/'requested_config.json', c)


def attach_splits(c, core):
    """Optionally reuse image IDs, with fresh extraction/training in our output."""
    if not c.get('observation_splits_file'):
        return
    splits = read(c['observation_splits_file'])
    groups = [set(splits[k]) for k in ('train', 'val', 'test')]
    if any(len(groups[i]) != len(splits[k]) for i, k in enumerate(('train', 'val', 'test'))) or any(
            groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Duplicate or overlapping image IDs')
    original = core.Dataset
    def factory(cfg):
        ds = original(dict(cfg, max_train=0, max_val=0, max_test=0))
        if any(not set(splits[k]).issubset(ds.splits[k]) for k in splits):
            raise ValueError('Requested splits are outside eligible dataset pools')
        ds.splits = copy.deepcopy(splits)
        return ds
    core.Dataset = factory


def relation_sets(meta, width, grid=37):
    """Build exact shared-anchor triples, using labels/positions only.

    Matching bins have width `width` in patch units. Positive/negative must
    share anchor class and crop. Each supported anchor has equal total weight.
    No diagonal/background/cross-class positives; no score-based pair selection.
    """
    import numpy as np
    pos = np.asarray(meta['position'], dtype=np.int64)
    ids, sem = np.asarray(meta['instance']), np.asarray(meta['semantic'])
    valid = ~np.asarray(meta['background'], dtype=bool) & (sem > 0)
    crop, local = pos//(grid*grid), pos % (grid*grid)
    xy = np.stack((local//grid, local % grid), axis=1)
    distance = np.linalg.norm(xy[:, None]-xy[None, :], axis=-1)
    same_id = ids[:, None] == ids[None, :]
    same_class = sem[:, None] == sem[None, :]
    same_crop = crop[:, None] == crop[None, :]
    eligible = valid[:, None] & valid[None, :] & same_crop & same_class
    np.fill_diagonal(eligible, False)
    positive, negative = eligible & same_id, eligible & ~same_id
    bins = np.floor(distance/width).astype(np.int64)
    result = dict(distance=distance, positive=positive, negative=negative,
                  inconsistent_same_id_pairs=int(np.triu(same_id & ~same_class & valid[:, None] & valid[None, :], 1).sum()))
    for name, matched in [('unmatched', False), ('matched', True)]:
        triples = []
        for a in range(len(pos)):
            groups = np.unique(bins[a, positive[a] | negative[a]]) if matched else [0]
            for group in groups:
                mask = bins[a] == group if matched else np.ones(len(pos), dtype=bool)
                pp, nn = np.flatnonzero(positive[a] & mask), np.flatnonzero(negative[a] & mask)
                if len(pp) and len(nn):
                    triples.append(np.column_stack((np.full(len(pp)*len(nn), a), np.repeat(pp, len(nn)), np.tile(nn, len(pp)))))
        triples = np.concatenate(triples) if triples else np.empty((0, 3), dtype=np.int64)
        anchors = np.unique(triples[:, 0])
        counts = np.bincount(triples[:, 0], minlength=len(pos))
        weights = 1/(len(anchors)*counts[triples[:, 0]]) if len(triples) else np.zeros(0)
        result[name] = dict(triples=triples, weights=weights, anchors=int(len(anchors)),
                            classes=sorted(set(int(sem[a]) for a in anchors)))
    return result


def ordering(score, relations, name):
    import numpy as np
    group = relations[name]; t = group['triples']
    if not len(t):
        return None
    delta = score[t[:, 0], t[:, 1]]-score[t[:, 0], t[:, 2]]
    return float(np.sum(((delta > 0)+.5*(delta == 0))*group['weights']))


def ranking_loss(score, relations):
    import torch
    t = relations['matched']['triples']
    if not len(t):
        return None
    index = torch.as_tensor(t, dtype=torch.long, device=score.device)
    weights = torch.as_tensor(relations['matched']['weights'], dtype=score.dtype, device=score.device)
    return (torch.nn.functional.softplus(score[index[:, 0], index[:, 2]]-score[index[:, 0], index[:, 1]])*weights).sum()


def sampled(c, meta, row):
    import numpy as np
    return np.random.default_rng(100000+c['split_seed']+int(meta['image_id'][row])).choice(
        c['saved_patches'], c['eval_patches'], replace=False)


def selected(meta, row, picks):
    return {k: meta[k][row, picks] for k in ('instance', 'semantic', 'background', 'position')}


def eval_bank(c, core, channel, split):
    acts, meta = core.cached(p.channel_config(c, channel), split, 23)
    picks = [sampled(c, meta, r) for r in range(len(acts))]
    relations = [relation_sets(selected(meta, r, q), c['observation']['distance_bin_patches']) for r, q in enumerate(picks)]
    return acts, meta, picks, relations


def evaluate(c, model, bank, save=None):
    import numpy as np
    import torch
    acts, meta, picks, relations = bank
    scores, records = [], []
    model.eval()
    with torch.inference_mode():
        for row, (q, rel) in enumerate(zip(picks, relations)):
            x = torch.as_tensor(np.asarray(acts[row, q]), device=c['device'])
            score = model.forward_pairwise(x, x).cpu().numpy()
            if not np.isfinite(score).all():
                raise ValueError('Non-finite evaluation scores')
            scores.append(score)
            records.append(dict(image_id=int(meta['image_id'][row]),
                matched_ordering=ordering(score, rel, 'matched'), unmatched_ordering=ordering(score, rel, 'unmatched'),
                distance_baseline_matched=ordering(-rel['distance'], rel, 'matched'),
                distance_baseline_unmatched=ordering(-rel['distance'], rel, 'unmatched'),
                matched_triples=len(rel['matched']['triples']), unmatched_triples=len(rel['unmatched']['triples']),
                matched_anchors=rel['matched']['anchors'], classes=rel['matched']['classes'],
                inconsistent_same_id_pairs=rel['inconsistent_same_id_pairs']))
    supported = [r['matched_ordering'] for r in records if r['matched_ordering'] is not None]
    summary = dict(macro_matched_ordering=float(np.mean(supported)) if supported else None,
                   images=len(records), supported_images=len(supported), per_image=records)
    if save is not None:
        save = Path(save); save.parent.mkdir(parents=True, exist_ok=True)
        rows = np.arange(len(acts))[:, None]; q = np.stack(picks)
        np.savez_compressed(save, score=np.stack(scores), patches=q, image_id=meta['image_id'],
                            **{k: meta[k][rows, q] for k in ('instance', 'semantic', 'background', 'position')})
    return summary


def task(c, objective, variant, channel):
    return Path(c['output'])/'observation'/objective/variant['name']/channel


def verify_artifacts(dest):
    files = ('checkpoint.pt', 'history.json', 'selection.json', 'test_predictions.npz', 'metrics.json')
    if any(not (dest/f).is_file() for f in files):
        raise ValueError(f'Incomplete task: {dest}')
    record = read(dest/'artifacts.json')
    if any(digest(dest/f) != record[f] for f in files):
        raise ValueError(f'Task artifact changed: {dest}')


def audit_cache(c, core):
    import numpy as np
    ds = core.Dataset(c)
    if read(Path(c['output'])/'splits.json') != ds.splits:
        raise ValueError('Run/dataset splits differ')
    signature = core.cache_signature(c, ds)
    inventory = dict(signature=signature, files={})
    for split in ('train', 'val', 'test'):
        first = None
        for channel in c['channels']:
            cfg = p.channel_config(c, channel)
            m = read(Path(cfg['output'])/'cache'/split/'manifest.json')
            acts, meta = core.cached(cfg, split, 23)
            if m.get('signature') != signature or m.get('schema') != 2 or not m.get('complete') or m.get('channel') != channel or m.get('location') != p.channel_location(channel):
                raise ValueError('Cache identity or endpoint differs')
            if acts.shape != (len(ds.splits[split]), c['saved_patches'], c['embed_dim']):
                raise ValueError('Incorrect feature dimensions')
            if not np.array_equal(meta['image_id'], ds.splits[split]):
                raise ValueError('Incorrect cache image IDs')
            if first is not None and any(not np.array_equal(meta[k], first[k]) for k in first):
                raise ValueError('Endpoint metadata is not paired')
            folder = Path(cfg['output'])/'cache'/split
            feature = folder/'layer_23.npy'; stat = feature.stat()
            inventory['files'][str(feature.relative_to(Path(c['output'])))] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            for name in ('labels.npz', 'manifest.json'):
                inventory['files'][str((folder/name).relative_to(Path(c['output'])))] = dict(sha256=digest(folder/name))
            first = meta
    file = Path(c['output'])/'observation/cache_audit.json'
    if file.exists() and read(file) != inventory:
        raise ValueError('Cache files changed after observation audit; use a new output')
    write(file, inventory)


def train(c, core):
    import numpy as np
    import torch
    import validate_decline as v
    audit_cache(c, core)
    audit_file = Path(c['output'])/'observation/runtime_position_audit.json'
    if not audit_file.exists():
        v.verify_positions(c, dict(channels=c['channels'], layers=c['layers'], verify_images=3), audit_file.parent, core)
    if not read(audit_file).get('allclose'):
        raise ValueError('Endpoint numerical audit failed')
    s = c['observation']
    for variant in s['variants']:
        for objective in s['objectives']:
            for channel in c['channels']:
                dest = task(c, objective, variant, channel)
                if (dest/'metrics.json').exists():
                    verify_artifacts(dest)
                    print('skip completed', dest, flush=True)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                core.seed_all(variant['seed'])
                model = v.make_readout(c, variant, core)
                opt = torch.optim.Adam(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
                scheduler = torch.optim.lr_scheduler.StepLR(opt, c['scheduler_step'], c['scheduler_gamma'])
                acts, meta = core.cached(p.channel_config(c, channel), 'train', 23)
                bank = eval_bank(c, core, channel, 'val')
                rng = np.random.default_rng(s['sampling_seed']); best = -1.; history = []
                for epoch in range(c['epochs']):
                    model.train(); order = rng.permutation(len(acts))
                    loss_sum = 0.; active_images = 0; steps = 0; supported_images = 0; triples = 0
                    sampling_hash = hashlib.sha256()
                    for start in range(0, len(order), c['batch_size']):
                        rows = order[start:start+c['batch_size']]
                        picks = [rng.choice(c['saved_patches'], c['sampled_patches'], replace=False) for _ in rows]
                        sampling_hash.update(np.asarray(rows, dtype=np.int64).tobytes())
                        sampling_hash.update(np.asarray(picks, dtype=np.int64).tobytes())
                        x = torch.as_tensor(np.stack([acts[r, q] for r, q in zip(rows, picks)]), device=c['device'])
                        logits = core.batch_logits(model, x)
                        losses = []
                        for j, (row, q) in enumerate(zip(rows, picks)):
                            rel = relation_sets(selected(meta, row, q), s['distance_bin_patches'])
                            triples += len(rel['matched']['triples'])
                            supported_images += bool(len(rel['matched']['triples']))
                            if objective == 'matched_anchor':
                                loss = ranking_loss(logits[j], rel)
                            else:
                                ids = torch.as_tensor(meta['instance'][row, q], device=c['device'])
                                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[j], (ids[:, None] == ids[None, :]).float())
                            if loss is not None:
                                losses.append(loss)
                        if losses:
                            loss = torch.stack(losses).mean()
                            if not torch.isfinite(loss):
                                raise ValueError('Non-finite training loss')
                            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                            loss_sum += float(loss.detach())*len(losses); active_images += len(losses); steps += 1
                    if not steps:
                        raise ValueError('No eligible training triples; inspect coverage instead of relaxing bins')
                    scheduler.step()
                    val = evaluate(c, model, bank)
                    score = val['macro_matched_ordering']
                    if score is None:
                        raise ValueError('Validation has no distance-matched relations')
                    history.append(dict(epoch=epoch+1, train_loss=loss_sum/active_images, optimizer_steps=steps,
                        active_image_visits=active_images, matched_supported_image_visits=supported_images,
                        matched_triples=triples, sampling_sha256=sampling_hash.hexdigest(),
                        val_macro_matched_ordering=score, val_supported_images=val['supported_images']))
                    if score > best:
                        best = score
                        torch.save(model.state_dict(), dest/'checkpoint.pt')
                        write(dest/'selection.json', dict(epoch=epoch+1, criterion='val image-macro distance-matched anchor ordering; earliest tie', value=score))
                    print(f'{objective} {variant["name"]} {channel} epoch={epoch+1} val={score:.5f} active={active_images}', flush=True)
                model.load_state_dict(torch.load(dest/'checkpoint.pt', map_location=c['device'], weights_only=True))
                metrics = evaluate(c, model, eval_bank(c, core, channel, 'test'), dest/'test_predictions.npz')
                write(dest/'history.json', history)
                write(dest/'metrics.json', metrics)
                write(dest/'artifacts.json', {f: digest(dest/f) for f in ('checkpoint.pt', 'history.json', 'selection.json', 'test_predictions.npz', 'metrics.json')})


def paired_summary(first, second, field, repeats, seed):
    import numpy as np
    if [r['image_id'] for r in first] != [r['image_id'] for r in second]:
        raise ValueError('Unpaired image records')
    if any((a[field] is None) != (b[field] is None) for a, b in zip(first, second)):
        raise ValueError('Endpoint coverage differs')
    delta = np.array([b[field]-a[field] for a, b in zip(first, second) if a[field] is not None])
    if not len(delta):
        return dict(supported_images=0, delta=None, ci95=None)
    rng = np.random.default_rng(seed); values = []
    for start in range(0, repeats, 256):
        picks = rng.integers(len(delta), size=(min(256, repeats-start), len(delta)))
        values.extend(delta[picks].mean(1).tolist())
    return dict(supported_images=len(delta), delta=float(delta.mean()), ci95=np.quantile(values, [.025, .975]).tolist(),
                improving=int((delta > 0).sum()), declining=int((delta < 0).sum()), tied=int((delta == 0).sum()))


def load_predictions(dest):
    import numpy as np
    verify_artifacts(dest)
    with np.load(dest/'test_predictions.npz', allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def paired_predictions(a, b):
    import numpy as np
    for k in ('patches', 'image_id', 'instance', 'semantic', 'background', 'position'):
        if not np.array_equal(a[k], b[k]):
            raise ValueError(f'Unpaired predictions: {k}')


def report(c):
    import numpy as np
    s = c['observation']; measurements = []; comparisons = []
    # Summaries describe these archived evaluations; source cache is not required.
    for variant in s['variants']:
        records = {}
        for objective in s['objectives']:
            predictions = []
            for channel in c['channels']:
                dest = task(c, objective, variant, channel)
                pred = load_predictions(dest); predictions.append(pred)
                m = read(dest/'metrics.json'); records[objective, channel] = m['per_image']
                measurements.append(dict(variant=variant['name'], objective=objective, channel=channel,
                    matched_macro=m['macro_matched_ordering'], supported_images=m['supported_images'],
                    unmatched_macro=float(np.mean([x['unmatched_ordering'] for x in m['per_image'] if x['unmatched_ordering'] is not None]))
                                    if any(x['unmatched_ordering'] is not None for x in m['per_image']) else None,
                    selection=read(dest/'selection.json')))
            paired_predictions(*predictions)
            z, h = [records[objective, ch] for ch in c['channels']]
            for field in ('matched_ordering', 'unmatched_ordering'):
                comparisons.append(dict(variant=variant['name'], contrast=objective+' h-minus-z', metric=field,
                    **paired_summary(z, h, field, s['bootstrap_repeats'], s['bootstrap_seed'])))
        for channel in c['channels']:
            a = records['all_pairs', channel]; b = records['matched_anchor', channel]
            paired_predictions(load_predictions(task(c, 'all_pairs', variant, channel)), load_predictions(task(c, 'matched_anchor', variant, channel)))
            comparisons.append(dict(variant=variant['name'], contrast=channel+' matched-training-minus-all-pairs', metric='matched_ordering',
                **paired_summary(a, b, 'matched_ordering', s['bootstrap_repeats'], s['bootstrap_seed'])))
        hashes = [[x['sampling_sha256'] for x in read(task(c, obj, variant, ch)/'history.json')]
                  for obj in s['objectives'] for ch in c['channels']]
        if any(h != hashes[0] for h in hashes[1:]):
            raise ValueError('Training sampling differs across objectives/endpoints')
        a = records['all_pairs', 'post_attention']; h = records['all_pairs', 'residual']
        mz = records['matched_anchor', 'post_attention']; mh = records['matched_anchor', 'residual']
        # Direct difference of local effects; shrinking a gap is not itself recovery.
        first = [dict(image_id=x['image_id'], effect=(y['matched_ordering']-x['matched_ordering']) if x['matched_ordering'] is not None else None) for x, y in zip(a, h)]
        second = [dict(image_id=x['image_id'], effect=(y['matched_ordering']-x['matched_ordering']) if x['matched_ordering'] is not None else None) for x, y in zip(mz, mh)]
        comparisons.append(dict(variant=variant['name'], contrast='matched-training local-effect minus all-pairs local-effect', metric='matched_ordering',
            **paired_summary(first, second, 'effect', s['bootstrap_repeats'], s['bootstrap_seed'])))
    coverage = records['all_pairs', 'post_attention']
    result = dict(scope='Exploratory observation; no independent-confirmation claim; CIs condition on fitted probes, no multiplicity correction',
        weighting='Equal supported anchors inside each image, then equal supported images. Distance bins control proximity only.',
        training_note='Matched training uses image-mean softplus(negative-positive), all-pairs uses BCE including diagonal/background. Same batches/epochs/initial seed and selector; active updates can differ, recorded in history.',
        measurements=measurements, comparisons=comparisons,
        coverage=dict(total_images=len(coverage), matched_images=sum(x['matched_ordering'] is not None for x in coverage),
            unmatched_images=sum(x['unmatched_ordering'] is not None for x in coverage),
            matched_triples=sum(x['matched_triples'] for x in coverage), matched_anchors=sum(x['matched_anchors'] for x in coverage),
            supported_classes=sorted({cl for x in coverage for cl in x['classes']}),
            inconsistent_same_id_pairs=sum(x['inconsistent_same_id_pairs'] for x in coverage)),
        distance_baseline={field: float(np.mean([x[field] for x in coverage if x[field] is not None])) if any(x[field] is not None for x in coverage) else None
                           for field in ('distance_baseline_matched', 'distance_baseline_unmatched')})
    write(Path(c['output'])/'observation/report.json', result)
    print(json.dumps(dict(coverage=result['coverage'], comparisons=comparisons), ensure_ascii=False, indent=2))
    return result


def visualize(c, core):
    """Display actual crops and only saved token scores; no interpolated dense map."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from types import SimpleNamespace as NS
    s = c['observation']; ds = core.Dataset(c)
    cls = core.load_author_classes(core.UPSTREAM/'src/data_extractor.py', ['ImageProcessor'])['ImageProcessor']
    proc = cls(NS(model=NS(patch_size=14), data_extractor=NS(image_processor=NS(mean=[.485,.456,.406], std=[.229,.224,.225], shortest_edge=518, crop_size=[518,518], normalize=False))))
    catalogue = []; support = []
    for variant in s['variants']:
        for objective in s['objectives']:
            a, b = [load_predictions(task(c, objective, variant, ch)) for ch in c['channels']]
            paired_predictions(a, b)
            candidates = {'win_to_loss': [], 'loss_to_win': [], 'stable': []}
            for row, image_id in enumerate(a['image_id']):
                meta = {k: a[k][row] for k in ('instance', 'semantic', 'background', 'position')}
                rel = relation_sets(meta, s['distance_bin_patches']); t = rel['matched']['triples']
                if len(t) < s['case_min_triples']:
                    continue
                d = [pred['score'][row, t[:, 0], t[:, 1]]-pred['score'][row, t[:, 0], t[:, 2]] for pred in (a, b)]
                masks = {'win_to_loss': (d[0] > 0) & (d[1] < 0), 'loss_to_win': (d[0] < 0) & (d[1] > 0),
                         'stable': ((d[0] > 0) & (d[1] > 0)) | ((d[0] < 0) & (d[1] < 0))}
                for group, mask in masks.items():
                    if mask.any():
                        candidates[group].append((row, t[mask]))
            rng = np.random.default_rng(s['case_seed'])
            for group, pool in candidates.items():
                support.append(dict(objective=objective, variant=variant['name'], group=group,
                                    eligible_images=len(pool), requested=s['cases_per_group'], exported=min(len(pool), s['cases_per_group'])))
                for choice in rng.permutation(len(pool))[:s['cases_per_group']]:
                    row, ts = pool[choice]; triple = ts[rng.integers(len(ts))]; anchor, positive, negative = map(int, triple)
                    image_id = int(a['image_id'][row]); pos = a['position'][row]; crop = int(pos[anchor]//1369)
                    rgb, sem, inst, bg = ds.read(image_id)
                    crops = proc.forward(rgb).numpy()
                    imask = proc.forward_patch(inst[:, :, None]).numpy()[crop, :, :, 0]
                    smask = proc.forward_patch(sem[:, :, None]).numpy()[crop, :, :, 0]
                    bmask = proc.forward_patch(bg[:, :, None]).numpy()[crop, :, :, 0].astype(bool)
                    local = pos % 1369; same = pos//1369 == crop
                    if not (np.array_equal(imask.reshape(-1)[local[same]], a['instance'][row, same]) and
                            np.array_equal(smask.reshape(-1)[local[same]], a['semantic'][row, same]) and
                            np.array_equal(bmask.reshape(-1)[local[same]], a['background'][row, same])):
                        raise ValueError('Visualized raw labels differ from cached labels')
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
                    axes[0].imshow(crops[crop].clip(0, 1)); axes[0].set_title(f'{image_id} crop {crop}')
                    target = np.where((imask == a['instance'][row, anchor]) & (smask == a['semantic'][row, anchor]) & ~bmask, 1,
                        np.where((imask == a['instance'][row, negative]) & (smask == a['semantic'][row, anchor]) & ~bmask, 2, 0))
                    axes[1].imshow(target, extent=(0,518,518,0), interpolation='nearest', vmin=0, vmax=2, cmap='viridis'); axes[1].set_title('GT: own / other instance')
                    for ax, pred, label in zip(axes[2:], (a, b), ('Before MLP', 'After MLP')):
                        score = pred['score'][row, anchor, same]
                        ranks = np.array([(np.sum(score < x)+.5*np.sum(score == x))/len(score) for x in score])
                        ax.imshow(crops[crop].clip(0, 1))
                        ax.scatter((local[same] % 37)*14+7, (local[same]//37)*14+7, c=ranks, cmap='magma', vmin=0, vmax=1, s=30)
                        ax.set_title(label+' (sampled score rank)')
                    for ax in axes:
                        for index, marker, color in [(anchor, '*', 'cyan'), (positive, '+', 'lime'), (negative, 'x', 'red')]:
                            ax.scatter((local[index] % 37)*14+7, (local[index]//37)*14+7, marker=marker, c=color, s=100)
                        ax.set_xlim(0,518); ax.set_ylim(518,0); ax.axis('off')
                    folder = Path(c['output'])/'observation/cases'/objective/variant['name']/group/str(image_id)
                    folder.mkdir(parents=True, exist_ok=True); fig.savefig(folder/'case.png', dpi=150); plt.close(fig)
                    record = dict(image_id=image_id, crop=crop, group=group, objective=objective, variant=variant['name'],
                        selected_saved_indices=[anchor, positive, negative], patch_positions=pos[triple].tolist(),
                        before_scores=a['score'][row, anchor, [positive, negative]].tolist(),
                        after_scores=b['score'][row, anchor, [positive, negative]].tolist(),
                        distances=[float(rel['distance'][anchor, positive]), float(rel['distance'][anchor, negative])],
                        selection='Uniform eligible image then uniform qualifying triple; post hoc examples, not representative prevalence.',
                        display='Only sampled tokens; within-crop score ranks, not calibrated probabilities.', path=str(folder/'case.png'))
                    write(folder/'case.json', record); catalogue.append(record)
    write(Path(c['output'])/'observation/cases/index.json', dict(cases=catalogue, coverage=support))
    print(f'Exported {len(catalogue)} observation cases', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(p.HERE/'config-observation.json'))
    parser.add_argument('--dataset-root'); parser.add_argument('--output'); parser.add_argument('--device')
    parser.add_argument('--splits-file', help='Optional existing image IDs only; features and probes belong to the independent output')
    parser.add_argument('--stage', choices=['plan', 'extract', 'train', 'report', 'visualize', 'all'], default='plan')
    args = parser.parse_args()
    c = p.read_config(args.config, args.dataset_root, args.output)
    if args.device: c['device'] = args.device
    if args.splits_file:
        c['observation_splits_file'] = str(Path(args.splits_file).resolve())
        splits = read(args.splits_file)
        for k in ('train','val','test'): c['max_'+k] = len(splits[k])
    plan = validate(c)
    if args.stage == 'plan':
        print(json.dumps(dict(resources=plan, objectives=c['observation']['objectives'],
            matched_metric='Distance-matched shared-anchor ordering, image macro; fixed bins, no test tuning',
            output=c['output'], old_artifacts_required=False, scope='Exploratory; image IDs may overlap previous runs'), indent=2))
        return
    prepare(c)
    if args.stage == 'report':
        report(c)
        return
    core = p.core_module(c); attach_splits(c, core)
    if args.stage in ('extract','all'):
        p.extract(c, core)
        import validate_decline as v
        v.verify_positions(c, dict(channels=c['channels'], layers=c['layers'], verify_images=3), Path(c['output'])/'observation', core)
    if args.stage in ('train','all'):
        train(c, core)
    if args.stage == 'all':
        report(c)
    if args.stage in ('visualize','all'):
        visualize(c, core)


if __name__ == '__main__':
    main()

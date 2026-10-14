"""Paired image bootstrap of saved semantic class counts; no model or training."""
from pathlib import Path
import json


def record_arrays(records):
    import numpy as np
    if not records: raise ValueError('No semantic per-image records')
    ids=np.array([r['image_id'] for r in records])
    if not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=len(ids): raise ValueError('Invalid or duplicate semantic image IDs')
    support=np.asarray([r['class_support'] for r in records])
    correct=np.asarray([r['class_correct'] for r in records])
    if support.ndim!=2 or support.shape[1]==0 or support.shape!=correct.shape:
        raise ValueError('Semantic class-count dimensions differ')
    if any(not np.issubdtype(a.dtype,np.integer) for a in (support,correct)) or (support<0).any() or (correct<0).any() or (correct>support).any():
        raise ValueError('Semantic class counts must be nonnegative integers, correct <= support')
    for row,r in enumerate(records):
        if int(support[row].sum())!=r['n']: raise ValueError('Semantic n differs from class support')
    return ids,support,correct


def metrics_from_counts(support,correct):
    import numpy as np
    support=np.asarray(support,dtype=float); correct=np.asarray(correct,dtype=float)
    total=support.sum(axis=-1); represented=(support>0).sum(axis=-1)
    accuracy=np.divide(correct.sum(axis=-1),total,out=np.full_like(total,np.nan),where=total>0)
    recall=np.divide(correct,support,out=np.zeros_like(correct),where=support>0)
    macro=np.divide(recall.sum(axis=-1),represented,out=np.full_like(total,np.nan),where=represented>0)
    return accuracy,macro,represented


def paired_semantic_bootstrap(first,second,repeats=2000,seed=27182):
    import numpy as np
    if type(repeats) is not int or repeats<2: raise ValueError('At least two bootstrap replicates required')
    ids,support,ca=record_arrays(first); bids,bsupport,cb=record_arrays(second)
    if set(ids.tolist())!=set(bids.tolist()): raise ValueError('Semantic comparisons use different images')
    lookup={int(value):i for i,value in enumerate(bids)}
    order=[lookup[int(value)] for value in ids]; reordered=not np.array_equal(ids,bids)
    bsupport,cb=bsupport[order],cb[order]
    if not np.array_equal(support,bsupport): raise ValueError('Semantic per-image class support differs; labels or evaluation samples changed')
    fingerprints_checked=0
    for a,b in zip(first,[second[i] for i in order]):
        fa,fb=a.get('evaluation_fingerprint'),b.get('evaluation_fingerprint')
        if fa is not None and fb is not None:
            if fa!=fb: raise ValueError('Semantic evaluation patch/label fingerprint differs')
            fingerprints_checked+=1
    total=support.sum(0); first_correct=ca.sum(0); second_correct=cb.sum(0)
    if not total.sum(): raise ValueError('No known foreground semantic samples')
    exact_a=metrics_from_counts(total,first_correct); exact_b=metrics_from_counts(total,second_correct)
    rng=np.random.default_rng(seed); n=len(ids); samples_a=[]; samples_b=[]; coverage=[]
    for start in range(0,repeats,64):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(64,repeats-start)).astype(float)
        sampled_support=weights@support
        aa=metrics_from_counts(sampled_support,weights@ca); bb=metrics_from_counts(sampled_support,weights@cb)
        samples_a.append(np.stack(aa[:2],axis=1)); samples_b.append(np.stack(bb[:2],axis=1)); coverage.append(aa[2])
    a,b=np.concatenate(samples_a),np.concatenate(samples_b); coverage=np.concatenate(coverage)
    valid=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    if valid.sum()<.95*repeats: raise ValueError('Too many bootstrap resamples lack semantic support')
    result={}
    for index,key in enumerate(('accuracy','macro_recall')):
        delta=b[valid,index]-a[valid,index]
        result[key]={'first_exact':float(exact_a[index]),'second_exact':float(exact_b[index]),
            'delta_exact':float(exact_b[index]-exact_a[index]),
            'first_ci95':np.quantile(a[valid,index],[.025,.975]).tolist(),
            'second_ci95':np.quantile(b[valid,index],[.025,.975]).tolist(),
            'delta_ci95':np.quantile(delta,[.025,.975]).tolist(),'delta_standard_error':float(delta.std(ddof=1))}
    classes=[]; count=int((total>0).sum())
    for cl in np.flatnonzero(total>0):
        ra=float(first_correct[cl]/total[cl]); rb=float(second_correct[cl]/total[cl])
        classes.append({'semantic_label':int(cl)+1,'support':int(total[cl]),'images_with_support':int((support[:,cl]>0).sum()),
            'first_recall':ra,'second_recall':rb,'delta_recall':rb-ra,'contribution_to_macro_delta':(rb-ra)/count})
    classes.sort(key=lambda row:abs(row['contribution_to_macro_delta']),reverse=True)
    return {'estimand':'pooled known-foreground patch accuracy and macro class recall, recomputed from image-cluster counts',
        'images':n,'replicates':repeats,'valid_replicates':int(valid.sum()),'seed':seed,
        'second_reordered_by_image_id':reordered,'evaluation_fingerprint_images_checked':fingerprints_checked,
        'pairing_note':'Image IDs and per-image class support are checked; old records without fingerprints additionally rely on shared extraction/evaluation provenance.',
        'supported_classes_exact':count,'supported_classes_bootstrap_ci95':np.quantile(coverage[valid],[.025,.975]).tolist(),
        'macro_definition':'Average over classes with nonzero support in each resample; rare classes may disappear, coverage is reported.',
        'uncertainty_scope':'Conditional on fitted probes and this test split; excludes training seed uncertainty; exploratory comparisons without multiplicity correction.',
        'metrics':result,'per_class':classes}


def compare_folders(first,second,repeats=2000,seed=27182):
    import numpy as np
    folders=[Path(first),Path(second)]
    records=[json.loads((folder/'per_image.json').read_text()) for folder in folders]
    result=paired_semantic_bootstrap(*records,repeats=repeats,seed=seed)
    for label,folder in zip(('first','second'),folders):
        metrics=json.loads((folder/'metrics.json').read_text())
        for key in ('accuracy','macro_recall'):
            if metrics.get(key) is None or not np.isclose(metrics[key],result['metrics'][key][label+'_exact'],atol=1e-12,rtol=0):
                raise ValueError(f'Saved semantic {key} disagrees with per-image counts: {folder}')
        _,support,correct=record_arrays(json.loads((folder/'per_image.json').read_text()))
        if metrics.get('class_support')!=support.sum(0).tolist() or metrics.get('class_correct')!=correct.sum(0).tolist():
            raise ValueError('Saved aggregate semantic class counts disagree')
    return {'first_source':str(folders[0]),'second_source':str(folders[1]),**result}

"""Distance leakage checks, real ranking gradients, and a tiny eight-task run."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast
import copy
import contextlib
import io
import math
import tempfile
import unittest
import numpy as np
import torch
import observe_binding as o


class Observation(unittest.TestCase):
    def meta(self):
        return dict(position=np.arange(6), instance=np.array([1,1,2,2,3,3]),
                    semantic=np.ones(6, dtype=int), background=np.zeros(6, dtype=bool))

    def test_matching_excludes_shortcut_cross_crop_background_and_cross_class(self):
        m = dict(position=np.array([0,1,3]), instance=np.array([1,1,2]), semantic=np.ones(3), background=np.zeros(3, bool))
        rel = o.relation_sets(m, 2)
        self.assertGreater(len(rel['unmatched']['triples']), 0)
        self.assertEqual(len(rel['matched']['triples']), 0)
        self.assertEqual(o.ordering(-rel['distance'], rel, 'unmatched'), 1)
        m = self.meta(); m['position'][5] += 1369; m['background'][4] = True; m['semantic'][3] = 2
        rel = o.relation_sets(m, 2)
        t = rel['matched']['triples']
        self.assertGreater(len(t), 0)
        for a,p,n in t:
            self.assertEqual(len({a,p,n}), 3)
            self.assertFalse(m['background'][[a,p,n]].any())
            self.assertEqual(len(set(m['semantic'][[a,p,n]])), 1)
            self.assertEqual(len(set(m['position'][[a,p,n]]//1369)), 1)
            self.assertEqual(m['instance'][a], m['instance'][p])
            self.assertNotEqual(m['instance'][a], m['instance'][n])
            self.assertEqual(int(rel['distance'][a,p]//2), int(rel['distance'][a,n]//2))
        self.assertGreater(rel['inconsistent_same_id_pairs'], 0)

    def test_anchor_weighting_ties_and_real_ranking_gradient(self):
        rel = o.relation_sets(self.meta(), 2)
        g = rel['matched']; t = g['triples']
        self.assertAlmostEqual(g['weights'].sum(), 1)
        for anchor in np.unique(t[:,0]):
            self.assertAlmostEqual(g['weights'][t[:,0]==anchor].sum(), 1/g['anchors'])
        self.assertEqual(o.ordering(np.zeros((6,6)), rel, 'matched'), .5)
        score = torch.zeros(6,6, requires_grad=True)
        loss = o.ranking_loss(score, rel); loss.backward()
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
        self.assertTrue((score.grad[rel['positive']] <= 0).all())
        self.assertTrue((score.grad[rel['negative']] >= 0).all())
        improved = np.where(rel['positive'], 2., 0.)
        self.assertEqual(o.ordering(improved, rel, 'matched'), 1.)

    def test_bootstrap_pairing_and_empty_coverage(self):
        a = [dict(image_id=1, metric=.4), dict(image_id=2, metric=None), dict(image_id=3, metric=.8)]
        b = copy.deepcopy(a)
        r = o.paired_summary(a,b,'metric',100,5)
        self.assertEqual(r['delta'], 0); self.assertEqual(r['ci95'], [0,0])
        self.assertEqual(r['supported_images'], 2)
        b[1]['metric'] = .5
        with self.assertRaisesRegex(ValueError, 'coverage'):
            o.paired_summary(a,b,'metric',100,5)
        self.assertIsNone(o.paired_summary(a[1:2],a[1:2],'metric',100,5)['delta'])

    def fixture(self, root):
        c = o.p.read_config(o.p.HERE/'config-observation.json', output=str(root/'run'))
        c.update(embed_dim=8, saved_patches=6, sampled_patches=6, eval_patches=6,
                 epochs=2, batch_size=2, device='cpu', max_train=3, max_val=2, max_test=2)
        c['observation'].update(bootstrap_repeats=40, cases_per_group=1, case_min_triples=1)
        c['observation']['variants'][1]['factor_width'] = 4
        o.prepare(c)
        splits = dict(train=[10,11,12], val=[20,21], test=[30,31]); o.write(Path(c['output'])/'splits.json', splits)
        rng = np.random.default_rng(9)
        for split, ids in splits.items():
            acts = rng.normal(size=(len(ids),6,8)).astype(np.float32)
            meta = {k: np.tile(v, (len(ids),1)) for k,v in self.meta().items()}; meta['image_id'] = np.array(ids)
            for ch in c['channels']:
                folder = Path(o.p.channel_config(c,ch)['output'])/'cache'/split; folder.mkdir(parents=True)
                np.save(folder/'layer_23.npy', acts if ch=='post_attention' else acts+.1)
                np.savez_compressed(folder/'labels.npz', **meta)
                o.write(folder/'manifest.json', dict(signature='tiny',schema=2,complete=True,channel=ch,location=o.p.channel_location(ch)))
        def cached(cfg,split,layer):
            folder = Path(cfg['output'])/'cache'/split
            with np.load(folder/'labels.npz') as z: meta = {k:z[k].copy() for k in z.files}
            return np.load(folder/f'layer_{layer}.npy',mmap_mode='r'),meta
        model_tree = ast.parse((o.p.HERE.parent/'vit-object-binding/src/utils/models.py').read_text())
        nodes = [n for n in model_tree.body if isinstance(n,ast.ClassDef) and n.name in ('QuadraticProbe','QuadraticFixedRankProbe')]
        namespace = dict(torch=torch,nn=torch.nn,math=math)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_probes','exec'),namespace)
        core = NS(cached=cached, Dataset=lambda cfg:NS(splits=splits), cache_signature=lambda cfg,ds:'tiny',
                  seed_all=lambda seed:torch.manual_seed(seed),
                  probe=lambda cfg,mode:namespace['QuadraticProbe'](NS(probe=NS(in_dim=8))),
                  load_author_classes=lambda file,names:{n:namespace[n] for n in names},
                  UPSTREAM=o.p.HERE.parent/'vit-object-binding',
                  batch_logits=lambda model,x:torch.stack([model.forward_pairwise(a,a) for a in x]))
        return c,core

    def test_eight_real_tasks_sampling_reports_restart_and_corruption(self):
        import validate_decline as v
        with tempfile.TemporaryDirectory(dir=o.p.HERE/'tmp') as tmp:
            c,core = self.fixture(Path(tmp))
            def numerical(cfg, settings, out, core):
                o.write(out/'runtime_position_audit.json',dict(allclose=True, synthetic_test=True))
            with patch.object(v,'verify_positions',side_effect=numerical), contextlib.redirect_stdout(io.StringIO()):
                o.train(c,core)
            with contextlib.redirect_stdout(io.StringIO()):
                result = o.report(c)
            self.assertEqual(len(result['measurements']),8)
            self.assertEqual(len(result['comparisons']),14)
            self.assertEqual(result['coverage']['matched_images'],2)
            files = list(Path(c['output']).rglob('checkpoint.pt'))
            hashes = [o.digest(f) for f in files]
            with patch.object(core,'probe',side_effect=AssertionError('completed task retrained')), contextlib.redirect_stdout(io.StringIO()):
                o.train(c,core)
            self.assertEqual(hashes,[o.digest(f) for f in files])
            changed = copy.deepcopy(c); changed['epochs'] += 1
            with self.assertRaisesRegex(ValueError,'protocol changed'):
                o.prepare(changed)
            dest = o.task(c,'all_pairs',c['observation']['variants'][0],'post_attention')
            with (dest/'test_predictions.npz').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                o.report(c)

    def test_visualization_uses_actual_multicrop_labels_and_sparse_scores(self):
        import torchvision
        from torchvision.transforms import functional as TF
        with tempfile.TemporaryDirectory(dir=o.p.HERE/'tmp') as tmp:
            c,core = self.fixture(Path(tmp))
            c['observation']['cases_per_group'] = 2
            tree = ast.parse((core.UPSTREAM/'src/data_extractor.py').read_text(encoding='utf-8'))
            nodes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImageProcessor']
            ns = dict(torch=torch,np=np,math=math,torchvision=torchvision,TF=TF,F=torch.nn.functional)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'real_processor', 'exec'),ns)
            core.load_author_classes = lambda file,names: {'ImageProcessor':ns['ImageProcessor']}
            inst = np.full((518,518),3,dtype=np.int32)
            inst[:,:28] = 1; inst[:,28:56] = 2
            rgb = np.full((518,518,3),128,dtype=np.uint8)
            inst_second = np.full_like(inst,3)
            inst_second[:,:56] = 1; inst_second[:,56:112] = 2
            core.Dataset = lambda cfg:NS(splits=dict(test=[30,31]), read=lambda image_id:(rgb,np.ones_like(inst),inst if image_id==30 else inst_second,np.zeros_like(inst)))
            for variant in c['observation']['variants']:
                for objective in c['observation']['objectives']:
                    for channel in c['channels']:
                        dest = o.task(c,objective,variant,channel); dest.mkdir(parents=True)
                        meta = {k:np.tile(val,(2,1)) for k,val in self.meta().items()}
                        # Different geometries catch reuse of the final image's distance matrix.
                        meta['position'][1] *= 2
                        # Make positive edges win at both endpoints, so stable examples exist.
                        same = meta['instance'][:,:,None] == meta['instance'][:,None,:]
                        score = np.where(same,2.,0.).astype(np.float32)
                        np.savez_compressed(dest/'test_predictions.npz',score=score,patches=np.tile(np.arange(6),(2,1)),image_id=np.array([30,31]),**meta)
                        for name in ('checkpoint.pt','history.json','selection.json','metrics.json'): (dest/name).write_text('{}')
                        o.write(dest/'artifacts.json',{f:o.digest(dest/f) for f in ('checkpoint.pt','history.json','selection.json','test_predictions.npz','metrics.json')})
            with contextlib.redirect_stdout(io.StringIO()): o.visualize(c,core)
            index = o.read(Path(c['output'])/'observation/cases/index.json')
            self.assertEqual(len(index['cases']),8)
            from PIL import Image
            for record in index['cases']:
                self.assertEqual(record['group'],'stable')
                positions = np.asarray(record['patch_positions'])
                xy = np.stack((positions//37, positions%37),axis=1)
                expected = [float(np.linalg.norm(xy[0]-xy[i])) for i in (1,2)]
                np.testing.assert_allclose(record['distances'],expected,rtol=0,atol=1e-12)
                with Image.open(record['path']) as im: self.assertGreater(im.width,1000)

    def test_refuses_old_outputs_and_unpaired_cache(self):
        with tempfile.TemporaryDirectory(dir=o.p.HERE/'tmp') as tmp:
            root = Path(tmp); old = root/'old'; old.mkdir(); (old/'report.json').write_text('{}')
            c = o.p.read_config(o.p.HERE/'config-observation.json',output=str(old))
            with self.assertRaisesRegex(ValueError,'Independent output'):
                o.prepare(c)
            c,core = self.fixture(root)
            file = Path(o.p.channel_config(c,'residual')['output'])/'cache/test/labels.npz'
            with np.load(file) as z: meta = {k:z[k].copy() for k in z.files}
            meta['position'][0,0] += 1; np.savez_compressed(file,**meta)
            with self.assertRaisesRegex(ValueError,'not paired'):
                o.audit_cache(c,core)


if __name__ == '__main__':
    (o.p.HERE/'tmp').mkdir(exist_ok=True)
    unittest.main()

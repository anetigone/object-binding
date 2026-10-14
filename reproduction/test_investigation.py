"""CPU tests for conditional estimands, cache pairing and block-step hooks."""
import ast, copy, hashlib, json, math, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
import investigate_decline as inv
import validate_decline as v


def predictions():
    # Four patches: one positive pair, two same-class negative pairs, three cross-class pairs.
    return dict(image_id=np.array([7]),patches=np.array([[0,1,2,3]]),
        condition=np.array([[1,2,3,2,3,3]],dtype=np.uint8),
        label=np.array([[True,False,False,False,False,False]]),score=np.array([[2.,1.,-1.,1.,-1.,-1.]]))


class Measurements(unittest.TestCase):
    def test_anchor_rankings_are_invariant_to_monotonic_calibration(self):
        a=predictions(); b=copy.deepcopy(a); b['score']=np.exp(a['score'])
        r=inv.anchored_rankings(a,b,np.zeros((1,6),dtype=bool))
        self.assertEqual(r['triples'],2); self.assertEqual(r['delta'],0)
        b['score'][0,0]=0
        r=inv.anchored_rankings(a,b,np.zeros((1,6),dtype=bool))
        self.assertEqual(r['strict_win_to_loss'],2); self.assertEqual(r['delta'],-1)
        self.assertIsNone(inv.anchored_rankings(a,b,np.ones((1,6),dtype=bool))['delta'])

    def test_same_anchor_ties_and_pair_alignment(self):
        a=predictions(); b=copy.deepcopy(a); b['score'][0,:]=1
        self.assertEqual(inv.anchored_rankings(a,b,np.zeros((1,6),bool))['second_ordering'],.5)
        b['patches'][0,0]=2
        with self.assertRaisesRegex(ValueError,'not paired'): inv.anchored_rankings(a,b,np.zeros((1,6),bool))

    def test_competition_is_crop_local_and_counts_instances_not_patches(self):
        meta=dict(position=np.array([[0,1,2,3,4,5,6,7]]),semantic=np.ones((1,8),int),
            instance=np.array([[1,1,2,2,3,3,4,5]]),background=np.zeros((1,8),int))
        count=inv.sampled_competition(meta,np.array([[0,3,4,7]]),4)
        np.testing.assert_array_equal(count,[[2,2,3,3]])

    def test_condition_filters_keep_labels_and_report_empty_groups(self):
        a=predictions(); b=copy.deepcopy(a); b['score']=-a['score']
        geo=dict(distance=np.full((1,6),.1),boundary_max=np.zeros((1,6)),cross_crop=np.zeros((1,6),bool))
        masks=inv.condition_masks(a,geo,np.full((1,4),2),dict(near_distance=.25,boundary_threshold=.01))
        rows={r['condition']:r for r in inv.conditional_metrics(a,b,masks,1)}
        self.assertEqual(rows['hard_instance']['positive'],1)
        self.assertEqual(rows['hard_instance']['negative'],2)
        self.assertEqual(rows['hard_instance']['delta'],-1)
        self.assertIsNone(rows['far']['delta'])

    def test_prediction_labels_are_rechecked_against_cache(self):
        a=predictions(); meta=dict(image_id=np.array([7]),instance=np.array([[1,1,2,3]]),
            semantic=np.array([[1,1,1,2]]),background=np.zeros((1,4),int))
        inv.check_prediction_meta(a,meta)
        meta['semantic'][0,2]=2
        with self.assertRaisesRegex(ValueError,'pair labels'): inv.check_prediction_meta(a,meta)

    def test_independent_semantic_metric_excludes_unsupported_classes(self):
        r=inv.semantic_metrics([0,0,1],[0,1,1],3)
        self.assertAlmostEqual(r['accuracy'],2/3); self.assertEqual(r['macro_recall'],.75)
        self.assertEqual(r['supported_classes'],2)


class BlockSteps(unittest.TestCase):
    def test_hooks_capture_residual_additions_not_attention_branch(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.norm1=torch.nn.LayerNorm(8); self.norm2=torch.nn.LayerNorm(8)
            def forward(self,hidden_states):
                mid=hidden_states+2*self.norm1(hidden_states)
                result=mid+3*self.norm2(mid)
                return (result,)
        block=Block(); capture=inv.p.Capture(NS(encoder=NS(layer=[block])),[0],['block_input','post_attention','residual'])
        picks=np.array([0,2]); capture.indices=picks; x=torch.randn(1,5,8)
        with torch.inference_mode():
            result=block(hidden_states=x)[0]; mid=x+2*block.norm1(x)
        for name,expected in [('block_input',x),('post_attention',mid),('residual',result)]:
            np.testing.assert_allclose(capture.values[name,0],expected[0,picks+1].numpy())
        capture.close(); self.assertFalse(block._forward_pre_hooks); self.assertFalse(block.norm2._forward_pre_hooks)

    def test_missing_norm2_fails_before_installing_hooks(self):
        block=torch.nn.Identity()
        with self.assertRaisesRegex(ValueError,'norm2'): inv.p.Capture(NS(encoder=NS(layer=[block])),[0],['block_input','post_attention'])
        self.assertFalse(block._forward_pre_hooks)


class Pipeline(unittest.TestCase):
    def setUp(self):
        (inv.p.HERE/'tmp').mkdir(exist_ok=True)

    def fixture(self,root):
        c=inv.p.read_config(inv.p.HERE/'config-overnight.json')
        c.update(output=str(root/'run'),dataset_root=str(root/'data'),layers=[0,1],channels=['residual'],
            saved_patches=6,sampled_patches=6,eval_patches=6,embed_dim=8,device='cpu',epochs=2,batch_size=2,
            scheduler_step=1,max_train=3,max_val=2,max_test=2)
        s=inv.read_settings(inv.p.HERE/'config-investigation.json',c)
        s.update(baseline_layers=[0,1],depth_layers=[0,1],step_layers=[1],semantic_num_classes=2,semantic_epochs=2,
            bootstrap_repeats=20,bootstrap_bins=128)
        splits={'train':[0,1,2],'val':[3,4],'test':[5,6]}
        inv.write(Path(c['output'])/'splits.json',splits)
        source=hashlib.sha256((inv.p.HERE.parent/'vit-object-binding/src/data_extractor.py').read_bytes()).hexdigest()
        signature=inv.p.identity(c,splits,source)
        for split,ids in splits.items():
            folder=Path(c['output'])/'channels/residual/cache'/split; folder.mkdir(parents=True)
            n=len(ids); meta=dict(image_id=np.array(ids),position=np.tile(np.arange(6),(n,1)),
                instance=np.tile([1,1,2,2,3,3],(n,1)),semantic=np.tile([1,1,1,1,2,2],(n,1)),background=np.zeros((n,6),int))
            np.savez_compressed(folder/'labels.npz',**meta)
            inv.write(folder/'manifest.json',dict(schema=2,complete=True,location='post_block',channel='residual',
                signature=signature,shape=[n,6,8],layers=[0,1]))
            for layer in c['layers']:
                np.save(folder/f'layer_{layer}.npy',np.random.default_rng(layer+len(ids)).normal(size=(n,6,8)).astype(np.float32))
        tree=ast.parse((inv.p.HERE/'run_server.py').read_text(encoding='utf-8'))
        def cached(cfg,split,layer):
            f=Path(cfg['output'])/'cache'/split
            with np.load(f/'labels.npz') as z: m={k:z[k] for k in z.files}
            return np.load(f/f'layer_{layer}.npy',mmap_mode='r'),m
        from sklearn.metrics import roc_auc_score
        ns=dict(np=np,torch=torch,roc_auc_score=roc_auc_score,cached=cached)
        funcs=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name in ('batch_logits','metrics','eval_probe','semantic_baseline','seed_all')]
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'core_test','exec'),ns)
        tree=ast.parse((inv.p.HERE.parent/'vit-object-binding/src/utils/models.py').read_text())
        cls=[f for f in tree.body if isinstance(f,ast.ClassDef) and f.name=='QuadraticProbe']
        models=dict(torch=torch,nn=torch.nn,math=math)
        exec(compile(ast.Module(body=cls,type_ignores=[]),'probe_test','exec'),models)
        core=NS(**{key:ns[key] for key in ('cached','batch_logits','eval_probe','seed_all')},
            probe=lambda cfg,mode:models['QuadraticProbe'](NS(probe=NS(in_dim=8))))
        return c,s,core

    def test_cache_based_binding_semantic_and_report_without_backbone(self):
        scratch=inv.p.HERE/'tmp'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp); c,s,core=self.fixture(root); out=root/'investigation'
            with patch.object(inv.p,'core_module',return_value=core):
                inv.train_binding(c,s,out); inv.train_semantic(c,s,out); inv.report(c,s,out)
                # Completion markers prevent accidental retraining.
                with patch.object(core,'probe',side_effect=AssertionError('should reuse')):
                    inv.train_binding(c,s,out)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['binding_complete'],2); self.assertEqual(report['semantic_complete'],2)
            self.assertEqual(len(report['paired_comparisons']),1)
            self.assertTrue((out/'measurements.csv').exists())
            self.assertTrue((out/'measurements.png').exists())
            # The plan accounts only for absent layers and keeps existing caches.
            extended=dict(c,layers=[0,1,2])
            result=inv.plan(c,s,extended,out)
            self.assertEqual(len(result['missing_cache_entries']),3)
            self.assertAlmostEqual(result['additional_cache_gib_upper_estimate'],7*6*8*4/2**30)
            c['model_revision']='different'
            with self.assertRaisesRegex(ValueError,'identity'): inv.validate_caches(c)

    def test_conditions_reuse_validated_predictions_and_geometry(self):
        with tempfile.TemporaryDirectory(dir=inv.p.HERE/'tmp') as tmp:
            root=Path(tmp); c,s,core=self.fixture(root)
            model=core.probe(c,'quadratic').eval()
            for layer in s['baseline_layers']:
                folder=inv.files(c,s,'residual',layer); folder.mkdir(parents=True)
                inv.write(folder/'run.json',dict(config=c,layer=layer,sampling_seed=s['sampling_seed'],
                    variant=dict(kind='full',seed=s['probe_seed'],weight_decay=s['probe_weight_decay']),
                    selection='validation instance-vs-same-class AUROC, first epoch wins ties'))
                metrics,_,detail=core.eval_probe(inv.p.channel_config(c,'residual'),model,'test',layer,True)
                v.save_predictions(folder/'test_predictions.npz',detail,np.array([5,6]))
                inv.write(folder/'metrics.json',metrics)
            a,_=inv.baseline(c,s); shape=a['score'].shape
            geo=Path(c['output'])/'validation/test_geometry.npz'; geo.parent.mkdir(exist_ok=True)
            np.savez_compressed(geo,image_id=a['image_id'],patches=a['patches'],
                distance=np.full(shape,.1),area_mean=np.full(shape,.2),boundary_max=np.zeros(shape),cross_crop=np.zeros(shape,bool))
            out=root/'conditions'; inv.conditions(c,s,out)
            self.assertTrue((out/'conditions.csv').exists()); self.assertTrue((out/'anchored_rankings.json').exists())
            plan=inv.plan(c,s,dict(c,layers=[0,1,2]),out)
            self.assertEqual(plan['reusable_validated_binding_tasks'],2)
            self.assertEqual(plan['new_binding_tasks'],1)
            inv.conditions(c,s,out,ci=True)
            self.assertTrue((out/'condition_ci.json').exists())
            # Cache annotations must agree before any report is written.
            with np.load(geo) as z: data={k:z[k].copy() for k in z.files}
            data['cross_crop'][:]=True; np.savez_compressed(geo,**data)
            with self.assertRaisesRegex(ValueError,'crop labels'): inv.conditions(c,s,out)


if __name__=='__main__': unittest.main()

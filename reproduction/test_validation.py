"""Synthetic verification of paired CIs, geometry controls and readout panel."""
import copy, unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import validate_decline as v

def fixture():
    rng = np.random.default_rng(3)
    code = np.tile(np.array([1,1,2,2,3,3],dtype=np.uint8),(10,1))
    score = rng.normal(size=code.shape)
    return {'score':score,'condition':code,'label':code==1,
        'patches':np.tile(np.arange(4),(10,1)),'image_id':np.arange(10)}

class Bootstrap(unittest.TestCase):
    def test_identical_layers_have_zero_paired_interval(self):
        a = fixture()
        result = v.paired_bootstrap(a,copy.deepcopy(a),'instance',repeats=80,bins=256,max_error=1e-8)
        np.testing.assert_allclose(result['delta_ci95'],[0,0])
    def test_binned_auc_matches_explicit_cluster_weights(self):
        a = fixture(); pos,neg,exact,approx = v.image_histograms(a,'instance',256)
        weights = np.array([1,0,3,1,1,2,0,0,1,1])
        mask,y = v.task_arrays(a,'instance')
        expected = roc_auc_score(y[mask],a['score'][mask],sample_weight=np.repeat(weights,6).reshape(10,6)[mask])
        self.assertAlmostEqual(float(v.auc_histogram(weights@pos,weights@neg)),expected)
        self.assertAlmostEqual(exact,approx)
    def test_known_decline(self):
        a = fixture(); b = copy.deepcopy(a)
        a['score'] = np.where(a['condition']==1,2.,-2.)
        b['score'] = -a['score']
        result = v.paired_bootstrap(a,b,'instance',repeats=80,bins=32)
        self.assertEqual(result['delta_exact'],-1.)
        np.testing.assert_allclose(result['delta_ci95'],[-1,-1])
    def test_alignment_and_approximation_guard(self):
        a = fixture(); b = copy.deepcopy(a); b['patches'][0,0] = 2
        with self.assertRaisesRegex(ValueError,'not paired'): v.paired_bootstrap(a,b,'instance',80,32)
        a['score'] = np.where(a['condition']==1,1.,-1.)
        # Distinct high/low scores collapsed into one bin should be rejected.
        with self.assertRaisesRegex(ValueError,'approximation error'):
            v.paired_bootstrap(a,a,'instance',80,1,max_error=.001)

class Geometry(unittest.TestCase):
    def test_pixel_boundary_fraction(self):
        mask = np.ones((28,28),dtype=int)
        self.assertFalse(v.boundary_fraction(mask).any())
        mask[:,14:] = 2
        np.testing.assert_allclose(v.boundary_fraction(mask),np.full((2,2),1/14))
    def test_stratified_scores_use_same_geometry_weights(self):
        a = fixture(); b = copy.deepcopy(a)
        a['score'] = np.where(a['condition']==1,1.,-1.); b['score'] = -a['score']
        g = {'distance':np.full((10,6),.2),'area_mean':np.full((10,6),.1),
            'boundary_max':np.zeros((10,6)),'cross_crop':np.zeros((10,6),dtype=bool)}
        result = v.stratified_auc(a,b,g)
        self.assertEqual(result['overlap_weighted_delta'],-1.)
        self.assertEqual(result['included_positive'],20)
        g['cross_crop'][:] = True
        self.assertIsNone(v.stratified_auc(a,b,g)['overlap_weighted_delta'])

    def test_geometry_reads_exact_mask_area_not_patch_count(self):
        import torch
        mask = np.ones((28,28),dtype=np.int32); mask[:,14:] = 2
        modes = np.array([[1,2,1,2]])
        def forward_patch(image):
            t = torch.from_numpy(image[:,:,0]).reshape(2,14,2,14).permute(0,2,1,3).reshape(2,2,196)
            return t.mode(-1).values[None,:,:,None]
        proc = NS(resize_image=lambda image,mode:image,crop_image=lambda image:[image],forward_patch=forward_patch)
        meta = {'image_id':np.array([7]),'instance':modes,'position':np.array([[0,1,2,3]])}
        ds = NS(read=lambda i:(None,None,mask,None))
        core = NS(Dataset=lambda c:ds,processor=lambda c:proc,cached=lambda *args:(None,meta))
        c = dict(output='unused',crop_size=28,eval_patches=4)
        s = dict(channels=['residual'],layers=[18,23])
        pred = dict(image_id=np.array([7]),patches=np.array([[0,1,2,3]]))
        g = v.patch_geometry(c,s,core,pred)
        np.testing.assert_allclose(g['area_mean'],.5)
        np.testing.assert_allclose(g['boundary_max'],1/14)
        self.assertFalse(g['cross_crop'].any())

class RuntimeAudit(unittest.TestCase):
    def test_recomputed_hooks_and_mismatch_detection(self):
        import torch
        class Block(torch.nn.Module):
            def __init__(self,scale): super().__init__(); self.scale = scale
            def forward(self,x): return x*self.scale
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.encoder = torch.nn.Module(); self.encoder.layer = torch.nn.ModuleList([Block(1.01) for _ in range(24)])
            def forward(self,pixel_values):
                x = torch.arange(40,dtype=torch.float32).reshape(1,5,8)+pixel_values.mean()
                for block in self.encoder.layer: x = block(x)
                return x
        model = Model(); capture = v.p.Capture(model,[18,23],['residual']); capture.indices = np.arange(4)
        model(pixel_values=torch.ones(1,3,2,2)); arrays = {l:capture.values['residual',l][None].copy() for l in (18,23)}; capture.close()
        meta = {'image_id':np.array([7]),'position':np.array([[0,1,2,3]])}
        ds = NS(read=lambda i:(np.ones((2,2,3)),None,None,None))
        proc = NS(forward=lambda rgb:torch.ones(1,2,2,3))
        core = NS(Dataset=lambda c:ds,processor=lambda c:proc,load_backbone=lambda c:model,
            cached=lambda c,split,l:(arrays[l],meta))
        c = dict(output='unused',crop_size=28,device='cpu'); s = dict(channels=['residual'],layers=[18,23],verify_images=1)
        with patch.object(v.p,'core_module',return_value=core),patch.object(v,'write') as saved:
            v.verify_positions(c,s,Path('unused'))
            self.assertTrue(saved.call_args.args[1]['allclose'])
            arrays[23][0,0,0] += 1
            with self.assertRaisesRegex(ValueError,'Cached features differ'): v.verify_positions(c,s,Path('unused'))
        self.assertFalse(model.encoder.layer[18]._forward_hooks)

class Readouts(unittest.TestCase):
    def test_shared_layernorm_and_factor_rank(self):
        import torch
        from types import SimpleNamespace as NS
        import ast, math
        tree = ast.parse((v.p.HERE.parent/'vit-object-binding/src/utils/models.py').read_text())
        body = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('QuadraticProbe','QuadraticFixedRankProbe')]
        namespace = dict(torch=torch,nn=torch.nn,math=math)
        exec(compile(ast.Module(body=body,type_ignores=[]),'author_classes','exec'),namespace)
        core = NS(probe=lambda c,mode:namespace['QuadraticProbe'](NS(probe=NS(in_dim=c['embed_dim']))),
            UPSTREAM=v.p.HERE.parent/'vit-object-binding',load_author_classes=lambda file,names:{n:namespace[n] for n in names})
        c = dict(embed_dim=8,device='cpu'); norm = torch.nn.LayerNorm(8)
        full = v.make_readout(c,{'kind':'full'},core,norm)
        x = torch.randn(4,8)
        torch.testing.assert_close(full.forward_pairwise(x,x),full.readout.forward_pairwise(norm(x),norm(x)))
        self.assertFalse(any(z.requires_grad for z in full.norm.parameters()))
        factor = v.make_readout(c,{'kind':'factor','factor_width':2},core)
        w = (factor.W1.weight.T@factor.W2.weight+factor.W2.weight.T@factor.W1.weight)/2
        self.assertLessEqual(torch.linalg.matrix_rank(w,atol=1e-6).item(),4)
        loss = full.forward_pairwise(x,x).sum(); loss.backward()
        self.assertTrue(all(z.grad is None for z in full.norm.parameters()))

if __name__=='__main__': unittest.main()

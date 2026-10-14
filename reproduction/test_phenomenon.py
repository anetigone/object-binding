"""CPU/synthetic harness checks; no datasets, pretrained weights or CUDA needed."""
import ast, json, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
import phenomenon as p

class Planning(unittest.TestCase):
    def setUp(self): self.c = p.read_config(p.HERE/'config.json')
    def test_budget_and_expansion_identity(self):
        self.assertAlmostEqual(p.resource_plan(self.c)['fp32_cache_gib'],13.671875)
        self.assertEqual(p.resource_plan(self.c)['training_tasks'],5)
        expanded = dict(self.c,layers=[12,18,23],channels=['residual','q','k','v'],seeds=[0,1,2])
        self.assertEqual(p.identity(self.c,{'train':[1],'val':[2],'test':[3]},'hash'),
            p.identity(expanded,{'train':[1],'val':[2],'test':[3]},'hash'))
        with self.assertRaises(ValueError): p.enforce_budget(expanded)
        changed = dict(self.c,saved_patches=128)
        self.assertNotEqual(p.identity(self.c,{},'hash'),p.identity(changed,{},'hash'))
    def test_custom_unknown_model_fails(self):
        c = dict(self.c,model='unsupported/model')
        with patch.object(Path,'read_text',return_value=json.dumps(c)):
            with self.assertRaises(ValueError): p.read_config('unused.json')
    def test_base_dimensions(self):
        c = p.read_config(p.HERE/'config-base.json')
        self.assertEqual((c['embed_dim'],c['num_layers']),(768,12))
        self.assertEqual(p.resource_plan(c)['training_tasks'],5)

class Hooks(unittest.TestCase):
    def test_qkv_and_residual_are_actual_outputs(self):
        import torch
        from torch import nn
        from types import SimpleNamespace as NS
        import numpy as np
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(8)
                self.attention = nn.Module()
                self.attention.attention = nn.Module()
                for name in ('query','key','value'): setattr(self.attention.attention,name,nn.Linear(8,8))
            def forward(self,x):
                a = self.attention.attention
                z = self.norm(x)
                q,k,v = a.query(z),a.key(z),a.value(z)
                return x+q+k+v
        block = Block()
        capture = p.Capture(NS(encoder=NS(layer=[block])),[0],['residual','q','k','v'])
        indices = np.array([0,3]); capture.indices = indices
        x = torch.randn(1,6,8)
        with torch.inference_mode(): out = block(x)
        np.testing.assert_allclose(capture.values['residual',0],out[0,indices+1].numpy())
        with torch.inference_mode():
            for ch,name in [('q','query'),('k','key'),('v','value')]:
                expected = getattr(block.attention.attention,name)(block.norm(x))[0,indices+1].numpy()
                np.testing.assert_allclose(capture.values[ch,0],expected)
        capture.close()
        self.assertFalse(block._forward_hooks)
    def test_qk_symmetry_and_head_layout(self):
        import numpy as np
        q = np.arange(32,dtype=np.float32).reshape(4,8)
        k = q[::-1].copy()
        result = p.qk_scores(q,k,2)
        self.assertEqual(result.shape,(2,4,4))
        np.testing.assert_allclose(result,result.transpose(0,2,1))
        self.assertAlmostEqual(float(result[0,1,2]),float((q[1,:4]@k[2,:4]+q[2,:4]@k[1,:4])/4))

    def test_refactored_direct_projections(self):
        import torch
        from torch import nn
        from types import SimpleNamespace as NS
        import numpy as np
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(8)
                self.attention = nn.Module()
                for name in ('q_proj','k_proj','v_proj'): setattr(self.attention,name,nn.Linear(8,8))
            def forward(self,x):
                z = self.norm1(x)
                return x+self.attention.q_proj(z)+self.attention.k_proj(z)+self.attention.v_proj(z)
        block = Block()
        capture = p.Capture(NS(encoder=NS(layer=[block])),[0],['residual','q','k','v'])
        capture.indices = np.array([0,2])
        x = torch.randn(1,5,8)
        with torch.inference_mode():
            out = block(x)
            for ch in ('q','k','v'):
                expected = getattr(block.attention,ch+'_proj')(block.norm1(x))[0,[1,3]].numpy()
                np.testing.assert_allclose(capture.values[ch,0],expected)
        np.testing.assert_allclose(capture.values['residual',0],out[0,[1,3]].numpy())
        capture.close()

    def test_invalid_layout_leaves_no_hooks(self):
        from torch import nn
        from types import SimpleNamespace as NS
        block = nn.Module(); block.attention = nn.Module()
        with self.assertRaisesRegex(ValueError,'Unsupported DINOv2 QKV layout'):
            p.Capture(NS(encoder=NS(layer=[block])),[0],['residual','q'])
        self.assertFalse(block._forward_hooks)

class Evaluation(unittest.TestCase):
    def test_hard_instance_and_category_tasks_have_correct_labels(self):
        import numpy as np
        import torch
        from types import SimpleNamespace as NS
        from sklearn.metrics import roc_auc_score
        core_path = p.HERE/'run_server.py'
        tree = ast.parse(core_path.read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'metrics','eval_probe','semantic_baseline'}]
        # Two patches of object 1, one of object 2 (same class), one object 3 (different class).
        meta = {'image_id':np.array([7]),'instance':np.array([[1,1,2,3]]),
            'semantic':np.array([[2,2,2,3]]),'background':np.zeros((1,4),dtype=int)}
        acts = np.eye(4,dtype=np.float32)[None]
        cfg = dict(saved_patches=4,eval_patches=4,split_seed=0,device='cpu')
        scores = torch.tensor([[2.,2.,-1.,-2.],[2.,2.,-1.,-2.],[-1.,-1.,2.,-2.],[-2.,-2.,-2.,2.]])
        ns = dict(np=np,torch=torch,roc_auc_score=roc_auc_score,cached=lambda *args:(acts,meta))
        exec(compile(ast.Module(body=functions,type_ignores=[]),'evaluation', 'exec'),ns)
        model = NS(forward_pairwise=lambda x,y:x@scores@y.T)
        result,records,detail = ns['eval_probe'](cfg,model,'test',0,True)
        self.assertEqual(result['instance_vs_same_class']['n'],3)
        self.assertEqual(result['instance_vs_same_class']['auroc'],1)
        self.assertEqual(result['class_grouping_cross_instance']['n'],5)
        self.assertEqual(result['class_grouping_cross_instance']['auroc'],1)
        self.assertEqual(records[0]['foreground']['positive'],1)
        self.assertEqual(sorted(detail[0][3].tolist()),[1,2,2,3,3,3])

if __name__=='__main__': unittest.main()

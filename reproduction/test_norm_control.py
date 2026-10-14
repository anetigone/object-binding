"""CPU checks of tokenwise norm matching and a cache-only control pipeline."""
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import copy, io, json, tempfile, unittest
import numpy as np
import investigate_decline as inv
import norm_control as n
import test_mlp_update as fixtures


class NormMatching(unittest.TestCase):
    def test_preserves_h_direction_and_matches_each_token_not_whole_image(self):
        z=np.array([[[3.,4.],[1.,0.]],[[0.,2.],[0.,4.]]],dtype=np.float32)
        h=np.array([[[0.,10.],[0.,3.]],[[2.,0.],[3.,4.]]],dtype=np.float32)
        original_z,original_h=z.copy(),h.copy(); result=n.norm_match(z,h,1e-12)
        np.testing.assert_allclose(np.linalg.norm(result,axis=-1),np.linalg.norm(z,axis=-1),rtol=1e-6)
        normalized=lambda a:a/np.linalg.norm(a,axis=-1,keepdims=True)
        np.testing.assert_allclose(normalized(result),normalized(h),rtol=1e-6)
        np.testing.assert_array_equal(result[0,0],[0,5])
        np.testing.assert_array_equal(z,original_z); np.testing.assert_array_equal(h,original_h)
        self.assertEqual(result.dtype,np.float32)

    def test_noop_when_token_norms_match_and_lazy_slice(self):
        z=np.arange(96,dtype=np.float32).reshape(2,6,8)+1
        bank=n.NormMatchedFeatures(z,z,1e-12)
        self.assertEqual(bank.shape,z.shape); self.assertEqual(len(bank),2)
        np.testing.assert_array_equal(bank[1,[0,4]],z[1,[0,4]])
        with self.assertRaisesRegex(ValueError,'full feature dimension'): bank[1,[0,4],:3]

    def test_zero_nonfinite_and_incompatible_inputs(self):
        zero=np.zeros((2,8),dtype=np.float32)
        np.testing.assert_array_equal(n.norm_match(zero,zero,1e-12),zero)
        with self.assertRaisesRegex(ValueError,'Near-zero'): n.norm_match(np.ones_like(zero),zero,1e-12)
        bad=zero.copy(); bad[0,0]=np.inf
        with self.assertRaisesRegex(ValueError,'Non-finite'): n.norm_match(zero,bad,1e-12)
        with self.assertRaisesRegex(ValueError,'matching'): n.norm_match(zero,zero.astype(np.float64),1e-12)

    def test_reject_invalid_epsilon_layer_and_budget(self):
        c=inv.p.read_config(inv.p.HERE/'config-overnight.json')
        settings=json.loads((inv.p.HERE/'config-norm-control.json').read_text())
        for changes in (dict(epsilon=0),dict(epsilon=-1),dict(epsilon=True),dict(layer=24),dict(max_new_binding_tasks=0)):
            with patch.object(Path,'read_text',return_value=json.dumps(dict(settings,**changes))):
                with self.assertRaises(ValueError): n.read_settings('unused',c)


class CachePipeline(unittest.TestCase):
    def setUp(self): (inv.p.HERE/'tmp').mkdir(exist_ok=True)

    def test_control_resume_audit_and_ci_preserve_every_source_file(self):
        with tempfile.TemporaryDirectory(dir=inv.p.HERE/'tmp') as tmp:
            c,s,core,_,root=fixtures.CachedPipeline().fixture(Path(tmp))
            settings=n.read_settings(inv.p.HERE/'config-norm-control.json',c)
            settings.update(layer=1,semantic_bootstrap_repeats=30)
            out=root/'norm_control'
            source_files={str(file):(file.stat().st_size,file.stat().st_mtime_ns) for file in Path(c['output']).rglob('*') if file.is_file()}
            cfgfile=Path(tmp)/'norm.json'; inv.write(cfgfile,settings)
            argv=['norm_control.py','--settings',str(cfgfile),'--stage','all']
            with patch.object(inv.p,'read_config',return_value=c),patch.object(inv.p,'core_module',return_value=core),patch('sys.argv',argv):
                n.main()
            report=json.loads((out/'report.json').read_text()); audit=json.loads((out/'norm_audit.json').read_text())
            self.assertEqual(report['binding_complete'],3); self.assertEqual(report['semantic_complete'],3)
            self.assertEqual(len(report['paired_comparisons']),3)
            self.assertEqual([row['endpoint_reused'] for row in report['measurements']],[True,True,False])
            self.assertEqual(report['new_feature_cache_gib'],0); self.assertEqual(report['backbone_forward_images'],0)
            self.assertEqual(audit['test_images'],2)
            self.assertLess(audit['groups']['all']['relative_norm_error']['quantiles'][-1],2e-6)
            self.assertGreater(audit['groups']['all']['h_cosine_matched']['quantiles'][0],.999999)
            for file,(size,mtime) in source_files.items():
                self.assertEqual((Path(file).stat().st_size,Path(file).stat().st_mtime_ns),(size,mtime))
            self.assertFalse(list(out.rglob('*.npy'))); self.assertTrue((out/'measurements.png').exists())
            with patch.object(inv.p,'read_config',return_value=c),patch.object(inv.p,'core_module',return_value=core),patch('sys.argv',argv),patch.object(core,'probe',side_effect=AssertionError('already complete')):
                n.main()
            captured=io.StringIO()
            with patch.object(inv.p,'read_config',return_value=c),patch('sys.argv',argv[:-1]+['plan']),redirect_stdout(captured): n.main()
            plan=json.loads(captured.getvalue())
            self.assertEqual(plan['pending_binding_tasks'],0); self.assertEqual(plan['pending_semantic_tasks'],0)
            # Independent output protects the old alpha experiment and endpoints.
            with patch.object(inv.p,'read_config',return_value=c),patch('sys.argv',argv+['--output',str(root/'mlp_update')]):
                with self.assertRaisesRegex(ValueError,'separate'): n.main()
            bad=copy.deepcopy(settings); bad['epsilon']=1e-9; inv.write(cfgfile,bad)
            with patch.object(inv.p,'read_config',return_value=c),patch('sys.argv',argv):
                with self.assertRaisesRegex(ValueError,'source/settings changed'): n.main()


if __name__=='__main__': unittest.main()

"""Synthetic CPU tests for paired semantic CIs and lazy MLP interpolation."""
import copy, io, json, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mlp_update as m
import semantic_uncertainty as u
import investigate_decline as inv
import test_investigation as fixtures


def records():
    return [dict(image_id=7,n=100,class_support=[100,0],class_correct=[100,0]),
        dict(image_id=8,n=1,class_support=[0,1],class_correct=[0,0])]


class SemanticBootstrap(unittest.TestCase):
    def test_recompute_pooled_accuracy_not_mean_image_accuracy(self):
        a=records(); b=copy.deepcopy(a); b[0]['class_correct']=[0,0]; b[1]['class_correct']=[0,1]
        r=u.paired_semantic_bootstrap(a,b,repeats=100,seed=3)
        self.assertAlmostEqual(r['metrics']['accuracy']['delta_exact'],-99/101)
        self.assertEqual(r['metrics']['macro_recall']['delta_exact'],0)
        self.assertEqual(r['supported_classes_exact'],2)
        self.assertEqual(r['supported_classes_bootstrap_ci95'],[1,2])
        self.assertAlmostEqual(sum(cl['contribution_to_macro_delta'] for cl in r['per_class']),0)

    def test_identical_results_zero_interval_and_explicit_image_alignment(self):
        a=records(); r=u.paired_semantic_bootstrap(a,list(reversed(a)),100)
        self.assertTrue(r['second_reordered_by_image_id'])
        for metric in r['metrics'].values():
            self.assertEqual(metric['delta_exact'],0)
            np.testing.assert_array_equal(metric['delta_ci95'],[0,0])

    def test_pairing_and_class_count_guards(self):
        a=records(); b=copy.deepcopy(a); b[0]['class_support']=[99,1]; b[0]['class_correct']=[99,0]
        with self.assertRaisesRegex(ValueError,'support differs'): u.paired_semantic_bootstrap(a,b,20)
        b=copy.deepcopy(a); b[1]['image_id']=7
        with self.assertRaisesRegex(ValueError,'duplicate'): u.paired_semantic_bootstrap(a,b,20)
        b=copy.deepcopy(a); b[0]['class_correct']=[101,0]
        with self.assertRaisesRegex(ValueError,'correct <= support'): u.paired_semantic_bootstrap(a,b,20)
        for row in a: row['evaluation_fingerprint']='first'
        b=copy.deepcopy(a); b[1]['evaluation_fingerprint']='changed'
        with self.assertRaisesRegex(ValueError,'fingerprint'): u.paired_semantic_bootstrap(a,b,20)


class LazyInterpolation(unittest.TestCase):
    def test_materialize_only_requested_slice_and_preserve_endpoints(self):
        z=np.arange(96,dtype=np.float32).reshape(2,6,8); h=z*3+1
        np.testing.assert_array_equal(m.BlendedFeatures(z,h,0)[1,[0,4]],z[1,[0,4]])
        np.testing.assert_array_equal(m.BlendedFeatures(z,h,1)[1,[0,4]],h[1,[0,4]])
        blend=m.BlendedFeatures(z,h,.5)
        self.assertEqual(len(blend),2); self.assertEqual(blend.shape,(2,6,8))
        np.testing.assert_allclose(blend[1,[0,4]],(z[1,[0,4]]+h[1,[0,4]])/2)
        self.assertEqual(blend[0].dtype,np.float32)

    def test_budget_and_alpha_validation(self):
        c=inv.p.read_config(inv.p.HERE/'config-overnight.json')
        base=json.loads((inv.p.HERE/'config-mlp-update.json').read_text())
        for changes in (dict(alphas=[0,.25,.5,1]),dict(alphas=[0,1.1]),dict(alphas=[0,True]),dict(alphas=[0,.5,.5,1])):
            with patch.object(Path,'read_text',return_value=json.dumps(dict(base,**changes))):
                with self.assertRaises(ValueError): m.read_settings('unused',c)


class CachedPipeline(unittest.TestCase):
    def setUp(self): (inv.p.HERE/'tmp').mkdir(exist_ok=True)

    def fixture(self,root):
        c,s,core=fixtures.Pipeline().fixture(root)
        investigation=Path(c['output'])/'investigation'; stepout=investigation/'steps'
        inv.write(stepout/'requested_settings.json',dict(base_config=c,settings=s,scope='steps'))
        for split in ('train','val','test'):
            residual=Path(c['output'])/'channels/residual/cache'/split
            for channel,source_layer in [('post_attention',1),('block_input',0)]:
                folder=Path(c['output'])/'channels'/channel/'cache'/split; folder.mkdir(parents=True)
                (folder/'labels.npz').write_bytes((residual/'labels.npz').read_bytes())
                manifest=json.loads((residual/'manifest.json').read_text())
                manifest.update(channel=channel,location=p_location(channel),layers=[1])
                inv.write(folder/'manifest.json',manifest)
                features=np.load(residual/f'layer_{source_layer}.npy')
                if channel=='post_attention': features=features*.8+np.float32(.1)
                np.save(folder/'layer_1.npy',features)
        cfg=dict(c,layers=[1],channels=['block_input','post_attention','residual'])
        inv.train_binding(cfg,s,stepout,core=core)
        inv.train_semantic(cfg,s,stepout,core=core)
        settings=m.read_settings(inv.p.HERE/'config-mlp-update.json',c)
        settings.update(layer=1,semantic_bootstrap_repeats=30)
        return c,s,core,settings,investigation

    def test_end_to_end_midpoint_reuses_endpoints_and_never_writes_features(self):
        with tempfile.TemporaryDirectory(dir=inv.p.HERE/'tmp') as tmp:
            c,s,core,settings,root=self.fixture(Path(tmp)); out=root/'mlp_update'
            out.mkdir(); before={str(f):f.stat().st_size for f in Path(c['output']).glob('channels/*/cache/*/layer_*.npy')}
            m.check_endpoints(c,s,settings,root)
            with patch.object(inv.p,'core_module',return_value=core):
                m.train(c,s,settings,root,out,'binding'); m.train(c,s,settings,root,out,'semantic')
                # Resume must not construct a binding probe or overwrite old results.
                with patch.object(core,'probe',side_effect=AssertionError('should skip completed point')):
                    m.train(c,s,settings,root,out,'binding')
            m.semantic_ci(c,settings,root,out,'steps'); m.report(c,s,settings,root,out)
            inv.write(root/'depth/requested_settings.json',dict(base_config=c,settings=s,scope='depth'))
            for channel,layer in [('block_input',0),('residual',1)]:
                source=root/'steps/semantic'/channel/'layer_1'; dest=root/'depth/semantic/residual'/f'layer_{layer}'
                dest.mkdir(parents=True)
                for name in ('metrics.json','per_image.json','selection.json'): (dest/name).write_bytes((source/name).read_bytes())
            m.semantic_ci(c,settings,root,out,'both')
            ci=json.loads((out/'semantic_ci_both.json').read_text())
            self.assertEqual(len(ci['comparisons']),3)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['binding_complete'],3); self.assertEqual(report['semantic_complete'],3)
            self.assertEqual(len(report['paired_comparisons']),3)
            self.assertEqual(report['new_feature_cache_gib'],0)
            self.assertEqual([row['endpoint_reused'] for row in report['measurements']],[True,False,True])
            self.assertTrue((out/'semantic_ci_steps.csv').exists()); self.assertTrue((out/'measurements.png').exists())
            self.assertEqual(before,{str(f):f.stat().st_size for f in Path(c['output']).glob('channels/*/cache/*/layer_*.npy')})
            self.assertFalse(list(out.rglob('*.npy')))
            # Exercise the CLI provenance and completed-task resume path too.
            settings_file=Path(tmp)/'update.json'; inv.write(settings_file,settings)
            argv=['mlp_update.py','--settings',str(settings_file),'--stage','all']
            with patch.object(inv.p,'read_config',return_value=c),patch.object(inv.p,'core_module',return_value=core),patch('sys.argv',argv):
                m.main()
            self.assertTrue((out/'requested_settings.json').exists()); self.assertTrue((out/'code_provenance.json').exists())
            cli_output=io.StringIO()
            with patch.object(inv.p,'read_config',return_value=c),patch('sys.argv',argv[:-1]+['plan']),redirect_stdout(cli_output):
                m.main()
            plan=json.loads(cli_output.getvalue())
            self.assertEqual(plan['pending_binding_tasks'],0); self.assertEqual(plan['pending_semantic_tasks'],0)
            provider=m.BlendProvider(c,settings)
            try:
                with self.assertRaisesRegex(ValueError,'Unexpected'): provider.cached(dict(channel='residual'),'test',1)
            finally: provider.close()

    def test_source_settings_reject_incompatible_training_config(self):
        with tempfile.TemporaryDirectory(dir=inv.p.HERE/'tmp') as tmp:
            c,s,core=fixtures.Pipeline().fixture(Path(tmp)); root=Path(tmp)/'investigation'
            inv.write(root/'steps/requested_settings.json',dict(base_config=c,settings=s,scope='steps'))
            self.assertEqual(m.source_settings(c,root,'steps'),s)
            changed=dict(c,epochs=c['epochs']+1)
            with self.assertRaisesRegex(ValueError,'config differs'): m.source_settings(changed,root,'steps')


def p_location(channel): return inv.p.channel_location(channel)


if __name__=='__main__': unittest.main()

"""Small CPU end-to-end check, source mismatch and resume protections."""
import ast, copy, json, math, shutil, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
import local_readout as local
import investigate_decline as inv
import validate_decline as v
import test_investigation as fixtures


class LocalReadout(unittest.TestCase):
    def fixture(self, root):
        c, _, core = fixtures.Pipeline().fixture(root)
        s = local.read_settings(local.p.HERE / 'config-local-readout.json', c)
        s.update(layer=1, residual_source='validation/readouts/residual/factor64/layer_1',
            bootstrap_repeats=20, bootstrap_bins=128)
        # Keep the actual author factor class, at tiny width/dimension for CPU.
        tree = ast.parse((local.p.HERE.parent / 'vit-object-binding/src/utils/models.py').read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'QuadraticFixedRankProbe']
        namespace = dict(torch=torch, nn=torch.nn, math=math)
        exec(compile(ast.Module(body=classes, type_ignores=[]), 'factor_fixture', 'exec'), namespace)
        core.UPSTREAM = local.p.HERE.parent / 'vit-object-binding'
        core.load_author_classes = lambda file, names: namespace
        core.probe = lambda cfg, mode: v.make_readout(cfg, s['variant'], core)
        for split in ('train', 'val', 'test'):
            source = Path(local.p.channel_config(c, 'residual')['output']) / 'cache' / split
            target = Path(local.p.channel_config(c, 'post_attention')['output']) / 'cache' / split
            shutil.copytree(source, target)
            manifest = json.loads((target / 'manifest.json').read_text())
            manifest.update(channel='post_attention', location=local.p.channel_location('post_attention'))
            inv.write(target / 'manifest.json', manifest)
        # Produce a fitted after-MLP endpoint using the same real training loop.
        seed = dict(baseline_layers=[], probe_seed=0, probe_weight_decay=0., sampling_seed=0)
        scratch = root / 'baseline'
        inv.train_binding(dict(c, channels=['residual'], layers=[1]), seed, scratch, core=core)
        source = local.source_folder(c, s)
        source.parent.mkdir(parents=True)
        shutil.move(str(scratch / 'binding/residual/layer_1'), source)
        inv.write(source / 'run.json', dict(config=c, variant=s['variant'], layer=1,
            sampling_seed=0, selection=local.SELECTION))
        return c, s, core, Path(c['output']) / 'investigation/local_readout'

    def test_train_report_and_resume_without_backbone_or_new_cache(self):
        (local.p.HERE / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local.p.HERE / 'tmp') as tmp:
            c, s, core, out = self.fixture(Path(tmp))
            source_hashes = {name: local.digest(local.source_folder(c, s) / name) for name in local.SOURCE_FILES}
            before = {str(file): local.digest(file) for file in Path(c['output']).glob('channels/*/cache/*/*')}
            plan = local.plan(c, s, out)
            self.assertEqual(plan['new_binding_tasks'], 1)
            self.assertFalse(plan['missing_cache_files']); self.assertFalse(plan['missing_source_files'])
            self.assertFalse(out.exists())
            with patch.object(local.p, 'core_module', return_value=core) as load:
                local.train(c, s, out)
                result = local.report(c, s, out)
                local.train(c, s, out)
                self.assertEqual(load.call_count, 1)
            self.assertEqual(local.plan(c, s, out)['new_binding_tasks'], 0)
            self.assertEqual(result['binding']['images'], 2)
            self.assertEqual(result['binding']['delta_exact'], 0.)
            self.assertGreater(result['anchored_rankings']['triples'], 0)
            self.assertEqual(result['assessment'], 'ci_within_small_difference_band')
            self.assertEqual(source_hashes, {name: local.digest(local.source_folder(c, s) / name) for name in local.SOURCE_FILES})
            self.assertEqual(before, {str(file): local.digest(file) for file in Path(c['output']).glob('channels/*/cache/*/*')})
            changed = copy.deepcopy(s); changed['minimum_effect'] = .01
            with self.assertRaisesRegex(ValueError, 'inputs/code/settings changed'):
                local.prepare(c, changed, out)
            file = local.new_folder(s, out) / 'run.json'
            identity = json.loads(file.read_text()); identity['variant']['factor_width'] = 128
            inv.write(file, identity)
            with self.assertRaisesRegex(ValueError, 'Post-attention readout identity differs'):
                local.report(c, s, out)

    def test_mismatched_source_and_predictions_fail_before_training(self):
        with tempfile.TemporaryDirectory(dir=local.p.HERE / 'tmp') as tmp:
            c, s, core, out = self.fixture(Path(tmp))
            changed = copy.deepcopy(s); changed['variant']['factor_width'] = 128
            with patch.object(local.p, 'core_module', side_effect=AssertionError('must fail before loading')):
                with self.assertRaisesRegex(ValueError, 'variant/layer/sampling/selection differs'):
                    local.train(c, changed, out)
            self.assertFalse(out.exists())
            file = local.source_folder(c, s) / 'test_predictions.npz'
            prediction = v.predictions(file); prediction['condition'][0, 0] = 0
            np.savez_compressed(file, **prediction)
            with self.assertRaisesRegex(ValueError, 'pair labels'):
                local.train(c, s, out)
            self.assertFalse(out.exists())

    def test_ci_crossing_zero_is_not_small_difference_evidence(self):
        self.assertEqual(local.assessment([-.03, .01], .005), 'inconclusive_at_prespecified_effect')
        self.assertEqual(local.assessment([-.03, -.01], .005), 'meaningful_decline_supported')
        self.assertEqual(local.assessment([-.003, .004], .005), 'ci_within_small_difference_band')


if __name__ == '__main__': unittest.main()

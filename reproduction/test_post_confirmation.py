"""Real tiny probe integration and source/resume guards for cached follow-up."""
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import ast
import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
import numpy as np
import torch
import post_confirmation as m
import confirm_l23 as f
import investigate_decline as inv
import phenomenon as p
import test_investigation as fixtures


class Followup(unittest.TestCase):
    def setUp(self):
        (p.HERE / 'tmp').mkdir(exist_ok=True)

    def fixture(self, root, distinct=False):
        c, _, core = fixtures.Pipeline().fixture(root)
        c.update(layers=[23], channels=list(m.CHANNELS))
        run = Path(c['output'])
        splits = f.read_json(run / 'splits.json')
        source_hash = f.digest(p.HERE.parent / 'vit-object-binding/src/data_extractor.py')
        signature = p.identity(c, splits, source_hash)
        for ch in m.CHANNELS:
            for split in f.SPLITS:
                original = run / 'channels/residual/cache' / split
                dest = run / 'channels' / ch / 'cache' / split
                dest.mkdir(parents=True, exist_ok=True)
                labels = (original / 'labels.npz').read_bytes()
                (dest / 'labels.npz').write_bytes(labels)
                # Identical endpoints make all four cells exactly agree; any
                # nonzero delta exposes an evaluation or checkpoint pairing bug.
                values = np.load(original / 'layer_1.npy')
                if distinct and ch == 'residual':
                    values = values * np.float32(.7) + np.float32(.2)
                np.save(dest / 'layer_23.npy', values)
                manifest = f.read_json(original / 'manifest.json')
                manifest.update(channel=ch, location=p.channel_location(ch), layers=[23], signature=signature)
                f.write(dest / 'manifest.json', manifest)
        tree = ast.parse((p.HERE.parent / 'vit-object-binding/src/utils/models.py').read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QuadraticFixedRankProbe']
        ns = dict(torch=torch, nn=torch.nn, math=math)
        exec(compile(ast.Module(body=classes, type_ignores=[]), 'factor_fixture', 'exec'), ns)
        core.UPSTREAM = p.HERE.parent / 'vit-object-binding'
        core.load_author_classes = lambda file, names: ns
        checks = f.read_json(p.HERE / 'config-confirmation-checks.json')
        checks.update(bootstrap_repeats=20, bootstrap_bins=128)
        frozen = dict(config=c, checks=checks, splits_sha256=f.digest(run / 'splits.json'),
            implementation_and_index_sha256={str(p.HERE / 'run_server.py'): f.digest(p.HERE / 'run_server.py')})
        f.write(run / 'requested_config.json', c)
        f.write(run / 'frozen_protocol.json', frozen)
        f.write(run / 'confirmation/runtime_position_audit.json', dict(allclose=True, comparisons=[dict(allclose=True)]))
        f.train(c, checks, core)
        f.report(c, checks)
        s = m.read_settings(p.HERE / 'config-post-confirmation.json')
        s.update(semantic_num_classes=2, semantic_epochs=2, bootstrap_repeats=20, bootstrap_bins=128)
        return run, c, frozen, s, core, run / 'post_confirmation_v1'

    def test_four_cells_semantic_report_resume_and_immutable_source(self):
        with tempfile.TemporaryDirectory(dir=p.HERE / 'tmp') as d:
            run, c, frozen, s, core, out = self.fixture(Path(d))
            before = {str(x): m.digest(x) for x in run.rglob('*') if x.is_file()}
            # The identical-feature fixture intentionally has no decline. Gate
            # behavior is tested independently; only bypass it for integration.
            with patch.object(m, 'require_gate'):
                m.prepare(run, c, frozen, s, out)
                m.cross_readouts(c, frozen, s, out, core)
                m.semantic(c, s, out, core)
                report = m.make_report(c, frozen, s, out)
                self.assertTrue(report['status']['complete'])
                self.assertEqual(report['status']['cross_cells_completed'], 8)
                self.assertEqual(len(report['cross_comparisons']), 10)
                for row in report['cross_comparisons']:
                    self.assertEqual(row['binding']['delta_exact'], 0)
                    self.assertEqual(row['binding']['delta_ci95'], [0, 0])
                for metric in report['independent_semantic']['metrics'].values():
                    self.assertEqual(metric['delta_exact'], 0)
                self.assertEqual(report['independent_semantic']['evaluation_fingerprint_images_checked'], c['max_test'])
                m.prepare(run, c, frozen, s, out)
                m.cross_readouts(c, frozen, s, out, core)
                with patch.object(torch.nn, 'Linear', side_effect=AssertionError('must skip completed semantic probes')):
                    m.semantic(c, s, out, core)
                for name, digest in before.items():
                    self.assertEqual(m.digest(name), digest, name)
                records_path = out / 'semantic/residual/layer_23/per_image.json'
                saved_records = records_path.read_bytes()
                records = f.read_json(records_path)
                records[0]['evaluation_fingerprint'] = 'wrong-patches'
                f.write(records_path, records)
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    m.make_report(c, frozen, s, out)
                records_path.write_bytes(saved_records)
                changed = dict(s, semantic_epochs=3)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    m.prepare(run, c, frozen, changed, out)
                # A mismatched checkpoint must fail before new cross cells.
                src = f.task_folder(c, 'full_seed0', 'post_attention') / 'checkpoint.pt'
                state = torch.load(src, weights_only=True)
                key = next(iter(state)); state[key] = state[key] + 10
                torch.save(state, src)
                with self.assertRaisesRegex(ValueError, 'diagonal scores'):
                    m.cross_readouts(c, frozen, s, out, core)

    def test_distinct_endpoints_use_correct_checkpoint_and_feature_axes(self):
        import validate_decline as v
        with tempfile.TemporaryDirectory(dir=p.HERE / 'tmp') as d:
            run, c, frozen, s, core, out = self.fixture(Path(d), distinct=True)
            with patch.object(m, 'require_gate'):
                m.prepare(run, c, frozen, s, out)
            m.cross_readouts(c, frozen, s, out, core)
            for variant in frozen['checks']['variants']:
                for fitted in m.CHANNELS:
                    state = torch.load(f.task_folder(c, variant['name'], fitted) / 'checkpoint.pt', weights_only=True)
                    if variant['kind'] == 'full':
                        weight = state['W.weight'].numpy(); bias = state['W.bias'][0].item()
                    else:
                        a, b = state['W1.weight'].numpy(), state['W2.weight'].numpy()
                        weight = a.T @ b; bias = state['W1.bias'][0].item()
                    symmetric = (weight + weight.T) / (2 * math.sqrt(c['embed_dim']))
                    for evaluated in m.CHANNELS:
                        pred = v.predictions(m.cross_folder(out, variant['name'], fitted, evaluated) / 'test_predictions.npz')
                        mapped, _ = core.cached(p.channel_config(c, evaluated), 'test', 23)
                        acts = np.array(mapped, copy=True)
                        mapped._mmap.close()
                        del mapped
                        expected = []
                        i, j = np.triu_indices(c['eval_patches'], 1)
                        for row, picks in enumerate(pred['patches']):
                            x = np.asarray(acts[row, picks])
                            expected.append((x @ symmetric @ x.T + bias)[i, j])
                        np.testing.assert_allclose(pred['score'], expected, atol=2e-6, rtol=2e-5)

    def test_missing_source_fails_before_output_or_training(self):
        with tempfile.TemporaryDirectory(dir=p.HERE / 'tmp') as d:
            run, c, frozen, s, core, out = self.fixture(Path(d))
            (f.task_folder(c, 'factor64', 'residual') / 'checkpoint.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing source artifacts'):
                m.prepare(run, c, frozen, s, out)
            self.assertFalse(out.exists())

    def test_incomplete_semantic_completion_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=p.HERE / 'tmp') as d:
            run, c, frozen, s, core, out = self.fixture(Path(d))
            with patch.object(m, 'require_gate'):
                m.prepare(run, c, frozen, s, out)
                m.semantic(c, s, out, core)
                dest = out / 'semantic/residual/layer_23'
                (dest / 'checkpoint.pt').unlink()
                with self.assertRaisesRegex(ValueError, 'Incomplete semantic'):
                    m.semantic(c, s, out, core)

    def test_gate_and_output_boundaries(self):
        frozen = dict(checks=dict(primary_variant='full_seed0', minimum_effect=.005))
        report = dict(primary_variant='full_seed0', minimum_effect=.005,
            primary_assessment='meaningful_decline_confirmed', results=[dict(variant='full_seed0', binding=dict(delta_ci95=[-.03, -.01]))])
        m.require_gate(report, frozen)
        report['results'][0]['binding']['delta_ci95'] = [-.03, -.001]
        with self.assertRaisesRegex(ValueError, 'primary gate'):
            m.require_gate(report, frozen)
        root = (p.HERE / 'tmp/source').resolve()
        for out in (root, root/'confirmation/overwrite', root/'channels/overwrite', root.parent/'elsewhere'):
            with self.assertRaisesRegex(ValueError, 'separate'):
                m.output_path(root, out)

    def test_plan_is_readonly_and_lists_local_missing_binary_artifacts(self):
        run = p.HERE.parent / 'runs/confirmation_l23_v1'
        if not run.exists():
            self.skipTest('Downloaded confirmation metadata not present')
        capture = io.StringIO()
        args = ['post_confirmation.py', '--source-run', str(run), '--output', str(run/'plan_does_not_write'), '--stage', 'plan']
        with patch('sys.argv', args), patch.object(p, 'core_module', side_effect=AssertionError('plan must not load ML')):
            with redirect_stdout(capture):
                m.main()
        result = json.loads(capture.getvalue())
        self.assertEqual(result['new_binding_tasks'], 0)
        self.assertEqual(result['new_semantic_tasks'], 2)
        self.assertFalse(Path(result['output']).exists())


if __name__ == '__main__':
    unittest.main()

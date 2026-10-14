"""Small CPU checks of train-only statistics, congruence and joint gap inference."""
import copy, json, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import precondition_readout as pre
import investigate_decline as inv
import test_investigation as fixtures


class Preconditioning(unittest.TestCase):
    def setUp(self):
        (pre.p.HERE / 'tmp').mkdir(exist_ok=True)

    def settings(self, c, root):
        settings = json.loads((pre.p.HERE / 'config-precondition-readout.json').read_text())
        settings.update(layer=1, max_statistics_images=3, statistics_patches_per_image=6,
            statistics_batch_images=2, bootstrap_repeats=20, bootstrap_bins=128)
        path = root / 'test-settings.json'; inv.write(path, settings)
        return pre.read_settings(path, c)

    def test_invertible_transform_preserves_rms_and_full_bilinear_family(self):
        x = np.array([[1., 2., 3.], [2., 4., 6.]], dtype=np.float64)  # rank-deficient statistics
        moment = x.T @ x / len(x)
        transform, audit = pre.matrix_from_moment(moment, .01)
        self.assertGreater(np.linalg.eigvalsh(transform.astype(float)).min(), 0)
        self.assertEqual(audit['truncated_dimensions'], 0)
        self.assertFalse(audit['centered'])
        np.testing.assert_allclose(np.mean(np.sum((x @ transform.T)**2, axis=1)),
                                   np.mean(np.sum(x**2, axis=1)), rtol=1e-5)
        np.testing.assert_allclose((x @ transform.T) @ np.linalg.inv(transform.T), x, atol=1e-5)
        class Full(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([[1., 2., 0.], [2., 0., 3.], [0., 3., -1.]]))
            def forward_pairwise(self, a, b): return a @ self.weight @ b.T
        probe = Full(); wrapped = pre.wrap_probe(probe, transform)
        a = torch.tensor(x, dtype=torch.float32); t = torch.tensor(transform)
        expected = a @ (t.T @ probe.weight @ t) @ a.T
        torch.testing.assert_close(wrapped.forward_pairwise(a, a), expected, rtol=1e-5, atol=1e-4)
        self.assertNotIn('transform', dict(wrapped.named_parameters()))
        self.assertIn('transform', wrapped.state_dict())
        with self.assertRaisesRegex(ValueError, 'zero'):
            pre.matrix_from_moment(np.zeros((3, 3)), .01)

    def test_joint_gap_narrowing_and_alignment(self):
        a = fixtures.predictions(); b = copy.deepcopy(a); b['score'][0, 0] = 0.
        s = dict(bootstrap_repeats=20, bootstrap_bins=128, bootstrap_seed=0, bootstrap_max_auc_error=.0005)
        result = pre.joint_bootstrap([a, b, a, a], s)
        self.assertEqual(result['raw_mlp_delta']['delta_exact'], -1.)
        self.assertEqual(result['preconditioned_mlp_delta']['delta_exact'], 0.)
        self.assertEqual(result['gap_narrowing']['delta_exact'], 1.)
        self.assertEqual(result['gap_narrowing']['delta_ci95'], [1., 1.])
        bad = copy.deepcopy(a); bad['image_id'][0] += 1
        with self.assertRaisesRegex(ValueError, 'not paired'):
            pre.joint_bootstrap([a, b, a, bad], s)

    def fixture(self, root):
        c, training, core = fixtures.Pipeline().fixture(root)
        s = self.settings(c, root)
        for split in ('train', 'val', 'test'):
            source = Path(pre.p.channel_config(c, 'residual')['output']) / 'cache' / split
            target = Path(pre.p.channel_config(c, 'post_attention')['output']) / 'cache' / split
            shutil.copytree(source, target)
            manifest = json.loads((target / 'manifest.json').read_text())
            manifest.update(channel='post_attention', location=pre.p.channel_location('post_attention'))
            inv.write(target / 'manifest.json', manifest)
        root = pre.source_root(c, s)
        inv.write(root / 'steps/requested_settings.json', dict(base_config=c, settings=training, scope='steps'))
        inv.train_binding(dict(c, channels=list(pre.CHANNELS), layers=[1]),
                          dict(training, baseline_layers=[]), root / 'steps', core=core)
        residual = inv.files(c, training, 'residual', 1)
        residual.parent.mkdir(parents=True)
        shutil.move(str(root / 'steps/binding/residual/layer_1'), residual)
        inv.write(residual / 'run.json', dict(config=c, variant=dict(kind='full', seed=0, weight_decay=0.),
            layer=1, sampling_seed=0, selection=pre.local.SELECTION))
        return c, s, core, root / 'precondition_readout'

    def test_fit_train_report_resume_and_source_protection(self):
        with tempfile.TemporaryDirectory(dir=pre.p.HERE / 'tmp') as tmp:
            c, s, core, out = self.fixture(Path(tmp))
            cache_before = {str(file): pre.local.digest(file) for file in Path(c['output']).glob('channels/*/cache/*/*')}
            sources = pre.raw_sources(c, s)
            source_before = {str(folder / name): pre.local.digest(folder / name)
                for folder in sources.values() for name in pre.RAW_FILES}
            self.assertFalse(out.exists()); self.assertEqual(pre.plan(c, s, out)['pending_binding_tasks'], 2)
            pre.fit(c, s, out)
            train, _ = core.cached(pre.p.channel_config(c, 'residual'), 'train', 1)
            moment = np.asarray(train, dtype=float).reshape(-1, 8)
            expected, _ = pre.matrix_from_moment(moment.T @ moment / len(moment), s['ridge_relative'])
            train._mmap.close()  # Release the Windows file lock before scratch cleanup.
            del train
            transform, audit = pre.load_transform(c, s, out, 'residual')
            np.testing.assert_allclose(transform, expected, rtol=1e-6, atol=1e-6)
            self.assertEqual(audit['tokens'], 18); self.assertEqual(audit['fit_split'], 'train')
            self.assertEqual(audit['selected_training_image_ids'], [0, 1, 2])
            with patch.object(pre.p, 'core_module', return_value=core) as load:
                pre.train(c, s, out)
                report = pre.report(c, s, out)
                pre.fit(c, s, out); pre.train(c, s, out)
                self.assertEqual(load.call_count, 1)
            self.assertEqual(report['paired_comparisons']['gap_narrowing']['delta_exact'], 0.)
            self.assertEqual(pre.plan(c, s, out)['pending_binding_tasks'], 0)
            self.assertTrue((out / 'measurements.csv').is_file())
            self.assertEqual(cache_before, {str(file): pre.local.digest(file) for file in Path(c['output']).glob('channels/*/cache/*/*')})
            self.assertEqual(source_before, {str(folder / name): pre.local.digest(folder / name)
                for folder in sources.values() for name in pre.RAW_FILES})
            changed = dict(s, ridge_relative=.02)
            with self.assertRaisesRegex(ValueError, 'inputs/code/settings changed'):
                pre.fit(c, changed, out)
            file = pre.fit_folder(out, 'residual') / 'transform.npz'
            np.savez(file, transform=np.eye(8, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'audit/hash differs'):
                pre.train(c, s, out)

    def test_fit_uses_only_training_features_without_labels_selecting_tokens(self):
        with tempfile.TemporaryDirectory(dir=pre.p.HERE / 'tmp') as tmp:
            root = Path(tmp); c, _, _ = fixtures.Pipeline().fixture(root)
            s = self.settings(c, root)
            train_meta, _ = inv.load_meta(c, 'train', 'residual')
            # Supply only training arrays: fit cannot consult validation/test values.
            source = Path(pre.p.channel_config(c, 'residual')['output']) / 'cache/train'
            target = Path(pre.p.channel_config(c, 'post_attention')['output']) / 'cache/train'
            shutil.copytree(source, target)
            with patch.object(pre, 'prepare'), patch.object(inv, 'load_meta', return_value=(train_meta, {})) as metadata:
                pre.fit(c, s, root / 'fit_only')
                metadata.assert_called_once_with(c, 'train', 'residual')


if __name__ == '__main__': unittest.main()

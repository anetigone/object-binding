"""Disjoint selection, frozen-protocol guards, and tiny real probe integration."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import ast
import math
import tempfile
import unittest
import numpy as np
import torch
import confirm_l23 as f
import phenomenon as p


class Confirmation(unittest.TestCase):
    def setUp(self):
        (p.HERE / 'tmp').mkdir(exist_ok=True)

    def config(self, root):
        c = p.read_config(p.HERE / 'config-confirmation.json', output=str(root / 'new'))
        c.update(max_train=3, max_val=2, max_test=2)
        s = f.read_json(p.HERE / 'config-confirmation-checks.json')
        old = root / 'old/splits.json'
        f.write(old, dict(train=[0, 1], val=[20], test=[40]))
        s['exclusion_split_files'] = [str(old)]
        c['confirmation_index_file'] = str(root / 'index.pkl')
        Path(c['confirmation_index_file']).write_bytes(b'stable index')
        return c, s

    def pools(self):
        return dict(train=list(range(10)), val=list(range(20, 30)), test=list(range(40, 50)))

    def test_disjoint_and_deterministic(self):
        c = dict(split_seed=23, max_train=3, max_val=2, max_test=2)
        used = {0, 1, 20, 40}
        result = f.select_fresh(self.pools(), used, c)
        self.assertFalse(f.validate_splits(result) & used)
        self.assertEqual(result, f.select_fresh(self.pools(), used, c))
        with self.assertRaisesRegex(ValueError, 'Insufficient unseen'):
            f.select_fresh(self.pools(), set(range(10)), c)
        bad = self.pools(); bad['val'].append(0)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            f.select_fresh(bad, used, c)

    def test_freeze_guards_and_loader(self):
        with tempfile.TemporaryDirectory(dir=p.HERE / 'tmp') as d:
            root = Path(d); c, s = self.config(root)
            f.check_settings(c, s)
            core = SimpleNamespace(Dataset=lambda cfg: SimpleNamespace(splits=self.pools(), read=lambda i: i))
            f.freeze(c, s, core)
            result = f.verify_frozen(c, s)
            f.attach_frozen_dataset(c, s, core)
            self.assertEqual(core.Dataset(c).splits, result)
            self.assertEqual(core.Dataset(c).read(42), 42)
            changed = copy.deepcopy(c); changed['epochs'] += 1
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                f.verify_frozen(changed, s)
            saved = (root / 'new/splits.json').read_bytes()
            (root / 'new/splits.json').write_bytes(saved + b' ')
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                f.verify_frozen(c, s)
            (root / 'new/splits.json').write_bytes(saved)
            Path(c['confirmation_index_file']).write_bytes(b'changed ordering')
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                f.verify_frozen(c, s)

    def test_reject_output_and_incomplete_results(self):
        with tempfile.TemporaryDirectory(dir=p.HERE / 'tmp') as d:
            c, s = self.config(Path(d))
            c['output'] = str(Path(s['exclusion_split_files'][0]).parent)
            with self.assertRaisesRegex(ValueError, 'separate'):
                f.check_settings(c, s)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                f.verify_completed(Path(d))

    def test_primary_gate_does_not_treat_wide_ci_as_equivalence(self):
        self.assertEqual(f.gate([-.03, -.006], .005), 'meaningful_decline_confirmed')
        self.assertEqual(f.gate([-.03, -.001], .005), 'negative_direction_only')
        self.assertEqual(f.gate([-.03, .02], .005), 'inconclusive')
        self.assertEqual(f.gate([-.003, .003], .005), 'small_difference_supported')

    def test_real_full_and_factor_training_with_shared_samples(self):
        import test_investigation as fixtures
        import investigate_decline as inv
        with tempfile.TemporaryDirectory(dir=p.HERE / 'tmp') as d:
            _, _, core = fixtures.Pipeline().fixture(Path(d) / 'fixture')
            tree = ast.parse((p.HERE.parent / 'vit-object-binding/src/utils/models.py').read_text())
            classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QuadraticFixedRankProbe']
            namespace = dict(torch=torch, nn=torch.nn, math=math)
            exec(compile(ast.Module(body=classes, type_ignores=[]), 'factor_fixture', 'exec'), namespace)
            core.UPSTREAM = p.HERE.parent / 'vit-object-binding'
            core.load_author_classes = lambda file, names: namespace
            c, s = self.config(Path(d))
            c.update(embed_dim=8, saved_patches=4, sampled_patches=4, eval_patches=4,
                     epochs=2, batch_size=2, device='cpu')
            rng = np.random.default_rng(7)
            acts = rng.normal(size=(3, 4, 8)).astype(np.float32)
            meta = dict(image_id=np.array([100, 101, 102]),
                instance=np.tile([1, 1, 2, 3], (3, 1)),
                semantic=np.tile([1, 1, 1, 2], (3, 1)),
                background=np.zeros((3, 4), dtype=bool), position=np.tile(np.arange(4), (3, 1)))
            def cached(cfg, split, layer):
                return acts, meta
            with patch.object(inv, 'validate_caches'), patch.object(core, 'cached', side_effect=cached):
                f.train(c, s, core)
                for variant in s['variants']:
                    for ch in c['channels']:
                        dest = f.task_folder(c, variant['name'], ch)
                        f.verify_completed(dest)
                        self.assertEqual(len(f.read_json(dest / 'history.json')), 2)
                    a, b = [np.load(f.task_folder(c, variant['name'], ch) / 'test_predictions.npz') for ch in c['channels']]
                    np.testing.assert_array_equal(a['patches'], b['patches'])
                    np.testing.assert_array_equal(a['condition'], b['condition'])
                    np.testing.assert_allclose(a['score'], b['score'])
                    a.close(); b.close()
                # Complete jobs skip, incomplete jobs do not silently skip.
                f.train(c, s, core)
                s.update(bootstrap_repeats=20, bootstrap_bins=128)
                f.write(Path(c['output']) / 'frozen_protocol.json', {'synthetic': True})
                with patch.object(inv, 'load_meta', return_value=(meta, {})):
                    f.report(c, s)
                result = f.read_json(Path(c['output']) / 'confirmation/report.json')
                self.assertEqual(result['primary_assessment'], 'small_difference_supported')
                for row in result['results']:
                    self.assertEqual(row['binding']['delta_exact'], 0)
                    self.assertEqual(row['anchored_rankings']['delta'], 0)
                (f.task_folder(c, 'full_seed0', 'residual') / 'checkpoint.pt').unlink()
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    f.train(c, s, core)


if __name__ == '__main__':
    unittest.main()

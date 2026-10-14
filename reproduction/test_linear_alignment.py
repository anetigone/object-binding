"""Train-only linear fit, raw-checkpoint prerequisite and frozen evaluation."""
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
import linear_alignment as a
import post_confirmation as post
import confirm_l23 as src
import phenomenon as p
import test_post_confirmation as fixtures


class Alignment(unittest.TestCase):
    def setUp(self):
        (p.HERE/'tmp').mkdir(exist_ok=True)

    def fixture(self,root):
        run,c,frozen,fs,core,followup=fixtures.Followup().fixture(root,distinct=True)
        with patch.object(post,'require_gate'):
            post.prepare(run,c,frozen,fs,followup)
        post.cross_readouts(c,frozen,fs,followup,core)
        post.semantic(c,fs,followup,core)
        post.make_report(c,frozen,fs,followup)
        s=src.read_json(p.HERE/'config-linear-alignment.json')
        s.update(max_statistics_images=3,statistics_patches_per_image=6,bootstrap_repeats=20,bootstrap_bins=128)
        return run,c,frozen,followup,s,core,run/'linear_alignment_v1'

    def test_known_linear_relation_and_lazy_slices(self):
        rng=np.random.default_rng(3);h=rng.normal(size=(100,4));truth=rng.normal(size=(4,4));z=h@truth
        matrix=a.ridge_map(h,z,1e-8)
        np.testing.assert_allclose(matrix,truth,rtol=1e-6,atol=1e-6)
        bank=h.astype(np.float32).reshape(5,20,4)
        mapped=a.MappedFeatures(bank,matrix.astype(np.float32))
        self.assertEqual(mapped.shape,bank.shape)
        np.testing.assert_allclose(mapped[1,[0,5]],bank[1,[0,5]]@matrix.astype(np.float32))
        with self.assertRaisesRegex(ValueError,'Degenerate'):
            a.ridge_map(np.zeros_like(h),z,.001)

    def test_real_audit_fit_and_frozen_evaluation_preserve_sources(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'tmp') as tmp:
            run,c,frozen,followup,s,core,out=self.fixture(Path(tmp))
            before={str(f):post.digest(f) for f in run.rglob('*') if f.is_file()}
            with patch.object(post,'require_gate'):
                a.prepare(run,c,frozen,followup,s,out)
            with self.assertRaises(FileNotFoundError):
                a.fit(c,s,out,core)
            a.raw_audit(c,frozen,followup,out,core)
            # Cache provider records every split used during fitting: no val/test.
            original=core.cached;calls=[]
            def cached(cfg,split,layer):
                calls.append(split);return original(cfg,split,layer)
            with patch.object(core,'cached',side_effect=cached):
                a.fit(c,s,out,core)
            self.assertEqual(calls,['train','train'])
            fit=src.read_json(out/'fit_audit.json')
            self.assertTrue(set(fit['image_ids']).issubset(src.read_json(run/'splits.json')['train']))
            self.assertEqual(fit['tokens'],18)
            a.evaluate(c,frozen,followup,s,out,core)
            report=a.report(c,frozen,followup,s,out)
            self.assertEqual(len(report['measurements']),8)
            self.assertEqual(len(report['comparisons']),2)
            self.assertTrue(report['raw_audit_passed'])
            for item in report['reconstruction']:
                self.assertLess(item['aligned_mse'],item['identity_mse'])
            # No optimizer/probe fit is invoked. Repeated map fit skips work.
            with patch.object(core,'cached',side_effect=AssertionError('already fitted')):
                a.fit(c,s,out,core)
            a.evaluate(c,frozen,followup,s,out,core)
            for file,digest in before.items():self.assertEqual(post.digest(file),digest,file)
            with patch.object(post,'require_gate'):
                with self.assertRaisesRegex(ValueError,'changed'):
                    a.prepare(run,c,frozen,followup,dict(s,ridge_relative=.002),out)
            # Stored matrix tampering is refused before reuse.
            file=out/'alignment.npz';data=file.read_bytes();file.write_bytes(data+b'x')
            with self.assertRaisesRegex(ValueError,'changed'):a.load_map(c,out)

    def test_missing_raw_assets_fail_before_output(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'tmp') as tmp:
            run,c,frozen,followup,s,core,out=self.fixture(Path(tmp))
            (followup/'semantic/post_attention/layer_23/checkpoint.pt').unlink()
            with self.assertRaisesRegex(ValueError,'prerequisite missing'):
                a.prepare(run,c,frozen,followup,s,out)
            self.assertFalse(out.exists())

    def test_train_fit_does_not_change_with_test_features(self):
        # Direct fit harness with in-memory banks isolates the no-leakage property.
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=p.HERE/'tmp') as tmp:
            root=Path(tmp);rng=np.random.default_rng(4)
            h=rng.normal(size=(3,6,4)).astype(np.float32);z=h@np.diag([1.,2.,3.,4.]).astype(np.float32)
            meta=dict(image_id=np.array([1,2,3]),position=np.tile(np.arange(6),(3,1)))
            c=dict(output=str(root/'source'),embed_dim=4,saved_patches=6)
            src.write(Path(c['output'])/'splits.json',dict(train=[1,2,3],val=[4],test=[5]))
            s=src.read_json(p.HERE/'config-linear-alignment.json');s.update(max_statistics_images=2,statistics_patches_per_image=4)
            def cached(cfg,split,layer):
                self.assertEqual(split,'train')
                return (h if cfg['channel']=='residual' else z),meta
            core=SimpleNamespace(cached=cached)
            matrices=[]
            for name in ('first','second'):
                out=root/name;src.write(out/'requested_settings.json',dict(stable=True))
                src.write(out/'raw_audit.json',dict(raw_audit_passed=True,source_request_sha256=post.digest(out/'requested_settings.json')))
                a.fit(c,s,out,core);matrices.append(a.load_map(c,out)[0])
            np.testing.assert_array_equal(*matrices)


if __name__=='__main__':unittest.main()

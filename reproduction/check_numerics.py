"""Tiny CPU checks of the harness, not a paper experiment or pretrained run."""
import ast, math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score

here = Path(__file__).resolve().parent
tree = ast.parse((here/'run_server.py').read_text(encoding='utf-8'))
names = {'load_author_classes','probe','metrics','processor','batch_logits'}
namespace = dict(torch=torch,nn=nn,F=F,np=np,math=math,torchvision=torchvision,TF=TF,NS=NS,
    UPSTREAM=here.parent/'vit-object-binding',ast=ast,roc_auc_score=roc_auc_score)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'selected_harness','exec'),namespace)

torch.manual_seed(0)
x = torch.randn(2,4,1024)
for mode in ('quadratic','linear','diag_quadratic'):
    p = namespace['probe']({},mode)
    s = namespace['batch_logits'](p,x)
    torch.testing.assert_close(s,s.transpose(1,2))
    torch.testing.assert_close(s[0,0,1],p.forward(x[0,0:1],x[0,1:2]).reshape(()))
    F.binary_cross_entropy_with_logits(s,torch.eye(4).expand(2,4,4)).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters())
    print(mode, 'symmetry, pairwise equivalence, finite backward: OK')

m = namespace['metrics']([1,1,0,0],[3.,-1.,2.,-2.])
assert m['accuracy']==.5 and m['balanced_accuracy']==.5 and m['auroc']==.75
assert namespace['metrics']([0,0],[1.,-1.])['auroc'] is None
assert namespace['metrics']([],[])['n']==0
proc = namespace['processor']({})
mask = np.full((518,700,1),7,dtype=np.int32)
patched = proc.forward_patch(mask)
assert tuple(patched.shape)==(2,37,37,1) and (patched==7).all()
rgb = np.full((518,700,3),128,dtype=np.uint8)
pixel = proc.forward(rgb)
assert tuple(pixel.shape)==(2,518,518,3) and torch.isfinite(pixel).all()
print('metrics and author multicrop mask/image alignment: OK')

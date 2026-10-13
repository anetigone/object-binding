"""Fetch the authors' pinned source as an ignored dependency, not a nested gitlink."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
target = root/'vit-object-binding'
pin = '014c66b45ea262f9b6eec83ff388a1e1c10dfcaa'
if not target.exists():
    subprocess.run(['git','clone','https://github.com/liyihao0302/vit-object-binding.git',str(target)],check=True)
if not (target/'.git').exists():
    raise SystemExit('Existing source has no Git metadata; preserve it and use a fresh clone location.')
head = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
dirty = subprocess.check_output(['git','-C',str(target),'status','--porcelain'],text=True).strip()
if dirty:
    raise SystemExit('Upstream has local edits. Preserve those edits before changing the source version.')
if head!=pin:
    subprocess.run(['git','-C',str(target),'checkout','--detach',pin],check=True)
print('Upstream ready:',pin)

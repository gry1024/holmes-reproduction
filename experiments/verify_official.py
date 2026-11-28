"""Check a known official checkpoint in the new runtime and scoring adapter."""
from pathlib import Path
import json
import sys
import zipfile
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'ICML26-Holmes/src'))
import torch
from Configs.builder import get_configs
from Datasets.builder import get_datasets
from Models.builder import get_models
from Validations.builder import get_validations
from Utils.utils import set_seed
from adapters import extend_model,extend_validation
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
with zipfile.ZipFile(root/'downloads/Holmes-cha-parallel.zip') as z:
    names=[n for n in z.namelist() if n.endswith('.ckpt')]
    assert len(names)==1
    with z.open(names[0]) as f:
        ck=torch.load(f,map_location='cpu',weights_only=False)
cfg=get_configs('cha').copy()
cfg.update(data_root=str(root/'data'),num_workers=4)
set_seed(cfg['seed'])
cfg,_,vv,vq,_,_=get_datasets(cfg)
model=get_models(cfg)
model.load_state_dict(ck['state_dict'],strict=True)
reference=ck['model_val'][:5]
model.cuda().eval()
with torch.no_grad():
    original=get_validations(cfg)(model,vv,vq)[:5]
model.cpu()
model=extend_model(model,'gating',cfg).cuda().eval()
with torch.no_grad():
    adapted=extend_validation(get_validations(cfg))(model,vv,vq)[:5]
result=dict(reference=reference,original=original,adapted_fixed_initial_gate=adapted)
assert all(abs(x-y)<1e-6 for x,y in zip(original,reference)),result
assert all(abs(x-y)<1e-6 for x,y in zip(adapted,reference)),result
(root/'results/runtime-verification.json').write_text(json.dumps(result,indent=2))
print('OFFICIAL_AND_ADAPTER_MATCH',result,flush=True)

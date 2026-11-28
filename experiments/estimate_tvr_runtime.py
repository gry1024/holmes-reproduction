"""Short real-data timing probe; does not save or alter experiment checkpoints."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
p=argparse.ArgumentParser();p.add_argument('--multi',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'ICML26-Holmes'/('multisrc' if a.multi else 'src')))
import torch
from torch.nn import functional as F
from Configs.builder import get_configs
from Datasets.builder import get_datasets
from Models.builder import get_models
from Losses.builder import get_losses
from Opts.builder import get_opts
from Validations.builder import get_validations
from Utils.utils import gpu,set_seed
from adapters import extend_model,extend_validation
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cfg=get_configs('tvr').copy();cfg.update(data_root=str(root/'data'),num_workers=4)
set_seed(0)
cfg,train,vv,vq,_,_=get_datasets(cfg)
cfg.update(warmup_end=True,use_hard_negative=True)
for variant in ['baseline','combined']:
    set_seed(0)
    model=get_models(cfg)
    validator=get_validations(cfg)
    if variant!='baseline':
        model=extend_model(model,variant,cfg,a.multi)
        validator=extend_validation(validator)
    model.cuda().train();criterion=get_losses(cfg);opt=get_opts(cfg,model,train)
    timings=[];previous=time.time()
    for index,batch in enumerate(train):
        batch=gpu(batch);opt.zero_grad(set_to_none=True)
        outputs=model(batch);loss=criterion(outputs,batch)
        if variant=='combined':loss=loss+0.1*F.cross_entropy(model.last_fused/.1,torch.tensor(batch['text_labels'],device='cuda'))
        loss.backward();opt.step();torch.cuda.synchronize()
        timings.append(time.time()-previous)
        del outputs,loss,batch
        if variant=='combined':model.last_fused=None
        previous=time.time()
        if index==9:break
    start=time.time()
    with torch.no_grad():validator(model,vv,vq)
    torch.cuda.synchronize();validation=time.time()-start
    step=statistics.median(timings[2:])
    record=dict(multi=a.multi,variant=variant,train_batches=len(train),step_seconds=step,
                validation_seconds=validation,estimated_epoch_seconds=step*len(train)+validation,
                max_epochs=cfg['n_epoch'],timing_only=True)
    print('TIMING',json.dumps(record),flush=True)
    target=root/'results'/f'tvr-timing-{a.multi}-{variant}.json'
    target.write_text(json.dumps(record,indent=2))
    del model,criterion,opt,validator
    torch.cuda.empty_cache()

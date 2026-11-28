"""Full-size synthetic backward check, not a benchmark result."""
import argparse
from pathlib import Path
import sys
import time
p=argparse.ArgumentParser()
p.add_argument('--multi',action='store_true')
p.add_argument('--dataset',default='cha')
a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ICML26-Holmes'/('multisrc' if a.multi else 'src')))
import torch
from torch.nn import functional as F
from Configs.builder import get_configs
from Models.builder import get_models
from Losses.builder import get_losses
from Opts.builder import get_opts
from Utils.utils import set_seed
from adapters import extend_model
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
print('GPU',torch.cuda.get_device_name(),torch.__version__,flush=True)
cfg=get_configs(a.dataset).copy()
cfg['warmup_end']=True
cfg['use_hard_negative']=True
for variant in ['pooling','gating','combined']:
    set_seed(0)
    model=extend_model(get_models(cfg),variant,cfg,a.multi).cuda()
    criterion=get_losses(cfg)
    opt=get_opts(cfg,model,[0]*100)
    b=128
    nq=5 if a.dataset=='tvr' else 4
    batch=dict(clip_video_features=torch.randn(b,32,cfg['visual_feat_dim'],device='cuda'),
        frame_video_features=torch.randn(b,128,cfg['visual_feat_dim'],device='cuda'),
        videos_mask=torch.ones(b,128,device='cuda'),text_feat=torch.randn(b*nq,30,cfg['q_feat_size'],device='cuda'),
        text_mask=torch.ones(b*nq,30,device='cuda'),text_labels=[i for i in range(b) for _ in range(nq)])
    batch['videos_mask'][0,80:]=0
    torch.cuda.reset_peak_memory_stats()
    start=time.time()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        output=model(batch)
        loss=criterion(output,batch)
        if model.gating:
            loss=loss+0.1*F.cross_entropy(model.last_fused/0.1,torch.tensor(batch['text_labels'],device='cuda'))
        assert torch.isfinite(loss)
        loss.backward()
        if model.gating:
            assert model.scale_gate[-1].weight.grad.abs().sum()>0
        opt.step()
        model.last_fused=None
        del loss,output
    torch.cuda.synchronize()
    print('PASSED',a.dataset,a.multi,variant,'GiB',torch.cuda.max_memory_allocated()/2**30,'seconds',time.time()-start,flush=True)
    del model,criterion,opt,batch
    torch.cuda.empty_cache()

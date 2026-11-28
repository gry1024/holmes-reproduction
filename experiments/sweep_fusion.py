"""Official-checkpoint inference sensitivity to clip/frame fusion weight."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['cha','act','tvr'], required=True)
p.add_argument('--multi', action='store_true')
p.add_argument('--checkpoint', type=Path, help='Retrained original-model best.pt')
p.add_argument('--data-root', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'ICML26-Holmes'/('multisrc' if a.multi else 'src')))
import torch
from Configs.builder import get_configs
from Datasets.builder import get_datasets
from Models.builder import get_models
from Utils.utils import gpu,set_seed
from Validations.builder import get_validations
from Validations.validations import get_gt,cal_perf

def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
label=a.dataset+('-multi' if a.multi else '')
checkpoint_path=a.checkpoint or root/'reproduction/checkpoints'/f'{label}.ckpt'
checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
reference=checkpoint['metrics'][:5] if a.checkpoint else checkpoint['model_val'][:5]
if a.checkpoint:
    training=json.loads((checkpoint_path.parent/'config.json').read_text())
    assert training['variant']=='baseline' and training['multi']==a.multi
    cfg=training['config'].copy()
else:
    cfg=get_configs(a.dataset).copy()
default=cfg['clip_scale_w']
cfg.update(data_root=str(a.data_root or root/'reproduction/data'),num_workers=0)
set_seed(cfg['seed'])
cfg,_,videos,queries,_,_=get_datasets(cfg)
model=get_models(cfg)
model.load_state_dict(checkpoint['state_dict'],strict=True)
del checkpoint
model.cuda().eval()
validator=get_validations(cfg)
start=time.time()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    contexts=validator.compute_context_info(model,videos)
    clip_rows,frame_rows,query_ids=[],[],[]
    for index,batch in enumerate(queries):
        batch=gpu(batch)
        query_ids.extend(batch[-1])
        args=[batch[0],batch[1],None,contexts['video_proposal_feat'],contexts['video_feat']]
        if a.multi:
            args.append(contexts['video_mask'])
        clip,frame=model.get_pred_from_raw_query(*args)
        clip_rows.append(clip.cpu())
        frame_rows.append(frame.cpu())
        if index % 50 == 0:
            print('SCORES',label,index,len(queries),flush=True)
    clip=torch.cat(clip_rows)
    frame=torch.cat(frame_rows)
    video_ids=contexts['video_metas']
    _,ground_truth=get_gt(video_ids,query_ids)
    del contexts,clip_rows,frame_rows,model,batch,args
    torch.cuda.empty_cache()
    clip,frame=clip.cuda(),frame.cuda()
    points=[]
    baseline=None
    for weight in sorted(set([0.,0.2,0.4,0.6,0.8,1.,default])):
        fused=weight*clip+(1-weight)*frame
        metrics=list(cal_perf(-fused,ground_truth))
        metrics=[float(x) for x in metrics]
        metrics.append(sum(metrics))
        row=dict(clip_weight=weight,frame_weight=1-weight,
                 R1=metrics[0],R5=metrics[1],R10=metrics[2],R100=metrics[3],SumR=metrics[4])
        if weight==default:
            baseline=row
            assert all(abs(x-y)<1e-8 for x,y in zip(metrics,reference)),(label,metrics,reference)
        if weight in [0.,0.2,0.4,0.6,0.8,1.]:
            points.append(row)
        print('METRICS',label,json.dumps(row),flush=True)
        del fused
collection=cfg['collection']
textdir=Path(cfg['data_root'])/collection/'TextData'
val_hash=file_hash(textdir/f'{collection}val.caption.txt')
test_hash=file_hash(textdir/f'{collection}test.caption.txt')
result=dict(dataset=label,split='val',queries=len(query_ids),videos=len(video_ids),
    checkpoint_sha256=file_hash(checkpoint_path),
    torch=torch.__version__,gpu=torch.cuda.get_device_name(),precision='FP32',
    checkpoint_source='retrained_baseline' if a.checkpoint else 'official',
    checkpoint_path=str(checkpoint_path),
    original_clip_weight=default,baseline_matches_checkpoint=True,
    baseline_matches_official=not bool(a.checkpoint),baseline=baseline,
    points=points,seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    val_caption_sha256=val_hash,test_caption_sha256=test_hash,
    val_test_caption_files_identical=val_hash==test_hash,
    protocol='Fixed checkpoint selected at default fusion weight; cached per-branch scores; author full-candidate ranking metrics; no per-weight retraining or reselection.')
out=a.output or root/'experiments/results/fusion-sensitivity'
out.mkdir(parents=True,exist_ok=True)
(out/f'{label}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('COMPLETE',label,flush=True)

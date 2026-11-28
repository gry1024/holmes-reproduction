"""Build the consolidated report from completed retraining records."""
import csv
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
res=root/'results'
runs={r['run']:r for r in json.loads((res/'completed-training.json').read_text())['runs']}
names={'cha':'Charades-STA','act':'ActivityNet','tvr':'TVR 单查询编码','tvr-multi':'TVR 双查询编码'}
variants={'baseline':'原版','pooling':'改动一','gating':'改动二','combined':'两项组合'}
metrics=['R@1','R@5','R@10','R@100','SumR']
assert len(runs)==16
with tarfile.open(res/'completed-training-records.tar.gz') as archive:
    configs={key:json.load(archive.extractfile(f'runs/{key}/config.json')) for key in runs}
    epochs={key:[json.loads(line) for line in archive.extractfile(f'runs/{key}/epochs.jsonl')] for key in runs}
for dataset in names:
    reference=configs[f'{dataset}/baseline']
    for variant in variants:
        key=f'{dataset}/{variant}'
        run=runs[key]
        assert run['result']['status']=='completed'
        assert run['result']['val']==run['result']['test']
        assert abs(sum(run['result']['val'][:4])-run['result']['val'][4])<1e-8
        other=configs[key]
        assert reference['torch']==other['torch'] and reference['precision']==other['precision']
        assert {k:v for k,v in reference['config'].items() if k!='root'}=={k:v for k,v in other['config'].items() if k!='root'}
sweeps={key:json.loads((res/'fusion-sensitivity-retrained'/f'{key}.json').read_text()) for key in names}
for key,sweep in sweeps.items():
    assert sweep['checkpoint_source']=='retrained_baseline'
    assert sweep['baseline_matches_checkpoint']
    assert abs(sweep['baseline']['SumR']-runs[f'{key}/baseline']['result']['val'][4])<1e-8
    assert [x['clip_weight'] for x in sweep['points']]==[0,.2,.4,.6,.8,1]

lines=['# Holmes 实验报告','',
'2026-11-28｜16 组训练与重训权重敏感性分析均已完成。','',
'## 1. 实验设置','',
'全部对照使用自行从头训练的 Holmes 检索模型，输入为作者提供的预提取视频、文本特征；不重训底层特征提取器。每个数据集独立训练，TVR 另含双查询编码版本。两个版本都输入一句查询：单编码版本共享整句表示，双编码版本另用词级表示进行帧匹配。','',
'原版与改进版共用训练循环、数据划分、优化器、学习率、选模及早停规则。环境为 RTX 5090 32GB、Python 3.12、PyTorch 2.7.1+cu128；FP32、关闭 TF32，seed=0、batch size=128、前 20 轮预热，沿用作者 BertAdam 与 warmup_linear 调度。原版和改进队列分别使用一张同型号显卡。','',
'| 设置 | 训练视频 | 评测查询 / 候选视频 | 学习率 | 最多轮次 | 默认片段权重 |',
'|---|---:|---:|---:|---:|---:|']
for key,name in names.items():
    cfg=configs[f'{key}/baseline']['config']; sw=sweeps[key]
    lines.append(f"| {name} | {runs[f'{key}/baseline']['train_videos']:,} | {sw['queries']:,} / {sw['videos']:,} | {cfg['lr']:g} | {cfg['n_epoch']} | {cfg['clip_scale_w']:.1f} |")
lines+=['',
'使用完整候选库，按验证集 SumR 选出最佳权重并评测。R@K 是正确视频进入前 K 名的查询比例（%）；SumR 为 R@1、R@5、R@10、R@100 之和。官方数据包的 val/test 描述文件相同，故只报告一套结果；本轮只有一个随机种子，不据此声称统计显著性或独立测试泛化。','',
'## 2. 两项改动','',
'**改动一：多局部证据聚合。** 将最高局部分数替换为 `0.5 × 最高分 + 0.5 × 前 3 个有效位置分数的均值`，同时用于片段和帧分支；不足 3 个位置时取实际数量，排除 padding，不要求位置相邻。双查询编码版本在词—帧匹配时应用相同聚合。','',
'**改动二：查询自适应融合。** 小型网络 `384 → 64 → GELU → 1 → sigmoid` 预测片段权重 w，最终分数为 `w × 片段分数 + (1−w) × 帧分数`，初始值等于原默认比例。训练增加 `0.1 × CE(融合分数 / 0.1, 原标注)`；评价的是门控与附加损失这一整体改动。','',
'分别训练原版、仅改动一、仅改动二、两项组合，在四种设置上共 16 组。','',
'## 3. 重训与改进结果','',
'Δ 为相对同数据集重训原版的 SumR 差值；所有数值由未舍入结果计算后保留两位小数。','',
'| 设置 | 模型 | R@1 | R@5 | R@10 | R@100 | SumR | Δ |',
'|---|---|---:|---:|---:|---:|---:|---:|']
csvrows=[]
for key,name in names.items():
    base=runs[f'{key}/baseline']['result']['val'][4]
    for variant,label in variants.items():
        run=runs[f'{key}/{variant}']; rr=run['result']; vals=rr['val']; delta=vals[4]-base
        lines.append('| '+name+' | '+label+' | '+' | '.join(f'{v:.2f}' for v in vals)+f' | {delta:+.2f} |')
        csvrows.append(dict(dataset=key,variant=variant,**dict(zip(metrics,vals)),delta_sumr=delta,
                            best_epoch=rr['best_epoch']+1,trained_epochs=run['completed_epochs'],
                            seconds=rr['seconds_this_invocation'],peak_training_gib=max(e['peak_gib'] for e in epochs[f'{key}/{variant}'])))
lines+=['',
'- 改动一在 ActivityNet、TVR 单查询编码上分别提高 **1.26、2.06**；Charades 下降 0.24，TVR 双查询编码下降 0.07。收益并不覆盖所有设置。',
'- 改动二在四种设置上均下降，TVR 双查询编码下降 **19.72**；两项组合没有改善这一趋势，双查询编码下降 **21.07**。',
'- 训练中间检查曾观察到门控明显偏向片段分支，与融合扫描中单一路径较弱的现象一致。但附加损失和编码器也随之改变，尚不能把下降单独归因于门控权重。','',
'## 4. 融合权重敏感性分析','',
'固定各重训原版在默认比例下选出的最佳权重，只改变推理分数 `α × 片段分数 + (1−α) × 帧分数`。扫描 α=0、0.2、0.4、0.6、0.8、1.0；不为每个 α 重训或重新选模。TVR 双查询编码额外核验默认 α=0.5。该扫描是我们的补充实验，所用论文正文与附录未找到融合比例的敏感性分析。','',
'| 片段权重 α | Charades | ActivityNet | TVR 单查询编码 | TVR 双查询编码 |',
'|---:|---:|---:|---:|---:|']
for i,w in enumerate([0,.2,.4,.6,.8,1]):
    lines.append(f'| {w:.1f} | '+' | '.join(f"{sw['points'][i]['SumR']:.2f}" for sw in sweeps.values())+' |')
lines+=['',
'![重训原版融合权重敏感性曲线](results/fusion-sensitivity-retrained/sumr-curves.png)','',
'Charades 与 TVR 单查询编码在六个点中以 **0.6** 最好；ActivityNet 以 **0.4** 最好，相比默认提高 **0.30**。TVR 双查询编码的六点最佳为 0.4（195.40），仍低于默认 **0.5（195.85）**。四种设置中，适当融合都优于只用一路；六点最佳不代表连续区间的全局最优。','',
'## 5. 训练开销','',
'下表耗时包含训练和评测，为各运行进程的实际用时；三组改进耗时相加。两条队列并行运行，因此不能把两列相加作为墙钟等待时间。','',
'| 设置 | 原版耗时 | 三组改进合计 | 实际训练轮次：原版 / 改动一 / 改动二 / 组合 |',
'|---|---:|---:|---|']
def duration(sec):
    mins=round(sec/60);return f'{mins//60} 小时 {mins%60:02d} 分'
for key,name in names.items():
    rows=[runs[f'{key}/{v}'] for v in variants]
    lines.append(f"| {name} | {duration(rows[0]['result']['seconds_this_invocation'])} | {duration(sum(r['result']['seconds_this_invocation'] for r in rows[1:]))} | "+' / '.join(str(r['completed_epochs']) for r in rows)+' |')
lines+=['',
'## 6. 数据与复跑入口','',
'- [16 组完整指标 CSV](results/training-comparison.csv) · [训练结果原始 JSON](results/completed-training.json) · [配置、逐轮指标和最终结果归档](results/completed-training-records.tar.gz)。',
'- [敏感性扫描全部召回率与 SumR](results/fusion-sensitivity-retrained/metrics.csv) · [各项召回率曲线](results/fusion-sensitivity-retrained/recall-curves.png) · [SumR 矢量图](results/fusion-sensitivity-retrained/sumr-curves.pdf)。',
'- 训练入口：[train.py](train.py)；后续权重扫描：[queue_retrained_fusion.py](queue_retrained_fusion.py)；数据获取：[下载说明](../reproduction/README.md)。',
'- 模型权重保存在服务器 `/data1/quant_team/holmes-20261127/runs/{cha,act,tvr,tvr-multi}/{baseline,pooling,gating,combined}/`；`best.pt` 为最佳权重，`last.pt` 为训练断点。','',
'单组重跑（双查询编码版本另加 `--multi`）：','',
'```bash',
'python experiments/train.py --dataset cha --variant baseline --data-root <数据目录> --output <新输出目录>',
'```','',
'`--variant` 可取 `baseline`、`pooling`、`gating`、`combined`；`--dataset` 可取 `cha`、`act`、`tvr`。','']
(root/'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
with (res/'training-comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
print('Verified 16 completed runs, matched training settings, and four retrained sensitivity scans.')
print(root/'RESULTS.md')

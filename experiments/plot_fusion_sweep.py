"""Export reproducible sensitivity plots, tables, and a short report."""
import csv
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path)
p.add_argument('--retrained',action='store_true')
a=p.parse_args()
out=a.output or root/'experiments/results/fusion-sensitivity'
names={'cha':'Charades-STA','act':'ActivityNet Captions','tvr':'TVR (single query)','tvr-multi':'TVR (dual query)'}
data={key:json.loads((out/f'{key}.json').read_text()) for key in names}
weights=[0.,.2,.4,.6,.8,1.]
for d in data.values():
    assert [r['clip_weight'] for r in d['points']]==weights
    assert d.get('baseline_matches_checkpoint',d.get('baseline_matches_official'))
    if a.retrained: assert d['checkpoint_source']=='retrained_baseline'
    for r in d['points']:
        assert r['R1']<=r['R5']<=r['R10']<=r['R100']
        assert abs(sum(r[k] for k in ['R1','R5','R10','R100'])-r['SumR'])<1e-8
with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['dataset','clip_weight','frame_weight','R1','R5','R10','R100','SumR'])
    writer.writeheader()
    for key,d in data.items():
        for row in d['points']:
            writer.writerow(dict(dataset=names[key],**row))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(13,8.5))
for ax,(key,d) in zip(axes.flat,data.items()):
    y=[r['SumR'] for r in d['points']]
    ax.plot(weights,y,'o-',color='#2364aa',lw=2,ms=6)
    base=d['baseline']
    ax.axhline(base['SumR'],color='#a8adb5',ls='--',lw=1)
    ax.scatter([base['clip_weight']],[base['SumR']],s=180,marker='*',color='#df8d20',edgecolors='#805510',zorder=5)
    best=max(d['points'],key=lambda r:r['SumR'])
    ax.scatter([best['clip_weight']],[best['SumR']],s=125,facecolors='none',edgecolors='#00866a',lw=2,zorder=6)
    for x,value in zip(weights,y):
        ax.annotate(f'{value:.2f}',(x,value),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    span=max(max(y+[base['SumR']])-min(y+[base['SumR']]),1)
    ax.set_ylim(min(y+[base['SumR']])-.16*span,max(y+[base['SumR']])+.25*span)
    ax.set_xlim(-.07,1.07)
    ax.set_xticks(weights)
    ax.set_title(names[key],loc='left',fontsize=12)
    ax.set_xlabel('Clip weight α  (frame weight = 1 − α)')
    ax.set_ylabel('SumR')
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Holmes: sensitivity to clip/frame fusion',fontsize=18,y=.985)
fig.text(.5,.945,('Retrained baseline checkpoints · fixed-model inference sweep' if a.retrained else 'Fixed official checkpoints · full candidate sets · no retraining'),ha='center',color='#555')
legend=[Line2D([0],[0],color='#2364aa',marker='o',label='Six requested weights'),
    Line2D([0],[0],color='#df8d20',marker='*',lw=0,ms=12,label='Default weight (0.6; dual-query TVR: 0.5)'),
    Line2D([0],[0],color='#00866a',marker='o',markerfacecolor='none',lw=0,ms=9,label='Best of six sampled weights')]
fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005),fontsize=9)
fig.tight_layout(rect=(0,.06,1,.925))
for ext in ['png','pdf','svg']:
    fig.savefig(out/f'sumr-curves.{ext}',dpi=220)
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(13,8))
colors=['#2364aa','#00866a','#9e4d9e','#dc762a']
for ax,metric in zip(axes.flat,['R1','R5','R10','R100']):
    for (key,d),color in zip(data.items(),colors):
        ax.plot(weights,[r[metric] for r in d['points']],'o-',color=color,lw=1.8,ms=4,label=names[key])
    ax.set_title(metric.replace('R','R@'),loc='left')
    ax.set_xticks(weights)
    ax.set_xlabel('Clip weight α  (frame weight = 1 − α)')
    ax.set_ylabel('Recall (%)')
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Holmes: recall at each fusion weight',fontsize=18,y=.99)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=10)
fig.tight_layout(rect=(0,.07,1,.95))
for ext in ['png','pdf','svg']:
    fig.savefig(out/f'recall-curves.{ext}',dpi=220)
plt.close(fig)

lines=['# Holmes 融合权重敏感性实验','',
'## 论文依据','',
'本地 arXiv v1 正文给出帧/片段分数的线性融合公式；附录 Parameter Analysis 只分析温度 τ、查询划分阈值 β、回收槽参数 z，未找到融合比例扫描。附录笼统说明沿用 HLFormer/GMMFormerV2 参数，没有解释六四开的具体选择依据，不能据此认定 0.6 是理论或全局最优值。',
'','## 实验口径','',
'α 表示片段权重，帧权重为 1−α。扫描 α=0.0、0.2、0.4、0.6、0.8、1.0；固定四套官方 checkpoint 和官方特征，只改变最终线性融合，缓存一次两路分数，再调用作者排序/指标代码评测完整候选库。不重新训练。',
'','普通版本官方 α=0.6；TVR 双查询官方 α=0.5，另计算该点用于核验，不混入六点扫描。四套模型在官方比例下均重新匹配 checkpoint 记录的五项原始精度指标。',
'','本次运行 val 入口；四套数据的 val/test 描述文件逐字节相同，不作为两个独立集合重复报告。所有数值来自实测。曲线以直线连接离散测量点，中间位置未经评测。',
'','Charades 在本机 RTX 4060 / PyTorch 2.0.1+cu117 上运行，其余三套在组内 GPU 2（RTX 5090 / PyTorch 2.7.1+cu128）上运行。每套内部固定同一环境；官方默认比例均通过原始指标核验。组内 GPU 1 的改进训练队列未修改。',
'','## SumR','',
'| 片段权重 α | Charades | ActivityNet | TVR 单查询 | TVR 双查询 |',
'|---:|---:|---:|---:|---:|']
for i,w in enumerate(weights):
    lines.append('| '+f'{w:.1f}'+' | '+' | '.join(f"{d['points'][i]['SumR']:.2f}" for d in data.values())+' |')
lines+=['','| 设置 | 官方比例 | 官方 SumR | 六点中最佳比例 | 最佳 SumR | 差值 |',
'|---|---:|---:|---:|---:|---:|']
for key,d in data.items():
    b=max(d['points'],key=lambda r:r['SumR'])
    lines.append(f"| {names[key]} | {d['original_clip_weight']:.1f} | {d['baseline']['SumR']:.2f} | {b['clip_weight']:.1f} | {b['SumR']:.2f} | {b['SumR']-d['baseline']['SumR']:+.2f} |")
lines+=['','![SumR 曲线](sumr-curves.png)','','![召回率曲线](recall-curves.png)','',
'六点中的最佳值仅用于描述当前公开评测集上的敏感性，不能作为独立测试提升或全局最优比例。它也不能直接解释重训门控模型的因果效果：门控模型的编码器已经改变。',
'','## 文件','',
'- `metrics.csv`：六个比例下四套模型的全部五项指标。',
'- `cha.json`、`act.json`、`tvr.json`、`tvr-multi.json`：权重哈希、环境、基准核验和原始结果。',
'- `sumr-curves.*`、`recall-curves.*`：PNG 预览，以及可用于汇报的 PDF/SVG 矢量图。',
'- 论文核对：[正文融合公式](../../../papers/holmes-zh/source/sections/Method.tex)、[附录 Parameter Analysis](../../../papers/holmes-zh/source/appendix_sections/Experiments.tex)。',
'- 复跑：`python -B experiments/run_fusion_sweep.py`，然后 `python -B experiments/plot_fusion_sweep.py`。','']
if a.retrained:
    lines=['# Holmes 重训原版融合权重敏感性实验','',
           '固定同环境、seed=0 重训原版在默认比例下选出的 best.pt，扫描片段权重 0、0.2、0.4、0.6、0.8、1.0；不是为每个比例重新训练或重新选模。TVR 双查询额外核验默认 0.5。',
           '默认比例的五项指标与重训 checkpoint 记录核验一致。四套均使用 RTX 5090、官方完整候选库和原评测代码；val/test 描述文件相同，不作为独立测试提升。','']+lines[lines.index('## SumR'):]
    lines=[line.replace('官方比例','默认比例').replace('官方 SumR','重训默认 SumR') for line in lines if not line.startswith('- 复跑：')]
    lines+=['- 复跑：`python experiments/queue_retrained_fusion.py --work <root> --gpu <UUID>`。','']
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines[lines.index('## SumR'):lines.index('![SumR 曲线](sumr-curves.png)')]))

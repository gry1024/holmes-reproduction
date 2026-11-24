"""Compare completed author-weight evaluations at the paper's display precision."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
metrics = ['R@1', 'R@5', 'R@10', 'R@100', 'SumR']
references = [
    ('cha', 'Charades-STA', [2.3, 9.5, 15.2, 53.6, 80.6]),
    ('act', 'ActivityNet Captions', [9.3, 27.8, 40.5, 79.1, 156.8]),
    ('tvr-multi', 'TVR 双查询（论文主表）', [18.4, 40.7, 52.0, 87.5, 198.6]),
    ('tvr', 'TVR 单查询（官方仓库补充）', [17.3, 39.0, 50.4, 87.4, 194.2]),
]
rows = []
report = ['# Holmes 作者权重复评', '',
          '使用官方预提取特征、公开权重及作者的完整检索评估代码。R@K 单位为百分比，SumR 为四项召回率数值之和。表中按论文精度保留一位小数，完整精度记录保存在 results 目录。', '',
          '| 数据集 | 结果来源 | R@1 | R@5 | R@10 | R@100 | SumR |',
          '|---|---|---:|---:|---:|---:|---:|']
counts = []
split_checks = []
checkpoint_matches = []
for key, label, reference in references:
    result = json.loads((root / 'results' / f'{key}-metrics.json').read_text())
    report.append('| ' + label + ' | 论文／官方仓库 | ' + ' | '.join(f'{v:.1f}' for v in reference) + ' |')
    for split in ['val', 'test']:
        entry = result['splits'][split]
        values = entry['metrics']
        assert len(values) == 5
        checkpoint_matches.append(values == result['checkpoint_reported_metrics'][:5])
        report.append('| ' + label + ' | 本机 ' + split + ' | ' + ' | '.join(f'{v:.1f}' for v in values) + ' |')
        counts.append((label, split, entry['queries'], entry['videos'], entry['seconds'], entry['peak_allocated_gpu_bytes']))
        for name, value, expected in zip(metrics, values, reference):
            rows.append({'dataset': label, 'split': split, 'metric': name,
                         'reference_source': 'official_repository' if key == 'tvr' else 'paper_main_table',
                         'reference_reported': expected,
                         'measured_raw': value, 'measured_rounded': round(value, 1),
                         'rounded_difference': round(round(value, 1) - expected, 1)})
    collection = result['config']['collection']
    manifest = json.loads((root / 'results' / f'{collection}-data-manifest.json').read_text())
    val = manifest['splits'][f'TextData/{collection}val.caption.txt']
    test = manifest['splits'][f'TextData/{collection}test.caption.txt']
    split_checks.append((label, val['sha256'] == test['sha256'],
                         val['query_set_sha256'] == test['query_set_sha256'],
                         val['video_set_sha256'] == test['video_set_sha256'],
                         val['caption_set_sha256'] == test['caption_set_sha256']))
report.extend(['', '差值按论文的一位小数精度计算，逐项对照见 [comparison.csv](comparison.csv)。SumR 先对未舍入的召回率求和再显示，因此不一定等于表中四个已舍入数值的和。', '',
               '| 数据集 | 划分 | 查询数 | 候选视频数 | 评测耗时（秒） | 峰值已分配显存（GiB） |',
               '|---|---|---:|---:|---:|---:|'])
for label, split, queries, videos, seconds, memory in counts:
    report.append(f'| {label} | {split} | {queries} | {videos} | {seconds:.2f} | {memory / 2**30:.2f} |')
mismatches = [row for row in rows if row['rounded_difference'] != 0]
report.extend(['', f'共比较 {len(rows)} 项指标（四套权重、两种文件划分、五项指标），其中 {len(mismatches)} 项在发表精度下不一致。', '',
               f'{sum(checkpoint_matches)}/{len(checkpoint_matches)} 次完整评测的五项原始精度结果与 checkpoint 内附带的作者记录逐项相同。对照表使用的是本机评测器实际返回的结果；checkpoint 记录仅用于额外核对。', '',
               '| 数据集 | val/test 文件相同 | 查询集合相同 | 候选视频集合相同 | 描述集合相同 |',
               '|---|---|---|---|---|'])
for label, *checks in split_checks:
    report.append('| ' + label + ' | ' + ' | '.join('是' if value else '否' for value in checks) + ' |')
report.extend(['', f"运行环境为 {result['platform']}，Python {result['python']}，PyTorch {result['torch']}，GPU 为 {result['gpu']}。表中耗时包含评测批次的数据读取、视频／文本编码、全候选打分、排序及指标计算，不包含下载、特征转换和模型加载。显存为 PyTorch 峰值已分配显存，不含驱动等额外占用。",
               '', f"作者代码版本为 `{result['source_commit']}`。将数据路径、输出路径适配到本机，数据加载 num_workers=0，PyTorch CPU 线程数为 4。模型结构、特征预处理、评测 batch size、打分权重及召回率计算沿用作者配置。",
               '', '各结果 JSON 记录权重 SHA-256、checkpoint 自带参考结果及配置差异。数据 manifest 记录压缩包 SHA-256、CRC 检查、查询／视频集合校验，以及特征转换逐值和帧顺序检查。', '',
               '本次完成范围是四份公开权重的完整检索评测，未重新训练模型、消融模型或特征提取器。作者的 Python 版本与 GPU 和本机不同，完整环境差异与重跑命令见 [README.md](README.md)。若 val/test 集合相同，分别运行只用于核对入口一致性，不代表在两个独立数据集上验证。', '',
               '[官方论文](https://arxiv.org/abs/2605.06083) · [官方代码与参考结果](https://github.com/lijun2005/ICML26-Holmes)', ''])
with (root / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
print(json.dumps({'comparisons': len(rows), 'rounded_mismatches': [row for row in rows if row['rounded_difference'] != 0]}, ensure_ascii=False), flush=True)

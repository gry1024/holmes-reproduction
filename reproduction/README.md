# Holmes 公开权重评测

本目录复评论文 *Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval* 的公开权重。范围为 Charades-STA、ActivityNet Captions、TVR 双查询主表结果，以及作者仓库公布的 TVR 单查询结果。各模型都在作者提供的完整查询集和候选视频集上执行原始检索评测。

这是公开权重的推理评测复现。依照本次“先不重新训练”的约定，没有重新训练模型，也没有重新训练论文消融实验中的模型。特征提取使用作者公开的预提取特征。因此，这次结果不能独立证明训练过程、消融实验或特征提取过程的可复现性。

## 指标和评测口径

- R@1、R@5、R@10、R@100：对于每条文本查询，将完整候选视频库按模型得分排序，正确视频进入前 K 名的查询占比，单位为百分比。
- SumR：上述四个召回率的原始数值先相加，再四舍五入显示，不是平均值，也不是准确率。因此它与表中四个已四舍五入数值的和可能相差 0.1。
- 任务是视频检索；没有把时间段定位的 IoU 指标或分类准确率混入比较。
- 沿用作者 `Datasets`、`Models`、`Validations`，每批视频 100、每批查询 50。Charades、ActivityNet、TVR 单查询的片段级／帧级分数权重为 0.6／0.4，TVR 双查询为 0.5／0.5。
- 32 个片段、最长 128 帧、最长 30 个文本 token，以及归一化、候选集合构建、排序与召回率计算均由作者代码执行。模型使用 FP32，没有启用量化或混合精度。
- `val` 与 `test` 文件分别运行，数据清单记录文件、查询集合、候选集合及完整描述集合的哈希，以辨认相同集合或文件命名造成的歧义。不能把文件名直接理解为独立的新测试集。

## 文件

- [工作区总导航](../README.md)：论文、笔记、作者代码及复现文件的入口。
- `RESULTS.md`：全部评测结束后生成的对照报告。
- `comparison.csv`：逐项指标的论文／仓库参考值及其来源、实测原值、一位小数值和按发表精度计算的差值。
- `results/*-metrics.json`：实际运行结果、耗时、显存、权重哈希、配置和环境。`checkpoint_reported_metrics` 是权重内附带的作者参考值；实际测量在 `splits` 中。
- `results/*-data-manifest.json`：原始压缩包和文件哈希、整包 CRC、数据划分与逐值特征转换检查。
- `results/*-evaluation.log`：原始评测输出。
- `results/source-audit.json`：38 个作者 Python 源文件与固定 Git 版本的逐文件核对，仅忽略 Windows 换行符差异。
- `environment.freeze.txt`：本次 Linux 评测环境的依赖版本。
- `scripts/download_drive.py`：校验 Range 响应的可续传下载器。
- `scripts/prepare_data.py`：整包校验、解压和特征转换。转换保留官方引用的 FAWL 转换器的 float64 中间存储，逐值核对所有原始 float32 特征及帧顺序。
- `scripts/evaluate.py`：调用原始评测器的入口，适配数据／输出路径，设置数据加载 `num_workers=0`、PyTorch CPU 线程数为 4。
- `scripts/run_ready.py`：按下载就绪状态执行四套评测。
- `scripts/compare_results.py`：从实际结果生成对照报告。
- `archive/references/`：数据清单和特征转换参考实现等来源材料。
- `archive/setup-logs/`：下载、环境搭建及初始检查的历史日志。
- `archive/待清理/`：未参与当前运行的旧分片与环境；本次自动删除被拦截，暂存于此。详见 [整理记录](archive/CLEANUP-2026-11-24.md)。

## 再次运行

以下命令在 WSL Ubuntu-22.04 内执行，工作目录为论文所在的 `pre` 文件夹：

```bash
cd /mnt/e/just_put_everything_you_need_here/多模态学习/pre
PY=/home/groy/.venvs/holmes-eval/bin/python
"$PY" reproduction/scripts/evaluate.py --dataset cha --checkpoint reproduction/checkpoints/cha.ckpt
"$PY" reproduction/scripts/evaluate.py --dataset act --checkpoint reproduction/checkpoints/act.ckpt
"$PY" reproduction/scripts/evaluate.py --dataset tvr --multi --checkpoint reproduction/checkpoints/tvr-multi.ckpt
"$PY" reproduction/scripts/evaluate.py --dataset tvr --checkpoint reproduction/checkpoints/tvr.ckpt
"$PY" reproduction/scripts/compare_results.py
```

全部数据下载及转换完成后，不需要再次下载或重新转换即可运行上述命令。结果 JSON 会覆盖同名文件，另行试验时请先保留现有结果。

## 环境与来源

本次使用 RTX 4060 Laptop 8GB、WSL2、Python 3.11.15、PyTorch 2.0.1+cu117。作者报告 Charades 训练环境为 Python 3.11.8／3080 Ti，另两套为 Python 3.9.17／A100 40GB；PyTorch 均为 2.0.1。本机与作者硬件及 Python 小版本不同，不能声称整个执行环境完全相同，也不能直接对比训练速度。

代码版本：`73bbbfcfca241418507a9a46691f0cb834e31d8b`。

配置审计保留了仓库默认配置与 checkpoint 保存配置的差异。已确认 ActivityNet 的 `loss_factor` 第四项在仓库中是 `0.0001`，checkpoint 中是 `0.0`；另外还有模型保存路径、困难负样本及 warmup 训练状态差异。本次原始评测器不执行训练损失，ActivityNet 实测结果仍与 checkpoint 完全一致。后续若重新训练，应单独核对这些训练配置，不能把本次推理评测当成训练过程已复现的证明。

- [论文](https://arxiv.org/abs/2605.06083)
- [作者代码、权重和参考指标](https://github.com/lijun2005/ICML26-Holmes)
- [作者发布的数据](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj)
- [作者引用的特征转换实现](https://github.com/BUAAPY/FAWL)

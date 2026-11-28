# 实验运行记录

16 组训练与 4 组重训原版融合权重扫描均已完成。最终结果见[实验总报告](RESULTS.md)，下文保留启动配置与运行方式。

## 重训原版融合权重扫描（已完成）

`queue_retrained_fusion.py` 等四套原版训练完成后，在释放的 GPU 2 依次读取各组 `baseline/best.pt`，扫描 α=0、0.2、0.4、0.6、0.8、1.0；TVR 双查询另核验默认 0.5。不为每个比例重训或重新选模。默认指标必须匹配重训 checkpoint 记录。

状态：远端 `retrained-fusion-status.json`；日志：`retrained-fusion-queue.log`；产物：`experiments/results/fusion-sensitivity-retrained/`，包括 JSON、CSV、PNG/PDF/SVG 曲线及说明。与官方权重扫描分开保存。若基线失败则明确记录失败，不使用官方权重替代。

## 原版重训基线（2026-11-28 补跑）

- 补跑四组：Charades、ActivityNet、TVR 单查询、TVR 双查询；结果位于 `runs/{cha,act,tvr,tvr-multi}/baseline/`。
- GPU 2，RTX 5090，UUID `GPU-d042fd90-2921-cab8-01d2-4ac455db0586`；独立队列，GPU 1 的改进实验继续执行。
- 与改进版共用 `train.py` 的数据、优化器、训练循环、种子 0、batch 128、FP32 和选模规则；使用未经适配的作者原模型、损失及评测器，不加载官方权重。
- 正式运行前自动核对已存在的 pooling 训练配置与 PyTorch 版本；除输出路径外不允许配置不一致。
- Charades 实际数据检查通过：预热前后反向传播、全量验证、权重保存与加载；训练检查显存约 6.30 GiB。检查权重不用于正式训练。
- 状态：`baseline-queue-status.json`；日志：`baseline-queue.log`；每组保存 `last.pt`、`best.pt`、`epochs.jsonl` 和完成后的 `result.json`。
- 改进效果必须与这些重训基线比较；官方 checkpoint 成绩保留为参考。相同随机种子不保证跨 GPU 非确定性计算逐位一致。

```bash
python -B -u experiments/run_suite.py --work /data1/quant_team/holmes-20261127 --gpu GPU-d042fd90-2921-cab8-01d2-4ac455db0586 --variants baseline --queue-name baseline-queue
```

## 原有改进实验

- 方案：[两项改动、三组实验](../notes/Holmes改进实验方案.md)。
- 远端：`quant-team`，`/data1/quant_team/holmes-20261127/`。
- 使用：GPU 1，RTX 5090 32GB，UUID `GPU-3d347ef7-e094-eae1-3405-bbd35d1a8f35`。
- 启动：2026-11-27 18:13（北京时间），已确认 `cha/pooling` 正式训练进程运行；后续按队列顺序执行。
- 环境：独立 Python 3.12 环境，PyTorch 2.7.1+cu128；FP32、关闭 TF32，不改公共环境。
- 三种设置：`pooling`、`gating`、`combined`；按数据集顺序串行运行。
- 状态：各运行目录的 `status.json`；逐轮指标：`epochs.jsonl`；最终评测：`result.json`。
- 断点：`last.pt`（含优化器与随机数状态）；最佳模型：`best.pt`。
- 原作者源码与官方权重复评结果保留，实验结果存入远端 `runs/`。

## 启动前验证

- 3 项单元检查通过：有效位置与短序列、Top-k 梯度、门控初始化/梯度/状态加载。
- batch size 128 的模拟完整训练通过：Charades 三种设置峰值约 7.2 GiB；TVR 双查询最高约 11.65 GiB。这是模拟训练峰值，不是完整数据训练上限。
- Charades 真实数据通过预热前后训练步骤、全量验证、checkpoint 保存加载检查。
- 在新环境复评 Charades 官方权重，原评测器与初始固定权重的适配评测器五项原始指标均与官方记录一致。

## 查看进度

```bash
ssh quant-team 'cat /data1/quant_team/holmes-20261127/queue-status.json'
ssh quant-team 'cat /data1/quant_team/holmes-20261127/runs/cha/pooling/status.json'
```

队列由 `run_suite.py` 脱离终端运行，逐任务检查显卡占用，逐轮保存断点；单任务失败记录原因后继续后续任务。剩余数据上传、转换完成后自动进入后续数据集。没有创建自动唤醒助手的定时任务；当前工具规则不支持用该类自动化跟踪单次作业。

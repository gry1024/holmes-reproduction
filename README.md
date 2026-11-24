# Holmes 论文与复现工作区

日常阅读先看论文和精读笔记，查看复现结论直接打开报告。

| 内容 | 入口 |
|---|---|
| 原始论文 | [Holmes_paper.pdf](Holmes_paper.pdf) |
| 中文精读笔记 | [Holmes论文精读笔记.pdf](notes/Holmes论文精读笔记.pdf) |
| 论文来源 | [作者仓库](https://github.com/lijun2005/ICML26-Holmes) · [原始链接记录](papers/source.txt) |
| 复现结果与论文对照 | [RESULTS.md](reproduction/RESULTS.md) |
| 逐项数值对照 | [comparison.csv](reproduction/comparison.csv) |
| 复现环境与重跑命令 | [复现说明](reproduction/README.md) |
| 本次整理与清理记录 | [整理记录](reproduction/archive/CLEANUP-2026-11-24.md) |

## 从 GitHub 获取工作区

```bash
gh auth login
gh repo clone gry1024/holmes-reproduction -- --recurse-submodules
cd holmes-reproduction
python reproduction/scripts/github_snapshot.py restore
```

代码、论文、笔记和评测报告在 Git 仓库内。`ICML26-Holmes` 是固定到作者代码版本的子模块。正式权重、完整压缩包、已转换的数据及旧下载分片放在 [Release 附件](https://github.com/gry1024/holmes-reproduction/releases/tag/workspace-files-v1)。出现 `snapshot-manifest.json` 表示完整快照已上传。恢复脚本需要 Python 3.9 或更新版本，会校验每个附件的 SHA-256，并恢复到原有相对路径。完整恢复需要约 126 GiB 磁盘空间。

```text
pre/
├── README.md                    总导航
├── Holmes_paper.pdf             原始论文
├── papers/
│   └── source.txt               作者仓库原始链接
├── notes/
│   ├── Holmes论文精读笔记.pdf    阅读版
│   ├── README.md                笔记编译说明
│   └── latex/                   main.tex 与配图
├── ICML26-Holmes/               作者 Git 仓库，保持原样
└── reproduction/
    ├── README.md                运行说明
    ├── RESULTS.md               完整评测报告
    ├── comparison.csv           数值对照
    ├── environment.freeze.txt   Linux 依赖版本
    ├── scripts/                 复现与 GitHub 快照恢复脚本
    ├── checkpoints/             4 份正式模型权重
    ├── data/                    3 套解压、转换后的数据
    ├── downloads/               7 个完整原始压缩包
    ├── results/                 实测指标、校验清单与运行日志
    
```

正式运行仍从本目录启动，原有重跑命令保持有效。数据、正式权重、完整下载包和作者仓库均保留原路径。完整压缩包供 `run_ready.py` 和数据准备流程使用，因此予以保留。



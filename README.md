# Holmes 论文与复现工作区

日常阅读先看论文和精读笔记，查看复现结论直接打开报告。

| 内容 | 入口 |
|---|---|
| 原始论文 | [Holmes_paper.pdf](Holmes_paper.pdf) |
| 中文译版 | [PDF、LaTeX 源码与致谢](papers/holmes-zh/README.md) |
| 中文精读笔记 | [Holmes论文精读笔记.pdf](notes/Holmes论文精读笔记.pdf) |
| 论文来源 | [作者仓库](https://github.com/lijun2005/ICML26-Holmes) · [原始链接记录](papers/source.txt) |
| 复现结果与论文对照 | [RESULTS.md](reproduction/RESULTS.md) |
| 逐项数值对照 | [comparison.csv](reproduction/comparison.csv) |
| 复现环境与重跑命令 | [复现说明](reproduction/README.md) |
| 本次整理与清理记录 | [整理记录](reproduction/archive/CLEANUP-2026-11-24.md) |

## Quickstart

需要 Linux 或 WSL2、Python 3.11 和 NVIDIA CUDA GPU。请先用有仓库访问权限的 GitHub 账号登录 `gh`。论文、笔记和已有评测结果可直接浏览。

### 1. 克隆并安装依赖

```bash
gh repo clone gry1024/holmes-reproduction -- --recurse-submodules
cd holmes-reproduction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r reproduction/environment.freeze.txt
mkdir -p reproduction/downloads
```

### 2. 下载数据与权重

把表中的七个 ZIP 按指定文件名保存到 `reproduction/downloads/`。下载量约 38.2 GiB，建议为下载及转换预留至少 130 GiB。

| 内容 | 官方下载 | 保存文件名 |
|---|---|---|
| Charades-STA 特征 | [Google Drive](https://drive.google.com/file/d/1NVM6wlBDsV2x6x4MswYmCWhzLYIhrhHs/view) | `charades-parallel.zip` |
| ActivityNet Captions 特征 | [Google Drive](https://drive.google.com/file/d/1frCLujoWK1Aj0xGWfMO0U1qQLUrLvXgy/view) | `activitynet.zip` |
| TVR 特征 | [Google Drive](https://drive.google.com/file/d/1RVG3qIy-nf9XdalMw8wpmWNSPcnOAaNK/view) | `tvr.zip` |
| Charades-STA 权重 | [Google Drive](https://drive.google.com/file/d/1qLLNnPx1WiEhQhjBrm9af4YgyDqjPCNg/view) | `Holmes-cha-parallel.zip` |
| ActivityNet Captions 权重 | [Google Drive](https://drive.google.com/file/d/1nNiGnyLLMMoeYeohp9JEcshVJyTWntLi/view) | `Holmes-act.zip` |
| TVR 双查询权重 | [Google Drive](https://drive.google.com/file/d/1BeHOYvYhil59mmDXDSGqhsBVL54W68cr/view) | `Holmes-tvr-multi.zip` |
| TVR 单查询权重 | [Google Drive](https://drive.google.com/file/d/1Zzw9l82enS2DHjMaS7ow_CaOhq5Ht1Fm/view) | `Holmes-tvr.zip` |

### 3. 运行评测

```bash
python -u reproduction/scripts/run_ready.py
```

脚本校验并转换特征、解包权重、执行四组评测，然后更新 `reproduction/RESULTS.md` 和 `reproduction/comparison.csv`。单项重跑命令见 [复现说明](reproduction/README.md)。数据与权重由作者托管。[官方数据目录](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj) · [官方权重与指标](https://github.com/lijun2005/ICML26-Holmes#32-retrieval-performance)

```text
pre/
├── README.md                    总导航
├── Holmes_paper.pdf             原始论文
├── papers/
│   ├── source.txt               作者仓库原始链接
│   └── holmes-zh/               中文译版及 LaTeX 源码
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
    ├── scripts/                 复现脚本
    ├── checkpoints/             解包后的权重，由脚本生成
    ├── data/                    转换后的数据，由脚本生成
    ├── downloads/               用户下载的官方 ZIP
    └── results/                 实测指标、校验清单与运行日志
```



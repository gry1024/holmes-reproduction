# 精读笔记

- [实验总报告](../experiments/RESULTS.md)：原版重训、三组改进、融合权重敏感性分析及训练开销。
- `Holmes论文精读笔记.pdf`：阅读版笔记。
- `latex/main.tex`：可编辑的 LaTeX 源文件。
- [Holmes 改进实验方案](Holmes改进实验方案.md)：两项改动、三组实验与运行口径。
- `latex/holmes_framework.png`：源文件引用的框架图，保持同目录相对路径。

在 Windows 下进入 `latex` 目录，使用 XeLaTeX 编译两次：

```powershell
xelatex main.tex
xelatex main.tex
```

源文件使用 `fontset=windows`。重新编译会生成 `latex/main.pdf`；确认内容后再更新上层阅读版 PDF。本次仅整理文件，没有重新编译。原有 `main.pdf` 与阅读版 SHA-256 完全相同，已删除该重复副本。

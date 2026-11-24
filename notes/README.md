# 精读笔记

- `Holmes论文精读笔记.pdf`：阅读版笔记。
- `latex/main.tex`：可编辑的 LaTeX 源文件。
- `latex/holmes_framework.png`：源文件引用的框架图，保持同目录相对路径。

在 Windows 下进入 `latex` 目录，使用 XeLaTeX 编译两次：

```powershell
xelatex main.tex
xelatex main.tex
```

源文件使用 `fontset=windows`。重新编译会生成 `latex/main.pdf`；确认内容后再更新上层阅读版 PDF。本次仅整理文件，没有重新编译。原有 `main.pdf` 与阅读版 SHA-256 完全相同，已删除该重复副本。

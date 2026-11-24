# 重新审视不确定性：部分相关视频检索中的证据学习

[阅读中文译版 PDF](<Revisiting Uncertainty_ On Evidential Learning for Partially Relevant Video Retrieval.pdf>) · [可编辑 LaTeX 源码](source/main.tex)

非官方中文译版，底本与工作区中的原文 PDF 一致，为 arXiv:2605.06083v1。译文覆盖摘要、正文、图表说明、原论文致谢和影响声明。附录、参考文献和图内文字沿用英文原文。公式、编号和实验数值沿用原稿。[原论文](https://arxiv.org/abs/2605.06083v1)

原论文作者为 Jun Li、Peifeng Lai、Xuhang Lou、Jinpeng Wang、Yuting Wang、Ke Chen、Yaowei Wang 和 Shu-Tao Xia。感谢作者公开论文和复现资源，也感谢 Leey21 提供 LaTeX 源码获取与翻译核查工作流。[作者代码](https://github.com/lijun2005/ICML26-Holmes) · [arxiv-translator](https://github.com/Leey21/arxiv-translator)

该译版对原稿进行了中文翻译和排版适配，按原论文的 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可发布。论文源码中的模板保留其原有声明。

需要重新编译时，进入 `source/`，在带 CTeX、Fandol 及 ICML 模板所需宏包的 TeX Live 环境中运行：

```bash
latexmk -xelatex -halt-on-error -interaction=nonstopmode main.tex
```

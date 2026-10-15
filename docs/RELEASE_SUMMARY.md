# SlideBank 0.1.0 — 首个 GitHub 预览版

日期：2026-10-14。

## 版本目标

把已有 clean backend 和最新 V5 可视化操作页整理成独立 GitHub 仓库，提供
可安装的 Python 包、可移植的配置与数据说明，以及能够部署到 GitHub Pages
的 project page。接入作者提供的论文主图，并发布可查看的 GitHub 与静态项目页。
保留现有推理方法，集中解决公开使用时的入口、路径与依赖问题。

## 与上个版本的差别

相对上一个本地提交 `171e765`，本次加入真实主图 PDF 与 200-DPI PNG，更新 README
和网页引用，清除概念占位图及等待主图的文案，并启用 GitHub Pages。模型和推理代码
没有修改，不增加新的实验结论。

整体 clean-code 基线是 Version 013 / `app_v5.py`，并保留当前 loader 后续已有的
大图分块 thumbnail 与 reg2026 修复。

| 项目 | 上个版本 | 当前版本 |
| --- | --- | --- |
| 操作页入口 | 多个历史 `app_v*.py` 并存 | 只发布最新 V5，统一为 `app.py` |
| 配置与路径 | 本机数据、模型缓存和 Quilt 路径 | 相对路径示例、环境变量、显式外部模型配置 |
| 本地模型选择 | 可能误用 API 配置中的模型 ID | Quilt/Qwen 本地 preset 独立于 API 配置；Send 与 Rebuild 对齐 |
| 图像路径 | 构建时的相对路径可能在导出后失效 | 新构建规范为绝对路径，旧路径仅在源文件可解析时转换 |
| 输入操作 | 上传文案暗示原始 WSI/image 均已支持 | 明确 UI-tree JSON 加载；WSI 构建走 CLI 或已配置切片重建 |
| 模型选项 | 包含未接通的 Anthropic/Mock 和过时 API 预设 | 仅保留支持的 provider，输入实际模型 ID，并前置校验 |
| 安装 | 单一未分层 requirements | `pyproject.toml`，按 UI/WSI/local/Gemini 分组依赖 |
| 项目展示 | Python 操作页 | 保留完整操作页，并加入论文页和浏览器本地 evidence viewer |
| 论文主图 | 本地整理版使用概念占位图 | 作者原始 PDF 与高分辨率 PNG，README/项目页同步展示 |
| 文档 | 实验历史与旧 README | 英文 README、citation、数据/操作页/部署说明、来源归属 |

批量 `scripts/infer.py` 仍采用 clean backend 的 tool-context retrieval；V5 操作页
采用 question policy → router/rule fallback → compact evidence context → final
VLM answer。两者没有合并成同一种推理流程。论文描述的 confidence-based
cross-level consensus 不应视为这两个入口已实现并验证的能力。

## 冗余项清理

只迁移当前运行必需的代码和 ontology，排除旧 UI、`ui_v2_backend.py`、旧 preview
inference、历史版本文档、loader 备份、baseline 环境、原始数据、模型权重、既有
bank、缓存、日志与本机 Gradio 证书。新增配置模板中不包含真实凭据。
本次移除已被论文主图替代的两个概念 SVG；上传的 `Fig1.pdf` 规范命名为
`assets/main-figure.pdf`，原始字节不变，避免保留重复源文件。
原实验目录保持原样。

## 预期实验结果

- 已有 bank 可以导出并加载全局、region、anchor 与 patch 层级，图像位置不会因
  JSON 导出目录改变而被错误重解释。
- 缺失 bank、API key 或模型配置时，操作页在模型调用前给出明确反馈。
- 相同模型、数据和参数下，公开整理版应保持对应旧入口的研究语义；本次修改没有
  提出新的精度提升假设。数值一致性仍需在实际模型和数据上做配对验证。
- 静态 project page 可独立浏览导出的 evidence；真实 bank 构建和 QA 仍需 Python
  服务。浏览器本地 viewer 不调用模型，也不上传选入的 JSON/图片。
- 不把论文报告的准确率或一致性数字当作这次 release 的复现实验结果。

## 验证与实际边界

已完成 18 项 backend/UI 回归、CLI 无第三方依赖的 `--help`、Gradio 实际启动与 HTTP
检查、Python wheel 构建、ontology 打包、配置解析、文档链接与路径/凭据模式扫描。
本次主图来自作者提供的单页 PDF，原件 SHA-256 为
`539a7e11e98c9d2f9a165ee807e18ebbc679a9ccbd5aadf650e0b9f17b50a731`，
按 200 DPI 渲染为 3701 × 1969 PNG；README 与项目页的原始 PDF、PNG 分别一致。

静态页面已用桌面与 390px 移动视口检查，覆盖 JSON 导入、region/anchor 选择、
本地图片匹配、空 bank、错误输入、文本转义、清空和 citation 复制。Chrome 交互通过
`file://` 验证；HTTP 首页和全部本地静态资源通过独立请求验证，均返回 200。

未运行 GPU 推理、付费 API 或完整 benchmark。公开的静态项目页提供论文、主图和
浏览器本地 evidence viewer；在线 Gradio 模型服务尚未部署。

仓库与项目页地址分别为 `https://github.com/gexinh/SlideBank` 和
`https://gexinh.github.io/SlideBank/`。发布后以 GitHub Actions 和远端 HEAD 核验结果
判断本次上传与部署是否成功。

## 许可与发布材料

自有代码按用户选定的 MIT 发布。WSIChat 派生文件保留独立来源说明，其许可仍需与
原作者确认，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。论文 citation 使用
[arXiv 当前元数据](https://arxiv.org/abs/2609.00342)；WACV 录用按作者提供的信息标注，
会议年份与正式 proceedings 条目不作推断。

需要更新主图时，替换 `assets/main-figure.pdf`，再运行
`python scripts/prepare_main_figure.py --dpi 200`，同步 README 和 project page。在线 Gradio
部署地址确定后，只需设置 `project/config.js` 的 `demoUrl`。

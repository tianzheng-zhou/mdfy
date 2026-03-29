# mdfy

AI 驱动的 PDF → Markdown 转换器，基于通义千问（Qwen）多模态大模型，支持**扫描件 OCR** 和**数字 PDF** 两种模式自动识别与转换。

## 特性

- **自动识别 PDF 类型**：采样前几页自动判断扫描件 / 数字 PDF，走不同处理管线
- **多模态 AI 转换**：逐页渲染为图片 + 提取文本层，一起发给 Qwen 视觉模型输出 Markdown
- **扫描件图表检测**：AI 视觉定位 + 像素级 bbox 精修，自动裁切保存图表
- **数字 PDF 混合提取**：嵌入图片提取 + AI 视觉检测，双路合并去重，不丢图
- **跨页智能拼接**：LLM Stitch 三层策略（快速判断 → flash 模型修正 → 正则回退），断句续接零割裂
- **滚动大纲上下文**：从已完成页面提取标题传给后续页，解决超长文档编号漂移
- **通用后处理管线**：11 项格式规范化（标题层级、图片引用、代码围栏、去重等）
- **双模式使用**：命令行 CLI + Web 可视化界面

## 快速开始

### 1. 安装依赖

```bash
pip install PyMuPDF openai Pillow numpy python-dotenv flask
```

### 2. 配置 API Key

本项目使用阿里云 DashScope API，需要设置环境变量：

```bash
# Windows
set DASHSCOPE_API_KEY=sk-xxxxx

# Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxxx
```

也可以在项目根目录创建 `.env` 文件：

```
DASHSCOPE_API_KEY=sk-xxxxx
```

### 3. 运行

**命令行模式：**

```bash
python pdf_to_md_ai.py your_file.pdf
```

**Web 界面模式：**

```bash
python web_app.py
# 浏览器访问 http://127.0.0.1:23504
```

## 命令行参数

```
python pdf_to_md_ai.py <pdf_path> [--model MODEL] [--output DIR]
```

| 参数 | 说明 |
|------|------|
| `pdf_path` | 要转换的 PDF 文件路径 |
| `--model`, `-m` | 模型选择：`qwen3.5-plus`（默认，质量更高）或 `qwen3.5-flash`（更快更便宜） |
| `--output`, `-o` | 输出目录，默认为 PDF 同目录下的同名文件夹 |

## 输出结构

```
your_file/
├── your_file.md     # 转换后的 Markdown 文件
└── images/          # 提取/裁切的图片
    ├── page1_fig1.png
    ├── page2_fig1.png
    └── ...
```

## Web 界面

Web 界面提供完整的可视化转换体验：

- **拖拽上传** PDF 文件（最大 200MB）
- **实时日志** SSE 推送转换进度
- **在线预览** 转换后的 Markdown 内容
- **一键下载** Markdown + 图片压缩包

启动后支持局域网访问，可通过环境变量自定义：

```bash
WEB_HOST=0.0.0.0  # 监听地址，默认 0.0.0.0
WEB_PORT=23504     # 端口号，默认 23504
```

## 模型选择

| 模型 | 特点 |
|------|------|
| `qwen3.5-plus` | 默认推荐。标题层级更稳定，图片引用更准确，基本一次到位 |
| `qwen3.5-flash` | 速度快、成本低。标题层级偶有不稳定，依赖后处理兜底 |

跨页拼接（Stitch）始终使用 `qwen3.5-flash`，因为该步骤仅处理文本，不需要视觉能力。

## 技术架构

```
PDF 输入
  │
  ├─ 自动检测 PDF 类型（扫描件 / 数字PDF）
  │
  ├─ 逐页处理循环
  │   ├─ 渲染页面为图片
  │   ├─ 提取文本层
  │   ├─ 图片提取/检测
  │   │   ├─ 扫描件：AI 视觉定位 → bbox 精修 → 裁切 → AI 验证
  │   │   └─ 数字PDF：嵌入提取 + AI 检测 → 合并去重
  │   ├─ 构建滚动上下文（大纲 + 上一页末尾）
  │   ├─ 调用 Qwen 模型生成 Markdown
  │   └─ 跨页 LLM Stitch 拼接
  │
  ├─ 通用后处理管线
  │   ├─ Markdown 围栏清理
  │   ├─ 图片引用修复
  │   ├─ 标题层级规范化
  │   ├─ 跨页重复标题去重
  │   ├─ 编号步骤提升
  │   └─ ...（共 11 项）
  │
  └─ 输出 .md + images/
```

## 依赖

| 包 | 用途 |
|----|------|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF 解析、页面渲染、图片提取 |
| [openai](https://github.com/openai/openai-python) | 调用 DashScope 兼容 API |
| [Pillow](https://pillow.readthedocs.io/) | 图片处理与压缩 |
| [numpy](https://numpy.org/) | 像素级 bbox 精修分析 |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | 从 .env 加载环境变量 |
| [Flask](https://flask.palletsprojects.com/) | Web 界面后端 |

## License

MIT

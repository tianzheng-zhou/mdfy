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
- **双模式转换**：管线模式（文本+视觉多阶段堈处理）和纯视觉模式（全页渲染+模型 OCR）
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

```
python pdf_to_md_ai.py your_file.pdf

# 指定纯视觉模式
python pdf_to_md_ai.py your_file.pdf --mode vision
```

**Web 界面模式：**

```bash
python web_app.py
# 浏览器访问 http://127.0.0.1:23504
```

## 命令行参数

```
python pdf_to_md_ai.py <pdf_path> [--model MODEL] [--mode MODE] [--output DIR]
```

| 参数 | 说明 |
|------|------|
| `pdf_path` | 要转换的 PDF 文件路径 |
| `--model`, `-m` | 模型选择：`qwen3.5-plus`（默认）、`qwen3.5-flash`、`qwen3.6-plus` |
| `--mode` | 转换模式：`pipeline`（默认，文本提取+图片检测管线）或 `vision`（纯视觉模式） |
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

## API 服务

项目提供完整的 RESTful API（`/api/v1/`），方便其他应用集成调用。

### 快速使用

```python
import requests, time

BASE = "http://localhost:23504/api/v1"
HEADERS = {"X-API-Key": "your-key"}  # 未设置 MDFY_API_KEY 可省略

# 1. 上传 PDF 开始转换
with open("doc.pdf", "rb") as f:
    resp = requests.post(f"{BASE}/convert", headers=HEADERS,
        files={"file": f}, data={"model": "qwen3.5-plus", "mode": "pipeline"})
task_id = resp.json()["data"]["task_id"]

# 2. 轮询等待
while True:
    status = requests.get(f"{BASE}/tasks/{task_id}", headers=HEADERS).json()["data"]
    if status["status"] in ("done", "error"):
        break
    time.sleep(3)

# 3. 获取结果
result = requests.get(f"{BASE}/tasks/{task_id}/result", headers=HEADERS).json()["data"]
print(result["markdown"])
```

### API 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查（无需认证） |
| GET | `/api/v1/models` | 列出可用模型 |
| POST | `/api/v1/convert` | 上传 PDF 并启动转换 |
| GET | `/api/v1/tasks/{id}` | 查询任务状态 |
| GET | `/api/v1/tasks/{id}/progress` | SSE 实时进度流 |
| GET | `/api/v1/tasks/{id}/result` | 获取完整结果（Markdown + 图片列表） |
| GET | `/api/v1/tasks/{id}/result/markdown` | 仅获取 Markdown |
| GET | `/api/v1/tasks/{id}/images` | 列出所有图片 |
| GET | `/api/v1/tasks/{id}/images/{name}` | 获取指定图片 |
| GET | `/api/v1/tasks/{id}/download` | 下载 ZIP 包 |
| GET | `/api/v1/tasks/{id}/download/markdown` | 下载 Markdown 文件 |

### 认证

设置环境变量 `MDFY_API_KEY` 后，所有请求需携带 API Key：

```bash
# Header 方式
curl -H "X-API-Key: your-key" http://localhost:23504/api/v1/models

# Query 参数方式
curl http://localhost:23504/api/v1/models?api_key=your-key
```

未设置 `MDFY_API_KEY` 则无需认证，适合本地 / 内网使用。

### 完整文档

启动服务后访问：
- **交互式文档页面**：http://localhost:23504/api/docs
- **JSON Schema**：http://localhost:23504/api/v1/docs
- **Markdown 文档**：项目根目录的 `API_DOCS.md`

## 模型选择

| 模型 | 特点 |
|------|------|
| `qwen3.5-plus` | 默认推荐。标题层级更稳定，图片引用更准确，基本一次到位 |
| `qwen3.5-flash` | 速度快、成本低。标题层级偶有不稳定，依赖后处理兜底 |
| `qwen3.6-plus` | 新一代模型，能力更强 |

跨页拼接（Stitch）始终使用 `qwen3.5-flash`，因为该步骤仅处理文本，不需要视觉能力。

## 转换模式

| 模式 | 说明 |
|------|------|
| `pipeline` | 默认模式。提取 PDF 文本层 + 图片检测裁切 + AI 视觉转换，多阶段管线处理 |
| `vision` | 纯视觉模式。将每页渲染为图片，完全依赖模型视觉能力进行 OCR 和版面理解，同时自动检测和裁切图片 |

## 技术架构

```
PDF 输入
  │
  ├─ 自动检测 PDF 类型（扫描件 / 数字PDF）
  │
  ├─ 选择转换模式
  │   ├─ pipeline（管线模式）：提取文本层 + 图片检测裁切 + AI 视觉转换
  │   └─ vision（纯视觉）：全页渲染图片 + 图片检测裁切 + AI 视觉 OCR
  │
  ├─ 逐页处理循环
  │   ├─ 渲染页面为图片
  │   ├─ 提取文本层（仅 pipeline 模式）
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

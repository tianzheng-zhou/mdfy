# mdfy

纯视觉 AI 驱动的 PDF → Markdown 转换器。基于阿里云通义千问（Qwen）多模态大模型，将 PDF 逐页渲染为图像，完全依靠视觉能力完成 OCR + 版面理解 + 图表检测与裁切 + 跨页智能拼接。

## 特性

- **极端纯视觉**：`fitz` 只做 PDF 渲染，文字 / 表格 / 公式 / 版面全部交给 Qwen 视觉模型。无文本层提取、无嵌入图解耦、无 scanned/digital 分支。
- **AI 图表检测**：JSON bbox + qwenvl markdown 双路检测 + 邻近聚类 + 多轮 accept/reject/adjust/split AI 精修。
- **跨页装饰图过滤**：数据驱动识别 logo/页眉图标（尺寸签名跨页重复 + 面积占比小）。
- **文档级上下文**：首页推断文档类型/主题/编号风格，贯穿整篇转换减少标题漂移与误判。
- **滚动大纲**：把已完成页面的标题传给后续页，超长文档编号稳定。
- **LLM 跨页拼接**：三层策略（快速判断 → flash 模型修正 → 正则回退 + 边界去重），断句续接零割裂。
- **质量自审 + 重试**：低分页面自动重跑并择优。
- **极简后处理**：只做 7 条结构性修复（图片引用、bbox 泄漏、表格合并、ghost 图片、空行压缩等），格式层级靠 prompt 兜住。
- **双入口**：命令行 CLI + Flask Web 可视化界面。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

本项目使用阿里云 DashScope 的 OpenAI 兼容端点：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxxxx"

# Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxxx
```

或在项目根目录创建 `.env`：

```
DASHSCOPE_API_KEY=sk-xxxxx
```

### 3. 运行

**命令行：**

```bash
python -m mdfy your_file.pdf
# 或
python run.py convert your_file.pdf -m qwen3.5-plus
```

**Web 界面：**

```bash
python run.py serve
# 浏览器访问 http://127.0.0.1:23504
```

## CLI 参数

```
python -m mdfy <pdf_path> [-m MODEL] [-o DIR]
```

| 参数 | 说明 |
|------|------|
| `pdf_path` | 要转换的 PDF 文件路径 |
| `-m`, `--model` | 模型：`qwen3.5-plus`（默认）/ `qwen3.5-flash` / `qwen3.6-plus` |
| `-o`, `--output` | 输出目录（默认：PDF 同目录下的同名文件夹） |

跨页拼接（stitch）固定使用 `qwen3.5-flash`——仅处理文本，不需要视觉能力。

## 输出结构

```
your_file/
├── your_file.md     # 转换后的 Markdown
└── images/          # AI 检测并裁切的图片
    ├── page1_fig1.png
    ├── page2_fig1.png
    └── ...
```

## Web 界面

- 拖拽上传 PDF（最大 200 MB）
- SSE 实时推送转换日志
- 渲染预览 / 源码 / 图片三栏切换
- 一键下载 Markdown 单文件 或 Markdown + 图片 ZIP 包

环境变量：

```
WEB_HOST=0.0.0.0   # 监听地址（默认 0.0.0.0）
WEB_PORT=23504      # 端口号（默认 23504）
```

## 模型选择

| 模型 | 特点 |
|------|------|
| `qwen3.5-plus` | **默认推荐**。视觉 OCR + 版面理解最稳定 |
| `qwen3.5-flash` | 速度快、成本低。小字/公式识别略弱 |
| `qwen3.6-plus` | 新一代模型，能力更强（可用时推荐） |

## 技术架构

```
PDF 输入
  │
  ├─ Phase A: 顺序渲染所有页为 PNG（fitz）
  ├─ Phase A.5: 首页推断文档全局上下文
  │
  ├─ Phase B: 并行图片检测（20 线程）
  │   ├─ JSON bbox 检测（主）
  │   ├─ qwenvl markdown 检测（副）
  │   ├─ 合并去重 + 邻近聚类
  │   ├─ 兜底检测（双路都空时）
  │   ├─ 裁切保存
  │   └─ AI 多轮精修（accept/reject/adjust/split）
  │
  ├─ Phase B.5: 跨页装饰图过滤
  ├─ Phase C: 计算图片位置与覆盖率
  │
  ├─ Phase D: 顺序 AI 转换（需 prev_tail 跨页上下文）
  │   ├─ 单一 SYSTEM_PROMPT + 文档上下文 + 滚动大纲
  │   ├─ 页级质量自审
  │   └─ 低分重试 + 择优
  │
  ├─ Phase E: 保底——未引用的裁切图追加到页末
  ├─ Phase F: 跨页 LLM stitch（边界去重 + 续接 + 回退）
  ├─ Phase G: 极简后处理（7 条结构性修复）
  └─ Phase H: 写 .md + images/
```

## 目录结构

```
mdfy/
├── config.py            # 模型列表、DPI、各阶段图像上限、并发线程数
├── client.py            # DashScope OpenAI 客户端
├── prompts.py           # 主 OCR / 检测 / 校验 / 拼接 / 上下文 / 审查 prompt
├── pdf_render.py        # 页面渲染 + 图像压缩 + bbox 工具 + 检测响应解析
├── figure_detect.py     # 检测 → 裁切 → AI 精修 → 跨页装饰过滤 → 位置计算
├── convert.py           # 单页转换 + 文档上下文推断 + 页级质量审查
├── stitch.py            # 大纲 + 断句判定 + LLM 拼接 + 正则去重回退
├── postprocess.py       # 极简后处理：图片引用/bbox 泄漏/表格/ghost 图片
├── orchestrator.py      # 主流程编排（Phase A → H）
├── __init__.py          # 导出 pdf_to_markdown_ai / AVAILABLE_MODELS / DEFAULT_MODEL
└── __main__.py          # python -m mdfy 入口
```

## 后续路线

- 回归集：在 `test-files/` 下建 8 类 ~30 份自建测试集，配 `tests/smoke_check.py` 做结构性 sanity check。
- 正式 benchmark：在基本流程稳定后拉 [OmniDocBench](https://github.com/opendatalab/OmniDocBench) 跑 981 页的 text edit distance / TEDS / formula CDM，与 MinerU / marker / docling 对比。

## 依赖

| 包 | 用途 |
|----|------|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF 打开与页面渲染 |
| [openai](https://github.com/openai/openai-python) | 调用 DashScope 兼容端点 |
| [Pillow](https://pillow.readthedocs.io/) | 图像压缩、bbox 绘制 |
| [numpy](https://numpy.org/) | 裁切像素级空白判定 |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | 从 .env 加载环境变量 |
| [Flask](https://flask.palletsprojects.com/) | Web 界面后端 |

## License

MIT

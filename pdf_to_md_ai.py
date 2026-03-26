"""
PDF to Markdown —— AI 增强转换（Qwen 多模态）

核心思路：
  PDF 每页渲染为图片 → 连同提取的文本层一起发给 Qwen → 模型直接输出 Markdown
  代码只负责喂数据、收结果、提取/保存图片，通用后处理仅做格式规范化。

使用方式：
  设置环境变量 DASHSCOPE_API_KEY，然后运行：
    python pdf_to_md_ai.py <pdf_path> [--model qwen3.5-flash|qwen3.5-plus]
  或直接修改底部的 pdf_file 路径运行。
"""

import argparse
import base64
import fitz  # PyMuPDF
import os
import re
import sys
import time
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


# ── Qwen API 配置 ──────────────────────────────────────────────────

def get_client():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows:  set DASHSCOPE_API_KEY=sk-xxxxx\n"
            "  Linux:    export DASHSCOPE_API_KEY=sk-xxxxx"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


AVAILABLE_MODELS = ["qwen3.5-flash", "qwen3.5-plus"]
DEFAULT_MODEL = "qwen3.5-flash"


# ── PDF 类型检测 ───────────────────────────────────────────────────

def _is_scan_tile_page(page, doc):
    """检测页面是否为扫描页（整页图或条带拼接）"""
    imgs = page.get_images(full=True)
    if not imgs:
        return False

    page_w_px = page.rect.width * 200 / 72  # 估算 200dpi 像素宽度

    # 情况1：单张整页图片
    if len(imgs) == 1:
        try:
            pix = fitz.Pixmap(doc, imgs[0][0])
            return pix.width > page_w_px * 0.8
        except Exception:
            return False

    # 情况2：多张等宽等高条带（≥3张）
    if len(imgs) >= 3:
        try:
            sizes = []
            for img in imgs:
                pix = fitz.Pixmap(doc, img[0])
                sizes.append((pix.width, pix.height))
            widths = set(s[0] for s in sizes)
            heights = set(s[1] for s in sizes)
            return len(widths) <= 1 and len(heights) <= 1
        except Exception:
            pass

    return False


def _detect_pdf_type(doc):
    """检测 PDF 类型：scanned（扫描件）或 digital（数字PDF）

    扫描件特征：文本层为空 + 图片为页面扫描条带或整页图片
    """
    sample_count = min(5, len(doc))
    text_chars = 0
    scan_page_count = 0

    for i in range(sample_count):
        page = doc[i]
        text_chars += len(page.get_text().strip())
        if _is_scan_tile_page(page, doc):
            scan_page_count += 1

    if text_chars < 50 * sample_count and scan_page_count >= sample_count * 0.6:
        return "scanned"
    return "digital"


# ── PDF 工具函数 ────────────────────────────────────────────────────

def render_page_to_image(page, dpi=200):
    """将 PDF 页面渲染为 PNG 字节"""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def extract_page_text(page):
    """提取页面的纯文本"""
    return page.get_text().strip()


def extract_and_save_images(doc, page, page_num, images_dir):
    """提取并保存页面中的原始图片，返回图片文件名列表"""
    saved = []
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace and pix.colorspace.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            filename = f"page{page_num + 1}_img{img_index + 1}.png"
            pix.save(os.path.join(images_dir, filename))
            saved.append(filename)
            pix = None
        except Exception as e:
            print(f"  ⚠ 第{page_num+1}页图片{img_index+1}提取失败: {e}")
    return saved


# ── 核心：调用 Qwen 多模态 ─────────────────────────────────────────

SYSTEM_PROMPT_SCANNED = """\
<role>你是扫描件 OCR 转 Markdown 助手。将扫描的书页精确识别并转写为结构化 Markdown。</role>

<critical_rules>
<rule id="ocr_accuracy">
精确识别页面上的所有文字，不要遗漏、不要添加。
如果有个别文字无法辨认，用 [?] 标注。
</rule>

<rule id="no_page_artifacts">
忽略页眉、页脚、页码等印刷辅助信息。
忽略水印、装订线等扫描伪影。
</rule>

<rule id="figures">
如果页面上有插图、图表等非文字内容：
- 插图/图片：用 `<!-- 图：[简要描述] -->` 标注其位置和内容
- 表格：尽量转写为 Markdown 表格
</rule>

<rule id="heading_levels">
# 仅用于整本书的书名（整篇文档只出现一次）。
## 用于章标题（如"第一章"、"Chapter 1"等主要章节）。
### 用于节标题（章内的小节）。
正文段落不加任何标题标记。
</rule>
</critical_rules>

<formatting>
- 合并因扫描分页和 PDF 换行造成的断句，还原为通顺的段落。
- 脚注用 [^n] 格式标注。
- 引用段落用 > 标记。
- 保留原文的加粗、斜体等强调格式。
- 不要添加任何原文没有的内容或解释。
- 直接输出 Markdown 正文，不要用 ```markdown 包裹。
</formatting>\
"""

SYSTEM_PROMPT = """\
<role>你是 PDF 转 Markdown 助手。将 PDF 页面截图精确转写为 Markdown。</role>

<critical_rules>
<rule id="no_fabrication">
只输出页面上用文字排版写出的正文内容。
截图/界面图中的 UI 文字（按钮名、标签页名、菜单项等）不属于正文，不要转写为标题或段落。
如果一页上正文文字极少或完全没有，只放图片引用即可。
</rule>

<rule id="no_page_artifacts">
忽略页眉、页脚、页码等印刷辅助信息，不要将它们输出到 Markdown 中。
常见页眉：文档标题、版本号、网址、章节名等在每页顶部/底部重复出现的内容。
</rule>

<rule id="images">
我会告诉你本页有哪些图片文件名。默认每张图片都必须用 ![](images/文件名) 引用。
唯一的例外：如果一张图片的内容**仅仅是代码/脚本文字**，且你已经完整转写为代码块，则可以省略该图片引用。
界面截图、操作步骤截图、示意图、对话框截图、任何包含 GUI 元素的图片，一律保留引用，即使你描述了其内容。
拿不准的时候，保留图片引用。
</rule>

<rule id="code_blocks">
代码、脚本、配置文件、数据数组等技术内容必须用 ``` 围栏包裹。
如果能识别语言，加上语言标识（如 ```python、```c 等）。
如果当前页的数据是上一页的延续，也必须用新的 ``` 围栏包裹。
</rule>

<rule id="heading_levels">
# 仅用于整篇文档的大标题（整篇文档只能出现一次 #，绝对不能有第二个 #）。

对于多级编号章节的文档（如技术手册 1 → 1.1 → 1.1.1）：
- ## 用于一级章节标题（单个编号：1、2、3... 或 "附录 N"）
- ### 用于二级及更深小节标题（如 1.1、2.3、2.3.1、2.3.1.1 等带点号的多级编号）

对于编号步骤的教程文档（如操作教程 1. 2. 3. 代表步骤）：
- ## 用于主步骤（"数字." 格式，如 1. 2. 13.）和无编号的章节标题
- ### 用于子步骤（"数字)" / "数字）" / "a)" / "b)" 格式）

通用规则：
- 目录/TOC 页的条目用列表格式（- 条目），不要用标题标记
- 表格标题（如 "表 1"、"Table 2"、"图 X"）不要用标题标记，用加粗
- 没有编号的说明性文字不要加任何标题标记，直接作为正文段落
</rule>
</critical_rules>

<formatting>
- 修正 PDF 换行造成的断句，合并成通顺的句子。
- 不要添加任何原文没有的内容或解释性文字。
- 直接输出 Markdown 正文，不要用 ```markdown 包裹整个输出。
</formatting>\
"""


def _build_outline(all_md_parts):
    """从已完成的 Markdown 中提取滚动大纲（所有标题）。

    返回类似：
      # 文档标题
      ## 1. xxx
      ### 1) xxx
      ## 2. xxx
      ...
    这个大纲传给每页，让模型知道全局位置，避免超长文档中编号漂移。
    为避免上下文过大，只保留最后 50 个标题。
    """
    headings = []
    for part in all_md_parts:
        for line in part.split('\n'):
            if re.match(r'^#{1,3} ', line):
                headings.append(line)
    # 截断：对于超长文档，只保留最后 50 个标题避免上下文膨胀
    if len(headings) > 50:
        headings = headings[-50:]
    return '\n'.join(headings)


def convert_page_with_ai(client, model, page_png_bytes, page_text, image_filenames,
                          page_num, total_pages, prev_md_tail="", outline="",
                          pdf_type="digital"):
    """调用 Qwen 多模态模型转换单页"""

    is_scanned = (pdf_type == "scanned")
    system_prompt = SYSTEM_PROMPT_SCANNED if is_scanned else SYSTEM_PROMPT

    # 图片清单（扫描件不需要，因为没有可提取的内容图片）
    if not is_scanned and image_filenames:
        img_list = '\n'.join(f'    <img>{f}</img>' for f in image_filenames)
        img_section = (
            f"<images count=\"{len(image_filenames)}\">\n"
            f"{img_list}\n"
            f"  </images>"
        )
    elif not is_scanned:
        img_section = '<images count="0"/>'
    else:
        img_section = ""

    # 文本层
    if page_text:
        text_section = f"<extracted_text>\n{page_text}\n  </extracted_text>"
    elif is_scanned:
        text_section = ""  # 扫描件无文本层，不发送空标签浪费 token
    else:
        text_section = "<extracted_text>（无文本层）</extracted_text>"

    # 滚动上下文：大纲 + 上一页末尾
    if outline:
        outline_section = f"<document_outline>\n{outline}\n  </document_outline>"
    else:
        outline_section = ""

    if prev_md_tail:
        tail_section = f"<previous_page_tail>\n{prev_md_tail}\n  </previous_page_tail>"
    else:
        tail_section = "<previous_page_tail>这是文档第一页</previous_page_tail>"

    # 根据 PDF 类型选择不同的指令
    if is_scanned:
        instructions = (
            f"  <instructions>\n"
            f"    - 精确 OCR 页面上的所有正文文字\n"
            f"    - 延续大纲中的标题层级，不要重复已有的标题\n"
            f"    - 【关键】查看 previous_page_tail，如果本页顶部的文字已经出现在 tail 中，跳过这些重复部分，从新内容开始\n"
            f"    - 合并因分页断开的句子，还原为通顺的段落\n"
            f"    - 忽略页眉、页脚、页码\n"
            f"    - 直接输出 Markdown，不要解释\n"
            f"  </instructions>"
        )
    else:
        instructions = (
            f"  <instructions>\n"
            f"    - 延续大纲中的标题层级和编号，不要重复已有的标题\n"
            f"    - 本页有 {len(image_filenames)} 张图片；仅代码截图已完整转写的可省略，其余必须引用\n"
            f"    - 代码/脚本/数据数组必须用 ``` 围栏包裹\n"
            f"    - 忽略页眉、页脚、页码，不要输出到 Markdown\n"
            f"    - 目录页的条目用列表（- 条目），不要用标题标记\n"
            f"    - 直接输出 Markdown，不要解释\n"
            f"  </instructions>"
        )

    # 组装 user_text，跳过空 section
    sections = [f"  <page number=\"{page_num + 1}\" total=\"{total_pages}\"/>"]
    if outline_section:
        sections.append(f"  {outline_section}")
    sections.append(f"  {tail_section}")
    if img_section:
        sections.append(f"  {img_section}")
    if text_section:
        sections.append(f"  {text_section}")
    sections.append(instructions)

    user_text = "<task>\n" + "\n".join(sections) + "\n</task>"

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(page_png_bytes).decode()}"
            },
        },
        {
            "type": "text",
            "text": user_text,
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=8192,
        extra_body={"enable_thinking": False},
    )

    return response.choices[0].message.content.strip()


# ── 主流程 ──────────────────────────────────────────────────────────

def pdf_to_markdown_ai(pdf_path, output_dir=None, model=None):
    """AI 增强 PDF 转 Markdown 主函数"""
    pdf_path = Path(pdf_path)
    model = model or DEFAULT_MODEL

    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # 检测 PDF 类型
    pdf_type = _detect_pdf_type(doc)

    print(f"📄 正在转换: {pdf_path.name}")
    print(f"📑 总页数: {total_pages}")
    print(f"📋 类型: {pdf_type} ({'扫描件OCR' if pdf_type == 'scanned' else '数字PDF'})")
    print(f"🤖 模型: {model}")
    print(f"📁 输出目录: {output_dir}\n")

    all_md_parts = []

    for page_num in range(total_pages):
        page = doc[page_num]

        # 1. 渲染页面为图片
        page_png = render_page_to_image(page)

        # 2. 提取文本层
        page_text = extract_page_text(page)

        # 3. 提取并保存嵌入的图片（扫描件跳过，因为图片是扫描条带）
        if pdf_type == "scanned":
            image_filenames = []
        else:
            image_filenames = extract_and_save_images(doc, page, page_num, str(images_dir))

        # 4. 构建滚动上下文：大纲 + 上一页末尾
        prev_tail = ""
        if all_md_parts:
            # 扫描件需要更长的尾部来帮助模型去重跨页文本
            tail_len = 800 if pdf_type == "scanned" else 500
            prev_tail = all_md_parts[-1][-tail_len:]
        outline = _build_outline(all_md_parts)

        print(f"  🔄 第 {page_num + 1}/{total_pages} 页...", end=" ", flush=True)
        start = time.time()
        try:
            md_part = convert_page_with_ai(
                client, model, page_png, page_text, image_filenames,
                page_num, total_pages,
                prev_md_tail=prev_tail, outline=outline,
                pdf_type=pdf_type,
            )
            elapsed = time.time() - start
            print(f"✅ ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ ({elapsed:.1f}s) {e}")
            # 降级：使用原始文本
            md_part = page_text if page_text else f"<!-- 第{page_num+1}页转换失败 -->"

        all_md_parts.append(md_part)

    doc.close()

    # 5. 拼接所有页面
    md_content = "\n\n".join(all_md_parts)

    # 统一换行符（API 可能返回 \r\n）
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

    # 清理 Qwen 可能输出的 markdown 代码围栏
    md_content = re.sub(r'^```markdown\s*\n', '', md_content)
    md_content = re.sub(r'\n```\s*$', '', md_content)
    # 也清理中间页面可能出现的围栏
    md_content = re.sub(r'\n```\s*\n\n```markdown\s*\n', '\n\n', md_content)

    # 修复模型生成的坏图片引用格式，如 ![](images/page9[](images/page9_img1.png)
    md_content = re.sub(
        r'!\[\]\(images/page\d+\[]\(images/(page\d+_img\d+\.png)\)',
        r'![](images/\1)',
        md_content,
    )

    # 通用后处理：修复缺少 images/ 前缀的图片引用
    md_content = re.sub(
        r'!\[([^\]]*)\]\((page\d+_img\d+\.png)\)',
        r'![\1](images/\2)',
        md_content,
    )

    # 通用后处理：将 ####+ 标题降为 ###（文档最多 3 级）
    md_content = re.sub(
        r'^#{4,} ',
        '### ',
        md_content,
        flags=re.MULTILINE,
    )

    # 通用后处理：确保整篇文档只有一个 # 标题，后续 # 降为 ##
    # （提前执行，让后续章标题合并等步骤能匹配到统一的 ## 标题）
    first_h1 = re.search(r'^# ', md_content, flags=re.MULTILINE)
    if first_h1:
        before = md_content[:first_h1.end()]
        after = md_content[first_h1.end():]
        after = re.sub(r'^# ', '## ', after, flags=re.MULTILINE)
        md_content = before + after

    # ── 通用后处理：目录页标题条目转为列表（在去重之前执行） ──
    # 检测方式1：寻找 ## 目录 标题
    # 检测方式2：以首个有正文的编号节标题为分界，前方无正文标题全部转为列表
    lines = md_content.split('\n')
    toc_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^## 目录\s*$', line):
            toc_idx = i
            break

    def _heading_has_body(lines, idx):
        """检查标题后 4 行内是否有正文（非标题、非列表、非空行）"""
        for j in range(idx + 1, min(idx + 5, len(lines))):
            stripped = lines[j].strip()
            if stripped and not re.match(r'^#{1,6} ', lines[j]) and not stripped.startswith('- '):
                return True
        return False

    if toc_idx is not None:
        # 方式1：从 ## 目录 往后找到正文开始位置
        toc_end = None
        for i in range(toc_idx + 1, len(lines)):
            if re.match(r'^#{1,6} ', lines[i]) and _heading_has_body(lines, i):
                toc_end = i
                break
            elif lines[i].strip() and not re.match(r'^#{1,6} ', lines[i]) and not lines[i].strip().startswith('- '):
                toc_end = i
                break
        if toc_end and (toc_end - toc_idx) > 5:
            for i in range(toc_idx + 1, toc_end):
                m = re.match(r'^#{1,6} (.+)$', lines[i])
                if m:
                    lines[i] = f'- {m.group(1)}'
            md_content = '\n'.join(lines)
            lines = md_content.split('\n')

    # 方式2：找到首个有正文的编号节标题（## N 文字），其前方所有无正文标题视为目录残留
    first_body_section = None
    for i, line in enumerate(lines):
        if re.match(r'^## \d+ ', line) and _heading_has_body(lines, i):
            first_body_section = i
            break

    if first_body_section:
        changed = False
        for i in range(first_body_section):
            if re.match(r'^#{2,6} ', lines[i]) and not _heading_has_body(lines, i):
                m = re.match(r'^#{2,6} (.+)$', lines[i])
                if m:
                    lines[i] = f'- {m.group(1)}'
                    changed = True
        if changed:
            md_content = '\n'.join(lines)

    # 通用后处理：去重重复出现的相同章节标题
    lines = md_content.split('\n')
    deduped = []
    seen_headings = set()  # 全局追踪所有见过的标题（去重非相邻的重复）
    prev_heading = None
    for line in lines:
        m = re.match(r'^(#{1,3}) (.+)$', line)
        if m:
            heading_key = (m.group(1), m.group(2).strip())
            if heading_key == prev_heading or heading_key in seen_headings:
                continue  # 跳过重复的标题（相邻或非相邻）
            prev_heading = heading_key
            seen_headings.add(heading_key)
        else:
            if line.strip():  # 非空行时重置相邻检查
                prev_heading = None
        deduped.append(line)
    md_content = '\n'.join(deduped)

    # ── 扫描件专用后处理 ──
    if pdf_type == "scanned":
        # 提升裸章号行为 ## 标题（如 "第六章\n## 语言是什么" → "## 第六章\n## 语言是什么"）
        md_content = re.sub(
            r'^(第.+?章)\s*$',
            r'## \1',
            md_content,
            flags=re.MULTILINE,
        )

        # 移除裸页码（独占一行的阿拉伯数字或小写罗马数字）
        md_content = re.sub(
            r'^(?:\d{1,3}|[ivxlc]+)$\n?',
            '',
            md_content,
            flags=re.MULTILINE,
        )

        # 合并拆分的章标题：## 第X章\n[空行]\n## 标题 → ## 第X章 标题
        md_content = re.sub(
            r'^(## 第.+?章)[ \t]*\n(?:\n)?## (.+)$',
            r'\1 \2',
            md_content,
            flags=re.MULTILINE,
        )
        # 也处理英文格式：## Chapter X\n[空行]\n## Title
        md_content = re.sub(
            r'^(## Chapter\s+\d+)[ \t]*\n(?:\n)?## (.+)$',
            r'\1 \2',
            md_content,
            flags=re.MULTILINE,
        )

    # ── 通用后处理：多级编号章节标题层级规范化 ──
    # 检测文档中是否存在多级编号标题（如 2.1、3.2.1 等）
    has_multilevel_sections = bool(re.search(
        r'^#{1,3} \d+\.\d+', md_content, flags=re.MULTILINE
    ))
    if has_multilevel_sections:
        # ## X.Y+ → ### X.Y+（二级及更深的编号用 ###）
        md_content = re.sub(
            r'^## (\d+\.\d+)',
            r'### \1',
            md_content,
            flags=re.MULTILINE,
        )
        # ### N 文字 → ## N 文字（单级编号应为 ##，仅限纯数字后跟空格+非数字）
        md_content = re.sub(
            r'^### (\d+) (\D)',
            r'## \1 \2',
            md_content,
            flags=re.MULTILINE,
        )

    # ── 以下后处理仅适用于数字 PDF（教程/手册），书籍扫描件跳过 ──
    if pdf_type != "scanned":
        # 修复子步骤标题层级（数字+括号应为 ###，不是 ##）
        md_content = re.sub(
            r'^## (\d+[\)）])',
            r'### \1',
            md_content,
            flags=re.MULTILINE,
        )

        # 裸编号子步骤（行首 数字+括号 无标题标记）提升为 ###
        md_content = re.sub(
            r'^(\d+[\)）])',
            r'### \1',
            md_content,
            flags=re.MULTILINE,
        )

        # 智能提升裸编号主步骤（仅当文档存在教程步骤格式 "## N." 时才执行）
        existing_step_nums = [
            int(m.group(1))
            for m in re.finditer(r'^## (\d+)\.', md_content, flags=re.MULTILINE)
        ]
        max_step = max(existing_step_nums) if existing_step_nums else 0

        if max_step > 0:  # 仅教程格式文档才执行步骤提升
            def _promote_step(m):
                num = int(m.group(1))
                if num >= max_step:
                    return f"## {m.group(1)}. "
                return m.group(0)

            md_content = re.sub(
                r'^(\d+)\. ',
                _promote_step,
                md_content,
                flags=re.MULTILINE,
            )

        # 字母编号子步骤层级修复（## a) → ### a)，裸 a) → ### a)）
        md_content = re.sub(
            r'^## ([a-zA-Z][\)）])',
            r'### \1',
            md_content,
            flags=re.MULTILINE,
        )
        md_content = re.sub(
            r'^([a-zA-Z][\)）])',
            r'### \1',
            md_content,
            flags=re.MULTILINE,
        )

    # 通用后处理：裸露的科学计数法数据行自动包裹代码围栏
    lines = md_content.split('\n')
    result_lines = []
    in_code_block = False
    in_bare_data = False
    data_pattern = re.compile(r'^\s*[-+]?\d+\.\d+E[+-]\d+(\s{2,}[-+]?\d+\.\d+E[+-]\d+)*\s*$', re.IGNORECASE)
    for line in lines:
        if line.startswith('```'):
            in_code_block = not in_code_block
            if in_bare_data:
                result_lines.append('```')
                in_bare_data = False
            result_lines.append(line)
            continue
        if not in_code_block and data_pattern.match(line):
            if not in_bare_data:
                result_lines.append('```')
                in_bare_data = True
            result_lines.append(line)
        else:
            if in_bare_data:
                result_lines.append('```')
                in_bare_data = False
            result_lines.append(line)
    if in_bare_data:
        result_lines.append('```')
    md_content = '\n'.join(result_lines)

    md_path = output_dir / f"{pdf_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    img_count = len(list(images_dir.glob("*.png")))
    print(f"\n🎉 转换完成！")
    print(f"   Markdown: {md_path}")
    print(f"   图片: {images_dir} ({img_count}张)")

    return str(md_path)


# ── 入口 ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PDF to Markdown —— AI 增强转换",
    )
    parser.add_argument("pdf", nargs="?",
                        default=r"d:\python_programs\mdfy\test-files\（mjy）Expert 使用教程.pdf",
                        help="要转换的 PDF 文件路径")
    parser.add_argument("--model", "-m", choices=AVAILABLE_MODELS,
                        default=DEFAULT_MODEL,
                        help=f"模型选择（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出目录（默认: PDF 同目录同名文件夹）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pdf_to_markdown_ai(args.pdf, output_dir=args.output, model=args.model)

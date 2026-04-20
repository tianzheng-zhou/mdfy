"""页面转换：单函数 convert_page，纯视觉 + 页级质量审查。"""

import base64
import json
import re

from .prompts import SYSTEM_PROMPT, DOC_CONTEXT_PROMPT, QUALITY_REVIEW_PROMPT_TEMPLATE
from .pdf_render import encode_data_url


# ══════════════════════════════════════════════════════════════════════
# 文档上下文推断
# ══════════════════════════════════════════════════════════════════════

def infer_document_context(client, model, first_page_bytes, first_page_mime):
    """Phase A.5：从首页快速推断文档类型、主题、编号风格等。"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_data_url(first_page_bytes, first_page_mime)}},
                {"type": "text", "text": DOC_CONTEXT_PROMPT},
            ]}],
            temperature=0.0,
            max_tokens=200,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ 文档上下文推断失败: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════
# 单页转换
# ══════════════════════════════════════════════════════════════════════

def convert_page(client, model, page_png_bytes, *,
                 page_num, total_pages,
                 prev_md_tail="", outline="", doc_context="",
                 image_filenames=None, image_positions=None, image_coverage=0,
                 page_image_mime="image/png"):
    """调用 Qwen 视觉模型将单页截图转为 Markdown。

    参数：
        page_png_bytes: 已压缩到 OCR_IMAGE_MAX_SIDE 的页面 PNG/JPEG 字节
        page_num: 0-based
        prev_md_tail: 上一页末尾若干字符（跨页续接）
        outline: 已完成页面的滚动大纲（全局位置感知）
        doc_context: 整篇文档的一句话描述（首页推断得到）
        image_filenames: 本页裁切图文件名列表（按从上到下顺序）
        image_positions: {filename: "~20%-50% 右半"}
        image_coverage: 图片总覆盖面积占比（0-100）

    返回模型输出的 Markdown 字符串。
    """
    image_filenames = image_filenames or []
    image_positions = image_positions or {}

    # ── 图片清单 ──
    if image_filenames:
        img_lines = []
        for f in image_filenames:
            pos_attr = f' pos="{image_positions.get(f, "")}"' if image_positions.get(f) else ''
            img_lines.append(f'    <img{pos_attr}>{f}</img>')
        img_list = '\n'.join(img_lines)
        coverage_attr = f' coverage="{image_coverage}%"' if image_coverage > 0 else ''
        coverage_warning = ""
        if image_coverage >= 30:
            coverage_warning = (
                f"\n    ⚠️ 本页约 {image_coverage}% 的面积被图片覆盖。"
                f"图片区域内的视觉内容已包含在裁切图片中，不要用 Markdown 文字重复描述。"
                f"仅 OCR 转写图片区域外的文字。"
            )
        img_section = (
            f'<images count="{len(image_filenames)}"{coverage_attr} note="已按页面从上到下排序">'
            f'{coverage_warning}\n'
            f'{img_list}\n'
            f'  </images>'
        )
    else:
        img_section = '<images count="0" note="本页未检测到独立图表"/>'

    # ── 滚动大纲 ──
    outline_section = ""
    if outline:
        outline_section = f"<document_outline>\n{outline}\n  </document_outline>"

    # ── 上一页尾巴（含表格/断句提示）──
    if prev_md_tail:
        stripped_tail = prev_md_tail.rstrip()
        is_mid_sentence = bool(stripped_tail) and not re.search(
            r'[。？！…」』\u201d]$|[.?!:;]$|\*\*$|```$',
            stripped_tail,
        )
        ends_with_table = False
        for _tl in reversed(stripped_tail.split('\n')):
            _tl_s = _tl.strip()
            if not _tl_s:
                continue
            if _tl_s.startswith('|') and '|' in _tl_s[1:]:
                ends_with_table = True
            break

        hints = []
        if ends_with_table:
            hints.append(
                "上一页末尾是 Markdown 表格。如果本页顶部是该表格的延续内容，"
                "请直接继续输出表格数据行（保持相同列数 | 分隔格式），不要重复表头和分隔行"
            )
        if is_mid_sentence:
            hints.append("上一页末尾的句子被分页截断，本页输出需要以续接文字开头")

        if hints:
            hints_xml = '\n'.join(f"  <!-- 注意：{h} -->" for h in hints)
            tail_section = (
                f"<previous_page_tail truncated=\"true\">\n{prev_md_tail}\n"
                f"{hints_xml}\n  </previous_page_tail>"
            )
        else:
            tail_section = f"<previous_page_tail>\n{prev_md_tail}\n  </previous_page_tail>"
    else:
        tail_section = "<previous_page_tail>（这是文档第一页）</previous_page_tail>"

    # ── 指令 ──
    instructions = (
        "  <instructions>\n"
        f"    - 这是第 {page_num + 1} 页（共 {total_pages} 页）\n"
        "    - 完全依靠截图进行 OCR，精确转写所有可见文字\n"
        "    - 延续 document_outline 中的标题层级与编号，不要重复已有的标题\n"
        "    - 已裁切的图片按 pos 位置插入 ![](images/文件名) 引用；图片范围内的文字/表格/数据不要重复转写\n"
        "    - 纯表格/纯代码截图已完整转写的可省略图片引用；含图表/示意图的混合内容一律保留引用\n"
        "    - 代码/脚本/数据数组用 ``` 围栏包裹，若能识别语言则加语言标识\n"
        "    - 数学公式用 LaTeX：行内 $...$，独立公式 $$...$$\n"
        "    - 表格用 Markdown 表格（| 列1 | 列2 |），列数一致\n"
        "    - 忽略页眉、页脚、页码\n"
        "    - 目录页条目用 `- 条目` 列表，不要用标题标记\n"
        "    - 绝对不要输出 bbox/坐标/像素注释\n"
        "    - 直接输出 Markdown 正文，不要用 ```markdown 包裹整个输出\n"
        "  </instructions>"
    )

    # ── 组装 ──
    sections = [f"  <page number=\"{page_num + 1}\" total=\"{total_pages}\"/>"]
    if doc_context:
        sections.append(f"  <document_context>{doc_context}</document_context>")
    if outline_section:
        sections.append(f"  {outline_section}")
    sections.append(f"  {tail_section}")
    sections.append(f"  {img_section}")
    sections.append(instructions)

    user_text = "<task>\n" + "\n".join(sections) + "\n</task>"

    user_content = [
        {"type": "image_url", "image_url": {"url": encode_data_url(page_png_bytes, page_image_mime)}},
        {"type": "text", "text": user_text},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=8192,
        extra_body={"enable_thinking": False},
    )

    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════
# 页级质量审查
# ══════════════════════════════════════════════════════════════════════

def review_page_quality(client, model, page_image_bytes, page_image_mime,
                        md_part, image_filenames, page_num):
    """让模型对比 PDF 页图与转换结果，评估质量。返回 (issues, score)。"""
    if not md_part or not md_part.strip():
        return ["空输出"], 0

    md_preview = md_part[:3000] if len(md_part) > 3000 else md_part

    img_info = ""
    if image_filenames:
        img_info = f"本页提供了 {len(image_filenames)} 张裁切图片: {', '.join(image_filenames)}"

    prompt = QUALITY_REVIEW_PROMPT_TEMPLATE.format(
        page_num=page_num + 1,
        img_info=img_info,
        md_preview=md_preview,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_data_url(page_image_bytes, page_image_mime)}},
                {"type": "text", "text": prompt},
            ]}],
            temperature=0.1,
            max_tokens=256,
            extra_body={"enable_thinking": False},
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # LaTeX 反斜杠可能破坏 JSON
            raw = raw.replace('\\', '\\\\')
            result = json.loads(raw)
        score = int(result.get("score", 80))
        issues = result.get("issues", [])
        if isinstance(issues, list):
            return [str(i) for i in issues], score
    except Exception as e:
        print(f"  ⚠ 模型质量审查失败: {e}")

    return [], 80

"""核心页面转换：pipeline 模式 + vision 模式单页 AI 转换。"""

import base64
import re

from .config import OCR_IMAGE_MAX_SIDE
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SCANNED, SYSTEM_PROMPT_VISION
from .pdf_utils import prepare_image_for_model, render_page_to_image
from .image_detect import detect_page_figures, crop_and_save_figures, _ai_verify_and_refine_crops
from .image_merge import _merge_ai_and_embedded_images, _compute_image_positions


def convert_page_with_ai(client, model, page_png_bytes, page_text, image_filenames,
                          page_num, total_pages, prev_md_tail="", outline="",
                          pdf_type="digital", page_image_mime="image/png",
                          image_positions=None, image_coverage=0,
                          text_in_images=None):
    """调用 Qwen 多模态模型转换单页"""

    is_scanned = (pdf_type == "scanned")
    system_prompt = SYSTEM_PROMPT_SCANNED if is_scanned else SYSTEM_PROMPT

    # 图片清单（带位置提示，按从上到下排序）
    if image_filenames:
        if image_positions:
            img_lines = []
            for f in image_filenames:
                pos_attr = f' pos="{image_positions.get(f, "")}"'
                # 标注图片内已有的文字（帮助模型避免重复转写）
                overlap_attr = ''
                if text_in_images and f in text_in_images:
                    overlap_attr = f' contains_text="{text_in_images[f]}"'
                img_lines.append(f'    <img{pos_attr}{overlap_attr}>{f}</img>')
            img_list = '\n'.join(img_lines)
        else:
            img_list = '\n'.join(f'    <img>{f}</img>' for f in image_filenames)
        # 图片覆盖率高时（>30%页面面积），加重警告避免重复转写
        # 但若文本层内容丰富（>100字符），说明图片与文字共存，不应压制文字转写
        coverage_attr = f' coverage="{image_coverage}%"' if image_coverage > 0 else ''
        coverage_warning = ""
        text_is_rich = len(page_text.strip()) > 50 if page_text else False
        if image_coverage >= 30 and not text_is_rich:
            coverage_warning = (
                f"\n    ⚠️ 本页约 {image_coverage}% 的面积被图片覆盖。"
                f"大部分可见内容已包含在图片中。严格遵守规则："
                f"仅输出图片区域外的标题/正文，然后引用图片。"
                f"图片区域内的表格、数据、公式、文字一律不要转写。"
            )
        elif image_coverage >= 30 and text_is_rich:
            coverage_warning = (
                f"\n    ⚠️ 本页约 {image_coverage}% 的面积被图片覆盖，但文本层内容丰富。"
                f"请正常转写文本层中的所有文字、公式和内容，在合适位置插入图片引用。"
                f"仅跳过 contains_text 属性中已列出的文字（避免重复）。"
            )
        img_section = (
            f"<images count=\"{len(image_filenames)}\"{coverage_attr} note=\"已按页面从上到下排序\">"
            f"{coverage_warning}\n"
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

    def _extract_leading_table_cells(raw_text, expected_cols):
        if not raw_text or expected_cols < 2:
            return None
        top_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not top_lines:
            return None

        start = 0
        while start < len(top_lines) and (
            '用户手册' in top_lines[start]
            or re.search(r'^www\.', top_lines[start], re.IGNORECASE)
        ):
            start += 1
        if start < len(top_lines) and re.fullmatch(r'\d+', top_lines[start]):
            start += 1

        candidates = top_lines[start:start + 12]
        if expected_cols == 3:
            for idx in range(0, len(candidates) - 2):
                cells = candidates[idx:idx + 3]
                if not re.fullmatch(r'\d+[A-Za-z]?', cells[0]):
                    continue
                if any(len(cell) > 60 for cell in cells):
                    continue
                next_line = candidates[idx + 3] if idx + 3 < len(candidates) else ''
                if next_line and not (
                    next_line.startswith(('注：', '注:')) or len(next_line) > 40
                ):
                    continue
                return [re.sub(r'\s+', ' ', cell).strip() for cell in cells]
        return None

    table_continuation_section = ""

    if prev_md_tail:
        # 检测 tail 末尾是否在句子中间被截断
        stripped_tail = prev_md_tail.rstrip()
        # 完整句子结束标志：中文句号、问号、感叹号、英文句点+空格/换行、冒号、引号闭合等
        is_mid_sentence = bool(stripped_tail) and not re.search(
            r'[。？！…」』\u201d]$|[.?!:;]$|\*\*$|```$',
            stripped_tail,
        )
        # 检测 tail 末尾是否在表格中（最后非空行以 | 开头且包含多个 |）
        ends_with_table = False
        prev_table_col_count = 0
        for _tl in reversed(stripped_tail.split('\n')):
            _tl_s = _tl.strip()
            if not _tl_s:
                continue
            if _tl_s.startswith('|') and '|' in _tl_s[1:]:
                ends_with_table = True
                prev_table_col_count = _tl_s.count('|') - 1
            break

        truncation_hints = []
        if ends_with_table:
            truncation_hints.append(
                "上一页末尾是一个 Markdown 表格。如果本页顶部是该表格的延续内容，"
                "请直接继续输出表格数据行（保持相同的 | 分隔列格式），不要重复表头和分隔行"
            )
            leading_cells = _extract_leading_table_cells(page_text, prev_table_col_count)
            if leading_cells:
                candidate_row = f"| {' | '.join(leading_cells)} |"
                truncation_hints.append(
                    f"当前页顶部检测到一行续表候选：{candidate_row}。"
                    "如果页面内容一致，必须完整输出这一整行表格，不能只保留后面的注释或正文"
                )
                table_continuation_section = (
                    "<detected_table_continuation>\n"
                    f"  <expected_row>{candidate_row}</expected_row>\n"
                    "</detected_table_continuation>"
                )
        if is_mid_sentence:
            truncation_hints.append("上一页末尾的句子被分页截断了，本页输出需要以续接文字开头")

        if truncation_hints:
            hints_xml = '\n'.join(f"  <!-- 注意：{h} -->" for h in truncation_hints)
            tail_section = (
                f"<previous_page_tail truncated=\"true\">\n{prev_md_tail}\n"
                f"{hints_xml}\n"
                f"  </previous_page_tail>"
            )
        else:
            tail_section = f"<previous_page_tail>\n{prev_md_tail}\n  </previous_page_tail>"
    else:
        tail_section = "<previous_page_tail>这是文档第一页</previous_page_tail>"

    # 根据 PDF 类型选择不同的指令
    if is_scanned:
        img_instructions = ""
        if image_filenames:
            img_instructions = f"    - 本页有 {len(image_filenames)} 张已裁切的图片（已按页面从上到下排序，pos 表示覆盖范围，contains_text 列出图内已有的文字）；请在对应位置放置图片引用 ![描述](images/文件名)，图片范围内的文字/表格/数据不要重复转写\n"
        instructions = (
            f"  <instructions>\n"
            f"    - 精确 OCR 页面上的所有正文文字\n"
            f"    - 延续大纲中的标题层级，不要重复已有的标题\n"
            f"    - 【关键·跨页连续】查看 previous_page_tail 的末尾：\n"
            f"      * 如果 tail 末尾不是完整句子（没有以。？！等结束），说明上一页句子被分页截断\n"
            f"      * 此时你的输出必须以本页续接的文字开头，直接接上被截断的句子，不要另起段落\n"
            f"      * ⚠️ 严禁重复：tail 中的文字已经被记录，你只输出本页新的内容。不要把 tail 最后一句再输出一遍。\n"
            f"      * 如果 tail 末尾是完整句子，正常从本页新内容开始即可\n"
            f"    - 续接完成后，如果页面上有章/节标题，仍必须使用 ## 或 ### 标记\n"
            f"    - 如果本页底部的句子没有说完（被下一页截断），照常输出到可见内容末尾\n"
            f"    - 忽略页眉、页脚、页码\n"
            f"{img_instructions}"
            f"    - 直接输出 Markdown，不要解释\n"
            f"  </instructions>"
        )
    else:
        # 通用：当文本层极薄时（<50字符），提示模型依赖截图 OCR
        ocr_hint = ""
        if len(page_text) < 50:
            ocr_hint = "    - ⚠️ 本页 PDF 文本层极少或缺失，请直接从页面截图中 OCR 读取所有可见的文字、公式和内容\n"
        instructions = (
            f"  <instructions>\n"
            f"{ocr_hint}"
            f"    - 延续大纲中的标题层级和编号，不要重复已有的标题\n"
            f"    - 本页有 {len(image_filenames)} 张图片（已按页面从上到下排序，pos 表示覆盖范围，contains_text 列出图内已有的文字）；图片范围内的文字/表格/数据不要重复转写；仅纯代码/纯表格截图已完整转写的可省略，混合内容图片一律保留引用\n"
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
    if table_continuation_section:
        sections.append(f"  {table_continuation_section}")
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
                "url": f"data:{page_image_mime};base64,{base64.b64encode(page_png_bytes).decode()}"
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
            {"role": "system", "content": [
                {"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=8192,
        extra_body={"enable_thinking": False},
    )

    return response.choices[0].message.content.strip()


def _detect_figures_for_page(client, model, page_png, page_num, images_dir,
                             pdf_type, embedded_filenames=None, doc_context=""):
    """并行阶段：检测 + 裁切 + AI 验证单页的图表区域。无 fitz 依赖。

    返回 fig_results (list of (filename, desc, pixel_bbox)) 或 None。
    """
    try:
        figures = detect_page_figures(client, model, page_png, page_num,
                                     doc_context=doc_context)
    except Exception as e:
        print(f"  ⚠ 第{page_num+1}页图片检测失败: {e}")
        figures = []

    if not figures:
        return None

    fig_results = crop_and_save_figures(page_png, figures, page_num, str(images_dir))
    if fig_results:
        kwargs = {}
        if pdf_type != "scanned" and embedded_filenames:
            kwargs["other_images"] = embedded_filenames
        fig_results = _ai_verify_and_refine_crops(
            client, model, page_png, fig_results,
            page_num, str(images_dir),
            doc_context=doc_context,
            **kwargs,
        )
    return fig_results if fig_results else None


def _convert_page_vision(client, model, page_png_bytes, page_num, total_pages,
                         prev_md_tail="", outline="", page_image_mime="image/png",
                         image_filenames=None, image_positions=None,
                         image_coverage=0, doc_context=""):
    """Vision 模式：纯视觉页面转换，完全依赖模型 OCR + 视觉理解。

    与 pipeline 模式的 convert_page_with_ai 类似，但不传文本层，
    完全依赖模型从截图中 OCR 读取文字。
    """
    image_filenames = image_filenames or []

    # 构建图片清单（与 pipeline 模式复用相同格式）
    img_section = ""
    if image_filenames:
        if image_positions:
            img_lines = []
            for f in image_filenames:
                pos_attr = f' pos="{image_positions.get(f, "")}"'
                img_lines.append(f'    <img{pos_attr}>{f}</img>')
            img_list = '\n'.join(img_lines)
        else:
            img_list = '\n'.join(f'    <img>{f}</img>' for f in image_filenames)
        coverage_attr = f' coverage="{image_coverage}%"' if image_coverage > 0 else ''
        coverage_warning = ""
        if image_coverage >= 30:
            coverage_warning = (
                f"\n    ⚠️ 本页约 {image_coverage}% 的面积被图片覆盖。"
                f"图片区域内的视觉内容已包含在裁切图片中，不需要用文字描述。"
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

    # 构建大纲部分
    outline_section = ""
    if outline:
        outline_section = f"  <document_outline>\n{outline}\n  </document_outline>"

    # 构建 prev_tail 部分
    tail_section = "  <previous_page_tail>（这是第一页）</previous_page_tail>"
    if prev_md_tail:
        tail_section = (
            f"  <previous_page_tail truncated=\"true\">\n"
            f"{prev_md_tail}\n"
            f"  </previous_page_tail>"
        )

    # 指令
    instructions = (
        f"  <instructions>\n"
        f"    - 这是第 {page_num + 1} 页，共 {total_pages} 页\n"
        f"    - 完全依靠截图进行 OCR，精确转写所有可见文字\n"
        f"    - 已裁切的图片按 pos 位置插入 ![](images/文件名) 引用\n"
        f"    - 表格必须转写为 Markdown 表格\n"
        f"    - 代码用 ``` 围栏包裹\n"
        f"    - 数学公式用 LaTeX 语法\n"
        f"    - 直接输出 Markdown，不要解释\n"
        f"  </instructions>"
    )

    # 组装 user_text
    sections = [f"  <page number=\"{page_num + 1}\" total=\"{total_pages}\"/>"]
    if doc_context:
        sections.append(f"  <document_context>{doc_context}</document_context>")
    if outline_section:
        sections.append(outline_section)
    sections.append(tail_section)
    if img_section:
        sections.append(f"  {img_section}")
    sections.append(instructions)

    user_text = "<task>\n" + "\n".join(sections) + "\n</task>"

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{page_image_mime};base64,{base64.b64encode(page_png_bytes).decode()}"
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
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_PROMPT_VISION,
                 "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=8192,
        extra_body={"enable_thinking": False},
    )

    return response.choices[0].message.content.strip()

"""
PDF to Markdown 转换脚本
使用 PyMuPDF 提取文本块和图片，按页面中的垂直位置排序，
将文本和图片交织生成 Markdown 文件。
"""

import fitz  # PyMuPDF
import os
import re
import sys
from pathlib import Path


def extract_page_elements(doc, page, page_num, images_dir):
    """提取单页中的所有元素（文本块+图片），按垂直位置排序"""
    elements = []  # (y_position, type, content)

    # 1. 提取文本块（带位置信息）
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block["type"] == 0:  # 文本块
            y0 = block["bbox"][1]
            # 组装文本块内容
            lines = []
            for line in block["lines"]:
                spans_text = ""
                for span in line["spans"]:
                    spans_text += span["text"]
                spans_text = spans_text.strip()
                if spans_text:
                    lines.append(spans_text)
            text = "\n".join(lines).strip()
            if text:
                elements.append((y0, "text", text, block))

    # 2. 提取图片（带位置信息）
    image_list = page.get_images(full=True)

    # 获取图片在页面上的位置
    for img_index, img in enumerate(image_list):
        xref = img[0]
        # 获取图片在页面上的矩形位置
        img_rects = page.get_image_rects(xref)
        if img_rects:
            rect = img_rects[0]
            y0 = rect.y0

            # 保存图片
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_filename = f"page{page_num + 1}_img{img_index + 1}.png"
                img_path = os.path.join(images_dir, img_filename)
                pix.save(img_path)
                pix = None

                elements.append((y0, "image", img_filename, rect))
            except Exception as e:
                print(f"  警告: 第{page_num+1}页图片{img_index+1}提取失败: {e}")

    # 按 y 坐标排序
    elements.sort(key=lambda x: x[0])
    return elements


def join_broken_lines(text):
    """合并因 PDF 换行导致的碎片化文本行"""
    lines = text.split('\n')
    if len(lines) <= 1:
        return text

    merged = []
    i = 0
    while i < len(lines):
        current = lines[i].strip()
        if not current:
            i += 1
            continue

        # 如果当前行不以标点结尾，且下一行存在且不是编号开头，则合并
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line:
                break
            # 如果下一行是编号开头，不合并
            if re.match(r'^\d+[\.\)）、]', next_line):
                break
            # 如果当前行是纯编号（如 "4."），不在此处合并，留给标题检测
            if re.match(r'^\d+[\.\)）、]\s*$', current):
                break
            # 如果当前行以中文标点或英文句末标点结尾，不合并
            if current and current[-1] in '。！？；…!?':
                break
            # 合并
            current = current + next_line
            i += 1

        merged.append(current)
        i += 1

    return '\n'.join(merged)


def detect_heading(text):
    """检测是否为标题/编号步骤，返回 (type, text)"""
    text = text.strip()

    # 匹配纯数字编号行如 "1." "2." "3)"
    if re.match(r'^\d+[\.\)）、]\s*$', text):
        return "step_number", text

    # 检查是否是 "数字.\n内容" 或 "数字)\n内容" 的多行文本块
    m = re.match(r'^(\d+[\.\)）、])\s*\n(.+)', text, re.DOTALL)
    if m:
        return "step_number_with_body", text

    # 匹配子步骤（如 "1）内容" "2）内容"）
    m = re.match(r'^(\d+)\s*[\)）]\s+(.+)', text)
    if m:
        return "substep", text

    # 匹配 "数字. 内容" 主步骤
    m = re.match(r'^(\d+)\.\s+(.+)', text)
    if m:
        return "step_with_content", text

    return None, text


def format_text_block(text, pending_step=None):
    """将文本块格式化为 Markdown"""
    lines = text.split('\n')
    result_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        result_lines.append(line)

    if not result_lines:
        return ""

    combined = "\n".join(result_lines)
    return combined


def post_process_markdown(md_content):
    """后处理 Markdown 内容，修复跨文本块断行等问题"""
    lines = md_content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 如果当前行是普通文本（非空、非标题、非图片），
        # 检查是否需要和下一个非空行合并
        if (line.strip()
                and not line.startswith('#')
                and not line.startswith('![')
                and not line.startswith('---')):

            # 查找下一个非空行
            j = i + 1
            blank_count = 0
            while j < len(lines) and lines[j].strip() == '':
                blank_count += 1
                j += 1

            # 如果有恰好1个空行分隔，且下一个非空行也是普通文本
            if (blank_count == 1 and j < len(lines)
                    and lines[j].strip()
                    and not lines[j].startswith('#')
                    and not lines[j].startswith('![')
                    and not lines[j].startswith('---')):

                current_stripped = line.rstrip()
                next_stripped = lines[j].strip()

                # 判断是否应该合并：当前行不以句末标点结尾，
                # 且下一行不以编号或特殊字符开头
                should_merge = False
                if current_stripped and current_stripped[-1] not in '。！？；…!?':
                    # 不以句末标点结尾
                    if not re.match(r'^\d+[\.\)）、]', next_stripped):
                        # 下一行不是编号开头
                        # 额外判断：如果当前行末尾是中文字符且下一行开头是中文字符，很可能是断行
                        if (len(current_stripped) > 0 and
                            (ord(current_stripped[-1]) > 0x4e00 or current_stripped[-1] in '，：、')):
                            should_merge = True

                if should_merge:
                    result.append(current_stripped + next_stripped)
                    i = j + 1
                    continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def pdf_to_markdown(pdf_path, output_dir=None):
    """主转换函数"""
    pdf_path = Path(pdf_path)

    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    md_lines = []
    pending_step_number = None  # 跟踪独立的步骤编号

    print(f"正在转换: {pdf_path.name}")
    print(f"总页数: {len(doc)}")
    print(f"输出目录: {output_dir}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        elements = extract_page_elements(doc, page, page_num, str(images_dir))

        print(f"  第{page_num+1}页: {len(elements)}个元素")

        for y0, elem_type, content, extra in elements:
            if elem_type == "text":
                # 先合并 PDF 断行
                content = join_broken_lines(content)
                heading_type, heading_text = detect_heading(content)

                if heading_type == "step_number":
                    pending_step_number = heading_text
                    continue

                if heading_type == "step_number_with_body":
                    # "4.\n长文本" 格式：拆分为标题+正文
                    m = re.match(r'^(\d+[\.\)）、])\s*\n(.+)', content, re.DOTALL)
                    if m:
                        step_label = m.group(1)
                        body = m.group(2).strip()
                        # 如果有待处理的步骤号，先清除
                        pending_step_number = None
                        md_lines.append("")
                        md_lines.append(f"### {step_label} {body.split(chr(10))[0]}")
                        md_lines.append("")
                        # 如果有多行正文
                        remaining = '\n'.join(body.split('\n')[1:]).strip()
                        if remaining:
                            md_lines.append(remaining)
                            md_lines.append("")
                        continue

                if pending_step_number:
                    # 上一个是纯编号行，合并为标题（取第一行作为标题，剩余作为正文）
                    lines = content.split('\n')
                    first_line = lines[0].strip()
                    rest = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

                    md_lines.append("")
                    md_lines.append(f"### {pending_step_number} {first_line}")
                    md_lines.append("")
                    if rest:
                        md_lines.append(rest)
                        md_lines.append("")
                    pending_step_number = None
                    continue

                if heading_type in ("substep", "step_with_content"):
                    # 提取编号和内容
                    m = re.match(r'^(\d+[\.\)）])\s*(.*)', content)
                    if m:
                        label = m.group(1)
                        body = m.group(2).strip()
                        md_lines.append("")
                        md_lines.append(f"### {label} {body}")
                        md_lines.append("")
                    else:
                        md_lines.append("")
                        md_lines.append(f"### {content}")
                        md_lines.append("")
                    continue

                # 普通文本
                formatted = format_text_block(content)
                if formatted:
                    if page_num == 0 and y0 < 100 and len(content) < 60:
                        md_lines.append(f"# {content}")
                        md_lines.append("")
                    else:
                        md_lines.append(formatted)
                        md_lines.append("")

            elif elem_type == "image":
                if pending_step_number:
                    md_lines.append("")
                    md_lines.append(f"### {pending_step_number}")
                    md_lines.append("")
                    pending_step_number = None

                md_lines.append(f"![](images/{content})")
                md_lines.append("")

    doc.close()

    # 写入 Markdown 文件
    md_content = "\n".join(md_lines)
    # 清理多余空行（3个及以上空行替换为2个）
    md_content = re.sub(r'\n{4,}', '\n\n\n', md_content)

    # 后处理：修复跨文本块的断行
    md_content = post_process_markdown(md_content)

    md_path = output_dir / f"{pdf_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n转换完成！")
    print(f"Markdown 文件: {md_path}")
    print(f"图片目录: {images_dir}")
    print(f"共提取图片: {len(list(images_dir.glob('*.png')))}张")

    return str(md_path)


if __name__ == "__main__":
    pdf_file = r"d:\python_programs\mdfy\test-files\（mjy）Expert 使用教程.pdf"
    pdf_to_markdown(pdf_file)

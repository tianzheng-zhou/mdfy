"""极简后处理：只修结构性缺陷，格式/标题层级交给 prompt + doc_context。"""

import re
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# 结构性修复：图片引用
# ══════════════════════════════════════════════════════════════════════

def _fix_image_refs(text):
    """修复模型输出中常见的图片引用格式错误。"""
    # 1. 剥掉整体被 ```markdown ... ``` 包裹的情况
    text = re.sub(r'^```markdown\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    text = re.sub(r'\n```\s*\n\n```markdown\s*\n', '\n\n', text)

    # 2. 模型循环接缝处产生的坏图片引用（page 后有数字）
    text = re.sub(
        r'!\[\]\(images/page\d*\[]\(images/(page\d+_(?:img|fig)\d+\.png)\)',
        r'![](images/\1)',
        text,
    )

    # 3. 缺少 ! 的图片引用：[](images/x.png) → ![](images/x.png)
    text = re.sub(r'(?<!\!)\[\]\(images/', '![](images/', text)

    # 4. 缺少 images/ 前缀：![](page1_fig1.png) → ![](images/page1_fig1.png)
    text = re.sub(
        r'!\[([^\]]*)\]\((page\d+_(?:img|fig)\d+\.png)\)',
        r'![\1](images/\2)',
        text,
    )

    # 5. 拆分粘连在同一行的多个图片引用
    text = re.sub(
        r'(!\[[^\]]*\]\(images/[^)]+\.png\))[ \t]*(!\[[^\]]*\]\(images/)',
        r'\1\n\2',
        text,
    )

    return text


# ══════════════════════════════════════════════════════════════════════
# 结构性修复：bbox 坐标泄漏 & HTML 注释
# ══════════════════════════════════════════════════════════════════════

def _strip_bbox_leak(text):
    """清理模型偶尔泄漏的 `<!-- Image (x,y,w,h) -->` 坐标注释。"""
    return re.sub(
        r'^<!--\s*Image\s*\([\d,\s.]+\)\s*-->\s*$',
        '',
        text,
        flags=re.MULTILINE,
    )


# ══════════════════════════════════════════════════════════════════════
# 结构性修复：单个 # 标题
# ══════════════════════════════════════════════════════════════════════

def _single_h1(text):
    """整篇文档只允许一个 `#` 标题，后续 `#` 降为 `##`。"""
    first_h1 = re.search(r'^# ', text, flags=re.MULTILINE)
    if not first_h1:
        return text
    before = text[:first_h1.end()]
    after = text[first_h1.end():]
    after = re.sub(r'^# ', '## ', after, flags=re.MULTILINE)
    return before + after


# ══════════════════════════════════════════════════════════════════════
# 结构性修复：Markdown 表格合并（跨页/跨行）
# ══════════════════════════════════════════════════════════════════════

def _merge_split_tables(text):
    """修复被分页与硬换行拆坏的 Markdown 表格。"""

    def _starts_table_like(line):
        s = line.strip()
        return s.startswith('|') and s.count('|') >= 2

    def _is_separator(line):
        return bool(re.match(r'^\s*\|[\s:\-|]+\|\s*$', line.strip()))

    def _normalize_row(row):
        row = row.strip()
        if not _starts_table_like(row):
            return row
        if row.endswith('|'):
            return row
        last_pipe = row.rfind('|')
        before = row[:last_pipe].rstrip()
        after = row[last_pipe + 1:].strip()
        if not after:
            return before + ' |'
        return before + '<br>' + after + ' |'

    def _append_last_cell(row, extra):
        row = _normalize_row(row).rstrip()
        last_pipe = row.rfind('|')
        before = row[:last_pipe].rstrip()
        return before + '<br>' + extra + ' |'

    def _is_block_boundary(line):
        stripped = line.strip()
        if not stripped:
            return False
        return (
            stripped.startswith(('#', '![', '- ', '* ', '> ', '```', '**'))
            or stripped.startswith(('注：', '注:', 'Note:', 'NOTE:'))
        )

    def _normalize_wrapped_table_rows(source):
        lines = source.split('\n')
        normalized = []
        current_row = None
        pending_break = False

        def _flush_row():
            nonlocal current_row, pending_break
            if current_row is None:
                return
            normalized.append(_normalize_row(current_row))
            current_row = None
            pending_break = False

        for line in lines:
            stripped = line.strip()
            if current_row is None:
                if _starts_table_like(stripped):
                    current_row = stripped
                else:
                    normalized.append(line)
                continue

            if _starts_table_like(stripped):
                _flush_row()
                current_row = stripped
                continue

            if not stripped:
                if current_row.strip().endswith('|'):
                    _flush_row()
                    normalized.append('')
                else:
                    pending_break = True
                continue

            if _is_block_boundary(stripped):
                _flush_row()
                normalized.append(line)
                continue

            if current_row.strip().endswith('|'):
                _flush_row()
                normalized.append(line)
                continue

            joiner = '<br>' if pending_break else ' '
            current_row += joiner + stripped
            pending_break = False

        _flush_row()
        return '\n'.join(normalized)

    def _header_row(block):
        if len(block) >= 2 and _is_separator(block[1]):
            return block[0].strip()
        return None

    def _table_cols(block):
        for line in block:
            if _starts_table_like(line) and not _is_separator(line):
                return line.count('|') - 1
        return None

    def _split_parts(lines):
        parts = []
        i = 0
        while i < len(lines):
            if _starts_table_like(lines[i]):
                block = []
                while i < len(lines) and _starts_table_like(lines[i]):
                    block.append(lines[i].strip())
                    i += 1
                parts.append(('table', block))
            else:
                block = []
                while i < len(lines) and not _starts_table_like(lines[i]):
                    block.append(lines[i])
                    i += 1
                parts.append(('text', block))
        return parts

    def _merge_blocks(source):
        parts = _split_parts(source.split('\n'))
        changed = True
        while changed:
            changed = False
            i = 0
            while i + 2 < len(parts):
                if parts[i][0] == 'table' and parts[i + 1][0] == 'text' and parts[i + 2][0] == 'table':
                    prev = parts[i][1]
                    gap = parts[i + 1][1]
                    nxt = parts[i + 2][1]
                    gap_nonempty = [line.strip() for line in gap if line.strip()]
                    safe_gap = (
                        len(gap_nonempty) <= 3 and
                        all(not any(line.startswith(prefix) for prefix in ('#', '![', '- ', '* ', '> ', '```'))
                            for line in gap_nonempty)
                    )
                    same_cols = _table_cols(prev) == _table_cols(nxt) and _table_cols(prev) is not None
                    prev_header = _header_row(prev)
                    next_header = _header_row(nxt)
                    next_has_mid_separator = len(nxt) >= 2 and _is_separator(nxt[1])
                    is_continuation = (
                        not gap_nonempty or
                        (safe_gap and (next_has_mid_separator or (prev_header and next_header == prev_header)))
                    )
                    if same_cols and is_continuation:
                        merged = prev[:]
                        if gap_nonempty:
                            merged[-1] = _append_last_cell(merged[-1], '<br>'.join(gap_nonempty))
                        skip = 0
                        if prev_header and next_header == prev_header:
                            skip = 2
                        for idx, line in enumerate(nxt):
                            if idx < skip:
                                continue
                            if _is_separator(line):
                                continue
                            merged.append(line)
                        parts[i:i + 3] = [('table', merged)]
                        changed = True
                        break
                i += 1
        return '\n'.join('\n'.join(block) for _, block in parts)

    def _clean_table_blocks(source):
        lines = source.split('\n')
        cleaned = []
        i = 0
        while i < len(lines):
            if _starts_table_like(lines[i]):
                block = []
                while i < len(lines) and _starts_table_like(lines[i]):
                    block.append(lines[i].strip())
                    i += 1
                first_header = block[0] if len(block) >= 2 and _is_separator(block[1]) else None
                output = []
                idx = 0
                if first_header:
                    output.extend([block[0], block[1]])
                    idx = 2
                while idx < len(block):
                    row = block[idx]
                    if _is_separator(row):
                        idx += 1
                        continue
                    if idx + 1 < len(block) and _is_separator(block[idx + 1]):
                        if first_header and row == first_header:
                            idx += 2
                            continue
                        output.append(row)
                        idx += 2
                        continue
                    output.append(row)
                    idx += 1
                cleaned.extend(output)
            else:
                cleaned.append(lines[i])
                i += 1
        return '\n'.join(cleaned)

    normalized = _normalize_wrapped_table_rows(text)
    merged = _merge_blocks(normalized)
    return _clean_table_blocks(merged)


# ══════════════════════════════════════════════════════════════════════
# 结构性修复：ghost 图片引用（引用了不存在的文件）
# ══════════════════════════════════════════════════════════════════════

def _remove_ghost_images(text, base_dir):
    """清理引用了不存在图片的 markdown 图片标记。"""
    def _check_img(m):
        img_path = base_dir / m.group(2)
        if img_path.exists():
            return m.group(0)
        return ''
    return re.sub(r'!\[([^\]]*)\]\((images/[^)]+)\)', _check_img, text)


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def postprocess_markdown(md_content: str, output_dir: Path) -> str:
    """运行极简后处理管线。

    只做 6 条结构性修复，格式/标题层级/目录识别交给 prompt + doc_context 在转换阶段解决。
    """
    # 1. 统一换行符
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

    # 2. 图片引用格式修复
    md_content = _fix_image_refs(md_content)

    # 3. 清理 bbox 坐标泄漏
    md_content = _strip_bbox_leak(md_content)

    # 4. 单个 # 标题
    md_content = _single_h1(md_content)

    # 5. 修复被分页拆坏的 Markdown 表格
    md_content = _merge_split_tables(md_content)

    # 6. 清理引用不存在的 ghost 图片
    md_content = _remove_ghost_images(md_content, output_dir)

    # 7. 压缩连续 3+ 空行为 2 空行
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)

    return md_content

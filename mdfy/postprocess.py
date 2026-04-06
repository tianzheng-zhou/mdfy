"""后处理管线：标题归一化、去重、表格修复、清理。"""

import re
from pathlib import Path

from .stitch import _is_continuation_start, _merge_split_list_item_paragraphs


# ── 模块级编译正则 ──────────────────────────────────────────────────

_IMG_REF_RE = re.compile(r'^!\[.*?\]\(images/.+?\)\s*$')
_TABLE_ROW_RE = re.compile(r'^\s*\|.+\|\s*$')

_DOTTED_HEADING_RE = re.compile(
    r'^(#{1,6})\s+(\d+(?:\.\d+)+)\s+(.*)',
)

_NUMBERED_HEADING_RE = re.compile(
    r'^(#{1,6}\s+)?([（(]\d+[）)]\s*.+)$'
)
_HEADING_NUM_RE = re.compile(r'[（(](\d+)[）)]')


# ══════════════════════════════════════════════════════════════════════
# 模块级函数（也被管线外部调用）
# ══════════════════════════════════════════════════════════════════════

def _remove_running_headers(text: str) -> str:
    """移除泄漏到正文中的页眉（书名/章节标题在页面顶部的重复出现）。

    三阶段处理：
    Phase 0: 修复同一行粘连的重复标题（## Foo## Foo → ## Foo）
    Phase 1: 移除正文中与文档主标题完全匹配的纯文本行（非标题行）
    Phase 2: 移除与前文更高级标题文本相同的重复标题（PPT running section headers）
    """
    _HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')

    # ── Phase 0: 修复同一行粘连的重复标题 ──
    _GLUED_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*(#{1,6}\s+.+)$')

    def _fix_glued(line):
        gm = _GLUED_RE.match(line)
        if not gm:
            return line
        h1_hashes, h1_text, h2_full = gm.group(1), gm.group(2).rstrip(), gm.group(3)
        m2 = _HEADING_RE.match(h2_full)
        if not m2:
            return line
        h2_text = m2.group(2).strip()
        if h1_text.strip().lower() == h2_text.lower():
            return f'{h1_hashes} {h1_text.strip()}'
        return f'{h1_hashes} {h1_text.strip()}\n\n{h2_full}'

    lines = text.split('\n')
    lines = [_fix_glued(l) for l in lines]
    text = '\n'.join(lines)
    lines = text.split('\n')  # re-split since _fix_glued may insert newlines

    # ── Phase 1: 文档主标题重复（非标题行） ──
    m = re.search(r'^# (.+)$', text, re.MULTILINE)
    doc_title = m.group(1).strip() if m else None

    to_remove = set()
    if doc_title and len(doc_title) >= 2:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == doc_title:
                if line.lstrip().startswith('#'):
                    continue
                if stripped.startswith(('|', '-', '*', '>', '`')):
                    continue
                ctx_lines = [lines[j].strip() for j in range(max(0, i - 2), min(len(lines), i + 3)) if j != i]
                if any('ISBN' in cl or '出版' in cl or '书 名' in cl or '书名' in cl for cl in ctx_lines):
                    continue
                to_remove.add(i)

    # ── Phase 2: 重复节标题（running section headers） ──
    def _norm_heading(t):
        t = t.strip().lower()
        for prefix in ('the ', 'a ', 'an '):
            if t.startswith(prefix):
                t = t[len(prefix):]
        return t

    seen = {}  # normalized_text → (line_index, level)

    for i, line in enumerate(lines):
        if i in to_remove:
            continue
        hm = _HEADING_RE.match(line)
        if not hm:
            continue
        level = len(hm.group(1))
        raw_text = hm.group(2).strip()
        norm = _norm_heading(raw_text)
        if len(norm) < 2:
            continue

        if norm in seen:
            prev_idx, prev_level = seen[norm]
            if level > prev_level:
                # 更深层级 = running header（如 ## Review → ### Review）
                to_remove.add(i)
                continue
            elif level == prev_level:
                # 同级：仅在两者之间无实质内容时移除（紧邻重复）
                has_content = False
                for k in range(prev_idx + 1, i):
                    if k in to_remove:
                        continue
                    if lines[k].strip():
                        has_content = True
                        break
                if not has_content:
                    to_remove.add(i)
                    continue

        seen[norm] = (i, level)

    if to_remove:
        lines = [l for i, l in enumerate(lines) if i not in to_remove]
        # 压缩连续 3+ 空行为 2 空行
        cleaned = []
        blank_count = 0
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append(line)
            else:
                blank_count = 0
                cleaned.append(line)
        lines = cleaned

    return '\n'.join(lines)


def _remove_redundant_table_images(text: str) -> str:
    """后处理：移除内容仅为表格且已被完整转写的嵌入图片引用。

    检测逻辑（两阶段）：
    1. 向上扫描 20 行寻找 markdown 表格行，遇到标题行则停止。
    2. 若第 1 阶段被标题阻断，执行扩展扫描：穿过标题继续向上扫描。
       若找到表格，额外检查两个保护条件：
       a. 图片周围是否有图表引用（"图 X" / "Figure X"）→ 保留
       b. 图片后方是否紧跟同级或更深标题 → 保留
          （更高级别标题 = 章节边界，不触发保护）
    """
    lines = text.split('\n')
    to_remove = set()
    _HEADING_LINE_RE = re.compile(r'^(#{1,6})\s+')
    _FIG_REF_RE = re.compile(r'[图Figure]\s*\d+|如图|见图|所示')

    i = 0
    while i < len(lines):
        if not _IMG_REF_RE.match(lines[i]):
            i += 1
            continue

        # 收集连续的图片引用块（可能夹杂空行）
        img_block_start = i
        img_block_end = i
        j = i + 1
        while j < len(lines):
            if _IMG_REF_RE.match(lines[j]):
                img_block_end = j
                j += 1
            elif lines[j].strip() == '':
                j += 1
            else:
                break

        # ── 第 1 阶段：20 行内扫描，遇到标题则停止 ──
        table_found_above = False
        heading_blocked = False
        headings_crossed = 0
        for k in range(img_block_start - 1, max(0, img_block_start - 21), -1):
            stripped = lines[k].strip()
            if not stripped:
                continue
            if _HEADING_LINE_RE.match(lines[k]):
                heading_blocked = True
                break
            if _TABLE_ROW_RE.match(lines[k]):
                table_found_above = True
                break

        # ── 第 2 阶段：标题阻断后的扩展扫描（穿透多层标题） ──
        if not table_found_above and heading_blocked:
            scan_start = img_block_start - 1
            scan_limit = max(0, img_block_start - 60)
            ext_table_found = False
            headings_crossed = 0
            crossed_levels = []
            for k in range(scan_start, scan_limit, -1):
                stripped = lines[k].strip()
                if not stripped:
                    continue
                hm = _HEADING_LINE_RE.match(lines[k])
                if hm:
                    headings_crossed += 1
                    crossed_levels.append(len(hm.group(1)))
                    if headings_crossed > 3:
                        break
                    continue
                if _TABLE_ROW_RE.match(lines[k]):
                    ext_table_found = True
                    break
            if ext_table_found:
                # 保护条件 A：图片周围有图表引用（"图 X"等）→ 保留
                has_fig_ref = False
                for m in range(max(0, img_block_start - 3),
                               min(len(lines), img_block_end + 4)):
                    if _FIG_REF_RE.search(lines[m]):
                        has_fig_ref = True
                        break
                if not has_fig_ref:
                    # 保护条件 B：后方标题层级 ≥ 穿越标题最小层级 → 保留
                    has_heading_after = False
                    heading_after_level = 0
                    for m in range(img_block_end + 1,
                                   min(len(lines), img_block_end + 6)):
                        h_after = _HEADING_LINE_RE.match(lines[m])
                        if h_after:
                            has_heading_after = True
                            heading_after_level = len(h_after.group(1))
                            break
                    if has_heading_after:
                        min_crossed = min(crossed_levels) if crossed_levels else 99
                        if heading_after_level < min_crossed:
                            # 后续标题层级更高（章节边界），不保护
                            table_found_above = True
                    else:
                        table_found_above = True

        if table_found_above:
            for idx in range(img_block_start, img_block_end + 1):
                if _IMG_REF_RE.match(lines[idx]):
                    to_remove.add(idx)

        i = max(j, i + 1)

    if not to_remove:
        return text

    result = [line for idx, line in enumerate(lines) if idx not in to_remove]

    # 压缩连续 3+ 空行为 2 空行
    cleaned = []
    blank_count = 0
    for line in result:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return '\n'.join(cleaned)


def _normalize_numbered_heading_levels(text: str) -> str:
    """根据节号中的点号数量规范化标题层级。

    LLM 逐页调用时常将所有编号标题输出为 ###，导致 3.5 和 3.5.1
    处于同一层级。本函数根据编号深度自动调整：
      X.Y → ###，X.Y.Z → ####，X.Y.Z.W → #####
    """
    lines = text.split('\n')
    result = []
    for line in lines:
        m = _DOTTED_HEADING_RE.match(line)
        if m:
            section_num = m.group(2)
            rest = m.group(3)
            segments = section_num.count('.') + 1
            target_level = min(segments + 1, 6)  # 1段=##, 2段=###, 3段=####, ...
            new_hashes = '#' * target_level
            result.append(f'{new_hashes} {section_num} {rest}')
        else:
            result.append(line)
    return '\n'.join(result)


def _normalize_sibling_headings(text: str) -> str:
    """后处理：归一化同级编号标题的 Markdown 层级。

    PDF 跨页转换时，同一组编号标题（如 (1)...(2)...(3)...）可能被
    不同 LLM 调用赋予不一致的标题级别（有的带 ###，有的没有)。
    本函数检测这类编号序列并统一为相同层级。
    """
    lines = text.split('\n')

    # 第一遍：收集所有编号标题行的信息
    numbered = []  # (line_idx, heading_level, number)
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = _NUMBERED_HEADING_RE.match(stripped)
        if not m:
            continue
        nm = _HEADING_NUM_RE.match(m.group(2))
        if not nm:
            continue
        prefix = m.group(1)
        level = len(prefix.rstrip()) if prefix else 0  # 0 = 无 # 前缀
        num = int(nm.group(1))
        numbered.append((i, level, num))

    if len(numbered) < 2:
        return text

    # 第二遍：按连续递增编号分组（间距 ≤ 150 行视为同组）
    groups: list[list[tuple]] = []
    cur_group = [numbered[0]]
    for j in range(1, len(numbered)):
        prev = cur_group[-1]
        item = numbered[j]
        # 编号连续递增 且 行距合理
        if item[2] == prev[2] + 1 and (item[0] - prev[0]) <= 150:
            cur_group.append(item)
        else:
            if len(cur_group) >= 2:
                groups.append(cur_group)
            cur_group = [item]
    if len(cur_group) >= 2:
        groups.append(cur_group)

    # 第三遍：归一化每组标题层级
    for group in groups:
        levels = [it[1] for it in group]
        if len(set(levels)) <= 1:
            continue  # 已一致，跳过
        # 选目标层级：取出现次数最多的非零层级；全为 0 则跳过
        non_zero = [l for l in levels if l > 0]
        if not non_zero:
            continue
        target = max(set(non_zero), key=non_zero.count)
        prefix = '#' * target + ' '
        for (line_idx, level, _num) in group:
            if level != target:
                # 去掉已有的 # 前缀（如果有），加上目标前缀
                content = re.sub(r'^#{1,6}\s+', '', lines[line_idx].strip())
                lines[line_idx] = prefix + content

    return '\n'.join(lines)


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
# 内部辅助函数
# ══════════════════════════════════════════════════════════════════════

def _fix_unclosed_comments(text):
    """修复未闭合的 HTML 注释（OCR 产出的注释可能缺少 -->）。"""
    lines = text.split('\n')
    in_comment = False
    for i, line in enumerate(lines):
        if '<!--' in line and '-->' not in line:
            in_comment = True
            # 找到注释起始行，查看后续行是否有 -->
            # 如果下一个非空行不含 -->，就在当前行末尾补上
            found_close = False
            for j in range(i + 1, min(i + 4, len(lines))):
                if '-->' in lines[j]:
                    found_close = True
                    break
            if not found_close:
                lines[i] = line + ' -->'
                in_comment = False
    return '\n'.join(lines)


def _dedup_consecutive_image_refs(text):
    """去除连续重复的图片引用行。"""
    lines = text.split('\n')
    result = []
    recent_refs = set()
    for line in lines:
        m = re.match(r'^!\[.*?\]\((images/.+?)\)\s*$', line)
        if m:
            ref = m.group(1)
            if ref in recent_refs:
                continue
            recent_refs.add(ref)
        else:
            if line.strip():
                recent_refs.clear()
        result.append(line)
    return '\n'.join(result)


def _dedup_consecutive_paragraphs(text):
    """去除连续重复段落（跨页拼接可能产生相邻的完全重复段落）。"""
    paragraphs = re.split(r'\n\n+', text)
    result = []
    for para in paragraphs:
        if result and para.strip() == result[-1].strip():
            continue
        result.append(para)
    return '\n\n'.join(result)


def _remove_orphan_boundary_fragments(text):
    """清理跨页拼接产生的孤儿碎片（如 "度。" 重复自上段末尾 "长度。"）。"""
    lines = text.split('\n')
    to_remove = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r'^([\u4e00-\u9fffa-zA-Z]{1,3})([。？！])(.*)', stripped)
        if not m:
            continue
        frag = m.group(1)
        rest = m.group(3)
        if not rest.strip():
            continue
        # 向上找最近的非空行
        prev_line = ''
        for k in range(i - 1, max(0, i - 5), -1):
            if lines[k].strip():
                prev_line = lines[k].strip()
                break
        if prev_line and prev_line.endswith(frag + m.group(2)):
            # 确认是孤儿碎片，移除开头的 "度。" 保留后续内容
            lines[i] = line[:len(line) - len(line.lstrip())] + rest.lstrip()
    return '\n'.join(lines)


def _remove_duplicate_image_blocks(text):
    """移除连续重复的图片块：同一图片引用连续出现两次且后续说明文字相同。"""
    return re.sub(
        r'(!\[[^\]]*\]\([^)]+\)\n\n[^\n]+\n)\n\1',
        r'\1',
        text,
    )


def _remove_ghost_images(text, base_dir):
    """清理引用了不存在图片的 markdown 图片标记。"""
    def _check_img(m):
        img_path = base_dir / m.group(2)
        if img_path.exists():
            return m.group(0)
        return ''  # 图片不存在，删除引用
    return re.sub(r'!\[([^\]]*)\]\((images/[^)]+)\)', _check_img, text)


def _heading_has_body(lines, idx):
    """检查标题后 4 行内是否有正文（非标题、非列表、非空行）。"""
    for j in range(idx + 1, min(idx + 5, len(lines))):
        stripped = lines[j].strip()
        if stripped and not re.match(r'^#{1,6} ', lines[j]) and not stripped.startswith('- '):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def postprocess_markdown(md_content: str, pdf_type: str, output_dir: Path,
                         page_images_map: dict) -> str:
    """运行完整的后处理管线。

    参数：
        md_content: 拼接后的原始 Markdown 文本
        pdf_type: "scanned" | "digital" | ...
        output_dir: 输出目录（Path），用于检查图片是否存在
        page_images_map: {页码: [图片文件名]} 映射

    返回：清洗后的 Markdown 文本
    """

    # ─── 1. 归一化跨页编号标题层级 ───
    md_content = _normalize_sibling_headings(md_content)

    # ─── 2. 根据编号深度规范化标题层级 ───
    md_content = _normalize_numbered_heading_levels(md_content)

    # ─── 3. 移除内容仅为表格的嵌入图片引用 ───
    md_content = _remove_redundant_table_images(md_content)

    # ─── 4. 移除泄漏到正文中的页眉 ───
    md_content = _remove_running_headers(md_content)

    # ─── 5. 统一换行符 ───
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

    # ─── 6. 清理 Qwen 可能输出的 markdown 代码围栏 ───
    md_content = re.sub(r'^```markdown\s*\n', '', md_content)
    md_content = re.sub(r'\n```\s*$', '', md_content)
    # 也清理中间页面可能出现的围栏
    md_content = re.sub(r'\n```\s*\n\n```markdown\s*\n', '\n\n', md_content)

    # ─── 7. 修复模型生成的坏图片引用格式 ───
    # 模式1: ![](images/page9[](images/page9_img1.png)  — page后有数字
    # 模式2: ![](images/page[](images/page10_fig1.png)  — page后无数字
    md_content = re.sub(
        r'!\[\]\(images/page\d*\[]\(images/(page\d+_(?:img|fig)\d+\.png)\)',
        r'![](images/\1)',
        md_content,
    )

    # ─── 8. 修复缺少 ! 的图片引用 ───
    # [](images/page1_img1.png) → ![](images/page1_img1.png)
    md_content = re.sub(r'(?<!\!)\[\]\(images/', '![](images/', md_content)

    # ─── 9. 拆分粘连在同一行的多个图片引用 ───
    # 注意：用 [ \t]* 而非 \s*，避免跨行匹配破坏正常的图片间距
    md_content = re.sub(
        r'(!\[[^\]]*\]\(images/[^)]+\.png\))[ \t]*(!\[[^\]]*\]\(images/)',
        r'\1\n\2',
        md_content,
    )

    # ─── 10. 修复未闭合的 HTML 注释 ───
    md_content = _fix_unclosed_comments(md_content)

    # ─── 11. 修复缺少 images/ 前缀的图片引用 ───
    md_content = re.sub(
        r'!\[([^\]]*)\]\((page\d+_(?:img|fig)\d+\.png)\)',
        r'![\1](images/\2)',
        md_content,
    )

    # ─── 12. ####+ 标题降为 ### ───
    md_content = re.sub(
        r'^#{4,} ',
        '### ',
        md_content,
        flags=re.MULTILINE,
    )

    # ─── 13. 确保整篇文档只有一个 # 标题，后续 # 降为 ## ───
    first_h1 = re.search(r'^# ', md_content, flags=re.MULTILINE)
    if first_h1:
        before = md_content[:first_h1.end()]
        after = md_content[first_h1.end():]
        after = re.sub(r'^# ', '## ', after, flags=re.MULTILINE)
        md_content = before + after

    # ─── 14. 拆分同一行内合并的多个目录条目 ───
    # 例如 "        - 4.3.1 启用双缓冲操作- 4.3.2 控制访问哪个缓冲区" → 两行（保留缩进）
    md_content = re.sub(
        r'([ \t]*)(- \d[\d.]+ [^\n]+?)- (\d[\d.]+ )',
        r'\1\2\n\1- \3',
        md_content,
    )

    # ─── 15. 去重 OCR 产生的重复节号 ───
    md_content = re.sub(r'(\d[\d.]+) \1 ', r'\1 ', md_content)

    # ─── 16. 目录页标题条目转为列表 ───
    lines = md_content.split('\n')
    toc_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^## 目录\s*$', line):
            toc_idx = i
            break

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

    # ─── 17. 按节号深度修正 TOC 条目缩进 ───
    lines = md_content.split('\n')
    i = 0
    while i < len(lines):
        if re.match(r'^[ \t]*- \d[\d.]* ', lines[i]):
            start = i
            entry_count = 0
            dotted_count = 0
            while i < len(lines):
                if re.match(r'^[ \t]*- ', lines[i]):
                    entry_count += 1
                    if re.match(r'^[ \t]*- \d+\.\d', lines[i]):
                        dotted_count += 1
                    i += 1
                elif not lines[i].strip():
                    i += 1
                else:
                    break
            # 至少 5 个条目且有层级结构（带点号的条目）才视为 TOC
            if entry_count >= 5 and dotted_count >= 2:
                for j in range(start, i):
                    m = re.match(r'^[ \t]*- (\d[\d.]*) ', lines[j])
                    if m:
                        depth = m.group(1).count('.')
                        lines[j] = '    ' * depth + lines[j].lstrip()
        else:
            i += 1
    md_content = '\n'.join(lines)

    # ─── 18. 修复 TOC 条目中重复的节号 ───
    md_content = re.sub(
        r'^([ \t]*- )(\d[\d.]*) \2 ',
        r'\1\2 ',
        md_content,
        flags=re.MULTILINE,
    )

    # ─── 19. 去重重复出现的相同章节标题 ───
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

    # ─── 20. 将 <!-- 图：... --> 注释转为可见的引用块 ───
    md_content = re.sub(
        r'^<!-- (图[：:].+?) -->$',
        r'> **[\1]**',
        md_content,
        flags=re.MULTILINE,
    )

    # ─── 21. 清理模型泄漏的 bbox 坐标注释 ───
    md_content = re.sub(
        r'^<!--\s*Image\s*\([\d,\s]+\)\s*-->$',
        '',
        md_content,
        flags=re.MULTILINE,
    )

    # ─── 22. 去除连续重复的图片引用行 ───
    md_content = _dedup_consecutive_image_refs(md_content)

    # ─── 23. 去除连续重复段落 ───
    md_content = _dedup_consecutive_paragraphs(md_content)

    # ─── 24. 合并跨页断裂的段落 ───
    lines = md_content.split('\n')
    merged_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 检查当前行是否是不完整的正文行（非空、非标题/列表/图片/代码/引用）
        stripped = line.strip()
        is_plain_text = (
            stripped
            and not re.match(r'^#{1,6} |^!\[|^```|^[-*] |^> |^<|^\d+\. |^\|', line)
            and not re.search(r'[。？！…」』\u201d；]$|[.?!;]$|\*\*$|```$', stripped)
        )
        if is_plain_text and i + 1 < len(lines) and lines[i + 1].strip() == '':
            # 往下找第一个非空行
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines):
                next_stripped = lines[j].strip()
                # 下一非空行是续接文字（非标题/列表/图片/代码/引用开头）且满足续接条件
                next_is_continuation = (
                    next_stripped
                    and not re.match(r'^#{1,6} |^!\[|^```|^[-*] |^> |^<|^\d+\. |^\|', lines[j])
                    and _is_continuation_start(next_stripped)
                )
                if next_is_continuation:
                    # 合并：保留当前行，跳过空行
                    merged_lines.append(line)
                    i = j  # 跳到续接行，下次循环会加入
                    continue
        merged_lines.append(line)
        i += 1
    md_content = '\n'.join(merged_lines)

    # ─── 25. 扫描件专用后处理 ───
    if pdf_type == "scanned":
        # 提升裸章号行为 ## 标题
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

    # ─── 26. 多级编号章节标题层级规范化 ───
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

    # ─── 27. Digital-PDF-only 后处理 ───
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

    # ─── 28. 裸露的科学计数法数据行自动包裹代码围栏 ───
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

    # ─── 29. 修复被分页与硬换行拆坏的 Markdown 表格 ───
    md_content = _merge_split_tables(md_content)

    # ─── 30. 合并被空行错误拆开的列表项续句 ───
    md_content = _merge_split_list_item_paragraphs(md_content)

    # ─── 31. 清理跨页拼接产生的孤儿碎片 ───
    md_content = _remove_orphan_boundary_fragments(md_content)

    # ─── 32. 移除连续重复的图片+说明文字块 ───
    md_content = _remove_duplicate_image_blocks(md_content)

    # ─── 33. 清理引用了不存在图片的 markdown 图片标记 ───
    md_content = _remove_ghost_images(md_content, output_dir)

    # ─── 34. 清理删除图片引用后可能产生的多余空行 ───
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)

    return md_content

"""跨页拼接：大纲构建、断句判定、LLM stitch、正则去重回退。"""

import re

from .config import STITCH_MODEL, STITCH_PREV_TAIL_CHARS, OUTLINE_MAX_HEADINGS
from .prompts import STITCH_SYSTEM


# ══════════════════════════════════════════════════════════════════════
# 滚动大纲（传给每页以感知全局位置）
# ══════════════════════════════════════════════════════════════════════

def build_outline(all_md_parts, max_headings=OUTLINE_MAX_HEADINGS):
    """从已完成的 Markdown 中提取滚动大纲（所有标题）。"""
    headings = []
    for part in all_md_parts:
        for line in part.split('\n'):
            if re.match(r'^#{1,4} ', line):
                headings.append(line)
    if len(headings) > max_headings:
        headings = headings[-max_headings:]
    return '\n'.join(headings)


# ══════════════════════════════════════════════════════════════════════
# 断句判定
# ══════════════════════════════════════════════════════════════════════

def is_incomplete_sentence(text):
    """判断文本末尾是否为未完成的句子（被分页截断）。"""
    stripped = text.rstrip()
    if not stripped:
        return False
    last_line = stripped.split('\n')[-1]
    if re.match(r'^#{1,6} ', last_line):
        return False
    if re.match(r'^!\[', last_line):
        return False
    if re.match(r'^```', last_line):
        return False
    if re.match(r'^>', last_line):
        return False
    list_match = re.match(r'^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(.+)$', last_line)
    sentence_tail = list_match.group(1).strip() if list_match else last_line.strip()
    if not sentence_tail:
        return False
    if re.search(r'[。？！…」』\u201d；：]$|[.?!;:]$|\*\*$', sentence_tail):
        return False
    return True


def is_continuation_start(text):
    """判断文本开头是否像是被截断句子的续接部分。"""
    stripped = text.lstrip()
    if not stripped:
        return False
    if re.match(r'^#{1,6} |^!\[|^```|^[-*] |^> |^<', stripped):
        return False
    first_char = stripped[0]
    # 高置信续接：中文功能词/助词/连接词开头
    if first_char in '和与及或但而且的了着过把被让给对向从到也都还又才就只已不没无' \
                     '，、；：）】」』"':
        return True
    # 小写英文字母（英文句子不会以小写开头）
    if first_char.islower():
        return True
    # 英文闭合标点续接
    if first_char in ',.;:)]\'"':
        return True
    # 中文汉字：前几个字内有句末/逗号标点
    if '\u4e00' <= first_char <= '\u9fff':
        head = stripped[:5]
        if re.search(r'[。？！]', head):
            return True
        if re.search(r'[，、]', head):
            return True
    return False


def merge_split_list_item_paragraphs(text):
    """合并被空行错误拆开的列表项续句。"""
    lines = text.split('\n')
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 2 < len(lines) and lines[i + 1] == '':
            match = re.match(r'^([ \t]*(?:[-*+]|\d+[.)]))[ \t]+(.+)$', line)
            next_line = lines[i + 2]
            next_stripped = next_line.lstrip()
            next_structural = bool(re.match(
                r'^#{1,6} |^!\[|^```|^[-*+] |^\d+[.)] |^> |^\||^---',
                next_stripped,
            ))
            if match and next_line.strip():
                item_text = match.group(2).strip()
                if (is_incomplete_sentence(item_text)
                        and not next_structural
                        and is_continuation_start(next_line)):
                    merged.append(line)
                    i += 2
                    continue
        merged.append(line)
        i += 1
    return '\n'.join(merged)


# ══════════════════════════════════════════════════════════════════════
# 快速边界判断
# ══════════════════════════════════════════════════════════════════════

def boundary_needs_stitch(prev_text, curr_text):
    """判断两页边界是否需要 LLM 拼接。返回 False 表示边界干净。"""
    prev_stripped = prev_text.rstrip()
    curr_stripped = curr_text.lstrip('\n')
    if not prev_stripped or not curr_stripped:
        return False

    last_line = prev_stripped.split('\n')[-1].strip()
    first_line = curr_stripped.split('\n')[0].strip()

    prev_complete = bool(re.search(
        r'[。？！…」』\u201d；]$|[.?!;:]$|\*\*$|```$', prev_stripped
    ))
    curr_structural = bool(re.match(
        r'^#{1,6} |^!\[|^```|^[-*] |^> |^---', first_line
    ))

    # 情况1：prev 完整 + curr 结构化 → 干净边界
    if prev_complete and curr_structural:
        return False

    # 情况2：prev 以结构化元素结尾 + curr 结构化
    prev_structural_end = bool(re.match(
        r'^#{1,6} |^```|^---|^!\[', last_line
    ))
    if prev_structural_end and curr_structural:
        return False

    # 情况3：表格续接
    if (last_line.startswith('|') and '|' in last_line[1:]
            and first_line.startswith('|') and '|' in first_line[1:]):
        prev_cols = last_line.count('|') - 1
        curr_cols = first_line.count('|') - 1
        if prev_cols == curr_cols and prev_cols > 0:
            return False

    # 检查重叠迹象
    prev_tail_lines = [l.strip() for l in prev_stripped.split('\n')[-5:] if l.strip()]
    curr_head_lines = [l.strip() for l in curr_stripped.split('\n')[:3] if l.strip()]
    has_overlap = any(cl in prev_tail_lines for cl in curr_head_lines if len(cl) > 10)

    if not prev_complete:
        return True
    if has_overlap:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
# 边界去重（正则回退）
# ══════════════════════════════════════════════════════════════════════

def dedup_page_boundary(prev_text, curr_text):
    """去除下一页开头与上一页末尾的重叠内容。"""
    if not prev_text or not curr_text:
        return curr_text

    prev_lines = prev_text.rstrip().split('\n')
    curr_lines = curr_text.lstrip('\n').split('\n')

    max_check = min(8, len(prev_lines), len(curr_lines))
    if max_check == 0:
        return curr_text

    # 策略1：逐行精确匹配
    overlap_lines = 0
    for n in range(1, max_check + 1):
        prev_tail_lines = [l.strip() for l in prev_lines[-n:]]
        curr_head_lines = [l.strip() for l in curr_lines[:n]]
        if prev_tail_lines == curr_head_lines and any(l for l in prev_tail_lines):
            overlap_lines = n

    if overlap_lines > 0:
        trimmed = '\n'.join(curr_lines[overlap_lines:])
        return trimmed.lstrip('\n')

    # 策略2：单行级重复检测
    first_curr_line = ''
    first_curr_idx = 0
    for idx, line in enumerate(curr_lines):
        if line.strip():
            first_curr_line = line.strip()
            first_curr_idx = idx
            break

    if first_curr_line and len(first_curr_line) > 15:
        for pl in prev_lines[-max_check:]:
            if pl.strip() == first_curr_line:
                trimmed = '\n'.join(curr_lines[first_curr_idx + 1:])
                return trimmed.lstrip('\n')

    return curr_text


# ══════════════════════════════════════════════════════════════════════
# LLM 拼接
# ══════════════════════════════════════════════════════════════════════

def stitch_boundary_with_llm(client, prev_tail, curr_head, stitch_model=STITCH_MODEL):
    """调用轻量 LLM 修正下一页开头（去重 + 续接）。

    返回修正后的 curr_head，或 None 表示失败/被拒。
    """
    user_msg = (
        f"<page_end>\n{prev_tail}\n</page_end>\n\n"
        f"<page_start>\n{curr_head}\n</page_start>"
    )

    try:
        response = client.chat.completions.create(
            model=stitch_model,
            messages=[
                {"role": "system", "content": STITCH_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=4096,
            extra_body={"enable_thinking": False},
        )
        result = response.choices[0].message.content.strip()
        result = re.sub(r'^```markdown\s*\n', '', result)
        result = re.sub(r'\n```\s*$', '', result)

        # 验证：长度不能膨胀
        if len(result) > len(curr_head) * 1.1 + len(prev_tail) * 0.3:
            print(f"  [stitch rejected: too long {len(result)} vs curr={len(curr_head)}, fallback]")
            return None
        if not result.strip():
            return None

        # 标题/图片/注释/表格行不能少
        orig_headings = len(re.findall(r'^#{1,6} ', curr_head, re.MULTILINE))
        result_headings = len(re.findall(r'^#{1,6} ', result, re.MULTILINE))
        if result_headings < orig_headings:
            print(f"  [stitch rejected: lost headings {result_headings}/{orig_headings}, fallback]")
            return None

        orig_imgs = len(re.findall(r'!\[', curr_head))
        result_imgs = len(re.findall(r'!\[', result))
        if result_imgs < orig_imgs:
            print(f"  [stitch rejected: lost images {result_imgs}/{orig_imgs}, fallback]")
            return None

        open_comments = len(re.findall(r'<!--', result))
        close_comments = len(re.findall(r'-->', result))
        if open_comments != close_comments:
            print(f"  [stitch rejected: unclosed HTML comment {open_comments}/{close_comments}, fallback]")
            return None

        orig_table_rows = len([l for l in curr_head.split('\n')
                               if l.strip().startswith('|') and '|' in l.strip()[1:]])
        result_table_rows = len([l for l in result.split('\n')
                                 if l.strip().startswith('|') and '|' in l.strip()[1:]])
        if result_table_rows < orig_table_rows:
            print(f"  [stitch rejected: lost table rows {result_table_rows}/{orig_table_rows}, fallback]")
            return None

        return result
    except Exception as e:
        print(f"  [stitch LLM failed: {e}, fallback to regex]")
        return None


# ══════════════════════════════════════════════════════════════════════
# 智能拼接主流程
# ══════════════════════════════════════════════════════════════════════

def join_pages_smart(parts, client=None):
    """智能拼接多页 Markdown 输出。

    策略分三层：
    1. 快速判断：边界干净 → 直接 \\n\\n 拼接（或表格续接 \\n）
    2. LLM 拼接：边界有问题 → 调用 flash 模型合并
    3. 正则回退：LLM 不可用时用 dedup_page_boundary + is_continuation_start
    """
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]

    result_parts = [parts[0]]
    stitch_count = 0

    for i in range(1, len(parts)):
        prev = parts[i - 1]
        curr = parts[i]

        if not curr.strip():
            continue

        if not boundary_needs_stitch(prev, curr):
            prev_last = prev.rstrip().split('\n')[-1].strip() if prev.rstrip() else ''
            curr_first = curr.lstrip('\n').split('\n')[0].strip() if curr.lstrip('\n') else ''
            # 表格续接：\n 衔接（同一表格块）；其他 \n\n
            if (prev_last.startswith('|') and '|' in prev_last[1:]
                    and curr_first.startswith('|') and '|' in curr_first[1:]):
                result_parts.append("\n" + curr)
            else:
                result_parts.append("\n\n" + curr)
            continue

        boundary_chars = STITCH_PREV_TAIL_CHARS
        prev_tail = prev[-boundary_chars:] if len(prev) > boundary_chars else prev
        curr_head = curr[:boundary_chars] if len(curr) > boundary_chars else curr
        curr_rest = curr[boundary_chars:] if len(curr) > boundary_chars else ""

        # LLM 拼接
        if client is not None:
            stitch_count += 1
            stitched = stitch_boundary_with_llm(client, prev_tail, curr_head)
            if stitched is not None:
                corrected_curr = stitched + curr_rest
                if is_incomplete_sentence(prev):
                    clean_start = stitched.lstrip('\n')
                    if clean_start and not re.match(r'^\s*#', clean_start):
                        if result_parts:
                            result_parts[-1] = result_parts[-1].rstrip('\n') + '\n'
                        result_parts.append(clean_start + curr_rest)
                    else:
                        result_parts.append("\n\n" + corrected_curr)
                else:
                    result_parts.append("\n\n" + corrected_curr)
                continue

        # 正则回退
        curr = dedup_page_boundary(prev, curr)
        if not curr.strip():
            continue

        if is_incomplete_sentence(prev) and is_continuation_start(curr):
            if result_parts:
                result_parts[-1] = result_parts[-1].rstrip('\n') + '\n'
            result_parts.append(curr.lstrip('\n'))
        else:
            result_parts.append("\n\n" + curr)

    if stitch_count > 0:
        print(f"\n  [LLM stitch: {stitch_count} boundaries processed]")

    return "".join(result_parts)

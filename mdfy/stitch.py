"""跨页边界处理：LLM stitch、去重、智能拼接。"""

import re

from .prompts import STITCH_SYSTEM
from .dedup import _dedup_page_boundary


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


def _is_incomplete_sentence(text):
    """判断文本末尾是否为未完成的句子（被分页截断）。

    返回 True 表示末尾句子不完整，需要与下一页拼接。
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    # 以标题、图片引用、列表项、代码块结尾的视为完整
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
    # 以中文/英文标点结束的句子视为完整
    if re.search(r'[。？！…」』\u201d；：]$|[.?!;:]$|\*\*$', sentence_tail):
        return False
    return True


def _is_continuation_start(text):
    """判断文本开头是否像是一个被截断句子的续接部分。

    返回 True 表示开头不是一个独立新段落的起始。
    结合 _is_incomplete_sentence 一起使用：只有前页末尾不完整时才调用此函数。
    """
    stripped = text.lstrip()
    if not stripped:
        return False
    # 以标题、图片、列表、代码块、引用、HTML标签开头的不是续接
    if re.match(r'^#{1,6} |^!\[|^```|^[-*] |^> |^<', stripped):
        return False
    first_char = stripped[0]
    # 高置信续接：以中文功能词、助词、连接词开头（几乎不可能是句子/段落的开头）
    if first_char in '和与及或但而且的了着过把被让给对向从到也都还又才就只已不没无' \
       '，、；：）】」』"':
        return True
    # 以小写英文字母开头（英文句子不会以小写开头）
    if first_char.islower():
        return True
    # 以英文闭合标点续接
    if first_char in ',.;:)]\'"':
        return True
    # 中文汉字开头：检查前几个字是否有句末标点
    # 如果续接文字在前3个字符内就出现了句末标点（。？！），说明是短续接尾巴
    if '\u4e00' <= first_char <= '\u9fff':
        head = stripped[:5]
        # 如果前5字内有句末标点，很可能是上一句的尾巴
        if re.search(r'[。？！]', head):
            return True
        # 如果前5字内有中文逗号/顿号，说明可能在句子中间
        if re.search(r'[，、]', head):
            return True
    return False


def _merge_split_list_item_paragraphs(text: str) -> str:
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
                if (_is_incomplete_sentence(item_text)
                        and not next_structural
                        and _is_continuation_start(next_line)):
                    merged.append(line)
                    i += 2
                    continue
        merged.append(line)
        i += 1
    return '\n'.join(merged)


def _boundary_needs_stitch(prev_text, curr_text):
    """快速判断两页边界是否需要 LLM 拼接。

    返回 False 的情况（边界干净，不需要 LLM）：
    - prev 以标题/代码块/图片/分隔符结尾，且 curr 以标题/图片等结构化元素开头
    - prev 以句末标点结尾，且 curr 以标题/新段落开头（无重叠迹象）
    """
    prev_stripped = prev_text.rstrip()
    curr_stripped = curr_text.lstrip('\n')
    if not prev_stripped or not curr_stripped:
        return False

    last_line = prev_stripped.split('\n')[-1].strip()
    first_line = curr_stripped.split('\n')[0].strip()

    # prev 以完整句末结尾
    prev_complete = bool(re.search(
        r'[。？！…」』\u201d；]$|[.?!;:]$|\*\*$|```$', prev_stripped
    ))
    # curr 以结构化元素开头（标题、图片、代码块、列表）
    curr_structural = bool(re.match(
        r'^#{1,6} |^!\[|^```|^[-*] |^> |^---', first_line
    ))

    # 情况1：prev 完整结尾 + curr 结构化开头 → 干净边界
    if prev_complete and curr_structural:
        return False

    # 情况2：prev 以标题/代码块/分隔线/图片引用结尾 + curr 结构化开头
    prev_structural_end = bool(re.match(
        r'^#{1,6} |^```|^---|^!\[', last_line
    ))
    if prev_structural_end and curr_structural:
        return False

    # 情况5：prev 以表格行结尾 + curr 以同列数表格行开头 → 干净的表格续接
    if (last_line.startswith('|') and '|' in last_line[1:]
            and first_line.startswith('|') and '|' in first_line[1:]):
        prev_cols = last_line.count('|') - 1
        curr_cols = first_line.count('|') - 1
        if prev_cols == curr_cols and prev_cols > 0:
            return False  # 表格续接，交给 \n 拼接 + 后处理合并

    # 检查是否有重叠迹象（curr 开头的内容在 prev 末尾出现过）
    prev_tail_lines = [l.strip() for l in prev_stripped.split('\n')[-5:] if l.strip()]
    curr_head_lines = [l.strip() for l in curr_stripped.split('\n')[:3] if l.strip()]
    has_overlap = any(cl in prev_tail_lines for cl in curr_head_lines if len(cl) > 10)

    # 情况3：prev 不完整（未以句末标点结尾）→ 需要拼接
    if not prev_complete:
        return True

    # 情况4：有重叠迹象 → 需要拼接
    if has_overlap:
        return True

    return False


def _stitch_boundary_with_llm(client, prev_tail, curr_head, stitch_model="qwen3.5-flash"):
    """调用轻量 LLM 修正下一页开头（去重 + 续接断句）。

    参数：
        client: OpenAI 客户端
        prev_tail: 上一页末尾 ~600 字
        curr_head: 下一页开头 ~600 字
        stitch_model: 用于拼接的模型（默认 flash，便宜快速）

    返回修正后的 curr_head（去掉了重复、处理了续接），或 None 表示失败。
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

        # 清理 LLM 可能输出的 markdown 围栏
        result = re.sub(r'^```markdown\s*\n', '', result)
        result = re.sub(r'\n```\s*$', '', result)

        # 验证：结果不能比 curr_head 长太多（防止 LLM 把 page_end 也输出了，或生成新内容）
        if len(result) > len(curr_head) * 1.1 + len(prev_tail) * 0.3:
            print(f"  [stitch rejected: too long {len(result)} vs curr={len(curr_head)}, fallback]")
            return None
        # 验证：结果不能为空
        if not result.strip():
            return None

        # 验证：结果中的标题和图片不能比 curr_head 少
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

        # 验证：HTML 注释完整性（有 <!-- 必须有配对的 -->）
        open_comments = len(re.findall(r'<!--', result))
        close_comments = len(re.findall(r'-->', result))
        if open_comments != close_comments:
            print(f"  [stitch rejected: unclosed HTML comment {open_comments}/{close_comments}, fallback]")
            return None

        # 验证：表格行不能丢失（防止 stitch 模型吞掉跨页表格的续接行）
        orig_table_rows = len([l for l in curr_head.split('\n')
                               if l.strip().startswith('|') and '|' in l.strip()[1:]])
        result_table_rows = len([l for l in result.split('\n')
                                 if l.strip().startswith('|') and '|' in l.strip()[1:]])
        if result_table_rows < orig_table_rows:
            print(f"  [stitch rejected: lost table rows {result_table_rows}/{orig_table_rows}, fallback]")
            return None

        return result
    except Exception as e:
        # LLM 拼接失败时回退到简单拼接
        print(f"  [stitch LLM failed: {e}, fallback to regex]")
        return None


def _join_pages_smart(parts, client=None):
    """智能拼接多页 Markdown 输出。

    策略分三层：
    1. 快速判断：边界干净（结构化分界）→ 直接 \\n\\n 拼接
    2. LLM 拼接：边界可能有问题（断句/重复）→ 调用 flash 模型合并
    3. 正则回退：LLM 不可用时用 _dedup_page_boundary + _is_continuation_start

    参数：
        parts: 每页的 Markdown 文本列表
        client: OpenAI 客户端（传入则启用 LLM 拼接，None 则纯正则）
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

        # 快速判断：是否需要处理这个边界
        if not _boundary_needs_stitch(prev, curr):
            # 表格续接时用 \n 拼接（保持同一个表格块），其他用 \n\n
            prev_last = prev.rstrip().split('\n')[-1].strip() if prev.rstrip() else ''
            curr_first = curr.lstrip('\n').split('\n')[0].strip() if curr.lstrip('\n') else ''
            if (prev_last.startswith('|') and '|' in prev_last[1:]
                    and curr_first.startswith('|') and '|' in curr_first[1:]):
                result_parts.append("\n" + curr)
            else:
                result_parts.append("\n\n" + curr)
            continue

        # 取边界区：prev 末尾 + curr 开头
        boundary_chars = 600
        prev_tail = prev[-boundary_chars:] if len(prev) > boundary_chars else prev
        curr_head = curr[:boundary_chars] if len(curr) > boundary_chars else curr
        curr_rest = curr[boundary_chars:] if len(curr) > boundary_chars else ""

        # 尝试 LLM 拼接（返回修正后的 curr_head，prev 不变）
        if client is not None:
            stitch_count += 1
            stitched = _stitch_boundary_with_llm(client, prev_tail, curr_head)
            if stitched is not None:
                # stitched 是修正后的 curr_head（去重 + 续接处理）
                corrected_curr = stitched + curr_rest
                # 判断拼接方式：如果 prev 以不完整句结尾，紧密拼接
                if _is_incomplete_sentence(prev):
                    # LLM 可能返回前导空行，strip 后再判断是否为标题开头
                    clean_start = stitched.lstrip('\n')
                    if clean_start and not re.match(r'^\s*#', clean_start):
                        # 紧密拼接：确保 prev 尾部只有一个 \n，curr 无前导空行
                        if result_parts:
                            result_parts[-1] = result_parts[-1].rstrip('\n') + '\n'
                        result_parts.append(clean_start + curr_rest)
                    else:
                        result_parts.append("\n\n" + corrected_curr)
                else:
                    result_parts.append("\n\n" + corrected_curr)
                continue

        # 回退：正则去重 + 续接
        curr = _dedup_page_boundary(prev, curr)
        if not curr.strip():
            continue

        if _is_incomplete_sentence(prev) and _is_continuation_start(curr):
            # 紧密拼接：确保 prev 尾部只有一个 \n，curr 无前导空行
            if result_parts:
                result_parts[-1] = result_parts[-1].rstrip('\n') + '\n'
            result_parts.append(curr.lstrip('\n'))
        else:
            result_parts.append("\n\n" + curr)

    if stitch_count > 0:
        print(f"\n  [LLM stitch: {stitch_count} boundaries processed]")

    return "".join(result_parts)

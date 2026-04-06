"""页内去重、跨页去重、页级质量审查。"""

import re


def _dedup_model_repetition(text):
    """检测并移除单页 AI 输出中的自重复（模型幻觉循环）。

    LLM 在处理图片密集页面时，容易在输出末尾重复之前的内容。
    典型表现：图片列表 + 说明文字整块重复输出第二遍，接缝处常丢失 '!' 导致
    broken image ref（如 '[](images/...)' 而非 '![](images/...)'）。

    策略：
    1. 修复 broken image ref、拆分粘连的图片引用行
    2. 查找末尾与前部匹配的最长重复块（tail-duplication）
    3. 若重复内容 >= 3 个有效行且 >= 15% 的总有效行数，剪除
    """
    # Step 1: 修复 broken image refs（缺少 !）
    text = re.sub(r'(?<!\!)\[\]\(images/', '![](images/', text)
    # Step 1b: 拆分粘连在同一行的多个图片引用
    # 注意：用 [ \t]* 而非 \s*，避免跨行匹配破坏正常的图片间距
    text = re.sub(
        r'(!\[[^\]]*\]\(images/[^)]+\.png\))[ \t]*(!\[[^\]]*\]\(images/)',
        r'\1\n\2',
        text,
    )
    # Step 1c: 拆分文字+图片引用粘连（模型循环接缝处：text![](images/...)）
    # 仅匹配非空白非!字符直接跟 ![，这不是正常 markdown 排版
    text = re.sub(
        r'([^\s!])(!\[[^\]]*\]\(images/)',
        r'\1\n\2',
        text,
    )

    lines = text.split('\n')
    n = len(lines)
    if n < 8:
        return text

    # 构建有效行索引 (行号, 去空白文本)
    content = [(i, lines[i].strip()) for i in range(n) if lines[i].strip()]
    nc = len(content)
    if nc < 6:
        return text

    # 对每个候选重复起点 k（从 25% 位置开始），检查 content[k:end] 是否
    # 与前面某位置 j 开始的序列匹配
    best_dup_ci = -1
    best_match_len = 0

    for k in range(max(3, nc // 4), nc - 2):
        target_line = content[k][1]
        if len(target_line) < 8:
            continue

        for j in range(0, k):
            if content[j][1] != target_line:
                continue

            # 统计从 j 和 k 开始的连续匹配行数
            ml = 0
            while j + ml < k and k + ml < nc:
                if content[j + ml][1] == content[k + ml][1]:
                    ml += 1
                else:
                    break

            # 重复必须延伸到末尾（或接近末尾）
            if k + ml >= nc - 1 and ml > best_match_len:
                best_match_len = ml
                best_dup_ci = k

    # 要求：至少 3 个匹配有效行 且 >= 15% 的总有效行
    if best_match_len >= 3 and best_match_len >= nc * 0.15:
        dup_start_line = content[best_dup_ci][0]
        trimmed = '\n'.join(lines[:dup_start_line]).rstrip()
        print(f"  [dedup: removed {best_match_len} repeated lines from page output]")
        return trimmed

    # Phase 2: 中间重复块检测 — 模型输出 A + A' + B（A' 是 A 的重复，B 是新内容）
    # 目标：去除 A'，保留 A + B
    best_mid_ci = -1
    best_mid_len = 0

    for k in range(max(3, nc // 4), nc - 3):
        target_line = content[k][1]
        if len(target_line) < 8:
            continue

        for j in range(0, k):
            if content[j][1] != target_line:
                continue

            ml = 0
            while j + ml < k and k + ml < nc:
                if content[j + ml][1] == content[k + ml][1]:
                    ml += 1
                else:
                    break

            if ml > best_mid_len and ml >= 5 and ml >= nc * 0.15:
                best_mid_len = ml
                best_mid_ci = k

    if best_mid_len >= 5 and best_mid_len >= nc * 0.15:
        dup_start_line = content[best_mid_ci][0]
        dup_end_ci = best_mid_ci + best_mid_len
        dup_end_line = content[dup_end_ci][0] if dup_end_ci < nc else len(lines)
        trimmed_lines = lines[:dup_start_line] + lines[dup_end_line:]
        trimmed = '\n'.join(trimmed_lines).rstrip()
        print(f"  [dedup: removed {best_mid_len} mid-repeated lines from page output]")
        return trimmed

    return text


def _review_page_quality(md_part, image_filenames, page_num):
    """页级质量自审：检测单页 AI 输出中的质量问题。

    返回 (issues, score)：
    - issues: list[str] 检测到的问题描述
    - score: 0-100 质量分，低于阈值触发重试
    """
    issues = []
    score = 100

    if not md_part or not md_part.strip():
        return ["空输出"], 0

    # 1. 检测 bbox 坐标泄漏：<!-- Image (x, y, x, y) --> 或类似格式
    bbox_leaks = re.findall(r'<!--\s*Image\s*\([\d,\s]+\)\s*-->', md_part)
    if bbox_leaks:
        issues.append(f"bbox坐标泄漏×{len(bbox_leaks)}")
        score -= 15 * len(bbox_leaks)

    # 2. 检测未闭合的 $$ 公式块（奇数个 $$ 表示有截断）
    dollar_blocks = re.findall(r'^\$\$', md_part, re.MULTILINE)
    if len(dollar_blocks) % 2 != 0:
        issues.append("公式块未闭合($$不配对)")
        score -= 25

    # 3. 检测 <!-- 图：... --> 占位符（模型未能引用已提供的图片）
    fig_placeholders = re.findall(r'<!--\s*图[：:].+?-->', md_part)
    if fig_placeholders and image_filenames:
        # 有图片文件名却生成了占位符，说明模型没正确引用
        issues.append(f"图片占位符×{len(fig_placeholders)}(有{len(image_filenames)}张图可用)")
        score -= 20 * len(fig_placeholders)

    # 4. 检测图片引用缺失：提供了图片但输出中完全没引用
    if image_filenames:
        unreferenced = [f for f in image_filenames if f not in md_part]
        if unreferenced:
            # 不能全怪模型——有些图可能被合理省略（纯表格/代码已转写）
            # 但超过一半未引用大概率是问题
            miss_ratio = len(unreferenced) / len(image_filenames)
            if miss_ratio > 0.5:
                issues.append(f"图片大量未引用({len(unreferenced)}/{len(image_filenames)})")
                score -= 15

    # 5. 检测连续重复的图片引用行
    img_refs = re.findall(r'!\[.*?\]\(images/[^)]+\)', md_part)
    if len(img_refs) != len(set(img_refs)):
        dup_count = len(img_refs) - len(set(img_refs))
        issues.append(f"重复图片引用×{dup_count}")
        score -= 10 * dup_count

    # 6. 检测行内公式截断（以 \frac{...+ 或 \sum_ 等结尾，缺少 $）
    lines = md_part.split('\n')
    for line in lines:
        stripped = line.rstrip()
        if stripped and re.search(r'\\(?:frac|sum|int|sqrt)\{[^}]*$', stripped):
            if not stripped.endswith('$') and not stripped.endswith('$$'):
                issues.append("行内公式截断")
                score -= 15
                break

    return issues, max(score, 0)


def _dedup_page_boundary(prev_text, curr_text):
    """去除下一页开头与上一页末尾的重叠内容。

    模型有时会重复 prev_tail 中的部分文字。此函数检测并移除
    curr_text 开头与 prev_text 末尾重叠的行。

    返回去重后的 curr_text。
    """
    if not prev_text or not curr_text:
        return curr_text

    # 取 prev 最后若干行和 curr 最前若干行进行比较
    prev_lines = prev_text.rstrip().split('\n')
    curr_lines = curr_text.lstrip('\n').split('\n')

    # 只检查末尾/开头一定行数范围（避免误匹配远距离内容）
    max_check = min(8, len(prev_lines), len(curr_lines))
    if max_check == 0:
        return curr_text

    # 策略1：逐行精确匹配（curr 开头的连续行 == prev 末尾的连续行）
    overlap_lines = 0
    for n in range(1, max_check + 1):
        prev_tail_lines = [l.strip() for l in prev_lines[-n:]]
        curr_head_lines = [l.strip() for l in curr_lines[:n]]
        # any() 允许含空行的块匹配（如 图片+空行+说明文字），但排除纯空行匹配
        if prev_tail_lines == curr_head_lines and any(l for l in prev_tail_lines):
            overlap_lines = n

    if overlap_lines > 0:
        trimmed = '\n'.join(curr_lines[overlap_lines:])
        return trimmed.lstrip('\n')

    # 策略2：单行级别的重复检测（curr 第一个非空行 == prev 末尾某行）
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

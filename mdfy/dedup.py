"""页内去重、跨页去重、页级质量审查。"""

import re
import json
import base64


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


def _model_review_page_quality(client, model, page_image_bytes, page_image_mime,
                                md_part, image_filenames, page_num):
    """用模型对比页面图片和生成的 Markdown，审查转换质量。

    返回 (issues: list[str], score: int)
    """
    if not md_part or not md_part.strip():
        return ["空输出"], 0

    # 截断过长的 markdown 避免浪费 token
    md_preview = md_part[:3000] if len(md_part) > 3000 else md_part

    img_info = ""
    if image_filenames:
        img_info = f"本页提供了 {len(image_filenames)} 张裁切图片: {', '.join(image_filenames)}"

    prompt = (
        f"对比这张PDF页面图片和下面的Markdown转换结果，评估转换质量。\n\n"
        f"页码: 第{page_num + 1}页\n"
        f"{img_info}\n\n"
        f"<markdown_output>\n{md_preview}\n</markdown_output>\n\n"
        "<evaluation_criteria>\n"
        "1. 文字完整性: Markdown是否完整转写了页面上的主要文字内容？\n"
        "2. 图片引用: 提供的图片是否在Markdown中被正确引用？\n"
        "3. 格式正确: 公式是否正确闭合？标题层级是否合理？\n"
        "4. 无坐标泄漏: 是否存在 bbox 坐标或内部标记泄漏到输出中？\n"
        "</evaluation_criteria>\n\n"
        '<output_format>\n'
        '返回 JSON: {"score": 0-100, "issues": ["问题1", "问题2"]}\n'
        '无问题时: {"score": 95, "issues": []}\n'
        '只输出JSON，不要其他文字。\n'
        '</output_format>'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{page_image_mime};base64,{base64.b64encode(page_image_bytes).decode()}"
                }},
                {"type": "text", "text": prompt},
            ]}],
            temperature=0.1,
            max_tokens=256,
            extra_body={"enable_thinking": False},
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        # 模型输出可能包含未转义的反斜杠（如 LaTeX \frac），需在 JSON 解析前修复
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            raw = raw.replace('\\', '\\\\')
            result = json.loads(raw)
        score = int(result.get("score", 80))
        issues = result.get("issues", [])
        if isinstance(issues, list):
            return [str(i) for i in issues], score
    except Exception as e:
        print(f"  ⚠ 模型质量审查失败: {e}")

    # 模型审查失败时默认通过
    return [], 80


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

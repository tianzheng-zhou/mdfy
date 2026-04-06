"""AI 图片检测管线：detect、crop、verify、refine。"""

import base64
import io
import os
import re

from .config import (
    DETECTION_IMAGE_MAX_SIDE, OCR_IMAGE_MAX_SIDE,
    MAX_CROP_VERIFY_ROUNDS,
)
from .pdf_utils import (
    prepare_image_for_model, bbox_to_pixels,
    _normalize_bbox_to_1000, parse_figure_detection_response,
    parse_qwenvl_markdown_figures, _request_qwenvl_markdown,
)


def _merge_figure_lists(*fig_lists):
    """合并多个 figure 列表，去除 bbox 重叠度 > 50% 的重复项。"""
    merged = []
    for figs in fig_lists:
        for fig in figs:
            bbox = fig["bbox"]
            is_dup = False
            for existing in merged:
                eb = existing["bbox"]
                ix1 = max(bbox[0], eb[0])
                iy1 = max(bbox[1], eb[1])
                ix2 = min(bbox[2], eb[2])
                iy2 = min(bbox[3], eb[3])
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area_b = (eb[2] - eb[0]) * (eb[3] - eb[1])
                    min_area = min(area_a, area_b)
                    if min_area > 0 and inter / min_area > 0.50:
                        is_dup = True
                        break
            if not is_dup:
                merged.append(fig)
    return merged


def detect_page_figures(client, model, page_png_bytes, page_num):
    """用 Qwen 视觉定位检测扫描页中的图片/图表区域。

    返回 [{"bbox": [x1,y1,x2,y2], "desc": "描述"}] 列表。
    策略：JSON bbox + qwenvl markdown 双路检测，合并去重。
    """
    detect_bytes, detect_mime, detect_size = prepare_image_for_model(
        page_png_bytes,
        max_side=DETECTION_IMAGE_MAX_SIDE,
    )

    prompt = (
        "<task>检测扫描书页中的视觉元素，返回精确坐标。</task>\n"
        "\n"
        "<target_elements>\n"
        "  照片、插图、图表、图形、地图、波形图、示意图、书法作品、绘画\n"
        "</target_elements>\n"
        "\n"
        "<exclude>\n"
        "  <item>纯文字行（正文段落、图题、图注、标题）</item>\n"
        "  <item>页眉、页脚、页码</item>\n"
        "  <item>表格（有边框的文字表格）</item>\n"
        "  <item>条形码、装饰线</item>\n"
        "</exclude>\n"
        "\n"
        "<precision_rules>\n"
        "  <rule>bbox 必须覆盖视觉元素的完整区域——从左边界到右边界、从上边界到下边界</rule>\n"
        "  <rule>不要只框住图片的局部或一角，必须包含整张图片</rule>\n"
        "  <rule>如果图片上方有文字段落，y1 从图片顶边开始，不包含文字</rule>\n"
        "  <rule>如果图片下方有文字段落，y2 到图片底边结束，不包含文字</rule>\n"
        "  <rule>x1 和 x2 必须覆盖图片的完整宽度</rule>\n"
        "  <rule>宁可在边缘稍微留白，也不要截掉图片的任何部分</rule>\n"
        "  <rule>封面页、扉页的装饰性背景不算图片，不要框选</rule>\n"
        "</precision_rules>\n"
        "\n"
        "<output_format>\n"
        '  返回 JSON 数组: [{"bbox": [x1, y1, x2, y2], "desc": "简短描述"}]\n'
        "  坐标系: 归一化 [0, 1000]，左上角(0,0)，右下角(1000,1000)\n"
        "  无视觉元素时返回: []\n"
        "  只输出 JSON，不要其他文字。\n"
        "</output_format>"
    )

    json_figures = []
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:{detect_mime};base64,{base64.b64encode(detect_bytes).decode()}"
                        }},
                        {"type": "text", "text": prompt},
                    ]},
                ],
                temperature=0.1 + attempt * 0.15,
                max_tokens=1024,
                extra_body={"enable_thinking": False},
            )
            raw = response.choices[0].message.content.strip()
            json_figures = parse_figure_detection_response(raw, detect_size)
            if json_figures:
                break
        except Exception:
            pass

    qwenvl_figures = []
    try:
        raw_qwenvl = _request_qwenvl_markdown(client, model, detect_bytes, detect_mime)
        qwenvl_figures = parse_qwenvl_markdown_figures(raw_qwenvl, detect_size)
    except Exception:
        pass

    all_figures = _merge_figure_lists(json_figures, qwenvl_figures)
    return all_figures


def _refine_bbox_with_pixels(img, x1, y1, x2, y2):
    """用像素分析扩展/收缩 bbox，使其精确匹配图片区域的实际边界。"""
    import numpy as np

    arr = np.array(img.convert("L"))
    h, w = arr.shape

    col_start = max(0, x1)
    col_end = min(w, x2)
    if col_end - col_start < 30:
        return x1, y1, x2, y2

    strip = arr[:, col_start:col_end]
    white_ratio = (strip > 230).mean(axis=1)
    is_text = white_ratio > 0.55

    cy = (y1 + y2) // 2
    _skip_y_expansion = False
    if is_text[min(cy, h - 1)]:
        best_s, best_e, best_l = cy, cy, 0
        s = None
        for i in range(max(0, y1), min(h, y2)):
            if not is_text[i]:
                if s is None:
                    s = i
            else:
                if s is not None:
                    if i - s > best_l:
                        best_s, best_e, best_l = s, i, i - s
                    s = None
        if s is not None and min(h, y2) - s > best_l:
            best_s, best_e, best_l = s, min(h, y2), min(h, y2) - s
        if best_l < 15:
            new_y1, new_y2 = y1, y2
            _skip_y_expansion = True
        else:
            cy = (best_s + best_e) // 2
    
    if not _skip_y_expansion:
        new_y1 = cy
        consecutive_text = 0
        for i in range(cy, -1, -1):
            if is_text[i]:
                consecutive_text += 1
                if consecutive_text >= 8:
                    new_y1 = i + consecutive_text
                    break
            else:
                consecutive_text = 0
                new_y1 = i
        else:
            new_y1 = 0

        new_y2 = cy
        consecutive_text = 0
        for i in range(cy, h):
            if is_text[i]:
                consecutive_text += 1
                if consecutive_text >= 8:
                    new_y2 = i - consecutive_text + 1
                    break
            else:
                consecutive_text = 0
                new_y2 = i + 1
        else:
            new_y2 = h

        new_y1 = max(0, new_y1 - 3)
        new_y2 = min(h, new_y2 + 3)

    row_start = max(0, new_y1)
    row_end = min(h, new_y2)
    if row_end - row_start < 30:
        return x1, new_y1, x2, new_y2

    h_strip = arr[row_start:row_end, :]
    col_white_ratio = (h_strip > 230).mean(axis=0)

    bbox_w = x2 - x1
    bbox_h = new_y2 - new_y1
    narrow_bbox = bbox_w < w * 0.35 and bbox_h > bbox_w * 1.5

    if narrow_bbox:
        content_col = col_white_ratio < 0.98
        content_cols = np.where(content_col)[0]
        if len(content_cols) > 5:
            new_x1 = int(content_cols[0])
            new_x2 = int(content_cols[-1]) + 1
        else:
            new_x1, new_x2 = x1, x2
    else:
        is_blank_col = col_white_ratio > 0.95
        cx = (x1 + x2) // 2
        cx = max(0, min(w - 1, cx))

        new_x1 = cx
        consecutive_blank = 0
        for i in range(cx, -1, -1):
            if is_blank_col[i]:
                consecutive_blank += 1
                if consecutive_blank >= 20:
                    new_x1 = i + consecutive_blank
                    break
            else:
                consecutive_blank = 0
                new_x1 = i

        new_x2 = cx
        consecutive_blank = 0
        for i in range(cx, w):
            if is_blank_col[i]:
                consecutive_blank += 1
                if consecutive_blank >= 20:
                    new_x2 = i - consecutive_blank + 1
                    break
            else:
                consecutive_blank = 0
                new_x2 = i + 1
        else:
            new_x2 = w

    new_x1 = min(new_x1, x1)
    new_x2 = max(new_x2, x2)
    new_x1 = max(0, new_x1 - 3)
    new_x2 = min(w, new_x2 + 3)

    return new_x1, new_y1, new_x2, new_y2


def crop_and_save_figures(page_png_bytes, figures, page_num, images_dir):
    """根据检测到的 bbox 裁切图片区域并保存。返回 [(文件名, 描述, 像素bbox)] 列表。"""
    from PIL import Image
    import numpy as np

    img = Image.open(io.BytesIO(page_png_bytes))
    saved = []
    page_area = img.width * img.height
    for fig_idx, fig in enumerate(figures):
        x1, y1, x2, y2 = bbox_to_pixels(fig["bbox"], img.width, img.height)
        desc = fig.get("desc", "")
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        fig_w = x2 - x1
        fig_h = y2 - y1
        if fig_w < 30 or fig_h < 30:
            continue
        fig_area = fig_w * fig_h
        if fig_area / page_area < 0.005:
            continue
        if fig_area / page_area > 0.70:
            continue
        x1, y1, x2, y2 = _refine_bbox_with_pixels(img, x1, y1, x2, y2)
        fig_w = x2 - x1
        fig_h = y2 - y1
        fig_area = fig_w * fig_h
        if fig_area / page_area > 0.70:
            continue
        aspect = fig_w / max(fig_h, 1)
        if aspect < 0.15 or aspect > 6.5:
            continue
        if fig_w < 80 or fig_h < 80:
            continue
        cropped = img.crop((x1, y1, x2, y2))
        arr = np.array(cropped.convert("L"))
        if (arr > 240).mean() > 0.95:
            continue
        cur_box = (x1, y1, x2, y2)
        is_dup = False
        for _, _, prev_box in saved:
            ix1 = max(cur_box[0], prev_box[0])
            iy1 = max(cur_box[1], prev_box[1])
            ix2 = min(cur_box[2], prev_box[2])
            iy2 = min(cur_box[3], prev_box[3])
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_cur = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
                area_prev = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                union = area_cur + area_prev - inter
                if union > 0 and inter / union > 0.5:
                    is_dup = True
                    break
        if is_dup:
            continue
        filename = f"page{page_num + 1}_fig{fig_idx + 1}.png"
        cropped.save(os.path.join(images_dir, filename))
        saved.append((filename, desc, (x1, y1, x2, y2)))
    return saved


def _draw_bbox_overlay(page_png_bytes, bboxes_px):
    """在页面图片上绘制带编号的矩形框，供 AI 审查参考。"""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(page_png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [
        (255, 0, 0), (0, 180, 0), (0, 0, 255), (255, 0, 255),
        (255, 160, 0), (0, 200, 200), (128, 0, 255), (255, 128, 0),
    ]
    for idx, (x1, y1, x2, y2) in enumerate(bboxes_px):
        color = colors[idx % len(colors)]
        for d in range(3):
            draw.rectangle([x1 - d, y1 - d, x2 + d, y2 + d], outline=color)
        label = str(idx + 1)
        draw.rectangle([x1, max(0, y1 - 22), x1 + 26, y1], fill=color)
        draw.text((x1 + 5, max(0, y1 - 20)), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ai_verify_crops(client, model, page_png_bytes, crop_results, page_num,
                     images_dir, *, round_num=0, other_images=None):
    """AI 验证裁切质量，返回每张裁切的动作列表。"""
    import json

    if not crop_results:
        return []

    bboxes_px = [cr[2] for cr in crop_results]
    overlay_bytes = _draw_bbox_overlay(page_png_bytes, bboxes_px)
    overlay_compressed, overlay_mime, _ = prepare_image_for_model(
        overlay_bytes, max_side=OCR_IMAGE_MAX_SIDE,
    )

    crops_desc = []
    for idx, (fname, desc, (x1, y1, x2, y2)) in enumerate(crop_results):
        crops_desc.append(
            f"  图{idx + 1}: {fname} | {x2 - x1}×{y2 - y1}px | 描述: {desc or '无'}"
        )

    content = [
        {"type": "image_url", "image_url": {
            "url": f"data:{overlay_mime};base64,"
                   f"{base64.b64encode(overlay_compressed).decode()}"
        }},
    ]
    for fname, _, _ in crop_results:
        crop_path = os.path.join(images_dir, fname)
        if not os.path.exists(crop_path):
            continue
        with open(crop_path, "rb") as f:
            crop_bytes = f.read()
        crop_compressed, crop_mime, _ = prepare_image_for_model(
            crop_bytes, max_side=800,
        )
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{crop_mime};base64,"
                   f"{base64.b64encode(crop_compressed).decode()}"
        }})

    other_info = ""
    if other_images:
        other_info = (
            f"\n注意：此页面已有 {len(other_images)} 张通过其他方式提取的图片：\n"
            f"  {', '.join(other_images)}\n"
            "如果某张裁切图与已有图片高度重叠（覆盖相同视觉元素），应 reject 该裁切。\n"
        )

    prompt = (
        f"这是第{page_num + 1}页的原始图片（带编号框标注裁切区域），"
        f"后面依次是 {len(crop_results)} 张裁切结果。\n\n"
        f"裁切信息：\n" + "\n".join(crops_desc) + "\n"
        f"{other_info}\n"
        "<task>严格评估每张裁切图的质量。对照原始页面仔细检查。</task>\n\n"
        "<critical_checks>\n"
        "  <check>【完整性】对照原始页面：裁切框是否覆盖了该图形的完整区域？\n"
        "    仔细检查框的上下左右是否有内容被截掉。\n"
        "    例如：一个包含多个子图（如 Read + Write）的大图，框是否只框了其中一部分？\n"
        "    如果截掉了内容 → adjust，提供覆盖完整图形的新 bbox。</check>\n"
        "  <check>【页眉页脚】裁切是否包含了页眉(如\"XX用户手册\")、页脚、页码等无关内容？\n"
        "    如果包含 → adjust，缩小 bbox 排除这些内容。</check>\n"
        "  <check>【误检】该区域是否根本不是图片/图表？（纯文字段落、表格等）\n"
        "    如果是误检 → reject。</check>\n"
        "  <check>【多图合一】一个框内是否包含了多个完全独立、不相关的图形？\n"
        "    如果是 → split。注意：同一个 Figure 的多个子图（如子图a、子图b）不需要拆分。</check>\n"
        "</critical_checks>\n\n"
        "<available_actions>\n"
        "  accept — 裁切完整、无多余内容、无误检，保留原样\n"
        "  reject — 误检或与已有图片重复，删除\n"
        "  adjust — 边界需要扩大或缩小，提供新 bbox（归一化 [0,1000]）\n"
        "  split  — 包含多个独立图形，拆分为多张，提供各子区域 bbox\n"
        "</available_actions>\n\n"
        "<output_format>\n"
        "返回 JSON 数组，每个元素对应一张裁切图：\n"
        '[\n'
        '  {"index": 1, "action": "accept"},\n'
        '  {"index": 2, "action": "reject", "reason": "与已有嵌入图重复"},\n'
        '  {"index": 3, "action": "adjust", "bbox": [x1,y1,x2,y2], '
        '"reason": "底部被截断，需要向下扩展"},\n'
        '  {"index": 4, "action": "split", "regions": [\n'
        '    {"bbox": [x1,y1,x2,y2], "desc": "上部图"},\n'
        '    {"bbox": [x1,y1,x2,y2], "desc": "下部图"}\n'
        '  ]}\n'
        ']\n'
        "坐标系: 归一化 [0,1000]，左上角(0,0)，右下角(1000,1000)。\n"
        "只输出 JSON 数组，不要其他文字。\n"
        "</output_format>"
    )
    content.append({"type": "text", "text": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1 + round_num * 0.1,
            max_tokens=2048,
            extra_body={"enable_thinking": False},
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        actions = json.loads(raw)
        if isinstance(actions, list):
            return actions
    except Exception as e:
        print(f"  ⚠ AI裁切验证失败: {e}")

    return [{"index": i + 1, "action": "accept"} for i in range(len(crop_results))]


def _execute_crop_actions(page_png_bytes, actions, crop_results, page_num, images_dir):
    """根据 AI 验证结果执行裁切动作，返回更新后的 crop_results。"""
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(page_png_bytes))

    action_map = {}
    for act in actions:
        idx = act.get("index", 0) - 1
        if 0 <= idx < len(crop_results):
            action_map[idx] = act

    new_results = []
    for idx, (fname, desc, pixel_bbox) in enumerate(crop_results):
        action = action_map.get(idx, {"action": "accept"})
        act_type = action.get("action", "accept")

        if act_type == "accept":
            new_results.append((fname, desc, pixel_bbox))

        elif act_type == "reject":
            try:
                os.remove(os.path.join(images_dir, fname))
            except OSError:
                pass

        elif act_type == "adjust":
            new_bbox = action.get("bbox")
            if not isinstance(new_bbox, list) or len(new_bbox) != 4:
                new_results.append((fname, desc, pixel_bbox))
                continue
            try:
                x1, y1, x2, y2 = bbox_to_pixels(new_bbox, img.width, img.height)
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img.width, x2 + padding)
                y2 = min(img.height, y2 + padding)
                if x2 - x1 < 80 or y2 - y1 < 80:
                    new_results.append((fname, desc, pixel_bbox))
                    continue
                cropped = img.crop((x1, y1, x2, y2))
                arr = np.array(cropped.convert("L"))
                if (arr > 240).mean() > 0.95:
                    new_results.append((fname, desc, pixel_bbox))
                    continue
                cropped.save(os.path.join(images_dir, fname))
                new_results.append((fname, desc, (x1, y1, x2, y2)))
            except Exception:
                new_results.append((fname, desc, pixel_bbox))

        elif act_type == "split":
            try:
                os.remove(os.path.join(images_dir, fname))
            except OSError:
                pass
            regions = action.get("regions", [])
            if not regions:
                try:
                    cropped = img.crop(pixel_bbox)
                    cropped.save(os.path.join(images_dir, fname))
                    new_results.append((fname, desc, pixel_bbox))
                except Exception:
                    pass
                continue
            for sub_idx, region in enumerate(regions):
                sub_bbox = region.get("bbox")
                sub_desc = region.get("desc", "")
                if not isinstance(sub_bbox, list) or len(sub_bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = bbox_to_pixels(sub_bbox, img.width, img.height)
                    padding = 5
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(img.width, x2 + padding)
                    y2 = min(img.height, y2 + padding)
                    if x2 - x1 < 80 or y2 - y1 < 80:
                        continue
                    cropped = img.crop((x1, y1, x2, y2))
                    arr = np.array(cropped.convert("L"))
                    if (arr > 240).mean() > 0.95:
                        continue
                    sub_fname = f"page{page_num + 1}_fig{idx + 1}_{chr(97 + sub_idx)}.png"
                    cropped.save(os.path.join(images_dir, sub_fname))
                    new_results.append((sub_fname, sub_desc, (x1, y1, x2, y2)))
                except Exception:
                    continue

        else:
            new_results.append((fname, desc, pixel_bbox))

    return new_results


def _ai_verify_and_refine_crops(client, model, page_png_bytes, crop_results,
                                page_num, images_dir, *,
                                max_rounds=MAX_CROP_VERIFY_ROUNDS,
                                other_images=None):
    """AI 验证 + 优化裁切结果的主循环。"""
    for round_num in range(max_rounds):
        if not crop_results:
            break

        actions = _ai_verify_crops(
            client, model, page_png_bytes, crop_results,
            page_num, images_dir, round_num=round_num,
            other_images=other_images,
        )

        for a in actions:
            act = a.get("action", "accept")
            if act != "accept":
                reason = a.get("reason", "")
                tag = f"{act}" + (f"({reason})" if reason else "")
                print(f"[fig{a.get('index', '?')}:{tag}]", end=" ", flush=True)

        if all(a.get("action", "accept") == "accept" for a in actions):
            break

        crop_results = _execute_crop_actions(
            page_png_bytes, actions, crop_results, page_num, images_dir,
        )

        if not any(a.get("action") in ("adjust", "split") for a in actions):
            break

    return crop_results

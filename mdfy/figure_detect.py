"""AI 图片检测管线：detect → crop → verify → refine。纯视觉版。

纯视觉模式下所有图片都来自 AI 检测 + 裁切；无嵌入图合并逻辑。
"""

import base64
import io
import json
import os
import re
from collections import defaultdict

from .config import (
    DETECTION_IMAGE_MAX_SIDE, OCR_IMAGE_MAX_SIDE, VERIFY_CROP_MAX_SIDE,
    MAX_CROP_VERIFY_ROUNDS,
)
from .pdf_render import (
    prepare_image_for_model, bbox_to_pixels, encode_data_url,
    parse_figure_detection_response, parse_qwenvl_markdown_figures,
    request_qwenvl_markdown,
)
from .prompts import (
    DETECT_SYSTEM, VERIFY_SYSTEM,
    build_detect_prompt, build_verify_prompt,
    DETECT_FALLBACK_PROMPT_TEMPLATE,
)


# ══════════════════════════════════════════════════════════════════════
# 合并 & 聚类
# ══════════════════════════════════════════════════════════════════════

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


def _cluster_nearby_bboxes(figures, gap_threshold=40):
    """将空间邻近的 bbox 聚类合并为包围框（归一化 [0,1000] 坐标系）。"""
    if len(figures) <= 1:
        return figures

    parent = list(range(len(figures)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(figures)):
        for j in range(i + 1, len(figures)):
            bi = figures[i]["bbox"]
            bj = figures[j]["bbox"]
            h_gap = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            v_gap = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            x_overlap = max(0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
            w_i = bi[2] - bi[0]
            w_j = bj[2] - bj[0]
            min_w = max(min(w_i, w_j), 1)
            x_overlap_ratio = x_overlap / min_w
            y_overlap = max(0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
            h_i = bi[3] - bi[1]
            h_j = bj[3] - bj[1]
            min_h = max(min(h_i, h_j), 1)
            y_overlap_ratio = y_overlap / min_h
            if max(h_gap, v_gap) < gap_threshold and (x_overlap_ratio > 0.3 or y_overlap_ratio > 0.3):
                union(i, j)

    clusters = defaultdict(list)
    for i in range(len(figures)):
        clusters[find(i)].append(i)

    result = []
    for indices in clusters.values():
        if len(indices) == 1:
            result.append(figures[indices[0]])
        else:
            x1 = min(figures[i]["bbox"][0] for i in indices)
            y1 = min(figures[i]["bbox"][1] for i in indices)
            x2 = max(figures[i]["bbox"][2] for i in indices)
            y2 = max(figures[i]["bbox"][3] for i in indices)
            descs = [figures[i].get("desc", "") for i in indices if figures[i].get("desc")]
            desc = "; ".join(descs[:3]) if descs else ""
            result.append({"bbox": [x1, y1, x2, y2], "desc": desc})
    return result


# ══════════════════════════════════════════════════════════════════════
# 检测
# ══════════════════════════════════════════════════════════════════════

def detect_page_figures(client, model, page_png_bytes, page_num, doc_context=""):
    """用视觉模型检测页面中的图片/图表区域。

    策略：JSON bbox + qwenvl markdown 双路检测，合并去重 + 邻近聚类。
    兜底：若两路都为空，用更宽松的 prompt 再试一次。

    返回 [{"bbox": [x1,y1,x2,y2], "desc": "描述"}]，坐标归一化 [0,1000]。
    """
    detect_bytes, detect_mime, detect_size = prepare_image_for_model(
        page_png_bytes, max_side=DETECTION_IMAGE_MAX_SIDE,
    )

    prompt = build_detect_prompt(doc_context)

    # ── Path 1: JSON bbox 检测 ──
    json_figures = []
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DETECT_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": encode_data_url(detect_bytes, detect_mime)}},
                        {"type": "text", "text": prompt},
                    ]},
                ],
                temperature=0.15 + attempt * 0.15,
                max_tokens=2048,
                extra_body={"enable_thinking": False},
            )
            raw = response.choices[0].message.content.strip()
            json_figures = parse_figure_detection_response(raw, detect_size)
            if json_figures:
                break
        except Exception as e:
            print(f"  ⚠ 第{page_num+1}页 JSON 检测尝试{attempt+1}失败: {e}")

    # ── Path 2: qwenvl markdown 检测 ──
    qwenvl_figures = []
    try:
        raw_qwenvl = request_qwenvl_markdown(client, model, detect_bytes, detect_mime)
        qwenvl_figures = parse_qwenvl_markdown_figures(raw_qwenvl, detect_size)
    except Exception as e:
        print(f"  ⚠ 第{page_num+1}页 qwenvl 检测失败: {e}")

    # ── Fallback: 双路都空时用更宽松的 prompt ──
    if not json_figures and not qwenvl_figures:
        ctx = f"<document_context>{doc_context}</document_context>\n\n" if doc_context else ""
        fallback_prompt = DETECT_FALLBACK_PROMPT_TEMPLATE.format(context=ctx)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": encode_data_url(detect_bytes, detect_mime)}},
                    {"type": "text", "text": fallback_prompt},
                ]}],
                temperature=0.3,
                max_tokens=2048,
                extra_body={"enable_thinking": False},
            )
            raw = response.choices[0].message.content.strip()
            fallback_figures = parse_figure_detection_response(raw, detect_size)
            if fallback_figures:
                print(f"  ✓ 第{page_num+1}页兜底检测发现{len(fallback_figures)}个图形", flush=True)
                json_figures = fallback_figures
        except Exception as e:
            print(f"  ⚠ 第{page_num+1}页兜底检测失败: {e}")

    all_figures = _merge_figure_lists(json_figures, qwenvl_figures)
    all_figures = _cluster_nearby_bboxes(all_figures)
    return all_figures


# ══════════════════════════════════════════════════════════════════════
# 裁切保存
# ══════════════════════════════════════════════════════════════════════

def crop_and_save_figures(page_png_bytes, figures, page_num, images_dir):
    """根据检测到的 bbox 裁切图片区域并保存。

    返回 [(文件名, 描述, 像素bbox)] 列表。
    """
    from PIL import Image

    img = Image.open(io.BytesIO(page_png_bytes))
    saved = []
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
        _tag = f"[p{page_num+1} fig{fig_idx+1}"

        if fig_w < 20 or fig_h < 20:
            print(f"{_tag}: skip tiny {fig_w}x{fig_h}]", end=" ", flush=True)
            continue

        # IoU 去重
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
            print(f"{_tag}: skip IoU-dup]", end=" ", flush=True)
            continue

        cropped = img.crop((x1, y1, x2, y2))
        filename = f"page{page_num + 1}_fig{fig_idx + 1}.png"
        cropped.save(os.path.join(images_dir, filename))
        saved.append((filename, desc, (x1, y1, x2, y2)))

    return saved


# ══════════════════════════════════════════════════════════════════════
# 裁切校验 & 优化
# ══════════════════════════════════════════════════════════════════════

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
                     images_dir, *, round_num=0, doc_context=""):
    """AI 验证裁切质量，返回每张裁切的 action 列表。"""
    if not crop_results:
        return []

    bboxes_px = [cr[2] for cr in crop_results]
    overlay_bytes = _draw_bbox_overlay(page_png_bytes, bboxes_px)
    overlay_compressed, overlay_mime, _ = prepare_image_for_model(
        overlay_bytes, max_side=OCR_IMAGE_MAX_SIDE,
    )

    crops_desc_lines = []
    for idx, (fname, desc, (x1, y1, x2, y2)) in enumerate(crop_results):
        crops_desc_lines.append(
            f"  图{idx + 1}: {fname} | {x2 - x1}×{y2 - y1}px | 描述: {desc or '无'}"
        )
    crops_desc = "\n".join(crops_desc_lines)

    content = [
        {"type": "image_url", "image_url": {"url": encode_data_url(overlay_compressed, overlay_mime)}},
    ]
    for fname, _, _ in crop_results:
        crop_path = os.path.join(images_dir, fname)
        if not os.path.exists(crop_path):
            continue
        with open(crop_path, "rb") as f:
            crop_bytes = f.read()
        crop_compressed, crop_mime, _ = prepare_image_for_model(
            crop_bytes, max_side=VERIFY_CROP_MAX_SIDE,
        )
        content.append({"type": "image_url", "image_url": {"url": encode_data_url(crop_compressed, crop_mime)}})

    prompt = build_verify_prompt(len(crop_results), crops_desc, page_num, doc_context)
    content.append({"type": "text", "text": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": content},
            ],
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
        print(f"  ⚠ AI 裁切验证失败: {e}")

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


def verify_and_refine_crops(client, model, page_png_bytes, crop_results,
                            page_num, images_dir, *,
                            max_rounds=MAX_CROP_VERIFY_ROUNDS, doc_context=""):
    """AI 验证 + 优化裁切结果的主循环。"""
    for round_num in range(max_rounds):
        if not crop_results:
            break

        actions = _ai_verify_crops(
            client, model, page_png_bytes, crop_results,
            page_num, images_dir, round_num=round_num, doc_context=doc_context,
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


# ══════════════════════════════════════════════════════════════════════
# 跨页装饰图过滤（logo/页眉图标等）
# ══════════════════════════════════════════════════════════════════════

def filter_cross_page_decorative(page_figures, page_data, total_pages):
    """过滤跨页重复的小装饰图（logo、校徽等）。

    判据（数据驱动，不靠主观判断）：
    - 相同尺寸签名（round 到 10px）出现在 > 50% 的页面上
    - 平均占页面面积 < 8%

    参数：
        page_figures: {page_num: [(filename, desc, (x1,y1,x2,y2)), ...]}
        page_data: [{'page_num', 'page_png', ...}, ...] 供获取页面尺寸
        total_pages: 总页数

    就地修改 page_figures 并删除对应文件，返回移除的图片数。
    """
    if total_pages < 3 or not page_figures:
        return 0

    from PIL import Image

    sig_pages = defaultdict(set)
    sig_crops = defaultdict(list)  # (page_num, fig_idx, area_ratio, filename, images_dir)

    # 构建 page_num → (png, images_dir) 的查找
    page_lookup = {}
    for pd in page_data:
        page_lookup[pd['page_num']] = pd

    for page_num, fig_results in page_figures.items():
        pd = page_lookup.get(page_num)
        if not pd:
            continue
        pg_img = Image.open(io.BytesIO(pd['page_png']))
        pg_area = pg_img.width * pg_img.height
        for idx, (fname, desc, (x1, y1, x2, y2)) in enumerate(fig_results):
            w, h = x2 - x1, y2 - y1
            sig = (round(w / 10) * 10, round(h / 10) * 10)
            sig_pages[sig].add(page_num)
            sig_crops[sig].append((page_num, idx, (w * h) / pg_area if pg_area > 0 else 1.0))

    threshold = total_pages * 0.5
    decorative_sigs = set()
    for sig, pages_set in sig_pages.items():
        if len(pages_set) > threshold:
            crops = sig_crops[sig]
            avg_area = sum(c[2] for c in crops) / len(crops) if crops else 1.0
            if avg_area < 0.08:
                decorative_sigs.add(sig)

    if not decorative_sigs:
        return 0

    removed = 0
    for page_num in list(page_figures.keys()):
        fig_results = page_figures[page_num]
        new_results = []
        images_dir = page_lookup[page_num].get('images_dir')
        for fname, desc, (x1, y1, x2, y2) in fig_results:
            sig = (round((x2 - x1) / 10) * 10, round((y2 - y1) / 10) * 10)
            if sig in decorative_sigs:
                if images_dir:
                    try:
                        os.remove(os.path.join(images_dir, fname))
                    except OSError:
                        pass
                removed += 1
            else:
                new_results.append((fname, desc, (x1, y1, x2, y2)))
        if new_results:
            page_figures[page_num] = new_results
        else:
            del page_figures[page_num]

    return removed


# ══════════════════════════════════════════════════════════════════════
# 图片位置计算（纯像素坐标版）
# ══════════════════════════════════════════════════════════════════════

def compute_image_positions(image_filenames, page_png_bytes, fig_bboxes):
    """按页面纵向位置排序图片并计算位置百分比。

    参数：
        image_filenames: list[str]
        page_png_bytes: 页面 PNG 字节
        fig_bboxes: {filename: (x1,y1,x2,y2)} 像素坐标

    返回 (sorted_filenames, positions_dict, coverage_pct)。
    positions 值形如 "~20%-50% 右半"；coverage_pct 图片总覆盖面积占比 (0-100)。
    """
    if not image_filenames:
        return [], {}, 0

    from PIL import Image
    pil_img = Image.open(io.BytesIO(page_png_bytes))
    page_width = pil_img.width
    page_height = pil_img.height

    items = []
    for fname in image_filenames:
        bbox = fig_bboxes.get(fname)
        if bbox:
            x_left, y_top, x_right, y_bottom = bbox
        else:
            # 缺失 bbox 时给一个居中默认值
            x_left = page_width * 0.1
            y_top = page_height * 0.3
            x_right = page_width * 0.9
            y_bottom = page_height * 0.7
        y_center = (y_top + y_bottom) / 2
        x_center = (x_left + x_right) / 2
        items.append((fname, y_center, y_top, y_bottom, x_left, x_right, x_center))

    items.sort(key=lambda x: x[1])

    page_area = page_width * page_height
    total_img_area = sum((xr - xl) * (yb - yt) for _, _, yt, yb, xl, xr, _ in items)
    coverage_pct = round(total_img_area / page_area * 100) if page_area > 0 else 0

    sorted_filenames = [it[0] for it in items]
    positions = {}
    for fname, _, y_top, y_bottom, _, _, x_center in items:
        top_pct = round(y_top / page_height * 100) if page_height > 0 else 30
        bot_pct = round(y_bottom / page_height * 100) if page_height > 0 else 70
        x_ratio = x_center / page_width if page_width > 0 else 0.5
        if x_ratio < 0.35:
            positions[fname] = f"~{top_pct}%-{bot_pct}% 左半"
        elif x_ratio > 0.65:
            positions[fname] = f"~{top_pct}%-{bot_pct}% 右半"
        else:
            positions[fname] = f"~{top_pct}%-{bot_pct}%"

    return sorted_filenames, positions, coverage_pct


# ══════════════════════════════════════════════════════════════════════
# 并行阶段调度
# ══════════════════════════════════════════════════════════════════════

def detect_and_refine_page(client, model, page_png, page_num, images_dir, doc_context=""):
    """并行阶段入口：检测 + 裁切 + AI 校验单页图表区域。

    返回 [(filename, desc, pixel_bbox), ...] 或空列表。
    """
    try:
        figures = detect_page_figures(
            client, model, page_png, page_num, doc_context=doc_context,
        )
    except Exception as e:
        print(f"  ⚠ 第{page_num+1}页图片检测失败: {e}")
        figures = []

    if not figures:
        return []

    fig_results = crop_and_save_figures(page_png, figures, page_num, str(images_dir))
    if fig_results:
        fig_results = verify_and_refine_crops(
            client, model, page_png, fig_results,
            page_num, str(images_dir), doc_context=doc_context,
        )
    return fig_results or []

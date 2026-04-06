"""AI/嵌入图合并去重、位置计算。"""

import io
import os
import re


def _merge_ai_and_embedded_images(page, doc, ai_filenames, ai_pixel_bboxes, embedded_filenames,
                                    page_png_bytes, images_dir):
    """合并AI检测图和嵌入图，双向去重。

    ai_pixel_bboxes: 成功裁切的AI图的像素坐标列表 [(x1,y1,x2,y2), ...]
    返回去重后的图片文件名列表（不重叠的AI图 + 不重叠的嵌入图）。
    双向去重逻辑：
    1. AI裁切被嵌入图覆盖>50% → 移除AI裁切（嵌入图质量更高）
    2. 嵌入图被AI裁切覆盖>50% → 移除嵌入图
    """
    from PIL import Image

    if not ai_filenames or not embedded_filenames:
        return ai_filenames + embedded_filenames

    img = Image.open(io.BytesIO(page_png_bytes))
    scale_x = img.width / page.rect.width
    scale_y = img.height / page.rect.height

    # ── 第1步：计算所有嵌入图的像素坐标 ──
    # 注意：embedded_filenames 经过装饰图/全页图过滤，其位置 ≠ page.get_images() 索引。
    # 必须从文件名解析原始 img_index（page1_img3.png → img_index=2）来正确映射 xref。
    image_list = page.get_images(full=True)
    embedded_pixel_bboxes = []  # 与 embedded_filenames 一一对应，None 表示无法获取
    for filename in embedded_filenames:
        m = re.search(r'_img(\d+)\.png$', filename)
        if not m:
            embedded_pixel_bboxes.append(None)
            continue
        img_idx = int(m.group(1)) - 1  # 0-based index into original image_list
        if img_idx >= len(image_list):
            embedded_pixel_bboxes.append(None)
            continue
        xref = image_list[img_idx][0]
        try:
            rects = page.get_image_rects(xref)
            if not rects:
                embedded_pixel_bboxes.append(None)
                continue
            rect = rects[0]
            ex1 = rect.x0 * scale_x
            ey1 = rect.y0 * scale_y
            ex2 = rect.x1 * scale_x
            ey2 = rect.y1 * scale_y
            embedded_pixel_bboxes.append((ex1, ey1, ex2, ey2))
        except Exception:
            embedded_pixel_bboxes.append(None)

    # ── 第2步：反向去重 — 移除被嵌入图覆盖的AI裁切 ──
    # 关键：一个AI裁切可能横跨多张嵌入图（如logo+chart），需要将所有嵌入图的
    # 覆盖面积加总来判断，而不是只看单张嵌入图是否 >50%。
    kept_ai_indices = []
    for ai_idx, (ax1, ay1, ax2, ay2) in enumerate(ai_pixel_bboxes):
        ai_area = max((ax2 - ax1) * (ay2 - ay1), 1)
        total_inter = 0
        for eb in embedded_pixel_bboxes:
            if eb is None:
                continue
            ex1, ey1, ex2, ey2 = eb
            ix1, iy1 = max(ax1, ex1), max(ay1, ey1)
            ix2, iy2 = min(ax2, ex2), min(ay2, ey2)
            if ix1 < ix2 and iy1 < iy2:
                total_inter += (ix2 - ix1) * (iy2 - iy1)
        is_covered_by_embed = total_inter / ai_area > 0.50
        if is_covered_by_embed:
            fname = ai_filenames[ai_idx]
            try:
                os.remove(os.path.join(images_dir, fname))
            except OSError:
                pass
            print(f"[{fname}:reject(与嵌入图重叠)]", end=" ", flush=True)
        else:
            kept_ai_indices.append(ai_idx)

    kept_ai = [ai_filenames[i] for i in kept_ai_indices]
    kept_ai_bboxes = [ai_pixel_bboxes[i] for i in kept_ai_indices]

    # ── 第3步：正向去重 — 移除被(剩余)AI裁切覆盖的嵌入图 ──
    kept_embedded = []
    for idx, filename in enumerate(embedded_filenames):
        eb = embedded_pixel_bboxes[idx]
        if eb is None:
            kept_embedded.append(filename)
            continue
        ex1, ey1, ex2, ey2 = eb
        emb_area = max((ex2 - ex1) * (ey2 - ey1), 1)
        is_covered = False
        for ax1, ay1, ax2, ay2 in kept_ai_bboxes:
            ix1, iy1 = max(ex1, ax1), max(ey1, ay1)
            ix2, iy2 = min(ex2, ax2), min(ey2, ay2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / emb_area > 0.50:
                    is_covered = True
                    break
        if is_covered:
            try:
                os.remove(os.path.join(images_dir, filename))
            except OSError:
                pass
        else:
            kept_embedded.append(filename)

    # ── 第4步：文字区域误检过滤 — 移除 AI 裁切中以文字为主的区域 ──
    # AI 视觉检测有时把彩色文字或段落区域误判为图/表，这类区域不与嵌入图重叠
    # 所以不会被前面的 dedup 捕获。用 PyMuPDF 文本块检测文字覆盖率。
    text_blocks = [b for b in page.get_text("blocks") if b[6] == 0]  # type 0 = text
    final_ai = []
    for fname, bbox_px in zip(kept_ai, kept_ai_bboxes):
        ax1, ay1, ax2, ay2 = bbox_px
        # 像素坐标 → 页面坐标
        px0, py0 = ax1 / scale_x, ay1 / scale_y
        px1, py1 = ax2 / scale_x, ay2 / scale_y
        ai_page_area = max((px1 - px0) * (py1 - py0), 1)
        text_overlap = 0
        for b in text_blocks:
            bx0, by0, bx1, by1 = b[:4]
            ix0, iy0 = max(px0, bx0), max(py0, by0)
            ix1, iy1 = min(px1, bx1), min(py1, by1)
            if ix0 < ix1 and iy0 < iy1:
                text_overlap += (ix1 - ix0) * (iy1 - iy0)
        text_ratio = text_overlap / ai_page_area
        if text_ratio > 0.40:
            try:
                os.remove(os.path.join(images_dir, fname))
            except OSError:
                pass
            print(f"[{fname}:reject(文字区域{text_ratio:.0%})]", end=" ", flush=True)
        else:
            final_ai.append(fname)

    return final_ai + kept_embedded


def _compute_image_positions(image_filenames, page_png_bytes, fig_bboxes=None, page=None):
    """按页面纵向位置排序图片并计算位置百分比。

    fig_bboxes: {filename: (x1,y1,x2,y2)} AI检测/裁切的像素坐标
    page: PyMuPDF 页面对象（用于获取嵌入图位置）
    返回 (sorted_filenames, positions_dict, coverage_pct, bboxes_pdf)
      positions_dict 值格式如 "~20%-50% 右半"
      coverage_pct: 图片总覆盖面积占页面比例 (0-100)
      bboxes_pdf: {filename: (x0,y0,x1,y1)} PDF坐标系的bbox
    """
    if not image_filenames:
        return [], {}, 0, {}

    from PIL import Image
    pil_img = Image.open(io.BytesIO(page_png_bytes))
    page_width = pil_img.width
    page_height = pil_img.height

    items = []  # [(filename, y_top, y_bottom, x_left, x_right)]
    bboxes_pdf = {}  # {filename: (x0, y0, x1, y1)} in PDF coordinate space

    # 计算像素→PDF坐标的转换因子
    pdf_scale_x = page.rect.width / page_width if page is not None and page_width > 0 else 1.0
    pdf_scale_y = page.rect.height / page_height if page is not None and page_height > 0 else 1.0

    for fname in image_filenames:
        y_top = None
        y_bottom = None
        x_left = None
        x_right = None
        pdf_bbox = None  # (x0, y0, x1, y1) in PDF coords

        # 1. AI/crop bbox（像素坐标）
        if fig_bboxes and fname in fig_bboxes:
            bbox = fig_bboxes[fname]
            x_left = bbox[0]
            y_top = bbox[1]
            x_right = bbox[2]
            y_bottom = bbox[3]
            # 转换为 PDF 坐标
            pdf_bbox = (x_left * pdf_scale_x, y_top * pdf_scale_y,
                        x_right * pdf_scale_x, y_bottom * pdf_scale_y)

        # 2. 嵌入图：从 PDF 页面获取位置
        elif page is not None:
            m = re.search(r'_img(\d+)\.png$', fname)
            if m:
                img_idx = int(m.group(1)) - 1  # 0-based
                try:
                    image_list = page.get_images(full=True)
                    if img_idx < len(image_list):
                        xref = image_list[img_idx][0]
                        rects = page.get_image_rects(xref)
                        if rects:
                            scale_x = page_width / page.rect.width
                            scale_y = page_height / page.rect.height
                            rect = rects[0]
                            # 保存 PDF 坐标 bbox
                            pdf_bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                            x_left = rect.x0 * scale_x
                            y_top = rect.y0 * scale_y
                            x_right = rect.x1 * scale_x
                            y_bottom = rect.y1 * scale_y
                except Exception:
                    pass

        if y_top is None:
            y_top = page_height * 0.3
            y_bottom = page_height * 0.7
            x_left = page_width * 0.1
            x_right = page_width * 0.9
            pdf_bbox = None  # 无法确定，不标注

        y_center = (y_top + y_bottom) / 2
        x_center = (x_left + x_right) / 2
        items.append((fname, y_center, y_top, y_bottom, x_left, x_right, x_center))
        if pdf_bbox is not None:
            bboxes_pdf[fname] = pdf_bbox

    items.sort(key=lambda x: x[1])

    # 计算图片覆盖面积（简单求和，不计重叠）
    page_area = page_width * page_height
    total_img_area = 0
    for _, _, yt, yb, xl, xr, _ in items:
        total_img_area += (xr - xl) * (yb - yt)
    coverage_pct = round(total_img_area / page_area * 100) if page_area > 0 else 0

    sorted_filenames = [it[0] for it in items]
    positions = {}
    for fname, y_center, y_top, y_bottom, x_left, x_right, x_center in items:
        top_pct = round(y_top / page_height * 100) if page_height > 0 else 30
        bot_pct = round(y_bottom / page_height * 100) if page_height > 0 else 70
        # 水平位置提示
        x_ratio = x_center / page_width if page_width > 0 else 0.5
        if x_ratio < 0.35:
            positions[fname] = f"~{top_pct}%-{bot_pct}% 左半"
        elif x_ratio > 0.65:
            positions[fname] = f"~{top_pct}%-{bot_pct}% 右半"
        else:
            positions[fname] = f"~{top_pct}%-{bot_pct}%"

    return sorted_filenames, positions, coverage_pct, bboxes_pdf


def _compute_text_in_images(page, bboxes_pdf):
    """计算每张图片bbox内包含的文本块。

    page: PyMuPDF 页面对象
    bboxes_pdf: {filename: (x0, y0, x1, y1)} PDF坐标系的bbox
    返回 {filename: "摘要文本"} — 每张图片内文字的摘要（截断到80字符）
    """
    if not bboxes_pdf or page is None:
        return {}

    text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    result = {}

    for fname, (ix0, iy0, ix1, iy1) in bboxes_pdf.items():
        overlapping_texts = []
        for b in text_blocks:
            if b[6] != 0:  # 跳过图片块
                continue
            bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
            text = b[4].strip()
            if not text:
                continue
            # 计算文本块与图片bbox的重叠面积
            overlap_x = max(0, min(bx1, ix1) - max(bx0, ix0))
            overlap_y = max(0, min(by1, iy1) - max(by0, iy0))
            block_area = max((bx1 - bx0) * (by1 - by0), 1)
            overlap_area = overlap_x * overlap_y
            # 文本块50%以上面积在图片内 → 认为该文本属于图片
            if overlap_area / block_area > 0.5:
                first_line = text.split('\n')[0].strip()
                if first_line:
                    overlapping_texts.append(first_line[:40])
        if overlapping_texts:
            summary = ', '.join(overlapping_texts)
            result[fname] = summary[:80]

    return result

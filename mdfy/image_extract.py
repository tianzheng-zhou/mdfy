"""嵌入图片提取、碎片合并、装饰图过滤。"""

import os
import re

import fitz

from .pdf_utils import _normalize_bbox_to_1000


def _detect_image_rotation(transform):
    """从 PDF 图片变换矩阵检测旋转角度，返回 0/±90/180。"""
    import math
    a, b = transform[0], transform[1]
    angle = math.degrees(math.atan2(b, a))
    rounded = round(angle / 90) * 90
    return rounded if rounded % 360 != 0 else 0


def _find_decorative_xrefs(doc):
    """预扫描全文档，找出跨页重复的装饰性图片 xref（logo/页脚图标等）。

    判断条件（数据驱动）：
    - 同一 xref 出现在 >50% 的页面上
    - 页面面积占比 <10%（排除真正的内容图片被复用的情况）
    """
    from collections import Counter, defaultdict

    total_pages = len(doc)
    if total_pages < 3:
        return set()
    xref_page_count = Counter()
    xref_area_pct = {}
    for pg_idx in range(total_pages):
        page = doc[pg_idx]
        page_area = page.rect.width * page.rect.height
        seen_on_page = set()
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_on_page:
                continue
            seen_on_page.add(xref)
            xref_page_count[xref] += 1
            if xref not in xref_area_pct and page_area > 0:
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        r = rects[0]
                        xref_area_pct[xref] = (r.width * r.height) / page_area
                except Exception:
                    pass
    threshold = total_pages * 0.5
    decorative = set()
    # Phase 1: 按 xref 计数
    for xref, count in xref_page_count.items():
        if count > threshold and xref_area_pct.get(xref, 1.0) < 0.10:
            decorative.add(xref)

    # Phase 2: 按 (width, height) 聚合
    xref_dims = {}
    for pg_idx in range(total_pages):
        page = doc[pg_idx]
        for img in page.get_images(full=True):
            xref = img[0]
            if xref not in xref_dims:
                try:
                    info = doc.extract_image(xref)
                    xref_dims[xref] = (info["width"], info["height"])
                except Exception:
                    pass
    dim_groups = defaultdict(set)
    for xref, dims in xref_dims.items():
        dim_groups[dims].add(xref)
    for dims, xrefs in dim_groups.items():
        if len(xrefs) < 2:
            continue
        total_count = sum(xref_page_count.get(x, 0) for x in xrefs)
        avg_area = sum(xref_area_pct.get(x, 1.0) for x in xrefs if x in xref_area_pct)
        n_with_area = sum(1 for x in xrefs if x in xref_area_pct)
        avg_area = avg_area / n_with_area if n_with_area else 1.0
        if total_count > threshold and avg_area < 0.10:
            decorative.update(xrefs)

    if decorative:
        print(f"  [decorative xrefs: {decorative} (skipped on {len(decorative)} repeated images)]")
    return decorative


def extract_and_save_images(doc, page, page_num, images_dir, *, decorative_xrefs=None):
    """提取并保存页面中的原始图片，返回图片文件名列表。"""
    if decorative_xrefs is None:
        decorative_xrefs = set()
    saved = []
    image_list = page.get_images(full=True)
    page_area = page.rect.width * page.rect.height
    image_info = page.get_image_info(xrefs=True)
    xref_transforms = {}
    for info in image_info:
        xr = info.get('xref', 0)
        tf = info.get('transform')
        if xr and tf:
            xref_transforms[xr] = tf

    candidates = []
    for img_index, img in enumerate(image_list):
        xref = img[0]
        if xref in decorative_xrefs:
            continue
        try:
            rects = page.get_image_rects(xref)
            if rects:
                rect = rects[0]
                img_area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                if page_area > 0 and img_area / page_area > 0.80:
                    print(f"  [skip fullpage img{img_index+1} ({img_area/page_area:.0%})]", end=" ", flush=True)
                    continue
                if rect.width > 0 and rect.height > 0:
                    aspect = max(rect.width / rect.height, rect.height / rect.width)
                    is_h_separator = aspect > 15 and rect.width / page.rect.width > 0.40
                    is_v_separator = aspect > 15 and rect.height / page.rect.height > 0.40
                    if is_h_separator or is_v_separator:
                        print(f"  [skip separator img{img_index+1} (aspect={aspect:.0f}:1)]", end=" ", flush=True)
                        continue
                if page_area > 0 and img_area / page_area > 0.30:
                    try:
                        pix_check = fitz.Pixmap(doc, xref)
                        src_pixels = pix_check.width * pix_check.height
                        ref_dpi = 150
                        display_pixels = (rect.width * ref_dpi / 72) * (rect.height * ref_dpi / 72)
                        pix_check = None
                        if display_pixels > 0 and src_pixels / display_pixels < 0.05:
                            print(f"  [skip bg-texture img{img_index+1} (density={src_pixels/display_pixels:.1%})]", end=" ", flush=True)
                            continue
                    except Exception:
                        pass
                candidates.append((img_index, xref, rect))
            else:
                candidates.append((img_index, xref, None))
        except Exception:
            candidates.append((img_index, xref, None))

    groups = _group_adjacent_images(candidates)

    for group in groups:
        if len(group) > 1 and all(r is not None for _, _, r in group):
            rects = [r for _, _, r in group]
            union = fitz.Rect(rects[0])
            for r in rects[1:]:
                union |= r
            render_dpi = 200
            for _, xr, r in group:
                try:
                    pix = fitz.Pixmap(doc, xr)
                    dpi_x = pix.width / (r.width / 72) if r.width > 0 else 200
                    render_dpi = max(render_dpi, dpi_x)
                    pix = None
                except Exception:
                    pass
            render_dpi = min(render_dpi, 400)
            mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
            merged_pix = page.get_pixmap(matrix=mat, clip=union)
            first_idx = group[0][0]
            filename = f"page{page_num + 1}_img{first_idx + 1}.png"
            filepath = os.path.join(images_dir, filename)
            merged_pix.save(filepath)
            merged_pix = None
            merged_indices = [str(i + 1) for i, _, _ in group]
            print(f"  [merge img{'&'.join(merged_indices)} → {filename}]", end=" ", flush=True)
            saved.append(filename)
        else:
            for img_index, xref, rect in group:
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace and pix.colorspace.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    filename = f"page{page_num + 1}_img{img_index + 1}.png"
                    filepath = os.path.join(images_dir, filename)
                    pix.save(filepath)
                    pix = None
                    tf = xref_transforms.get(xref)
                    if tf:
                        rotation = _detect_image_rotation(tf)
                        if rotation != 0:
                            from PIL import Image
                            with Image.open(filepath) as im:
                                corrected = im.rotate(-rotation, expand=True)
                                corrected.save(filepath)
                    saved.append(filename)
                except Exception as e:
                    print(f"  ⚠ 第{page_num+1}页图片{img_index+1}提取失败: {e}")
    return saved


def _group_adjacent_images(candidates):
    """将空间上垂直相邻的碎片图分到同一组。"""
    GAP_Y_MAX = 5
    GAP_Y_MIN = -2
    OVERLAP_X = 0.5

    with_rect = [(i, x, r) for i, x, r in candidates if r is not None]
    without_rect = [(i, x, r) for i, x, r in candidates if r is None]
    if not with_rect:
        return [[(i, x, r)] for i, x, r in candidates] if candidates else []
    with_rect.sort(key=lambda t: t[2].y0)
    groups = [[with_rect[0]]]
    for item in with_rect[1:]:
        prev_rect = groups[-1][-1][2]
        curr_rect = item[2]
        gap = curr_rect.y0 - prev_rect.y1
        if GAP_Y_MIN <= gap <= GAP_Y_MAX:
            overlap_x0 = max(prev_rect.x0, curr_rect.x0)
            overlap_x1 = min(prev_rect.x1, curr_rect.x1)
            overlap_w = max(0, overlap_x1 - overlap_x0)
            min_w = min(prev_rect.width, curr_rect.width)
            if min_w > 0 and overlap_w / min_w > OVERLAP_X:
                groups[-1].append(item)
                continue
        groups.append([item])
    for item in without_rect:
        groups.append([item])
    return groups

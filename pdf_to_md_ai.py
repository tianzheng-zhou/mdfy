"""
PDF to Markdown —— AI 增强转换（Qwen 多模态）

核心思路：
  PDF 每页渲染为图片 → 连同提取的文本层一起发给 Qwen → 模型直接输出 Markdown
  代码只负责喂数据、收结果、提取/保存图片，通用后处理仅做格式规范化。

使用方式：
  设置环境变量 DASHSCOPE_API_KEY，然后运行：
    python pdf_to_md_ai.py <pdf_path> [--model qwen3.5-flash|qwen3.5-plus]
  或直接修改底部的 pdf_file 路径运行。
"""

import argparse
import base64
import fitz  # PyMuPDF
import io
import os
import re
import sys
import time
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


# ── Qwen API 配置 ──────────────────────────────────────────────────

def get_client():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows:  set DASHSCOPE_API_KEY=sk-xxxxx\n"
            "  Linux:    export DASHSCOPE_API_KEY=sk-xxxxx"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


AVAILABLE_MODELS = ["qwen3.5-flash", "qwen3.5-plus"]
DEFAULT_MODEL = "qwen3.5-plus"
MODEL_IMAGE_MAX_BYTES = 7_500_000
DETECTION_IMAGE_MAX_SIDE = 1600
OCR_IMAGE_MAX_SIDE = 2200
MIN_MODEL_IMAGE_SIDE = 640


# ── PDF 类型检测 ───────────────────────────────────────────────────

def _is_scan_tile_page(page, doc):
    """检测页面是否为扫描页（整页图或条带拼接）"""
    imgs = page.get_images(full=True)
    if not imgs:
        return False

    page_w_px = page.rect.width * 200 / 72  # 估算 200dpi 像素宽度

    # 情况1：单张整页图片
    if len(imgs) == 1:
        try:
            pix = fitz.Pixmap(doc, imgs[0][0])
            return pix.width > page_w_px * 0.8
        except Exception:
            return False

    # 情况2：多张等宽等高条带（≥3张）
    if len(imgs) >= 3:
        try:
            sizes = []
            for img in imgs:
                pix = fitz.Pixmap(doc, img[0])
                sizes.append((pix.width, pix.height))
            widths = set(s[0] for s in sizes)
            heights = set(s[1] for s in sizes)
            return len(widths) <= 1 and len(heights) <= 1
        except Exception:
            pass

    return False


def _detect_pdf_type(doc):
    """检测 PDF 类型：scanned（扫描件）或 digital（数字PDF）

    扫描件特征：文本层为空 + 图片为页面扫描条带或整页图片
    """
    sample_count = min(5, len(doc))
    text_chars = 0
    scan_page_count = 0

    for i in range(sample_count):
        page = doc[i]
        text_chars += len(page.get_text().strip())
        if _is_scan_tile_page(page, doc):
            scan_page_count += 1

    if text_chars < 50 * sample_count and scan_page_count >= sample_count * 0.6:
        return "scanned"
    return "digital"


# ── PDF 工具函数 ────────────────────────────────────────────────────

def render_page_to_image(page, dpi=200):
    """将 PDF 页面渲染为 PNG 字节"""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _serialize_pil_image(image, fmt, **save_kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def prepare_image_for_model(image_bytes, *, max_side, max_bytes=MODEL_IMAGE_MAX_BYTES,
                            min_side=MIN_MODEL_IMAGE_SIDE):
    """将图片压缩到适合发送给视觉模型的尺寸与体积。

    返回 (encoded_bytes, mime_type, (width, height))。
    优先保留 PNG；若体积仍过大，则降尺度并回退到 JPEG。
    """
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.LANCZOS

    current = image
    current_max_side = min(max(image.size), max_side)

    while True:
        if max(current.size) > current_max_side:
            scale = current_max_side / max(current.size)
            resized_size = (
                max(1, int(round(current.width * scale))),
                max(1, int(round(current.height * scale))),
            )
            candidate = current.resize(resized_size, resample)
        else:
            candidate = current

        png_bytes = _serialize_pil_image(candidate, "PNG", optimize=True)
        if len(png_bytes) <= max_bytes:
            return png_bytes, "image/png", candidate.size

        jpeg_source = candidate.convert("RGB") if candidate.mode != "RGB" else candidate
        smallest_jpeg = None
        for quality in (90, 85, 80, 75, 70, 60, 50, 40):
            jpeg_bytes = _serialize_pil_image(
                jpeg_source,
                "JPEG",
                quality=quality,
                optimize=True,
            )
            if smallest_jpeg is None or len(jpeg_bytes) < len(smallest_jpeg):
                smallest_jpeg = jpeg_bytes
            if len(jpeg_bytes) <= max_bytes:
                return jpeg_bytes, "image/jpeg", candidate.size

        if max(candidate.size) <= min_side:
            return smallest_jpeg, "image/jpeg", candidate.size

        next_max_side = max(min_side, int(max(candidate.size) * 0.85))
        if next_max_side >= max(candidate.size):
            return smallest_jpeg, "image/jpeg", candidate.size

        current = candidate
        current_max_side = next_max_side


def extract_page_text(page):
    """提取页面的纯文本"""
    return page.get_text().strip()


def bbox_to_pixels(bbox, image_width, image_height):
    """将 bbox 转为像素坐标。

    优先按官方定位能力常见的归一化坐标 [0, 1000] 解释；
    若超出范围，则回退为绝对像素坐标，兼容旧测试输出。
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]

    if all(0 <= v <= 1000 for v in (x1, y1, x2, y2)):
        x1 = x1 / 1000 * image_width
        x2 = x2 / 1000 * image_width
        y1 = y1 / 1000 * image_height
        y2 = y2 / 1000 * image_height

    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))

    x1 = max(0, min(image_width, x1))
    x2 = max(0, min(image_width, x2))
    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))
    return x1, y1, x2, y2


def _normalize_bbox_to_1000(bbox, image_size, *, from_pixels=False):
    """统一将 bbox 转为 [0, 1000] 归一化坐标。"""
    x1, y1, x2, y2 = [float(v) for v in bbox]

    if image_size and (from_pixels or any(v > 1000 for v in (x1, y1, x2, y2))):
        width, height = image_size
        x1 = x1 / width * 1000
        x2 = x2 / width * 1000
        y1 = y1 / height * 1000
        y2 = y2 / height * 1000

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        max(0.0, min(1000.0, x1)),
        max(0.0, min(1000.0, y1)),
        max(0.0, min(1000.0, x2)),
        max(0.0, min(1000.0, y2)),
    ]


def parse_figure_detection_response(raw_text, image_size):
    """解析模型返回的 JSON 检测结果，兼容 bbox / bbox_2d。"""
    import json

    raw = raw_text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(data, list):
        return []

    figures = []
    for item in data:
        if not isinstance(item, dict):
            continue

        bbox_key = None
        if "bbox" in item:
            bbox_key = "bbox"
        elif "bbox_2d" in item:
            bbox_key = "bbox_2d"

        if bbox_key is None:
            continue

        bbox = item.get(bbox_key)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not all(isinstance(v, (int, float)) for v in bbox):
            continue

        figures.append({
            "bbox": _normalize_bbox_to_1000(
                bbox,
                image_size,
                from_pixels=(bbox_key == "bbox_2d"),
            ),
            "desc": item.get("desc") or item.get("label") or "",
        })

    return figures


def parse_qwenvl_markdown_figures(raw_text, image_size):
    """解析 qwenvl markdown 中的图片坐标注释。"""
    raw = raw_text.strip()
    raw = re.sub(r'^```(?:markdown)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    figures = []
    seen = set()
    for match in re.finditer(r'<!--\s*Image\s*\(([^)]+)\)\s*-->', raw, flags=re.IGNORECASE):
        parts = [part.strip() for part in match.group(1).split(',')]
        if len(parts) != 4:
            continue
        try:
            pixel_bbox = [float(part) for part in parts]
        except ValueError:
            continue

        norm_bbox = _normalize_bbox_to_1000(pixel_bbox, image_size, from_pixels=True)
        bbox_key = tuple(int(round(v)) for v in norm_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        figures.append({"bbox": norm_bbox, "desc": ""})

    return figures


def _request_qwenvl_markdown(client, model, image_bytes, image_mime):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{image_mime};base64,{base64.b64encode(image_bytes).decode()}"
                }},
                {"type": "text", "text": "qwenvl markdown"},
            ]},
        ],
        temperature=0.1,
        max_tokens=4096,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content.strip()


def _detect_image_rotation(transform):
    """从 PDF 图片变换矩阵检测旋转角度，返回 0/±90/180。"""
    import math
    a, b = transform[0], transform[1]
    angle = math.degrees(math.atan2(b, a))
    rounded = round(angle / 90) * 90
    return rounded if rounded % 360 != 0 else 0


def extract_and_save_images(doc, page, page_num, images_dir):
    """提取并保存页面中的原始图片，返回图片文件名列表"""
    saved = []
    image_list = page.get_images(full=True)
    # 获取图片变换矩阵，用于检测旋转
    image_info = page.get_image_info(xrefs=True)
    xref_transforms = {}
    for info in image_info:
        xr = info.get('xref', 0)
        tf = info.get('transform')
        if xr and tf:
            xref_transforms[xr] = tf
    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace and pix.colorspace.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            filename = f"page{page_num + 1}_img{img_index + 1}.png"
            filepath = os.path.join(images_dir, filename)
            pix.save(filepath)
            pix = None
            # 检测并修正 PDF 变换矩阵中的旋转
            tf = xref_transforms.get(xref)
            if tf:
                rotation = _detect_image_rotation(tf)
                if rotation != 0:
                    from PIL import Image
                    with Image.open(filepath) as im:
                        # PDF 坐标系 y 轴向上，PIL y 轴向下，旋转方向取反
                        corrected = im.rotate(-rotation, expand=True)
                        corrected.save(filepath)
            saved.append(filename)
        except Exception as e:
            print(f"  ⚠ 第{page_num+1}页图片{img_index+1}提取失败: {e}")
    return saved


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
    image_list = page.get_images(full=True)
    embedded_pixel_bboxes = []  # 与 embedded_filenames 一一对应，None 表示无法获取
    for idx, filename in enumerate(embedded_filenames):
        if idx >= len(image_list):
            embedded_pixel_bboxes.append(None)
            continue
        xref = image_list[idx][0]
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
    kept_ai_indices = []
    for ai_idx, (ax1, ay1, ax2, ay2) in enumerate(ai_pixel_bboxes):
        ai_area = max((ax2 - ax1) * (ay2 - ay1), 1)
        is_covered_by_embed = False
        for eb in embedded_pixel_bboxes:
            if eb is None:
                continue
            ex1, ey1, ex2, ey2 = eb
            ix1, iy1 = max(ax1, ex1), max(ay1, ey1)
            ix2, iy2 = min(ax2, ex2), min(ay2, ey2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / ai_area > 0.50:
                    is_covered_by_embed = True
                    break
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

    return kept_ai + kept_embedded


def _compute_image_positions(image_filenames, page_png_bytes, fig_bboxes=None, page=None):
    """按页面纵向位置排序图片并计算位置百分比。

    fig_bboxes: {filename: (x1,y1,x2,y2)} AI检测/裁切的像素坐标
    page: PyMuPDF 页面对象（用于获取嵌入图位置）
    返回 (sorted_filenames, positions_dict)，positions_dict: {filename: "~XX%"}
    """
    if not image_filenames:
        return [], {}

    from PIL import Image
    pil_img = Image.open(io.BytesIO(page_png_bytes))
    page_height = pil_img.height

    items = []  # [(filename, y_center)]

    for fname in image_filenames:
        y_center = None

        # 1. AI/crop bbox（像素坐标）
        if fig_bboxes and fname in fig_bboxes:
            bbox = fig_bboxes[fname]
            y_center = (bbox[1] + bbox[3]) / 2

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
                            scale_y = page_height / page.rect.height
                            rect = rects[0]
                            ey1 = rect.y0 * scale_y
                            ey2 = rect.y1 * scale_y
                            y_center = (ey1 + ey2) / 2
                except Exception:
                    pass

        if y_center is None:
            y_center = page_height / 2  # fallback

        items.append((fname, y_center))

    items.sort(key=lambda x: x[1])

    sorted_filenames = [it[0] for it in items]
    positions = {}
    for fname, y_center in items:
        pct = round(y_center / page_height * 100) if page_height > 0 else 50
        positions[fname] = f"~{pct}%"

    return sorted_filenames, positions


def _merge_figure_lists(*fig_lists):
    """合并多个 figure 列表，去除 bbox 重叠度 > 50% 的重复项。"""
    merged = []
    for figs in fig_lists:
        for fig in figs:
            bbox = fig["bbox"]
            is_dup = False
            for existing in merged:
                eb = existing["bbox"]
                # 计算交集面积 / 较小框面积
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
    坐标优先使用归一化区间 [0, 1000]，便于映射回不同分辨率。

    策略：JSON bbox + qwenvl markdown 双路检测，合并去重结果，提高召回率。
    JSON bbox 检测失败时会重试一次。
    """
    detect_bytes, detect_mime, detect_size = prepare_image_for_model(
        page_png_bytes,
        max_side=DETECTION_IMAGE_MAX_SIDE,
    )

    # ── 路径1：JSON bbox（prompt 可控，精度更高）──
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
    for attempt in range(2):  # 最多重试一次
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
                temperature=0.1 + attempt * 0.15,  # 重试时稍微提高温度
                max_tokens=1024,
                extra_body={"enable_thinking": False},
            )
            raw = response.choices[0].message.content.strip()
            json_figures = parse_figure_detection_response(raw, detect_size)
            if json_figures:
                break
        except Exception:
            pass

    # ── 路径2：qwenvl markdown 兜底 / 补充 ──
    qwenvl_figures = []
    try:
        raw_qwenvl = _request_qwenvl_markdown(client, model, detect_bytes, detect_mime)
        qwenvl_figures = parse_qwenvl_markdown_figures(raw_qwenvl, detect_size)
    except Exception:
        pass

    # 合并两路结果，去重
    all_figures = _merge_figure_lists(json_figures, qwenvl_figures)
    return all_figures


def _refine_bbox_with_pixels(img, x1, y1, x2, y2):
    """用像素分析扩展/收缩 bbox，使其精确匹配图片区域的实际边界。

    策略：
    1. 从 bbox 中心区域出发，确认存在非文字内容（图片像素）
    2. 向上下扩展 bbox，直到碰到连续的文字/空白行
    3. 向左右扩展 bbox，直到碰到连续的空白列
    4. 收缩 bbox 边缘的纯文字/空白区域

    返回修正后的 (x1, y1, x2, y2)。
    """
    import numpy as np

    arr = np.array(img.convert("L"))
    h, w = arr.shape

    # ── Y 方向精修（基于行白色像素比）──
    col_start = max(0, x1)
    col_end = min(w, x2)
    if col_end - col_start < 30:
        return x1, y1, x2, y2

    strip = arr[:, col_start:col_end]
    white_ratio = (strip > 230).mean(axis=1)
    is_text = white_ratio > 0.55

    # 找 bbox 中心区域的"图片行"作为种子
    cy = (y1 + y2) // 2
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
            # Y 方向找不到足够的"图片行"种子，保留原始 Y 不精修，
            # 但仍然跳到 X 方向精修（不要直接 return）
            new_y1, new_y2 = y1, y2
            # 跳过 Y 方向扩展，直接进入 X 精修
            _skip_y_expansion = True
        else:
            cy = (best_s + best_e) // 2
            _skip_y_expansion = False
    else:
        _skip_y_expansion = False

    if not _skip_y_expansion:
        # 从种子行向上扩展
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

        # 从种子行向下扩展
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

    # ── X 方向精修（基于列白色像素比）──
    row_start = max(0, new_y1)
    row_end = min(h, new_y2)
    if row_end - row_start < 30:
        return x1, new_y1, x2, new_y2

    h_strip = arr[row_start:row_end, :]
    col_white_ratio = (h_strip > 230).mean(axis=0)

    # 判断原始 bbox 是否在 X 方向明显过窄（可能是模型只检测到了图片局部）
    bbox_w = x2 - x1
    bbox_h = new_y2 - new_y1
    narrow_bbox = bbox_w < w * 0.35 and bbox_h > bbox_w * 1.5

    if narrow_bbox:
        # 对于明显过窄的 bbox，采用更激进的扩展策略：
        # 从 bbox 区域的 Y 范围内, 找到所有"有内容"的列（白色比 < 0.98）
        # 使用更宽松的空白判定和更大的连续空白阈值
        content_col = col_white_ratio < 0.98
        # 找最左和最右的有内容列
        content_cols = np.where(content_col)[0]
        if len(content_cols) > 5:
            new_x1 = int(content_cols[0])
            new_x2 = int(content_cols[-1]) + 1
        else:
            new_x1, new_x2 = x1, x2
    else:
        is_blank_col = col_white_ratio > 0.95  # 空白列阈值（几乎纯白）

        cx = (x1 + x2) // 2
        cx = max(0, min(w - 1, cx))

        # 从中心向左扩展，找到图片左边界
        # 使用较大的连续空白阈值(20列)避免误判波形图等细密内容的间隙
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

        # 从中心向右扩展
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

    # X 方向取精修结果和原始 bbox 的并集（宁大勿小，避免截掉图片内容）
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
        # 添加少量 padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        fig_w = x2 - x1
        fig_h = y2 - y1
        # 跳过太小的区域（可能是误检）
        if fig_w < 30 or fig_h < 30:
            continue
        fig_area = fig_w * fig_h
        if fig_area / page_area < 0.005:
            continue
        # 跳过占页面面积过大的区域（>70% 通常是整页误检）
        if fig_area / page_area > 0.70:
            continue
        # 用像素分析精确扩展/收缩 bbox
        x1, y1, x2, y2 = _refine_bbox_with_pixels(img, x1, y1, x2, y2)
        fig_w = x2 - x1
        fig_h = y2 - y1
        # 再次检查扩展后面积
        fig_area = fig_w * fig_h
        if fig_area / page_area > 0.70:
            continue
        # 跳过极窄长条（宽高比异常，通常是边缘装饰误检）
        aspect = fig_w / max(fig_h, 1)
        if aspect < 0.15 or aspect > 6.5:
            continue
        # 扩展后仍然太小的跳过
        if fig_w < 80 or fig_h < 80:
            continue
        cropped = img.crop((x1, y1, x2, y2))
        # 过滤近乎全白的裁切（白色像素 >95% 说明是误裁空白区域）
        arr = np.array(cropped.convert("L"))
        if (arr > 240).mean() > 0.95:
            continue
        # IoU 去重：与已保存的图片比较，如果重叠 >50% 则跳过（避免近似重复）
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


# ── AI 裁切验证与优化 ──────────────────────────────────────────────

_MAX_CROP_VERIFY_ROUNDS = 2  # 防止死循环：最多验证+修正 2 轮


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
    """AI 验证裁切质量，返回每张裁切的动作列表。

    crop_results: [(filename, desc, (x1,y1,x2,y2))]
    other_images: 同页已有的其他图片文件名（如嵌入图），用于去重判断
    返回 [{"index": N, "action": "accept"|"reject"|"adjust"|"split", ...}]
    """
    import json

    if not crop_results:
        return []

    # 带编号框的页面图（用高分辨率，让 AI 看清楚内容边界）
    bboxes_px = [cr[2] for cr in crop_results]
    overlay_bytes = _draw_bbox_overlay(page_png_bytes, bboxes_px)
    overlay_compressed, overlay_mime, _ = prepare_image_for_model(
        overlay_bytes, max_side=OCR_IMAGE_MAX_SIDE,
    )

    # 裁切描述
    crops_desc = []
    for idx, (fname, desc, (x1, y1, x2, y2)) in enumerate(crop_results):
        crops_desc.append(
            f"  图{idx + 1}: {fname} | {x2 - x1}×{y2 - y1}px | 描述: {desc or '无'}"
        )

    # 构建多图内容：页面标注图 + 各裁切图
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

    # 已有嵌入图信息（用于去重判断）
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

    # 解析失败时全部 accept
    return [{"index": i + 1, "action": "accept"} for i in range(len(crop_results))]


def _execute_crop_actions(page_png_bytes, actions, crop_results, page_num, images_dir):
    """根据 AI 验证结果执行裁切动作，返回更新后的 crop_results。"""
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(page_png_bytes))

    # index → action 映射
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
                # AI 给的坐标只做 padding，不再执行像素精修（避免精修反而破坏 AI 的判断）
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
                # 无有效子区域，从原 bbox 恢复
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
                                max_rounds=_MAX_CROP_VERIFY_ROUNDS,
                                other_images=None):
    """AI 验证 + 优化裁切结果的主循环。

    max_rounds: 最多验证轮数，防止死循环。
    other_images: 同页已有的其他图片文件名（如嵌入图），传给 AI 做去重。
    返回最终的 [(filename, desc, pixel_bbox)] 列表。
    """
    for round_num in range(max_rounds):
        if not crop_results:
            break

        actions = _ai_verify_crops(
            client, model, page_png_bytes, crop_results,
            page_num, images_dir, round_num=round_num,
            other_images=other_images,
        )

        # 打印每张图的验证结果
        for a in actions:
            act = a.get("action", "accept")
            if act != "accept":
                reason = a.get("reason", "")
                tag = f"{act}" + (f"({reason})" if reason else "")
                print(f"[fig{a.get('index', '?')}:{tag}]", end=" ", flush=True)

        # 全部 accept → 无需继续
        if all(a.get("action", "accept") == "accept" for a in actions):
            break

        crop_results = _execute_crop_actions(
            page_png_bytes, actions, crop_results, page_num, images_dir,
        )

        # 仅 reject（无 adjust/split）→ 无新裁切需验证
        if not any(a.get("action") in ("adjust", "split") for a in actions):
            break

    return crop_results


# ── 核心：调用 Qwen 多模态 ─────────────────────────────────────────

SYSTEM_PROMPT_SCANNED = """\
<role>你是扫描件 OCR 转 Markdown 助手。将扫描的书页精确识别并转写为结构化 Markdown。</role>

<critical_rules>
<rule id="ocr_accuracy">
精确识别页面上的所有文字，不要遗漏、不要添加。
如果有个别文字无法辨认，用 [?] 标注。
</rule>

<rule id="no_page_artifacts">
忽略页眉、页脚、页码等印刷辅助信息。
忽略水印、装订线等扫描伪影。
</rule>

<rule id="figures">
如果页面上有插图、图表等非文字内容：
- 如果我提供了 <images> 列表中的图片文件名，严格按照每个 <img> 标签的 pos 属性（纵向位置百分比）将图片插入到对应位置的文本中间，用 ![描述](images/文件名) 引用
- 如果没有提供图片文件名，用 `<!-- 图：[简要描述] -->` 标注其位置和内容
- 表格：尽量转写为 Markdown 表格。如果一张图片的内容仅仅是表格，且你已完整转写为 Markdown 表格，则省略该图片引用
</rule>

<rule id="heading_levels">
# 仅用于整本书的书名（整篇文档只出现一次）。
## 用于章标题（如"第一章"、"Chapter 1"等主要章节）。
### 用于节标题（章内的小节）。
正文段落不加任何标题标记。
</rule>

<rule id="cross_page_continuity">
书籍文字经常在页面底部断开——一个句子、甚至一个词可能被分页截断。
- 查看 previous_page_tail：如果它的末尾是一个不完整的句子（没有以句号、问号、感叹号、冒号等结束），说明**上一页的末尾句子被分页截断了**。
- 此时你必须在输出的**最开头**直接接续这个被截断的句子的剩余部分（从本页顶部看到的续文开始），不要另起段落，不要加空行。
- **严禁重复**：绝对不要重复 previous_page_tail 中已经出现过的任何文字。tail 的内容已经被记录了，你只需要输出本页**新**的内容。
- 如果 tail 末尾刚好在一个汉字/单词中间断开，直接输出本页续接的文字即可。
- 如果本页底部的句子也没有说完，就正常输出到页面可见内容的末尾即可，下一页会接续。
- **标题不受跨页续接影响**：即使本页顶部有续接文字，续接完成后，如果页面上有章节标题，仍然必须使用正确的 ## 或 ### 标记。标题标记永远不能省略。
</rule>

<rule id="cross_page_table">
表格经常跨页。如果 previous_page_tail 末尾包含 Markdown 表格行（以 | 开头和结尾的行），说明上一页有一个表格可能延续到本页。
- 如果本页顶部确实是该表格的延续数据：直接继续输出表格数据行（保持相同的列数和 | 分隔格式）
- **不要重复表头行和分隔行**（| --- | 行），只输出新的数据行
- 如果上一页表格单元格内的文字被分页截断（previous_page_tail 最后一行是不完整的表格行），本页开头的续接文字应直接作为该单元格内容的延续，保持在同一个表格行内
- 表格续写完成后，页面上的其他内容（注释、正文、标题等）正常输出
</rule>
</critical_rules>

<formatting>
- 合并因扫描分页和 PDF 换行造成的断句，还原为通顺的段落。
- 脚注用 [^n] 格式标注。
- 引用段落用 > 标记。
- 保留原文的加粗、斜体等强调格式。
- 不要添加任何原文没有的内容或解释。
- 直接输出 Markdown 正文，不要用 ```markdown 包裹。
</formatting>\
"""

SYSTEM_PROMPT = """\
<role>你是 PDF 转 Markdown 助手。将 PDF 页面截图精确转写为 Markdown。</role>

<critical_rules>
<rule id="no_fabrication">
只输出页面上用文字排版写出的正文内容。
截图/界面图中的 UI 文字（按钮名、标签页名、菜单项等）不属于正文，不要转写为标题或段落。
如果一页上正文文字极少或完全没有，只放图片引用即可。
</rule>

<rule id="no_page_artifacts">
忽略页眉、页脚、页码等印刷辅助信息，不要将它们输出到 Markdown 中。
常见页眉：文档标题、版本号、网址、章节名等在每页顶部/底部重复出现的内容。
</rule>

<rule id="images">
我会告诉你本页有哪些图片文件名。默认每张图片都必须用 ![](images/文件名) 引用。
图片列表已按页面从上到下排序，每个 <img> 标签有 pos 属性表示纵向位置百分比——请严格按照 pos 位置将图片插入到对应位置的文本中间。
可以省略图片引用的情况：
- 如果一张图片的内容**仅仅是代码/脚本文字**，且你已经完整转写为代码块
- 如果一张图片的内容**仅仅是表格**（无论本页还是上一页已转写），且该表格已被完整转写为 Markdown 表格
以上两种情况下，省略图片引用，不要重复展示已转写的内容。
界面截图、操作步骤截图、示意图、对话框截图、任何包含 GUI 元素的图片，一律保留引用，即使你描述了其内容。
拿不准的时候，保留图片引用。
</rule>

<rule id="code_blocks">
代码、脚本、配置文件、数据数组等技术内容必须用 ``` 围栏包裹。
如果能识别语言，加上语言标识（如 ```python、```c 等）。
如果当前页的数据是上一页的延续，也必须用新的 ``` 围栏包裹。
</rule>

<rule id="cross_page_table">
表格经常跨页。如果 previous_page_tail 末尾包含 Markdown 表格行（以 | 开头和结尾的行），说明上一页有一个表格可能延续到本页。
- 如果本页顶部确实是该表格的延续数据：直接继续输出表格数据行（保持相同的列数和 | 分隔格式）
- **不要重复表头行和分隔行**（| --- | 行），只输出新的数据行
- 如果上一页表格单元格内的文字被分页截断，本页开头的续接文字应直接作为该单元格内容的延续，保持在同一个表格行内
- 表格续写完成后，页面上的其他内容（注释、正文、标题等）正常输出
</rule>

<rule id="heading_levels">
# 仅用于整篇文档的大标题（整篇文档只能出现一次 #，绝对不能有第二个 #）。

对于多级编号章节的文档（如技术手册 1 → 1.1 → 1.1.1）：
- ## 用于一级章节标题（单个编号：1、2、3... 或 "附录 N"）
- ### 用于二级及更深小节标题（如 1.1、2.3、2.3.1、2.3.1.1 等带点号的多级编号）

对于编号步骤的教程文档（如操作教程 1. 2. 3. 代表步骤）：
- ## 用于主步骤（"数字." 格式，如 1. 2. 13.）和无编号的章节标题
- ### 用于子步骤（"数字)" / "数字）" / "a)" / "b)" 格式）

通用规则：
- 目录/TOC 页的条目用列表格式（- 条目），不要用标题标记
- 表格标题（如 "表 1"、"Table 2"、"图 X"）不要用标题标记，用加粗
- 没有编号的说明性文字不要加任何标题标记，直接作为正文段落
</rule>
</critical_rules>

<formatting>
- 修正 PDF 换行造成的断句，合并成通顺的句子。
- 不要添加任何原文没有的内容或解释性文字。
- 直接输出 Markdown 正文，不要用 ```markdown 包裹整个输出。
</formatting>\
"""


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


def convert_page_with_ai(client, model, page_png_bytes, page_text, image_filenames,
                          page_num, total_pages, prev_md_tail="", outline="",
                          pdf_type="digital", page_image_mime="image/png",
                          image_positions=None):
    """调用 Qwen 多模态模型转换单页"""

    is_scanned = (pdf_type == "scanned")
    system_prompt = SYSTEM_PROMPT_SCANNED if is_scanned else SYSTEM_PROMPT

    # 图片清单（带位置提示，按从上到下排序）
    if image_filenames:
        if image_positions:
            img_list = '\n'.join(
                f'    <img pos="{image_positions.get(f, "")}">{f}</img>'
                for f in image_filenames
            )
        else:
            img_list = '\n'.join(f'    <img>{f}</img>' for f in image_filenames)
        img_section = (
            f"<images count=\"{len(image_filenames)}\" note=\"已按页面从上到下排序\">\n"
            f"{img_list}\n"
            f"  </images>"
        )
    elif not is_scanned:
        img_section = '<images count="0"/>'
    else:
        img_section = ""

    # 文本层
    if page_text:
        text_section = f"<extracted_text>\n{page_text}\n  </extracted_text>"
    elif is_scanned:
        text_section = ""  # 扫描件无文本层，不发送空标签浪费 token
    else:
        text_section = "<extracted_text>（无文本层）</extracted_text>"

    # 滚动上下文：大纲 + 上一页末尾
    if outline:
        outline_section = f"<document_outline>\n{outline}\n  </document_outline>"
    else:
        outline_section = ""

    def _extract_leading_table_cells(raw_text, expected_cols):
        if not raw_text or expected_cols < 2:
            return None
        top_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not top_lines:
            return None

        start = 0
        while start < len(top_lines) and (
            '用户手册' in top_lines[start]
            or re.search(r'^www\.', top_lines[start], re.IGNORECASE)
        ):
            start += 1
        if start < len(top_lines) and re.fullmatch(r'\d+', top_lines[start]):
            start += 1

        candidates = top_lines[start:start + 12]
        if expected_cols == 3:
            for idx in range(0, len(candidates) - 2):
                cells = candidates[idx:idx + 3]
                if not re.fullmatch(r'\d+[A-Za-z]?', cells[0]):
                    continue
                if any(len(cell) > 60 for cell in cells):
                    continue
                next_line = candidates[idx + 3] if idx + 3 < len(candidates) else ''
                if next_line and not (
                    next_line.startswith(('注：', '注:')) or len(next_line) > 40
                ):
                    continue
                return [re.sub(r'\s+', ' ', cell).strip() for cell in cells]
        return None

    table_continuation_section = ""

    if prev_md_tail:
        # 检测 tail 末尾是否在句子中间被截断
        stripped_tail = prev_md_tail.rstrip()
        # 完整句子结束标志：中文句号、问号、感叹号、英文句点+空格/换行、冒号、引号闭合等
        is_mid_sentence = bool(stripped_tail) and not re.search(
            r'[。？！…」』\u201d]$|[.?!:;]$|\*\*$|```$',
            stripped_tail,
        )
        # 检测 tail 末尾是否在表格中（最后非空行以 | 开头且包含多个 |）
        ends_with_table = False
        prev_table_col_count = 0
        for _tl in reversed(stripped_tail.split('\n')):
            _tl_s = _tl.strip()
            if not _tl_s:
                continue
            if _tl_s.startswith('|') and '|' in _tl_s[1:]:
                ends_with_table = True
                prev_table_col_count = _tl_s.count('|') - 1
            break

        truncation_hints = []
        if ends_with_table:
            truncation_hints.append(
                "上一页末尾是一个 Markdown 表格。如果本页顶部是该表格的延续内容，"
                "请直接继续输出表格数据行（保持相同的 | 分隔列格式），不要重复表头和分隔行"
            )
            leading_cells = _extract_leading_table_cells(page_text, prev_table_col_count)
            if leading_cells:
                candidate_row = f"| {' | '.join(leading_cells)} |"
                truncation_hints.append(
                    f"当前页顶部检测到一行续表候选：{candidate_row}。"
                    "如果页面内容一致，必须完整输出这一整行表格，不能只保留后面的注释或正文"
                )
                table_continuation_section = (
                    "<detected_table_continuation>\n"
                    f"  <expected_row>{candidate_row}</expected_row>\n"
                    "</detected_table_continuation>"
                )
        if is_mid_sentence:
            truncation_hints.append("上一页末尾的句子被分页截断了，本页输出需要以续接文字开头")

        if truncation_hints:
            hints_xml = '\n'.join(f"  <!-- 注意：{h} -->" for h in truncation_hints)
            tail_section = (
                f"<previous_page_tail truncated=\"true\">\n{prev_md_tail}\n"
                f"{hints_xml}\n"
                f"  </previous_page_tail>"
            )
        else:
            tail_section = f"<previous_page_tail>\n{prev_md_tail}\n  </previous_page_tail>"
    else:
        tail_section = "<previous_page_tail>这是文档第一页</previous_page_tail>"

    # 根据 PDF 类型选择不同的指令
    if is_scanned:
        img_instructions = ""
        if image_filenames:
            img_instructions = f"    - 本页有 {len(image_filenames)} 张已裁切的图片（已按页面从上到下排序，pos 属性为纵向位置百分比）；请严格按照 pos 位置将每张图片插入到对应位置的文本中间，用 ![描述](images/文件名) 引用\n"
        instructions = (
            f"  <instructions>\n"
            f"    - 精确 OCR 页面上的所有正文文字\n"
            f"    - 延续大纲中的标题层级，不要重复已有的标题\n"
            f"    - 【关键·跨页连续】查看 previous_page_tail 的末尾：\n"
            f"      * 如果 tail 末尾不是完整句子（没有以。？！等结束），说明上一页句子被分页截断\n"
            f"      * 此时你的输出必须以本页续接的文字开头，直接接上被截断的句子，不要另起段落\n"
            f"      * ⚠️ 严禁重复：tail 中的文字已经被记录，你只输出本页新的内容。不要把 tail 最后一句再输出一遍。\n"
            f"      * 如果 tail 末尾是完整句子，正常从本页新内容开始即可\n"
            f"    - 续接完成后，如果页面上有章/节标题，仍必须使用 ## 或 ### 标记\n"
            f"    - 如果本页底部的句子没有说完（被下一页截断），照常输出到可见内容末尾\n"
            f"    - 忽略页眉、页脚、页码\n"
            f"{img_instructions}"
            f"    - 直接输出 Markdown，不要解释\n"
            f"  </instructions>"
        )
    else:
        instructions = (
            f"  <instructions>\n"
            f"    - 延续大纲中的标题层级和编号，不要重复已有的标题\n"
            f"    - 本页有 {len(image_filenames)} 张图片（已按页面从上到下排序，pos 属性为纵向位置百分比）；请严格按照 pos 位置将每张图片插入到对应位置的文本中间；仅代码截图已完整转写的可省略，其余必须引用\n"
            f"    - 代码/脚本/数据数组必须用 ``` 围栏包裹\n"
            f"    - 忽略页眉、页脚、页码，不要输出到 Markdown\n"
            f"    - 目录页的条目用列表（- 条目），不要用标题标记\n"
            f"    - 直接输出 Markdown，不要解释\n"
            f"  </instructions>"
        )

    # 组装 user_text，跳过空 section
    sections = [f"  <page number=\"{page_num + 1}\" total=\"{total_pages}\"/>"]
    if outline_section:
        sections.append(f"  {outline_section}")
    sections.append(f"  {tail_section}")
    if table_continuation_section:
        sections.append(f"  {table_continuation_section}")
    if img_section:
        sections.append(f"  {img_section}")
    if text_section:
        sections.append(f"  {text_section}")
    sections.append(instructions)

    user_text = "<task>\n" + "\n".join(sections) + "\n</task>"

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{page_image_mime};base64,{base64.b64encode(page_png_bytes).decode()}"
            },
        },
        {
            "type": "text",
            "text": user_text,
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=8192,
        extra_body={"enable_thinking": False},
    )

    return response.choices[0].message.content.strip()


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
        if prev_tail_lines == curr_head_lines and all(l for l in prev_tail_lines):
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


# ── 已转写表格的图片引用清理 ────────────────────────────────────────

_IMG_REF_RE = re.compile(r'^!\[.*?\]\(images/.+?\)\s*$')
_TABLE_ROW_RE = re.compile(r'^\s*\|.+\|\s*$')


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


# ── 编号标题层级规范化 ──────────────────────────────────────────────

_DOTTED_HEADING_RE = re.compile(
    r'^(#{1,6})\s+(\d+(?:\.\d+)+)\s+(.*)',
)


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


# ── 同级编号标题归一化 ──────────────────────────────────────────────

# 匹配可能的编号子标题行：  可选的 # 前缀 + 括号编号 + 标题文字
_NUMBERED_HEADING_RE = re.compile(
    r'^(#{1,6}\s+)?([（(]\d+[）)]\s*.+)$'
)
_HEADING_NUM_RE = re.compile(r'[（(](\d+)[）)]')


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


# ── LLM 辅助拼接 ────────────────────────────────────────────────────

_STITCH_SYSTEM = """\
你是一个文本拼接助手。你的唯一任务是修正"下一页"的开头。

输入：
<page_end> 上一页的末尾文字（已经确认正确，你不需要改动它）
<page_start> 下一页的开头文字（可能有问题需要修正）

你需要输出**修正后的 page_start 文字**，规则如下：
1. **去重**：如果 page_start 的开头重复了 page_end 末尾已有的句子或段落，删除重复的部分。
2. **续接断句**：如果 page_end 的最后一句话没有说完（被分页截断），page_start 的开头应该直接续接那个断句。确保 page_start 的输出开头就是续接文字，不要加空行。
3. **严格保留原文**：
   - 保留 page_start 中所有 Markdown 格式（## ### 标题、![alt](url) 图片引用、- 列表、> 引用）不变
   - 保留所有 HTML 注释 <!-- ... --> 完整不变
   - **绝对不要改写、润色、总结或添加原文没有的文字**
   - 你只能做：删除 page_start 开头重复的内容
4. **只输出修正后的 page_start**，不要输出 page_end 的内容。
5. **表格续接**：如果 page_end 末尾是 Markdown 表格行（包含 | 分隔列），而 page_start 也包含该表格的续行：
   - 表格续行是**新内容**（不是重复），必须完整保留
   - 确保续行保持相同的 | 分隔格式
   - 如果 page_end 最后一行是未完成的表格行（单元格内文字被分页截断），将 page_start 开头的续接文字合并到该表格行中

直接输出，不要解释。"""


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

    # 情况2：prev 以标题/代码块/分隔线结尾 + curr 结构化开头
    prev_structural_end = bool(re.match(
        r'^#{1,6} |^```|^---', last_line
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
                {"role": "system", "content": _STITCH_SYSTEM},
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


# ── 主流程 ──────────────────────────────────────────────────────────

def pdf_to_markdown_ai(pdf_path, output_dir=None, model=None):
    """AI 增强 PDF 转 Markdown 主函数"""
    pdf_path = Path(pdf_path)
    model = model or DEFAULT_MODEL

    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    # 清理旧的图片文件，避免前次运行的残留
    if images_dir.exists():
        for old_img in images_dir.glob("*.png"):
            old_img.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # 检测 PDF 类型
    pdf_type = _detect_pdf_type(doc)

    print(f"[*] 正在转换: {pdf_path.name}")
    print(f"    总页数: {total_pages}")
    print(f"    类型: {pdf_type} ({'扫描件OCR' if pdf_type == 'scanned' else '数字PDF'})")
    print(f"    模型: {model}")
    print(f"    输出目录: {output_dir}\n")

    all_md_parts = []
    page_images_map = {}  # page_num -> [filename, ...] 跟踪每页裁切的图片

    for page_num in range(total_pages):
        page = doc[page_num]

        # 1. 渲染页面为图片
        page_png = render_page_to_image(page)
        page_api_image, page_api_mime, _ = prepare_image_for_model(
            page_png,
            max_side=OCR_IMAGE_MAX_SIDE,
        )

        # 2. 提取文本层
        page_text = extract_page_text(page)

        # 3. 提取并保存图片
        if pdf_type == "scanned":
            # 扫描件：用 AI 视觉定位检测页面中的图表/插图区域，裁切保存
            try:
                figures = detect_page_figures(client, model, page_png, page_num)
            except Exception as e:
                print(f"  ⚠ 第{page_num+1}页图片检测失败: {e}")
                figures = []
            if figures:
                fig_results = crop_and_save_figures(page_png, figures, page_num, str(images_dir))
                # AI 验证裁切质量：可 accept/reject/adjust/split
                if fig_results:
                    fig_results = _ai_verify_and_refine_crops(
                        client, model, page_png, fig_results,
                        page_num, str(images_dir),
                    )
                image_filenames = [f[0] for f in fig_results]
                # 构建图片文件名→描述的映射，传给 OCR 调用
                fig_desc_map = {f[0]: f[1] for f in fig_results}
                fig_bboxes = {f[0]: f[2] for f in fig_results}
                print(f"[img:{len(image_filenames)}]", end=" ", flush=True)
            else:
                image_filenames = []
                fig_bboxes = {}
            # 按纵向位置排序并计算位置百分比
            image_filenames, image_positions = _compute_image_positions(
                image_filenames, page_png, fig_bboxes=fig_bboxes, page=page)
        else:
            # 数字PDF：嵌入图片提取 + AI 视觉检测（混合策略，确保不丢失图片信息）
            embedded_filenames = extract_and_save_images(doc, page, page_num, str(images_dir))

            # AI 检测页面中的图表/图形区域（补充嵌入提取无法覆盖的矢量图/组合图）
            try:
                figures = detect_page_figures(client, model, page_png, page_num)
            except Exception as e:
                print(f"  ⚠ 第{page_num+1}页AI图片检测失败: {e}")
                figures = []

            if figures:
                fig_results = crop_and_save_figures(page_png, figures, page_num, str(images_dir))
                # AI 验证裁切质量：可 accept/reject/adjust/split
                if fig_results:
                    fig_results = _ai_verify_and_refine_crops(
                        client, model, page_png, fig_results,
                        page_num, str(images_dir),
                        other_images=embedded_filenames,
                    )
                ai_filenames = [f[0] for f in fig_results]
                ai_pixel_bboxes = [f[2] for f in fig_results]
                # 合并：AI检测图 + 非重叠嵌入图（去除被AI覆盖的嵌入图）
                image_filenames = _merge_ai_and_embedded_images(
                    page, doc, ai_filenames, ai_pixel_bboxes, embedded_filenames,
                    page_png, str(images_dir),
                )
                fig_bboxes = {f[0]: f[2] for f in fig_results}
                print(f"[AI:{len(ai_filenames)} embed:{len(embedded_filenames)}->{len(image_filenames) - len(ai_filenames)}]", end=" ", flush=True)
            else:
                image_filenames = embedded_filenames
                fig_bboxes = {}
            # 按纵向位置排序并计算位置百分比
            image_filenames, image_positions = _compute_image_positions(
                image_filenames, page_png, fig_bboxes=fig_bboxes, page=page)

        # 4. 构建滚动上下文：大纲 + 上一页末尾
        prev_tail = ""
        if all_md_parts:
            # 扫描件需要更长的尾部来帮助模型识别跨页截断
            tail_len = 1200 if pdf_type == "scanned" else 500
            prev_tail = all_md_parts[-1][-tail_len:]
        outline = _build_outline(all_md_parts)

        print(f"  [{page_num + 1}/{total_pages}]", end=" ", flush=True)
        start = time.time()
        try:
            md_part = convert_page_with_ai(
                client, model, page_api_image, page_text, image_filenames,
                page_num, total_pages,
                prev_md_tail=prev_tail, outline=outline,
                pdf_type=pdf_type,
                page_image_mime=page_api_mime,
                image_positions=image_positions,
            )
            elapsed = time.time() - start
            print(f"OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ ({elapsed:.1f}s) {e}")
            # 降级：使用原始文本
            md_part = page_text if page_text else f"<!-- 第{page_num+1}页转换失败 -->"

        all_md_parts.append(md_part)
        if image_filenames:
            page_images_map[page_num] = image_filenames

    doc.close()

    # 4.5 保底：检查每页裁切的图片是否都被模型引用了，未引用的追加到该页 MD 末尾
    for pg, filenames in page_images_map.items():
        md_part = all_md_parts[pg]
        for fname in filenames:
            if fname not in md_part:
                all_md_parts[pg] += f"\n\n![](images/{fname})"

    # 5. 智能拼接所有页面：LLM 辅助去重 + 断句合并
    print(f"\n  拼接 {len(all_md_parts)} 页...", end="", flush=True)
    md_content = _join_pages_smart(all_md_parts, client=client)

    # 5.5 归一化跨页编号标题层级（不同 LLM 调用可能给同级标题不同的 # 层级）
    md_content = _normalize_sibling_headings(md_content)

    # 5.5b 根据编号深度规范化标题层级（X.Y → ###, X.Y.Z → ####, ...）
    md_content = _normalize_numbered_heading_levels(md_content)

    # 5.6 移除内容仅为表格的嵌入图片引用（表格已转写为 markdown，图片冗余）
    md_content = _remove_redundant_table_images(md_content)

    # 统一换行符（API 可能返回 \r\n）
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

    # 清理 Qwen 可能输出的 markdown 代码围栏
    md_content = re.sub(r'^```markdown\s*\n', '', md_content)
    md_content = re.sub(r'\n```\s*$', '', md_content)
    # 也清理中间页面可能出现的围栏
    md_content = re.sub(r'\n```\s*\n\n```markdown\s*\n', '\n\n', md_content)

    # 修复模型生成的坏图片引用格式
    # 模式1: ![](images/page9[](images/page9_img1.png)  — page后有数字
    # 模式2: ![](images/page[](images/page10_fig1.png)  — page后无数字
    md_content = re.sub(
        r'!\[\]\(images/page\d*\[]\(images/(page\d+_(?:img|fig)\d+\.png)\)',
        r'![](images/\1)',
        md_content,
    )

    # 通用后处理：修复未闭合的 HTML 注释（OCR 产出的注释可能缺少 -->）
    def _fix_unclosed_comments(text):
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
    md_content = _fix_unclosed_comments(md_content)

    # 通用后处理：修复缺少 images/ 前缀的图片引用
    md_content = re.sub(
        r'!\[([^\]]*)\]\((page\d+_(?:img|fig)\d+\.png)\)',
        r'![\1](images/\2)',
        md_content,
    )

    # 通用后处理：将 ####+ 标题降为 ###（文档最多 3 级）
    md_content = re.sub(
        r'^#{4,} ',
        '### ',
        md_content,
        flags=re.MULTILINE,
    )

    # 通用后处理：确保整篇文档只有一个 # 标题，后续 # 降为 ##
    # （提前执行，让后续章标题合并等步骤能匹配到统一的 ## 标题）
    first_h1 = re.search(r'^# ', md_content, flags=re.MULTILINE)
    if first_h1:
        before = md_content[:first_h1.end()]
        after = md_content[first_h1.end():]
        after = re.sub(r'^# ', '## ', after, flags=re.MULTILINE)
        md_content = before + after

    # 通用后处理：拆分同一行内合并的多个目录条目
    # 例如 "        - 4.3.1 启用双缓冲操作- 4.3.2 控制访问哪个缓冲区" → 两行（保留缩进）
    md_content = re.sub(
        r'([ \t]*)(- \d[\d.]+ [^\n]+?)- (\d[\d.]+ )',
        r'\1\2\n\1- \3',
        md_content,
    )

    # 通用后处理：去重 OCR 产生的重复节号（如 "2.3.1 2.3.1 状态图" → "2.3.1 状态图"）
    md_content = re.sub(r'(\d[\d.]+) \1 ', r'\1 ', md_content)

    # ── 通用后处理：目录页标题条目转为列表（在去重之前执行） ──
    # 检测方式1：寻找 ## 目录 标题
    # 检测方式2：以首个有正文的编号节标题为分界，前方无正文标题全部转为列表
    lines = md_content.split('\n')
    toc_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^## 目录\s*$', line):
            toc_idx = i
            break

    def _heading_has_body(lines, idx):
        """检查标题后 4 行内是否有正文（非标题、非列表、非空行）"""
        for j in range(idx + 1, min(idx + 5, len(lines))):
            stripped = lines[j].strip()
            if stripped and not re.match(r'^#{1,6} ', lines[j]) and not stripped.startswith('- '):
                return True
        return False

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

    # 通用后处理：按节号深度修正 TOC 条目缩进
    # OCR 模型在跨页后常丢失缩进上下文，用节号层深（点号数量）推断正确缩进
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

    # 通用后处理：修复 TOC 条目中重复的节号（如 "2.3.1 2.3.1 状态图" → "2.3.1 状态图"）
    md_content = re.sub(
        r'^([ \t]*- )(\d[\d.]*) \2 ',
        r'\1\2 ',
        md_content,
        flags=re.MULTILINE,
    )

    # 通用后处理：去重重复出现的相同章节标题
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

    # ── 通用后处理：合并跨页断裂的段落 ──
    # 检测模式：正文行（不以句末标点结尾）\n\n续接文字 → 合并为同一段落
    # 仅合并两个都是纯正文行（非标题/列表/图片/代码/引用）的情况
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

    # ── 扫描件专用后处理 ──
    if pdf_type == "scanned":
        # 提升裸章号行为 ## 标题（如 "第六章\n## 语言是什么" → "## 第六章\n## 语言是什么"）
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

    # ── 通用后处理：多级编号章节标题层级规范化 ──
    # 检测文档中是否存在多级编号标题（如 2.1、3.2.1 等）
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

    # ── 以下后处理仅适用于数字 PDF（教程/手册），书籍扫描件跳过 ──
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

    # 通用后处理：裸露的科学计数法数据行自动包裹代码围栏
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

    # 通用后处理：修复被分页与硬换行拆坏的 Markdown 表格
    def _merge_split_tables(text):
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
    md_content = _merge_split_tables(md_content)

    # 合并被空行错误拆开的列表项续句
    md_content = _merge_split_list_item_paragraphs(md_content)

    # 清理跨页拼接产生的孤儿碎片（如 "度。" 重复自上段末尾 "长度。"）
    def _remove_orphan_boundary_fragments(text):
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
    md_content = _remove_orphan_boundary_fragments(md_content)

    # 通用后处理：清理引用了不存在图片的 markdown 图片标记
    def _remove_ghost_images(text, base_dir):
        def _check_img(m):
            img_path = base_dir / m.group(2)
            if img_path.exists():
                return m.group(0)
            return ''  # 图片不存在，删除引用
        return re.sub(r'!\[([^\]]*)\]\((images/[^)]+)\)', _check_img, text)
    md_content = _remove_ghost_images(md_content, output_dir)
    # 清理删除图片引用后可能产生的多余空行
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)

    md_path = output_dir / f"{pdf_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    img_count = len(list(images_dir.glob("*.png")))
    print(f"\n[DONE] 转换完成！")
    print(f"   Markdown: {md_path}")
    print(f"   图片: {images_dir} ({img_count}张)")

    return str(md_path)


# ── 入口 ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PDF to Markdown —— AI 增强转换",
    )
    parser.add_argument("pdf", nargs="?",
                        default=r"d:\python_programs\mdfy\test-files\（mjy）Expert 使用教程.pdf",
                        help="要转换的 PDF 文件路径")
    parser.add_argument("--model", "-m", choices=AVAILABLE_MODELS,
                        default=DEFAULT_MODEL,
                        help=f"模型选择（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出目录（默认: PDF 同目录同名文件夹）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pdf_to_markdown_ai(args.pdf, output_dir=args.output, model=args.model)

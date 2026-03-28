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


def extract_and_save_images(doc, page, page_num, images_dir):
    """提取并保存页面中的原始图片，返回图片文件名列表"""
    saved = []
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace and pix.colorspace.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            filename = f"page{page_num + 1}_img{img_index + 1}.png"
            pix.save(os.path.join(images_dir, filename))
            saved.append(filename)
            pix = None
        except Exception as e:
            print(f"  ⚠ 第{page_num+1}页图片{img_index+1}提取失败: {e}")
    return saved


def _merge_ai_and_embedded_images(page, doc, ai_filenames, ai_pixel_bboxes, embedded_filenames,
                                    page_png_bytes, images_dir):
    """合并AI检测图和嵌入图，去除被AI图覆盖的嵌入图。

    ai_pixel_bboxes: 成功裁切的AI图的像素坐标列表 [(x1,y1,x2,y2), ...]
    返回去重后的图片文件名列表（AI图在前，不重叠的嵌入图在后）。
    被AI覆盖的嵌入图文件会被删除以避免冗余。
    """
    from PIL import Image

    if not ai_filenames or not embedded_filenames:
        return ai_filenames + embedded_filenames

    img = Image.open(io.BytesIO(page_png_bytes))
    scale_x = img.width / page.rect.width
    scale_y = img.height / page.rect.height

    # 获取嵌入图的页面位置
    image_list = page.get_images(full=True)
    kept_embedded = []
    for idx, filename in enumerate(embedded_filenames):
        if idx >= len(image_list):
            kept_embedded.append(filename)
            continue
        xref = image_list[idx][0]
        try:
            rects = page.get_image_rects(xref)
            if not rects:
                kept_embedded.append(filename)
                continue
            rect = rects[0]
            # PDF坐标(pt) → 渲染像素坐标
            ex1 = rect.x0 * scale_x
            ey1 = rect.y0 * scale_y
            ex2 = rect.x1 * scale_x
            ey2 = rect.y1 * scale_y
            emb_area = max((ex2 - ex1) * (ey2 - ey1), 1)

            # 检查是否被某个AI图覆盖 >50%
            is_covered = False
            for ax1, ay1, ax2, ay2 in ai_pixel_bboxes:
                ix1, iy1 = max(ex1, ax1), max(ey1, ay1)
                ix2, iy2 = min(ex2, ax2), min(ey2, ay2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter / emb_area > 0.50:
                        is_covered = True
                        break
            if is_covered:
                # 删除被覆盖的嵌入图文件
                try:
                    os.remove(os.path.join(images_dir, filename))
                except OSError:
                    pass
            else:
                kept_embedded.append(filename)
        except Exception:
            kept_embedded.append(filename)

    return ai_filenames + kept_embedded


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
- 如果我提供了 <images> 列表中的图片文件名，在图出现的位置用 ![描述](images/文件名) 引用对应的图片
- 如果没有提供图片文件名，用 `<!-- 图：[简要描述] -->` 标注其位置和内容
- 表格：尽量转写为 Markdown 表格
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
唯一的例外：如果一张图片的内容**仅仅是代码/脚本文字**，且你已经完整转写为代码块，则可以省略该图片引用。
界面截图、操作步骤截图、示意图、对话框截图、任何包含 GUI 元素的图片，一律保留引用，即使你描述了其内容。
拿不准的时候，保留图片引用。
</rule>

<rule id="code_blocks">
代码、脚本、配置文件、数据数组等技术内容必须用 ``` 围栏包裹。
如果能识别语言，加上语言标识（如 ```python、```c 等）。
如果当前页的数据是上一页的延续，也必须用新的 ``` 围栏包裹。
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
                          pdf_type="digital", page_image_mime="image/png"):
    """调用 Qwen 多模态模型转换单页"""

    is_scanned = (pdf_type == "scanned")
    system_prompt = SYSTEM_PROMPT_SCANNED if is_scanned else SYSTEM_PROMPT

    # 图片清单
    if image_filenames:
        img_list = '\n'.join(f'    <img>{f}</img>' for f in image_filenames)
        img_section = (
            f"<images count=\"{len(image_filenames)}\">\n"
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

    if prev_md_tail:
        # 检测 tail 末尾是否在句子中间被截断
        stripped_tail = prev_md_tail.rstrip()
        # 完整句子结束标志：中文句号、问号、感叹号、英文句点+空格/换行、冒号、引号闭合等
        is_mid_sentence = bool(stripped_tail) and not re.search(
            r'[。？！…」』\u201d]$|[.?!:;]$|\*\*$|```$',
            stripped_tail,
        )
        if is_mid_sentence:
            tail_section = (
                f"<previous_page_tail truncated=\"true\">\n{prev_md_tail}\n"
                f"  <!-- 注意：上一页末尾的句子被分页截断了，本页输出需要以续接文字开头 -->\n"
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
            img_instructions = f"    - 本页有 {len(image_filenames)} 张已裁切的图片；在图出现的位置用 ![描述](images/文件名) 引用\n"
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
            f"    - 本页有 {len(image_filenames)} 张图片；仅代码截图已完整转写的可省略，其余必须引用\n"
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
    if re.match(r'^[-*] ', last_line):
        return False
    if re.match(r'^>', last_line):
        return False
    # 以中文/英文标点结束的句子视为完整
    if re.search(r'[。？！…」』\u201d；：]$|[.?!;:]$|\*\*$', stripped):
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
                # 判断拼接方式：如果 prev 以不完整句结尾且 stitched 不以换行/标题开头，紧密拼接
                if _is_incomplete_sentence(prev) and not re.match(r'\s*\n|^\s*#', stitched):
                    result_parts.append(corrected_curr)
                else:
                    result_parts.append("\n\n" + corrected_curr)
                continue

        # 回退：正则去重 + 续接
        curr = _dedup_page_boundary(prev, curr)
        if not curr.strip():
            continue

        if _is_incomplete_sentence(prev) and _is_continuation_start(curr):
            result_parts.append(curr)
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
                image_filenames = [f[0] for f in fig_results]
                # 构建图片文件名→描述的映射，传给 OCR 调用
                fig_desc_map = {f[0]: f[1] for f in fig_results}
                print(f"[img:{len(image_filenames)}]", end=" ", flush=True)
            else:
                image_filenames = []
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
                ai_filenames = [f[0] for f in fig_results]
                ai_pixel_bboxes = [f[2] for f in fig_results]
                # 合并：AI检测图 + 非重叠嵌入图（去除被AI覆盖的嵌入图）
                image_filenames = _merge_ai_and_embedded_images(
                    page, doc, ai_filenames, ai_pixel_bboxes, embedded_filenames,
                    page_png, str(images_dir),
                )
                print(f"[AI:{len(ai_filenames)} embed:{len(embedded_filenames)}->{len(image_filenames) - len(ai_filenames)}]", end=" ", flush=True)
            else:
                image_filenames = embedded_filenames

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

    # 通用后处理：清理引用了不存在图片的 markdown 图片标记
    def _remove_ghost_images(text, img_dir):
        def _check_img(m):
            img_path = img_dir / m.group(2)
            if img_path.exists():
                return m.group(0)
            return ''  # 图片不存在，删除引用
        return re.sub(r'!\[([^\]]*)\]\((images/[^)]+)\)', _check_img, text)
    md_content = _remove_ghost_images(md_content, images_dir)
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

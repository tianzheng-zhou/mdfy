"""PDF 渲染、图像压缩、bbox 归一化与检测响应解析。

纯视觉版：fitz 只用于打开 PDF + 渲染为 PNG。无文本层提取、无 PDF 类型检测。
"""

import base64
import io
import re

import fitz

from .config import MODEL_IMAGE_MAX_BYTES, MIN_MODEL_IMAGE_SIDE, RENDER_DPI


# ── 渲染 ─────────────────────────────────────────────────────────────

def render_page_to_image(page, dpi=RENDER_DPI):
    """将 PDF 页面渲染为 PNG 字节。"""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ── 图像压缩 ─────────────────────────────────────────────────────────

def _serialize_pil_image(image, fmt, **save_kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def prepare_image_for_model(image_bytes, *, max_side, max_bytes=MODEL_IMAGE_MAX_BYTES,
                            min_side=MIN_MODEL_IMAGE_SIDE):
    """将图片压缩到适合发送给视觉模型的尺寸与体积。

    返回 (encoded_bytes, mime_type, (width, height))。
    优先保留 PNG；若体积仍过大则降尺度并回退到 JPEG。
    """
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    resample = getattr(Image, "Resampling", Image).LANCZOS

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
                jpeg_source, "JPEG", quality=quality, optimize=True,
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


def encode_data_url(image_bytes, mime):
    """把图像字节编码为 data URL，供 OpenAI image_url 字段使用。"""
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


# ── bbox 坐标 ────────────────────────────────────────────────────────

def bbox_to_pixels(bbox, image_width, image_height):
    """将 bbox 转为像素坐标。

    优先按归一化坐标 [0, 1000] 解释；若超出范围则回退为绝对像素。
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


def normalize_bbox_to_1000(bbox, image_size, *, from_pixels=False):
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


# ── 检测响应解析 ─────────────────────────────────────────────────────

def parse_figure_detection_response(raw_text, image_size):
    """解析模型返回的 JSON 检测结果，兼容 bbox / bbox_2d 两种字段名。"""
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
            "bbox": normalize_bbox_to_1000(
                bbox,
                image_size,
                from_pixels=(bbox_key == "bbox_2d"),
            ),
            "desc": item.get("desc") or item.get("label") or "",
        })

    return figures


def parse_qwenvl_markdown_figures(raw_text, image_size):
    """解析 qwenvl markdown 模式输出中的 `<!-- Image (x,y,w,h) -->` 坐标注释。"""
    raw = raw_text.strip()
    raw = re.sub(r'^```(?:markdown)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    figures = []
    seen = set()
    for match in re.finditer(r'<!--\s*Image\s*\(([^)]+)\)\s*-->', raw, flags=re.IGNORECASE):
        parts = [p.strip() for p in match.group(1).split(',')]
        if len(parts) != 4:
            continue
        try:
            pixel_bbox = [float(p) for p in parts]
        except ValueError:
            continue

        norm_bbox = normalize_bbox_to_1000(pixel_bbox, image_size, from_pixels=True)
        bbox_key = tuple(int(round(v)) for v in norm_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        figures.append({"bbox": norm_bbox, "desc": ""})

    return figures


def request_qwenvl_markdown(client, model, image_bytes, image_mime):
    """走 qwenvl 内置的 markdown 模式，用于抓取 `<!-- Image (...) -->` 坐标。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_data_url(image_bytes, image_mime)}},
                {"type": "text", "text": "qwenvl markdown"},
            ]},
        ],
        temperature=0.1,
        max_tokens=4096,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content.strip()

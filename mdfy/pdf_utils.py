"""PDF 工具函数：渲染、文本提取、图片压缩、bbox 转换。"""

import base64
import io
import re

import fitz

from .config import MODEL_IMAGE_MAX_BYTES, MIN_MODEL_IMAGE_SIDE


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


def _is_scan_tile_page(page, doc):
    """检测页面是否为扫描页（整页图或条带拼接）"""
    imgs = page.get_images(full=True)
    if not imgs:
        return False

    page_w_px = page.rect.width * 200 / 72

    if len(imgs) == 1:
        try:
            pix = fitz.Pixmap(doc, imgs[0][0])
            return pix.width > page_w_px * 0.8
        except Exception:
            return False

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

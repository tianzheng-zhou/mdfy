"""
测试脚本：对扫描书页进行图片区域检测与裁切。
对比两种方案：
  A) 直接让模型输出 bbox JSON（当前方案）
  B) Qwen 原生 qwenvl markdown 文档解析
测试文件：缤纷的语言学-前39page.pdf
"""

import base64
import fitz
import json
import os
import re
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

# ── 配置 ──
PDF_PATH = r"d:\python_programs\mdfy\test-files\缤纷的语言学-前39page.pdf"
OUTPUT_DIR = Path(r"d:\python_programs\mdfy\test-files\_crop_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen3.5-plus"
# 明确有图的页面（0-indexed）：
#   15 = 第16页: 图1 波形图 (整页图)
#   26 = 第27页: 图2 黑猩猩 (文字+大图)
#   28 = 第29页: 图3 手势 (大图+文字)
#   33 = 第34页: 图4 汉语卷轴 (大图+说明)
# 加一个纯文字页做对照
TEST_PAGES = [15, 24, 26, 28, 33]  # 24是纯文字对照


def get_client():
    return OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def render_page(page, dpi=200):
    """渲染页面为 PNG 字节"""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ── 方案 A：JSON bbox 检测 ──

def detect_figures_json(client, page_png_bytes, page_num):
    """方案 A：让模型直接输出图片区域的 JSON bbox"""
    prompt = (
        "检测这个扫描书页中所有非文字的视觉元素（插图、照片、图表、图形、地图、波形图等）。\n"
        "不要包括纯文字区域、页眉页脚、页码、条形码。\n"
        "返回JSON数组，每个元素格式：\n"
        '{"bbox": [x1, y1, x2, y2], "desc": "简短中文描述"}\n'
        "其中 x1,y1 是左上角像素坐标，x2,y2 是右下角像素坐标（相对于输入图片的像素）。\n"
        "如果页面上没有图片/图表，返回空数组：[]\n"
        "只输出JSON，不要其他文字。"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(page_png_bytes).decode()}"
                }},
                {"type": "text", "text": prompt},
            ]},
        ],
        temperature=0.1,
        max_tokens=1024,
        extra_body={"enable_thinking": False},
    )
    raw = response.choices[0].message.content.strip()
    return raw


# ── 方案 B：Qwen qwenvl markdown 文档解析 ──

def detect_figures_qwenvl_md(client, page_png_bytes, page_num):
    """方案 B：用 qwenvl markdown 进行文档解析，获取图片位置信息"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(page_png_bytes).decode()}"
                }},
                {"type": "text", "text": "qwenvl markdown"},
            ]},
        ],
        temperature=0.1,
        max_tokens=4096,
        extra_body={"enable_thinking": False},
    )
    raw = response.choices[0].message.content.strip()
    return raw


# ── 方案 C：Qwen 原生 grounding 检测 ──

def detect_figures_grounding(client, page_png_bytes, page_num):
    """方案 C：用 Qwen 原生物体定位能力检测图片区域"""
    prompt = (
        "请检测这个扫描书页中所有非文字的视觉元素（插图、照片、图表、图形、地图、波形图等），"
        "并以 JSON 格式输出每个元素的 bbox 坐标。"
        "不要包括纯文字区域。\n"
        "输出格式示例：\n"
        '[{"bbox": [x1, y1, x2, y2], "desc": "描述"}]\n'
        "坐标为归一化坐标（0-1000范围），左上角为(0,0)，右下角为(1000,1000)。\n"
        "如果没有非文字元素，返回 []"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(page_png_bytes).decode()}"
                }},
                {"type": "text", "text": prompt},
            ]},
        ],
        temperature=0.1,
        max_tokens=1024,
        extra_body={"enable_thinking": False},
    )
    raw = response.choices[0].message.content.strip()
    return raw


def parse_bbox_json(raw_text):
    """解析 JSON 格式的 bbox 输出"""
    text = raw_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def crop_figure(page_png_bytes, bbox, img_width, img_height, is_normalized=False):
    """从渲染图中裁切指定区域"""
    img = Image.open(io.BytesIO(page_png_bytes))
    x1, y1, x2, y2 = bbox

    if is_normalized:
        # 归一化坐标 [0, 1000] → 像素坐标
        x1 = int(x1 / 1000 * img.width)
        y1 = int(y1 / 1000 * img.height)
        x2 = int(x2 / 1000 * img.width)
        y2 = int(y2 / 1000 * img.height)
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # padding
    pad = 5
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.width, x2 + pad)
    y2 = min(img.height, y2 + pad)

    if (x2 - x1) < 30 or (y2 - y1) < 30:
        return None
    return img.crop((x1, y1, x2, y2))


def main():
    client = get_client()
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)

    print(f"📄 PDF: {PDF_PATH}")
    print(f"📑 总页数: {total_pages}")
    print(f"🤖 模型: {MODEL}")
    print(f"📁 输出: {OUTPUT_DIR}\n")

    # 去重并限制在有效页码范围内
    pages_to_test = sorted(set(p for p in TEST_PAGES if 0 <= p < total_pages))

    for page_num in pages_to_test:
        page = doc[page_num]
        page_png = render_page(page)

        # 获取渲染后的图片尺寸
        img = Image.open(io.BytesIO(page_png))
        img_w, img_h = img.width, img.height

        print(f"{'='*60}")
        print(f"📄 第 {page_num + 1} 页 (渲染尺寸: {img_w}x{img_h})")
        print(f"{'='*60}")

        # 保存原始渲染图以便对比
        orig_path = OUTPUT_DIR / f"page{page_num+1}_orig.png"
        with open(orig_path, "wb") as f:
            f.write(page_png)

        # ── 方案 A：JSON bbox（像素坐标）──
        print("\n🅰️ 方案A: JSON bbox (像素坐标)")
        try:
            start = time.time()
            raw_a = detect_figures_json(client, page_png, page_num)
            elapsed = time.time() - start
            print(f"  ⏱ {elapsed:.1f}s")
            print(f"  📝 原始输出:\n    {raw_a[:500]}")
            figures_a = parse_bbox_json(raw_a)
            print(f"  📊 解析到 {len(figures_a)} 个区域")
            for i, fig in enumerate(figures_a):
                print(f"    [{i}] bbox={fig.get('bbox')}, desc={fig.get('desc', '')}")
                cropped = crop_figure(page_png, fig["bbox"], img_w, img_h, is_normalized=False)
                if cropped:
                    save_path = OUTPUT_DIR / f"page{page_num+1}_A_fig{i+1}.png"
                    cropped.save(str(save_path))
                    print(f"    ✅ 已保存: {save_path.name} ({cropped.width}x{cropped.height})")
        except Exception as e:
            print(f"  ❌ 异常: {e}")

        # ── 方案 C：归一化坐标 [0, 1000] ──
        print("\n🅲 方案C: 归一化坐标 (0-1000)")
        try:
            start = time.time()
            raw_c = detect_figures_grounding(client, page_png, page_num)
            elapsed = time.time() - start
            print(f"  ⏱ {elapsed:.1f}s")
            print(f"  📝 原始输出:\n    {raw_c[:500]}")
            figures_c = parse_bbox_json(raw_c)
            print(f"  📊 解析到 {len(figures_c)} 个区域")
            for i, fig in enumerate(figures_c):
                bbox = fig.get("bbox", [])
                print(f"    [{i}] bbox={bbox}, desc={fig.get('desc', '')}")
                # 判断是否为归一化坐标（所有值 <= 1000）
                if bbox and all(0 <= v <= 1000 for v in bbox):
                    cropped = crop_figure(page_png, bbox, img_w, img_h, is_normalized=True)
                else:
                    cropped = crop_figure(page_png, bbox, img_w, img_h, is_normalized=False)
                if cropped:
                    save_path = OUTPUT_DIR / f"page{page_num+1}_C_fig{i+1}.png"
                    cropped.save(str(save_path))
                    print(f"    ✅ 已保存: {save_path.name} ({cropped.width}x{cropped.height})")
        except Exception as e:
            print(f"  ❌ 异常: {e}")

        # ── 方案 B：qwenvl markdown ──
        # 只在有图页测试
        if page_num in [15, 26, 28, 33]:
            print("\n🅱️ 方案B: qwenvl markdown")
            try:
                start = time.time()
                raw_b = detect_figures_qwenvl_md(client, page_png, page_num)
                elapsed = time.time() - start
                print(f"  ⏱ {elapsed:.1f}s")
                # 只打印前1000字符
                print(f"  📝 原始输出 (前1000字符):\n    {raw_b[:1000]}")
                # 检查是否有 <img> 或 bbox 之类的标记
                if '<img' in raw_b.lower() or 'bbox' in raw_b.lower() or '<figure' in raw_b.lower():
                    print(f"  ✅ 检测到图片/位置标记！")
            except Exception as e:
                print(f"  ❌ 异常: {e}")

        print()

    doc.close()
    print(f"\n🎉 测试完成！请检查 {OUTPUT_DIR} 中的裁切结果")


if __name__ == "__main__":
    main()

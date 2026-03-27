"""
测试脚本 v2：聚焦测试扫描书页的图片区域检测与裁切。
针对「缤纷的语言学-前39page.pdf」，测试前几页 + 已知有图的页面。

已知有图的页面（0-indexed）：
  - p15: 图1 - 波形图 (those three oranges)
  - p26: 图2 - 黑猩猩整饰照片
  - p28: 图3 - 一个手势
  - p33: 表格 + 中国卷轴

纯文字页面（不应检测到图）：
  - p0~p4: 封面、版权页等
  - p8~p10: 纯文字正文
"""

import base64
import fitz
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io

from pdf_to_md_ai import (
    bbox_to_pixels,
    parse_figure_detection_response,
    parse_qwenvl_markdown_figures,
    prepare_image_for_model,
    detect_page_figures,
    crop_and_save_figures,
)

load_dotenv()

# ── 配置 ──
PDF_PATH = r"d:\python_programs\mdfy\test-files\缤纷的语言学-前39page.pdf"
OUTPUT_DIR = Path(r"d:\python_programs\mdfy\test-files\_crop_test_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen3.5-plus"

# 纯文字页 (期望 0 检测) + 含图页 (期望 ≥1 检测)
TEXT_ONLY_PAGES = [0, 1, 2, 5, 8]
FIGURE_PAGES = [15, 26, 28, 33]
ALL_TEST_PAGES = sorted(set(TEXT_ONLY_PAGES + FIGURE_PAGES))


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


def main():
    client = get_client()
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)

    print(f"📄 PDF: {PDF_PATH}")
    print(f"📑 总页数: {total_pages}")
    print(f"🤖 模型: {MODEL}")
    print(f"📁 输出: {OUTPUT_DIR}")
    print(f"🧪 测试页: {[p+1 for p in ALL_TEST_PAGES]}\n")

    # 创建 images 子目录用于 crop_and_save_figures
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    results = {}

    for page_num in ALL_TEST_PAGES:
        if page_num >= total_pages:
            continue

        page = doc[page_num]
        page_png = render_page(page)

        # 获取渲染后的图片尺寸（不保留完整图到内存）
        img = Image.open(io.BytesIO(page_png))
        img_w, img_h = img.width, img.height
        img.close()

        is_text_page = page_num in TEXT_ONLY_PAGES
        page_type = "📝纯文字" if is_text_page else "🖼️含图"

        print(f"{'='*60}")
        print(f"📄 第 {page_num + 1} 页 [{page_type}] (渲染: {img_w}x{img_h})")
        print(f"{'='*60}")

        # 使用 pdf_to_md_ai 的核心检测函数
        start = time.time()
        try:
            figures = detect_page_figures(client, MODEL, page_png, page_num)
            elapsed = time.time() - start
            print(f"  ⏱ 检测耗时: {elapsed:.1f}s")
            print(f"  📊 检测到 {len(figures)} 个区域")

            for i, fig in enumerate(figures):
                bbox = fig.get("bbox", [])
                desc = fig.get("desc", "")
                print(f"    [{i}] bbox={[round(v,1) for v in bbox]}, desc={desc}")

            # 裁切并保存
            if figures:
                saved = crop_and_save_figures(page_png, figures, page_num, str(images_dir))
                print(f"  💾 保存了 {len(saved)} 张裁切图")
                for fname, desc in saved:
                    fpath = images_dir / fname
                    crop_img = Image.open(str(fpath))
                    size_kb = os.path.getsize(str(fpath)) / 1024
                    print(f"    ✅ {fname}: {crop_img.width}x{crop_img.height}, {size_kb:.0f} KB")
                    crop_img.close()

            # 评估
            if is_text_page and len(figures) > 0:
                print(f"  ⚠️ 误检！纯文字页检测到 {len(figures)} 个区域")
            elif not is_text_page and len(figures) == 0:
                print(f"  ⚠️ 漏检！含图页未检测到图片")
            else:
                print(f"  ✅ 检测结果符合预期")

            results[page_num] = {
                "type": "text" if is_text_page else "figure",
                "detected": len(figures),
                "saved": len(saved) if figures else 0,
                "time": elapsed,
            }

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 检测失败 ({elapsed:.1f}s): {e}")
            results[page_num] = {
                "type": "text" if is_text_page else "figure",
                "detected": -1,
                "saved": 0,
                "time": elapsed,
                "error": str(e),
            }

        print()

    doc.close()

    # 汇总
    print(f"\n{'='*60}")
    print("📊 汇总")
    print(f"{'='*60}")
    print(f"{'页码':>4} {'类型':>8} {'检测数':>6} {'保存数':>6} {'耗时':>6} {'状态':>6}")
    print("-" * 50)

    false_pos = 0
    false_neg = 0
    for pn in sorted(results.keys()):
        r = results[pn]
        if r["detected"] < 0:
            status = "❌错误"
        elif r["type"] == "text" and r["detected"] > 0:
            status = "⚠️误检"
            false_pos += r["detected"]
        elif r["type"] == "figure" and r["detected"] == 0:
            status = "⚠️漏检"
            false_neg += 1
        else:
            status = "✅"
        print(f"  {pn+1:>3} {r['type']:>8} {r['detected']:>6} {r['saved']:>6} {r['time']:>5.1f}s {status}")

    print(f"\n误检数: {false_pos}, 漏检数: {false_neg}")
    print(f"\n🎉 测试完成！请检查 {OUTPUT_DIR / 'images'} 中的裁切结果")


if __name__ == "__main__":
    main()

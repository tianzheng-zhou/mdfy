"""Mini 端到端转换测试：转换第 15-29 页（包含波形图、黑猩猩照片、手势照片）"""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from pdf_to_md_ai import (
    get_client, DEFAULT_MODEL, render_page_to_image, prepare_image_for_model,
    extract_page_text, detect_page_figures, crop_and_save_figures,
    convert_page_with_ai, _detect_pdf_type, _build_outline,
    OCR_IMAGE_MAX_SIDE,
)
import fitz

PDF_PATH = Path("test-files/缤纷的语言学-前39page.pdf")
OUTPUT_DIR = Path("test-files/_mini_e2e_test")
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 只转换 page 15-29（0-indexed: 14-28）
START_PAGE = 14
END_PAGE = 28  # inclusive

client = get_client()
model = DEFAULT_MODEL
doc = fitz.open(str(PDF_PATH))
total_pages = len(doc)
pdf_type = _detect_pdf_type(doc)

print(f"📄 PDF: {PDF_PATH}")
print(f"📑 总页数: {total_pages}, 类型: {pdf_type}")
print(f"🧪 转换范围: 第{START_PAGE+1}-{END_PAGE+1}页")
print(f"📁 输出: {OUTPUT_DIR}\n")

all_md_parts = []
total_images = 0

for page_num in range(START_PAGE, END_PAGE + 1):
    page = doc[page_num]
    page_png = render_page_to_image(page)
    page_api_image, page_api_mime, _ = prepare_image_for_model(
        page_png, max_side=OCR_IMAGE_MAX_SIDE
    )
    page_text = extract_page_text(page)

    # 图片检测
    try:
        figures = detect_page_figures(client, model, page_png, page_num)
    except Exception as e:
        print(f"  ⚠ 第{page_num+1}页图片检测失败: {e}")
        figures = []

    if figures:
        fig_results = crop_and_save_figures(page_png, figures, page_num, str(IMAGES_DIR))
        image_filenames = [f[0] for f in fig_results]
        total_images += len(image_filenames)
        print(f"  📷 第{page_num+1}页: {len(image_filenames)}张图片")
    else:
        image_filenames = []

    prev_tail = all_md_parts[-1][-800:] if all_md_parts else ""
    outline = _build_outline(all_md_parts)

    print(f"  🔄 第{page_num+1}页 OCR...", end=" ", flush=True)
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
        print(f"✅ ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ ({elapsed:.1f}s) {e}")
        md_part = f"<!-- 第{page_num+1}页转换失败 -->"

    all_md_parts.append(md_part)

doc.close()

# 拼接输出
md_content = "\n\n".join(all_md_parts)
md_content = md_content.replace('\r\n', '\n')
md_content = re.sub(r'^```markdown\s*\n', '', md_content)
md_content = re.sub(r'\n```\s*$', '', md_content)

md_path = OUTPUT_DIR / "mini_test.md"
md_path.write_text(md_content, encoding="utf-8")

# 检查图片引用
img_refs = re.findall(r'!\[.*?\]\(images/[^)]+\)', md_content)
comment_refs = re.findall(r'<!-- 图：.*?-->', md_content)

print(f"\n{'='*60}")
print(f"📊 转换完成")
print(f"  总页数: {END_PAGE - START_PAGE + 1}")
print(f"  裁切图片: {total_images}张")
print(f"  MD中图片引用: {len(img_refs)}个")
print(f"  MD中图注释: {len(comment_refs)}个")
print(f"  输出: {md_path}")
for ref in img_refs:
    print(f"    {ref}")

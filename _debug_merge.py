"""调试第4页AI检测和嵌入图的bbox关系"""
import fitz, io
from PIL import Image
from pdf_to_md_ai import render_page_to_image, detect_page_figures, bbox_to_pixels
from openai import OpenAI

doc = fitz.open('d:/python_programs/mdfy/test-files/（mjy）Expert 使用教程.pdf')
page = doc[3]  # page 4

page_png = render_page_to_image(page)
img = Image.open(io.BytesIO(page_png))
print(f'Page 4 rendered: {img.width}x{img.height}')

# Get embedded image rects
image_list = page.get_images(full=True)
scale_x = img.width / page.rect.width
scale_y = img.height / page.rect.height
for idx, im in enumerate(image_list):
    xref = im[0]
    rects = page.get_image_rects(xref)
    if rects:
        r = rects[0]
        px = (r.x0*scale_x, r.y0*scale_y, r.x1*scale_x, r.y1*scale_y)
        area = (px[2]-px[0]) * (px[3]-px[1])
        print(f'  Embedded {idx}: pixel=({px[0]:.0f},{px[1]:.0f},{px[2]:.0f},{px[3]:.0f}) area={area:.0f}')

# Detect AI figures
client = OpenAI(
    api_key='sk-f4f2bbbb37024fa09e6ca3bff5e1067e',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
figures = detect_page_figures(client, 'qwen3.5-flash', page_png, 3)
print(f'AI detected {len(figures)} figures:')
for i, fig in enumerate(figures):
    bbox = fig['bbox']
    px = bbox_to_pixels(bbox, img.width, img.height)
    area = (px[2]-px[0]) * (px[3]-px[1])
    desc = fig.get("description", "")[:60]
    print(f'  Fig {i}: bbox={bbox}, pixel=({px[0]},{px[1]},{px[2]},{px[3]}) area={area}, desc={desc}')

# Check overlaps
print('\nOverlap analysis:')
for idx, im in enumerate(image_list):
    xref = im[0]
    rects = page.get_image_rects(xref)
    if not rects:
        continue
    r = rects[0]
    ex1, ey1 = r.x0*scale_x, r.y0*scale_y
    ex2, ey2 = r.x1*scale_x, r.y1*scale_y
    emb_area = max((ex2-ex1)*(ey2-ey1), 1)
    
    for fi, fig in enumerate(figures):
        bbox = fig['bbox']
        ax1, ay1, ax2, ay2 = bbox_to_pixels(bbox, img.width, img.height)
        ix1 = max(ex1, ax1)
        iy1 = max(ey1, ay1)
        ix2 = min(ex2, ax2)
        iy2 = min(ey2, ay2)
        if ix1 < ix2 and iy1 < iy2:
            inter = (ix2-ix1) * (iy2-iy1)
            ratio = inter / emb_area
            print(f'  Embed{idx} vs Fig{fi}: inter={inter:.0f} / emb_area={emb_area:.0f} = {ratio:.2%} {"COVERED" if ratio > 0.5 else ""}')
        else:
            print(f'  Embed{idx} vs Fig{fi}: no overlap')

doc.close()

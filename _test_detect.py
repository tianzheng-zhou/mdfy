"""Test detect_page_figures with VL model + crop verification."""
import fitz, json, os, sys, base64, re
from PIL import Image
import io
sys.path.insert(0, '.')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DASHSCOPE_API_KEY') or open('.env').read().split('=',1)[1].strip(),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

pdf_path = r'test-files/缤纷的语言学-前39page.pdf'
pdf = fitz.open(pdf_path)
model = 'qwen-vl-max'

prompt = (
    "检测这个扫描书页中所有非文字的视觉元素（插图、照片、图表、图形、地图、波形图等）。\n"
    "不要包括纯文字区域、页眉页脚、页码、条形码。\n"
    "返回JSON数组，每个元素格式：\n"
    '{"bbox": [x1, y1, x2, y2], "desc": "简短中文描述"}\n'
    "其中 x1,y1 是左上角像素坐标，x2,y2 是右下角像素坐标。\n"
    "如果页面上没有图片/图表，返回空数组：[]"
)

os.makedirs('test-files/_crop_test', exist_ok=True)

# Test on pages with figures: 15(waveform), 26(chimpanzee), 28(gesture), 33(calligraphy), 34(Latin Bible)
for page_idx in [15, 26, 28, 33, 34]:
    page = pdf[page_idx]
    pix = page.get_pixmap(dpi=200)
    png_bytes = pix.tobytes('png')
    w, h = pix.width, pix.height
    print(f'=== Page {page_idx+1} (idx {page_idx}) size={w}x{h} ===')
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"}},
            {"type": "text", "text": prompt},
        ]}],
        temperature=0.1,
        max_tokens=1024,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    print(f'Raw: {raw}')
    
    try:
        figures = json.loads(raw)
    except:
        print('JSON parse error')
        continue
    
    if not figures:
        print('No figures detected')
        continue
    
    img = Image.open(io.BytesIO(png_bytes))
    
    for i, fig in enumerate(figures):
        bbox = fig['bbox']
        desc = fig.get('desc', '')
        print(f'  Fig {i+1}: bbox={bbox} desc="{desc}"')
        
        # Try as 0-1000 normalized coords
        x1_norm = int(bbox[0] / 1000 * w)
        y1_norm = int(bbox[1] / 1000 * h)
        x2_norm = int(bbox[2] / 1000 * w)
        y2_norm = int(bbox[3] / 1000 * h)
        
        # Try as absolute pixel coords  
        x1_abs = int(bbox[0])
        y1_abs = int(bbox[1])
        x2_abs = int(bbox[2])
        y2_abs = int(bbox[3])
        
        # Crop with normalized interpretation
        crop_norm = img.crop((
            max(0, x1_norm), max(0, y1_norm),
            min(w, x2_norm), min(h, y2_norm)
        ))
        crop_norm.save(f'test-files/_crop_test/page{page_idx+1}_fig{i+1}_norm.png')
        
        # Crop with absolute interpretation
        crop_abs = img.crop((
            max(0, x1_abs), max(0, y1_abs),
            min(w, x2_abs), min(h, y2_abs)
        ))
        crop_abs.save(f'test-files/_crop_test/page{page_idx+1}_fig{i+1}_abs.png')
        
        print(f'    Norm crop: ({x1_norm},{y1_norm})-({x2_norm},{y2_norm}) size={crop_norm.size}')
        print(f'    Abs crop:  ({x1_abs},{y1_abs})-({x2_abs},{y2_abs}) size={crop_abs.size}')
    
    print()

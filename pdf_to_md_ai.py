"""
PDF to Markdown —— AI 增强转换（Qwen 3.5 Plus 多模态）

核心思路：
  PDF 每页渲染为图片 → 连同提取的文本层一起发给 Qwen → 模型直接输出 Markdown
  代码只负责喂数据、收结果、提取/保存图片，不做任何规则后处理。

使用方式：
  设置环境变量 DASHSCOPE_API_KEY，然后运行：
    python pdf_to_md_ai.py <pdf_path>
  或直接修改底部的 pdf_file 路径运行。
"""

import base64
import fitz  # PyMuPDF
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


MODEL = "qwen3.5-plus"


# ── PDF 工具函数 ────────────────────────────────────────────────────

def render_page_to_image(page, dpi=200):
    """将 PDF 页面渲染为 PNG 字节"""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def extract_page_text(page):
    """提取页面的纯文本"""
    return page.get_text().strip()


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


# ── 核心：调用 Qwen 多模态 ─────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一个 PDF 转 Markdown 的专业助手。你的任务是将用户提供的 PDF 页面截图精确转写成结构良好的 Markdown。

## 最重要的原则：绝不编造内容
- 只输出这一页**实际可见**的文字内容。
- 如果页面上主要是截图、几乎没有文字，就只放图片引用，不要编写任何描述。
- 绝对不要"脑补"或"猜测"截图里的设置含义、操作步骤等——如果原文没写，你就不写。
- 不要给没有编号的内容段落自行添加编号。

## 标题层级规则（严格遵守）
- `#` 仅用于文档大标题（整个文档只有一个）。
- `##` 用于主步骤编号（原文中明确写了 "1." "2." ... "7." 这类数字+句号的步骤）。
- `###` 用于子步骤编号（原文中明确写了 "1）" "2）" ... 这类数字+括号的步骤）。
- 非编号的段落小标题用 `####` 或加粗。
- 不要为原文中没有的步骤编号创造新编号。

## 内容规则
1. 完整保留所有文字内容，修正 PDF 换行造成的断句，合并成通顺的句子。
2. 页面中的截图/界面图，用 `![](images/{图片文件名})` 在合适位置引用。
3. 不要输出解释性文字，只输出 Markdown 正文。
4. 菜单路径用 `→` 连接，如 `Verification → DRC → Errors`。\
"""


def convert_page_with_ai(client, page_png_bytes, page_text, image_filenames,
                          page_num, total_pages, prev_md_tail=""):
    """调用 Qwen 多模态模型转换单页"""

    # 构造图片列表提示
    if image_filenames:
        img_list = '\n'.join(f'  - {f}' for f in image_filenames)
        img_hint = f"本页包含以下嵌入图片，请在合适位置用 ![](images/文件名) 引用：\n{img_list}"
    else:
        img_hint = "本页没有嵌入的图片。"

    # 构造文本辅助提示
    if page_text:
        text_hint = f"PDF 提取的原始文本层（供参考，可能有断行问题）：\n```\n{page_text}\n```"
    else:
        text_hint = "本页未提取到文本层，需完全依赖截图内容。"

    # 上下文提示：上一页末尾的 Markdown（帮助保持标题层级一致性）
    if prev_md_tail:
        context_hint = f"上一页末尾的 Markdown（帮你了解当前上下文和标题层级）：\n```\n{prev_md_tail}\n```"
    else:
        context_hint = "这是文档的第一页。"

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(page_png_bytes).decode()}"
            },
        },
        {
            "type": "text",
            "text": (
                f"这是 PDF 的第 {page_num + 1}/{total_pages} 页。\n\n"
                f"{context_hint}\n\n"
                f"{img_hint}\n\n"
                f"{text_hint}\n\n"
                "请将这一页转换为 Markdown，直接输出结果，延续上一页的标题层级。"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=4096,
        extra_body={"enable_thinking": False},
    )

    return response.choices[0].message.content.strip()


# ── 主流程 ──────────────────────────────────────────────────────────

def pdf_to_markdown_ai(pdf_path, output_dir=None):
    """AI 增强 PDF 转 Markdown 主函数"""
    pdf_path = Path(pdf_path)

    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    print(f"📄 正在转换: {pdf_path.name}")
    print(f"📑 总页数: {total_pages}")
    print(f"🤖 模型: {MODEL}")
    print(f"📁 输出目录: {output_dir}\n")

    all_md_parts = []

    for page_num in range(total_pages):
        page = doc[page_num]

        # 1. 渲染页面为图片
        page_png = render_page_to_image(page)

        # 2. 提取文本层
        page_text = extract_page_text(page)

        # 3. 提取并保存嵌入的图片
        image_filenames = extract_and_save_images(doc, page, page_num, str(images_dir))

        # 4. 调用 AI 转换（传入上一页末尾上下文）
        prev_tail = ""
        if all_md_parts:
            prev_tail = all_md_parts[-1][-500:]

        print(f"  🔄 第 {page_num + 1}/{total_pages} 页...", end=" ", flush=True)
        start = time.time()
        try:
            md_part = convert_page_with_ai(
                client, page_png, page_text, image_filenames,
                page_num, total_pages, prev_md_tail=prev_tail,
            )
            elapsed = time.time() - start
            print(f"✅ ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ ({elapsed:.1f}s) {e}")
            # 降级：使用原始文本
            md_part = page_text if page_text else f"<!-- 第{page_num+1}页转换失败 -->"

        all_md_parts.append(md_part)

    doc.close()

    # 5. 拼接所有页面
    md_content = "\n\n".join(all_md_parts)

    # 清理 Qwen 可能输出的 markdown 代码围栏
    md_content = re.sub(r'^```markdown\s*\n', '', md_content)
    md_content = re.sub(r'\n```\s*$', '', md_content)
    # 也清理中间页面可能出现的围栏
    md_content = re.sub(r'\n```\s*\n\n```markdown\s*\n', '\n\n', md_content)

    # 修复模型生成的坏图片引用格式，如 ![](images/page9[](images/page9_img1.png)
    md_content = re.sub(
        r'!\[\]\(images/page\d+\[]\(images/(page\d+_img\d+\.png)\)',
        r'![](images/\1)',
        md_content,
    )

    md_path = output_dir / f"{pdf_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    img_count = len(list(images_dir.glob("*.png")))
    print(f"\n🎉 转换完成！")
    print(f"   Markdown: {md_path}")
    print(f"   图片: {images_dir} ({img_count}张)")

    return str(md_path)


# ── 入口 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = r"d:\python_programs\mdfy\test-files\（mjy）Expert 使用教程.pdf"

    pdf_to_markdown_ai(pdf_file)

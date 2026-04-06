"""CLI 入口：python -m mdfy <pdf_path>"""

import argparse

from .config import AVAILABLE_MODELS, DEFAULT_MODEL, AVAILABLE_MODES, DEFAULT_MODE
from .pipeline import pdf_to_markdown_ai


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
    parser.add_argument("--mode", choices=AVAILABLE_MODES,
                        default=DEFAULT_MODE,
                        help=f"转换模式（默认: {DEFAULT_MODE}）：pipeline=管线模式, vision=纯视觉模式")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pdf_to_markdown_ai(args.pdf, output_dir=args.output, model=args.model, mode=args.mode)

"""CLI 入口：python -m mdfy <pdf_path>"""

import argparse

from .config import AVAILABLE_MODELS, DEFAULT_MODEL
from .orchestrator import pdf_to_markdown_ai


def parse_args():
    parser = argparse.ArgumentParser(
        prog="mdfy",
        description="mdfy — 纯视觉 PDF → Markdown 转换（基于 Qwen 多模态模型）",
    )
    parser.add_argument("pdf", help="要转换的 PDF 文件路径")
    parser.add_argument("--model", "-m", choices=AVAILABLE_MODELS, default=DEFAULT_MODEL,
                        help=f"模型选择（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出目录（默认: PDF 同目录下的同名文件夹）")
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_to_markdown_ai(args.pdf, output_dir=args.output, model=args.model)


if __name__ == "__main__":
    main()

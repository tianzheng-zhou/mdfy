"""mdfy 启动脚本 — Web 服务 / CLI 转换统一入口。"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="mdfy — AI 增强 PDF 转 Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python run.py serve                  # 启动 Web 服务 (默认 0.0.0.0:23504)\n"
            "  python run.py serve -p 5000           # 指定端口\n"
            "  python run.py convert doc.pdf          # CLI 转换单个 PDF\n"
            "  python run.py convert doc.pdf -m qwen3.6-plus --mode vision\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # ── serve ──
    srv = sub.add_parser("serve", help="启动 Web 服务")
    srv.add_argument("-H", "--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    srv.add_argument("-p", "--port", type=int, default=23504, help="监听端口 (默认 23504)")
    srv.add_argument("--debug", action="store_true", help="调试模式")

    # ── convert ──
    from mdfy.config import AVAILABLE_MODELS, DEFAULT_MODEL, AVAILABLE_MODES, DEFAULT_MODE

    cvt = sub.add_parser("convert", help="CLI 转换 PDF")
    cvt.add_argument("pdf", help="PDF 文件路径")
    cvt.add_argument("-m", "--model", choices=AVAILABLE_MODELS, default=DEFAULT_MODEL,
                     help=f"模型 (默认 {DEFAULT_MODEL})")
    cvt.add_argument("-o", "--output", default=None, help="输出目录")
    cvt.add_argument("--mode", choices=AVAILABLE_MODES, default=DEFAULT_MODE,
                     help=f"模式 (默认 {DEFAULT_MODE})")

    args = parser.parse_args()

    if args.command == "serve":
        from web_app import app
        app.run(host=args.host, port=args.port, debug=args.debug)

    elif args.command == "convert":
        from mdfy import pdf_to_markdown_ai
        pdf_to_markdown_ai(args.pdf, output_dir=args.output, model=args.model, mode=args.mode)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

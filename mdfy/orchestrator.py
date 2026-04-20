"""主流程编排：render → detect → convert → stitch → postprocess。纯视觉版。"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz

from .config import (
    DEFAULT_MODEL, RENDER_DPI, OCR_IMAGE_MAX_SIDE,
    FIGURE_DETECT_WORKERS, PREV_TAIL_CHARS_CONVERT,
)
from .client import get_client
from .pdf_render import render_page_to_image, prepare_image_for_model
from .figure_detect import (
    detect_and_refine_page,
    filter_cross_page_decorative,
    compute_image_positions,
)
from .convert import infer_document_context, convert_page, review_page_quality
from .stitch import build_outline, join_pages_smart
from .postprocess import postprocess_markdown


def pdf_to_markdown_ai(pdf_path, output_dir=None, model=None):
    """纯视觉 PDF → Markdown 主流程。

    参数：
        pdf_path: PDF 文件路径
        output_dir: 输出目录，默认 PDF 同目录下同名文件夹
        model: 模型名，默认 DEFAULT_MODEL

    流程（Phase A → H）：
        A. 顺序渲染所有页为 PNG + 压缩副本
        A.5 首页推断文档全局上下文
        B. 并行图片检测 + 裁切 + AI 校验
        B.5 跨页装饰图过滤（logo 等）
        C. 计算每页图片位置与覆盖率
        D. 顺序逐页 AI 转换 + 质量审查 + 重试
        E. 保底：未引用的裁切图追加到页末
        F. 跨页 LLM 拼接
        G. 极简后处理
        H. 写 .md + images/
    """
    pdf_path = Path(pdf_path)
    model = model or DEFAULT_MODEL

    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    # 清理旧图片避免残留
    if images_dir.exists():
        for old_img in images_dir.glob("*.png"):
            try:
                old_img.unlink()
            except OSError:
                pass
    images_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(model)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    print(f"[*] 正在转换: {pdf_path.name}")
    print(f"    总页数: {total_pages}")
    print(f"    模型: {model}")
    print(f"    输出目录: {output_dir}\n")

    # ══════════════════════════════════════════════════════════════════
    # Phase A: 顺序渲染所有页
    # ══════════════════════════════════════════════════════════════════
    print(f"  Phase A: 渲染 {total_pages} 页...", flush=True)
    page_data = []
    for page_num in range(total_pages):
        page = doc[page_num]
        page_png = render_page_to_image(page, dpi=RENDER_DPI)
        page_api_image, page_api_mime, _ = prepare_image_for_model(
            page_png, max_side=OCR_IMAGE_MAX_SIDE,
        )
        page_data.append({
            'page_num': page_num,
            'page_png': page_png,
            'page_api_image': page_api_image,
            'page_api_mime': page_api_mime,
            'images_dir': str(images_dir),
        })
        print(f"\r  Phase A: 渲染页面 [{page_num + 1}/{total_pages}]", end="", flush=True)
    print()

    doc.close()

    # ══════════════════════════════════════════════════════════════════
    # Phase A.5: 推断文档全局上下文
    # ══════════════════════════════════════════════════════════════════
    doc_context = infer_document_context(
        client, model,
        page_data[0]['page_api_image'], page_data[0]['page_api_mime'],
    )
    if doc_context:
        print(f"  文档上下文: {doc_context}", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # Phase B: 并行图片检测 + 裁切 + AI 校验
    # ══════════════════════════════════════════════════════════════════
    print(f"  Phase B: 并行检测图表...", flush=True)
    page_figures = {}  # page_num → [(filename, desc, pixel_bbox), ...]
    with ThreadPoolExecutor(max_workers=FIGURE_DETECT_WORKERS) as pool:
        futures = {}
        for pd in page_data:
            fut = pool.submit(
                detect_and_refine_page,
                client, model, pd['page_png'], pd['page_num'],
                images_dir, doc_context,
            )
            futures[pd['page_num']] = fut

        for page_num, fut in futures.items():
            try:
                fig_results = fut.result()
                if fig_results:
                    page_figures[page_num] = fig_results
                    print(f"    第{page_num + 1}页: {len(fig_results)} 张图", flush=True)
            except Exception as e:
                print(f"    ⚠ 第{page_num + 1}页图片检测失败: {e}")

    # ══════════════════════════════════════════════════════════════════
    # Phase B.5: 跨页装饰图过滤
    # ══════════════════════════════════════════════════════════════════
    removed = filter_cross_page_decorative(page_figures, page_data, total_pages)
    if removed:
        print(f"  [decorative filter: removed {removed} repeated small images]", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # Phase C: 计算图片位置与覆盖率
    # ══════════════════════════════════════════════════════════════════
    page_image_info = {}  # page_num → (filenames, positions, coverage)
    for pd in page_data:
        page_num = pd['page_num']
        fig_results = page_figures.get(page_num)
        if fig_results:
            image_filenames = [f[0] for f in fig_results]
            fig_bboxes = {f[0]: f[2] for f in fig_results}
            image_filenames, image_positions, img_coverage = compute_image_positions(
                image_filenames, pd['page_png'], fig_bboxes,
            )
            page_image_info[page_num] = (image_filenames, image_positions, img_coverage)
        else:
            page_image_info[page_num] = ([], {}, 0)

    # ══════════════════════════════════════════════════════════════════
    # Phase D: 顺序 AI 转换（需要 prev_tail 跨页上下文）
    # ══════════════════════════════════════════════════════════════════
    all_md_parts = []
    page_images_map = {}  # page_num → [filename, ...]
    for pd in page_data:
        page_num = pd['page_num']
        image_filenames, image_positions, img_coverage = page_image_info[page_num]

        prev_tail = ""
        if all_md_parts:
            prev_tail = all_md_parts[-1][-PREV_TAIL_CHARS_CONVERT:]
        outline = build_outline(all_md_parts)

        print(f"  [{page_num + 1}/{total_pages}]", end=" ", flush=True)
        if image_filenames:
            print(f"[img:{len(image_filenames)}]", end=" ", flush=True)
        start = time.time()
        try:
            md_part = convert_page(
                client, model, pd['page_api_image'],
                page_num=page_num, total_pages=total_pages,
                prev_md_tail=prev_tail, outline=outline, doc_context=doc_context,
                image_filenames=image_filenames,
                image_positions=image_positions,
                image_coverage=img_coverage,
                page_image_mime=pd['page_api_mime'],
            )
            if not md_part or not md_part.strip():
                print(f"⚠空 ({time.time() - start:.1f}s) 模型返回空内容")
                md_part = f"<!-- 第{page_num + 1}页转换失败 -->"
            else:
                elapsed = time.time() - start
                # 质量审查 → 低分重试
                issues, qscore = review_page_quality(
                    client, model, pd['page_api_image'], pd['page_api_mime'],
                    md_part, image_filenames, page_num,
                )
                if issues and qscore < 60:
                    print(f"⚠审({qscore}) {'; '.join(issues)} → 重试", end=" ", flush=True)
                    try:
                        md_part2 = convert_page(
                            client, model, pd['page_api_image'],
                            page_num=page_num, total_pages=total_pages,
                            prev_md_tail=prev_tail, outline=outline, doc_context=doc_context,
                            image_filenames=image_filenames,
                            image_positions=image_positions,
                            image_coverage=img_coverage,
                            page_image_mime=pd['page_api_mime'],
                        )
                        if md_part2 and md_part2.strip():
                            issues2, qscore2 = review_page_quality(
                                client, model, pd['page_api_image'], pd['page_api_mime'],
                                md_part2, image_filenames, page_num,
                            )
                            if qscore2 > qscore:
                                md_part = md_part2
                                print(f"✓审({qscore2})", end=" ", flush=True)
                            else:
                                print(f"✗审({qscore2})保留原版", end=" ", flush=True)
                    except Exception:
                        print("重试失败,保留原版", end=" ", flush=True)
                print(f"OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ ({elapsed:.1f}s) {e}")
            md_part = f"<!-- 第{page_num + 1}页转换失败: {e} -->"

        all_md_parts.append(md_part)
        if image_filenames:
            page_images_map[page_num] = image_filenames

    # ══════════════════════════════════════════════════════════════════
    # Phase E: 保底——未引用的裁切图追加到该页末
    # ══════════════════════════════════════════════════════════════════
    for pg, filenames in page_images_map.items():
        md_part = all_md_parts[pg]
        for fname in filenames:
            if fname not in md_part:
                all_md_parts[pg] += f"\n\n![](images/{fname})"

    # ══════════════════════════════════════════════════════════════════
    # Phase F: 跨页 LLM 拼接
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  拼接 {len(all_md_parts)} 页...", end="", flush=True)
    md_content = join_pages_smart(all_md_parts, client=client, stitch_model=model)

    # 内容完整性回退：stitch 若丢了整页图片引用，回退到纯 \n\n 拼接
    missing_pages = []
    for pg, filenames in page_images_map.items():
        if not any(fname in md_content for fname in filenames):
            missing_pages.append(pg + 1)
    if missing_pages:
        print(f"\n  ⚠ 检测到 {len(missing_pages)} 页图片全部丢失: {missing_pages}，执行回退拼接")
        md_content = "\n\n".join(part for part in all_md_parts if part.strip())

    # ══════════════════════════════════════════════════════════════════
    # Phase G: 极简后处理
    # ══════════════════════════════════════════════════════════════════
    md_content = postprocess_markdown(md_content, output_dir)

    # ══════════════════════════════════════════════════════════════════
    # Phase H: 写出
    # ══════════════════════════════════════════════════════════════════
    md_path = output_dir / f"{pdf_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    img_count = len(list(images_dir.glob("*.png")))
    print(f"\n[DONE] 转换完成！")
    print(f"   Markdown: {md_path}")
    print(f"   图片: {images_dir} ({img_count} 张)")

    return str(md_path)

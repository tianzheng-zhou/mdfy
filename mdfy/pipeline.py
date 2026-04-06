"""主流程编排：pipeline 模式和 vision 模式的入口。"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz

from .config import (
    AVAILABLE_MODELS, AVAILABLE_MODES, DEFAULT_MODEL, DEFAULT_MODE,
    OCR_IMAGE_MAX_SIDE, FIGURE_DETECT_WORKERS as _FIGURE_DETECT_WORKERS,
)
from .client import get_client
from .pdf_utils import (
    render_page_to_image, prepare_image_for_model, extract_page_text,
    _detect_pdf_type,
)
from .image_extract import extract_and_save_images, _find_decorative_xrefs
from .image_merge import (
    _merge_ai_and_embedded_images, _compute_image_positions,
    _compute_text_in_images,
)
from .image_detect import detect_page_figures
from .convert import convert_page_with_ai, _convert_page_vision, _detect_figures_for_page
from .dedup import _dedup_model_repetition, _review_page_quality
from .stitch import _build_outline, _join_pages_smart
from .postprocess import postprocess_markdown


def _vision_mode_convert(pdf_path, output_dir, model, client):
    """Vision 模式主流程：渲染所有页面为图片 → AI 检测并裁切图表 → 纯视觉 AI 转换。

    与 pipeline 模式的区别：
    - 不提取 PDF 文本层（纯视觉 OCR）
    - 不提取嵌入图片（所有图片从渲染页面裁切）
    - 不做嵌入图与 AI 检测图的合并
    - 复用 detect_page_figures + crop_and_save_figures + _ai_verify_and_refine_crops
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    images_dir = output_dir / "images"

    # ── Phase A: 渲染所有页面为图片（顺序，快速 fitz 操作）──
    print(f"  Vision Phase A: 渲染 {total_pages} 页为图片...", flush=True)
    page_data = []
    for page_num in range(total_pages):
        page = doc[page_num]
        page_png = render_page_to_image(page, dpi=200)
        page_api_image, page_api_mime, _ = prepare_image_for_model(
            page_png, max_side=OCR_IMAGE_MAX_SIDE,
        )
        page_data.append({
            'page_num': page_num,
            'page_png': page_png,
            'page_api_image': page_api_image,
            'page_api_mime': page_api_mime,
        })
        print(f"\r  Vision Phase A: 渲染页面 [{page_num + 1}/{total_pages}]", end="", flush=True)
    print()

    doc.close()

    # ── Phase B: 并行图片检测 + 裁切（复用 pipeline 的检测管线）──
    print(f"  Vision Phase B: 并行检测图表...", flush=True)
    page_figures = {}  # page_num -> fig_results
    with ThreadPoolExecutor(max_workers=_FIGURE_DETECT_WORKERS) as pool:
        futures = {}
        for pd in page_data:
            fut = pool.submit(
                _detect_figures_for_page,
                client, model, pd['page_png'], pd['page_num'],
                images_dir, "scanned",  # 视觉模式等同扫描件：无嵌入图
            )
            futures[pd['page_num']] = fut

        for page_num, fut in futures.items():
            try:
                fig_results = fut.result()
                if fig_results:
                    page_figures[page_num] = fig_results
                    print(f"    第{page_num+1}页: {len(fig_results)}张图", flush=True)
            except Exception as e:
                print(f"    ⚠ 第{page_num+1}页图片检测失败: {e}")

    # ── Phase C: 计算图片位置（快速，无 fitz 依赖）──
    page_image_info = {}  # page_num -> (filenames, positions, coverage)
    for pd in page_data:
        page_num = pd['page_num']
        fig_results = page_figures.get(page_num)
        if fig_results:
            image_filenames = [f[0] for f in fig_results]
            fig_bboxes = {f[0]: f[2] for f in fig_results}
            image_filenames, image_positions, img_coverage, _ = _compute_image_positions(
                image_filenames, pd['page_png'], fig_bboxes=fig_bboxes, page=None)
            page_image_info[page_num] = (image_filenames, image_positions, img_coverage)
        else:
            page_image_info[page_num] = ([], {}, 0)

    # ── Phase D: 顺序 AI 转换（需要 prev_tail 跨页上下文）──
    all_md_parts = []
    page_images_map = {}  # page_num -> [filename, ...]
    for pd in page_data:
        page_num = pd['page_num']
        image_filenames, image_positions, img_coverage = page_image_info[page_num]

        prev_tail = ""
        if all_md_parts:
            prev_tail = all_md_parts[-1][-1200:]
        outline = _build_outline(all_md_parts)

        print(f"  [{page_num + 1}/{total_pages}]", end=" ", flush=True)
        if image_filenames:
            print(f"[img:{len(image_filenames)}]", end=" ", flush=True)
        start = time.time()
        try:
            md_part = _convert_page_vision(
                client, model, pd['page_api_image'],
                page_num, total_pages,
                prev_md_tail=prev_tail, outline=outline,
                page_image_mime=pd['page_api_mime'],
                image_filenames=image_filenames,
                image_positions=image_positions,
                image_coverage=img_coverage,
            )
            if not md_part or not md_part.strip():
                print(f"⚠空 ({time.time() - start:.1f}s) 模型返回空内容")
                md_part = f"<!-- 第{page_num + 1}页转换失败 -->"
            else:
                elapsed = time.time() - start
                # 页级质量自审
                issues, qscore = _review_page_quality(md_part, image_filenames, page_num)
                if issues and qscore < 60:
                    print(f"⚠审({qscore}) {'; '.join(issues)} → 重试", end=" ", flush=True)
                    try:
                        md_part2 = _convert_page_vision(
                            client, model, pd['page_api_image'],
                            page_num, total_pages,
                            prev_md_tail=prev_tail, outline=outline,
                            page_image_mime=pd['page_api_mime'],
                            image_filenames=image_filenames,
                            image_positions=image_positions,
                            image_coverage=img_coverage,
                        )
                        if md_part2 and md_part2.strip():
                            issues2, qscore2 = _review_page_quality(md_part2, image_filenames, page_num)
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

        # 页内去重
        md_part = _dedup_model_repetition(md_part)
        all_md_parts.append(md_part)
        if image_filenames:
            page_images_map[page_num] = image_filenames

    return all_md_parts, page_images_map


# ── 主流程 ──────────────────────────────────────────────────────────

def pdf_to_markdown_ai(pdf_path, output_dir=None, model=None, mode=None):
    """AI 增强 PDF 转 Markdown 主函数

    参数:
        pdf_path: PDF 文件路径
        output_dir: 输出目录，默认与 PDF 同目录同名文件夹
        model: 模型名（默认 qwen3.5-plus）
        mode: 转换模式 - "pipeline"（默认，文本提取+图片检测管线）或 "vision"（纯视觉）
    """
    pdf_path = Path(pdf_path)
    model = model or DEFAULT_MODEL
    mode = mode or DEFAULT_MODE

    if output_dir is None:
        output_dir = pdf_path.parent / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    # 清理旧的图片文件，避免前次运行的残留
    if images_dir.exists():
        for old_img in images_dir.glob("*.png"):
            old_img.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # 检测 PDF 类型
    pdf_type = _detect_pdf_type(doc)

    print(f"[*] 正在转换: {pdf_path.name}")
    print(f"    总页数: {total_pages}")
    print(f"    模式: {mode} ({'纯视觉' if mode == 'vision' else '管线'})")
    print(f"    类型: {pdf_type} ({'扫描件OCR' if pdf_type == 'scanned' else '数字PDF'})")
    print(f"    模型: {model}")
    print(f"    输出目录: {output_dir}\n")

    # ══════════════════════════════════════════════════════════════
    # Vision 模式：纯视觉转换，跳过管线
    # ══════════════════════════════════════════════════════════════
    if mode == "vision":
        doc.close()
        all_md_parts, page_images_map = _vision_mode_convert(pdf_path, output_dir, model, client)

        # 保底：检查每页裁切的图片是否都被模型引用了，未引用的追加到该页 MD 末尾
        for pg, filenames in page_images_map.items():
            md_part = all_md_parts[pg]
            for fname in filenames:
                if fname not in md_part:
                    all_md_parts[pg] += f"\n\n![](images/{fname})"

        # 5. 智能拼接所有页面
        print(f"\n  拼接 {len(all_md_parts)} 页...", end="", flush=True)
        md_content = _join_pages_smart(all_md_parts, client=client)

        # 5.1 内容完整性检查：检测 stitch/后处理是否丢失了整页内容
        _missing_pages = []
        for pg, filenames in page_images_map.items():
            if not any(fname in md_content for fname in filenames):
                _missing_pages.append(pg + 1)
        if _missing_pages:
            print(f"\n  ⚠ 检测到 {len(_missing_pages)} 页图片全部丢失: {_missing_pages}，执行回退拼接")
            md_content = "\n\n".join(part for part in all_md_parts if part.strip())

    else:
        # ══════════════════════════════════════════════════════════
        # Pipeline 模式：完整管线
        # ══════════════════════════════════════════════════════════

        all_md_parts = []
        page_images_map = {}  # page_num -> [filename, ...] 跟踪每页裁切的图片

        # 预扫描：检测跨页重复的装饰性图片（logo/页眉页脚图标等）
        decorative_xrefs = _find_decorative_xrefs(doc)

        # ══════════════════════════════════════════════════════════════
        # Phase A: 预处理所有页面（顺序执行，fitz 操作，快速）
        # ══════════════════════════════════════════════════════════════
        page_data = []
        print("  Phase A: 预处理页面...", flush=True)
        for page_num in range(total_pages):
            page = doc[page_num]
            page_png = render_page_to_image(page)
            page_api_image, page_api_mime, _ = prepare_image_for_model(
                page_png, max_side=OCR_IMAGE_MAX_SIDE,
            )
            page_text = extract_page_text(page)

            embedded_filenames = []
            if pdf_type != "scanned":
                embedded_filenames = extract_and_save_images(
                    doc, page, page_num, str(images_dir),
                    decorative_xrefs=decorative_xrefs,
                )

            page_data.append({
                'page_num': page_num,
                'page_png': page_png,
                'page_api_image': page_api_image,
                'page_api_mime': page_api_mime,
                'page_text': page_text,
                'embedded_filenames': embedded_filenames,
            })
            print(f"\r  Phase A: 预处理页面 [{page_num+1}/{total_pages}]", end="", flush=True)
        print()

        # ══════════════════════════════════════════════════════════════
        # Phase B+C+D: 并行图片检测 + 顺序转换（流水线）
        #   - 工作线程：并行执行 detect_figures + crop + verify（API 调用）
        #   - 主线程：顺序执行 merge + convert（需要 prev_tail 跨页上下文）
        #   流水线效果：主线程等 convert API 响应时，工作线程处理后续页
        # ══════════════════════════════════════════════════════════════
        with ThreadPoolExecutor(max_workers=_FIGURE_DETECT_WORKERS) as pool:
            # 提交所有页面的图片检测任务
            futures = {}
            for pd in page_data:
                fut = pool.submit(
                    _detect_figures_for_page,
                    client, model, pd['page_png'], pd['page_num'],
                    images_dir, pdf_type, pd['embedded_filenames'],
                )
                futures[pd['page_num']] = fut

            # 顺序处理每页：等待图片检测 → merge → convert
            for page_num in range(total_pages):
                pd = page_data[page_num]
                page = doc[page_num]

                # 等待本页的图片检测完成
                try:
                    fig_results = futures[page_num].result()
                except Exception as e:
                    print(f"  ⚠ 第{page_num+1}页图片检测失败: {e}")
                    fig_results = None

                # Phase C: merge + compute_positions（快速，fitz 操作）
                if pdf_type == "scanned":
                    if fig_results:
                        image_filenames = [f[0] for f in fig_results]
                        fig_bboxes = {f[0]: f[2] for f in fig_results}
                        print(f"[img:{len(image_filenames)}]", end=" ", flush=True)
                    else:
                        image_filenames = []
                        fig_bboxes = {}
                    image_filenames, image_positions, img_coverage, img_bboxes_pdf = _compute_image_positions(
                        image_filenames, pd['page_png'], fig_bboxes=fig_bboxes, page=page)
                    text_in_images = _compute_text_in_images(page, img_bboxes_pdf)
                else:
                    embedded_filenames = pd['embedded_filenames']
                    if fig_results:
                        ai_filenames = [f[0] for f in fig_results]
                        ai_pixel_bboxes = [f[2] for f in fig_results]
                        image_filenames = _merge_ai_and_embedded_images(
                            page, doc, ai_filenames, ai_pixel_bboxes, embedded_filenames,
                            pd['page_png'], str(images_dir),
                        )
                        fig_bboxes = {f[0]: f[2] for f in fig_results}
                        print(f"[AI:{len(ai_filenames)} embed:{len(embedded_filenames)}->"
                              f"{len(image_filenames) - len(ai_filenames)}]", end=" ", flush=True)
                    else:
                        image_filenames = embedded_filenames
                        fig_bboxes = {}
                    image_filenames, image_positions, img_coverage, img_bboxes_pdf = _compute_image_positions(
                        image_filenames, pd['page_png'], fig_bboxes=fig_bboxes, page=page)
                    text_in_images = _compute_text_in_images(page, img_bboxes_pdf)

                # Phase D: 构建上下文 + AI 转换（顺序，需要 prev_tail）
                prev_tail = ""
                if all_md_parts:
                    tail_len = 1200 if pdf_type == "scanned" else 500
                    prev_tail = all_md_parts[-1][-tail_len:]
                outline = _build_outline(all_md_parts)

                print(f"  [{page_num + 1}/{total_pages}]", end=" ", flush=True)
                start = time.time()
                try:
                    md_part = convert_page_with_ai(
                        client, model, pd['page_api_image'], pd['page_text'], image_filenames,
                        page_num, total_pages,
                        prev_md_tail=prev_tail, outline=outline,
                        pdf_type=pdf_type,
                        page_image_mime=pd['page_api_mime'],
                        image_positions=image_positions,
                        image_coverage=img_coverage,
                        text_in_images=text_in_images,
                    )
                    if not md_part or not md_part.strip():
                        print(f"⚠空 ({time.time() - start:.1f}s) 模型返回空内容，使用文本层回退")
                        md_part = pd['page_text'] if pd['page_text'] else f"<!-- 第{page_num+1}页转换失败 -->"
                    else:
                        elapsed = time.time() - start
                        # 页级质量自审
                        issues, qscore = _review_page_quality(md_part, image_filenames, page_num)
                        if issues and qscore < 60:
                            print(f"⚠审({qscore}) {'; '.join(issues)} → 重试", end=" ", flush=True)
                            try:
                                md_part2 = convert_page_with_ai(
                                    client, model, pd['page_api_image'], pd['page_text'], image_filenames,
                                    page_num, total_pages,
                                    prev_md_tail=prev_tail, outline=outline,
                                    pdf_type=pdf_type,
                                    page_image_mime=pd['page_api_mime'],
                                    image_positions=image_positions,
                                    image_coverage=img_coverage,
                                    text_in_images=text_in_images,
                                )
                                if md_part2 and md_part2.strip():
                                    issues2, qscore2 = _review_page_quality(md_part2, image_filenames, page_num)
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
                    md_part = pd['page_text'] if pd['page_text'] else f"<!-- 第{page_num+1}页转换失败 -->"

                # 页内去重：检测并移除模型自重复（图片密集页面常见）
                md_part = _dedup_model_repetition(md_part)

                all_md_parts.append(md_part)
                if image_filenames:
                    page_images_map[page_num] = image_filenames

        doc.close()

        # 4.5 保底：检查每页裁切的图片是否都被模型引用了，未引用的追加到该页 MD 末尾
        for pg, filenames in page_images_map.items():
            md_part = all_md_parts[pg]
            for fname in filenames:
                if fname not in md_part:
                    all_md_parts[pg] += f"\n\n![](images/{fname})"

        # 5. 智能拼接所有页面：LLM 辅助去重 + 断句合并
        print(f"\n  拼接 {len(all_md_parts)} 页...", end="", flush=True)
        md_content = _join_pages_smart(all_md_parts, client=client)

        # 5.1 内容完整性检查：检测 stitch/后处理是否丢失了整页内容
        _missing_pages = []
        for pg, filenames in page_images_map.items():
            if not any(fname in md_content for fname in filenames):
                _missing_pages.append(pg + 1)
        if _missing_pages:
            print(f"\n  ⚠ 检测到 {len(_missing_pages)} 页图片全部丢失: {_missing_pages}，执行回退拼接")
            # 回退：不使用 LLM stitch，直接 \n\n 拼接
            md_content = "\n\n".join(part for part in all_md_parts if part.strip())

    # ── 后处理 ──
    md_content = postprocess_markdown(md_content, pdf_type, output_dir, page_images_map)

    md_path = output_dir / f"{pdf_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    img_count = len(list(images_dir.glob("*.png")))
    print(f"\n[DONE] 转换完成！")
    print(f"   Markdown: {md_path}")
    print(f"   图片: {images_dir} ({img_count}张)")

    return str(md_path)

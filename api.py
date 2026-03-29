"""
PDF to Markdown — RESTful API Blueprint
提供 /api/v1/ 下的完整 API 服务，供外部应用集成调用。
"""

import io
import os
import time
import uuid
import zipfile
import threading
import secrets
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Blueprint, request, jsonify, send_file,
    send_from_directory, Response, current_app,
)

import sys
from pdf_to_md_ai import pdf_to_markdown_ai, AVAILABLE_MODELS, DEFAULT_MODEL

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

# ── 任务存储（与 web_app 共享） ──────────────────────────────────────

_tasks: dict | None = None  # 延迟绑定，由 init_api() 注入
_upload_dir: Path | None = None


def init_api(tasks_dict: dict, upload_dir: Path):
    """由 web_app 调用，注入共享的任务字典和上传目录"""
    global _tasks, _upload_dir
    _tasks = tasks_dict
    _upload_dir = upload_dir


# ── LogCapture（复用 web_app 中的类） ────────────────────────────────

class _LogCapture(io.TextIOBase):
    def __init__(self):
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def write(self, text: str):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        if text.strip():
            with self._lock:
                self._lines.append(text.rstrip("\n"))
        return len(text)

    def flush(self):
        sys.__stdout__.flush()

    def get_lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)


class _APITaskInfo:
    def __init__(self, pdf_name: str, model: str):
        self.pdf_name = pdf_name
        self.model = model
        self.status = "pending"
        self.log = _LogCapture()
        self.result_md: str | None = None
        self.output_dir: str | None = None
        self.error: str | None = None
        self.start_time = time.time()
        self.end_time: float | None = None


# ── API Key 认证 ────────────────────────────────────────────────────

def _check_api_key():
    """验证 API Key（如果服务端配置了 MDFY_API_KEY 环境变量）"""
    required_key = os.environ.get("MDFY_API_KEY", "")
    if not required_key:
        return True  # 未配置则不强制认证

    # 支持 Header 和 Query Parameter 两种方式
    provided = request.headers.get("X-API-Key") or request.args.get("api_key")
    if not provided:
        return False
    return secrets.compare_digest(provided, required_key)


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _check_api_key():
            return jsonify({
                "success": False,
                "error": "认证失败：请提供有效的 API Key（Header: X-API-Key 或 Query: api_key）",
                "data": None,
            }), 401
        return f(*args, **kwargs)
    return decorated


def _ok(data, status=200):
    return jsonify({"success": True, "data": data, "error": None}), status


def _err(msg, status=400):
    return jsonify({"success": False, "data": None, "error": msg}), status


# ── API 路由 ─────────────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return _ok({"status": "ok", "timestamp": time.time()})


@api_bp.route("/models", methods=["GET"])
@require_api_key
def list_models():
    """列出可用模型"""
    return _ok({
        "models": AVAILABLE_MODELS,
        "default": DEFAULT_MODEL,
    })


@api_bp.route("/convert", methods=["POST"])
@require_api_key
def convert():
    """上传 PDF 并启动异步转换任务"""
    file = request.files.get("file")
    model = request.form.get("model", DEFAULT_MODEL)

    if not file or not file.filename:
        return _err("缺少 PDF 文件，请通过 multipart/form-data 的 'file' 字段上传")
    if not file.filename.lower().endswith(".pdf"):
        return _err("仅支持 PDF 格式文件")
    if model not in AVAILABLE_MODELS:
        return _err(f"不支持的模型: {model}，可选: {', '.join(AVAILABLE_MODELS)}")

    task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:12]
    task_dir = _upload_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    pdf_path = task_dir / safe_name
    file.save(str(pdf_path))

    task = _APITaskInfo(pdf_name=safe_name, model=model)
    _tasks[task_id] = task

    def _run():
        old_stdout = sys.stdout
        sys.stdout = task.log
        try:
            task.status = "converting"
            result = pdf_to_markdown_ai(str(pdf_path), model=model)
            task.result_md = result
            task.output_dir = str(Path(result).parent)
            task.status = "done"
        except Exception as exc:
            task.error = str(exc)
            task.status = "error"
        finally:
            sys.stdout = old_stdout
            task.end_time = time.time()

    threading.Thread(target=_run, daemon=True).start()

    return _ok({
        "task_id": task_id,
        "pdf_name": safe_name,
        "model": model,
        "status": "pending",
    }, 202)


@api_bp.route("/tasks/<task_id>", methods=["GET"])
@require_api_key
def task_status(task_id: str):
    """查询任务状态"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)

    elapsed = None
    if task.end_time:
        elapsed = round(task.end_time - task.start_time, 1)
    elif task.status == "converting":
        elapsed = round(time.time() - task.start_time, 1)

    return _ok({
        "task_id": task_id,
        "pdf_name": task.pdf_name,
        "model": task.model,
        "status": task.status,
        "elapsed_seconds": elapsed,
        "error": task.error,
    })


@api_bp.route("/tasks/<task_id>/progress", methods=["GET"])
@require_api_key
def task_progress(task_id: str):
    """SSE 实时进度推送"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)

    def _stream():
        sent = 0
        while True:
            lines = task.log.get_lines()
            for line in lines[sent:]:
                yield f"data: {line}\n\n"
            sent = len(lines)

            if task.status in ("done", "error"):
                if task.status == "done":
                    elapsed = (task.end_time or time.time()) - task.start_time
                    yield f'event: done\ndata: {{"elapsed": {elapsed:.1f}}}\n\n'
                else:
                    yield f'event: error\ndata: {{"error": "{task.error}"}}\n\n'
                break
            time.sleep(0.5)

    return Response(_stream(), mimetype="text/event-stream")


@api_bp.route("/tasks/<task_id>/result", methods=["GET"])
@require_api_key
def task_result(task_id: str):
    """获取转换结果：Markdown 内容 + 图片列表"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)
    if task.status != "done":
        return _err(f"任务尚未完成，当前状态: {task.status}", 409)

    md_path = task.result_md
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    images_dir = Path(task.output_dir) / "images"
    images = []
    if images_dir.exists():
        images = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() == ".png")

    return _ok({
        "task_id": task_id,
        "markdown": content,
        "images": images,
        "image_count": len(images),
    })


@api_bp.route("/tasks/<task_id>/result/markdown", methods=["GET"])
@require_api_key
def task_result_markdown_only(task_id: str):
    """仅获取 Markdown 文本内容（不含图片列表，适合轻量调用）"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)
    if task.status != "done":
        return _err(f"任务尚未完成，当前状态: {task.status}", 409)

    md_path = task.result_md
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    return _ok({"markdown": content})


@api_bp.route("/tasks/<task_id>/images", methods=["GET"])
@require_api_key
def task_images_list(task_id: str):
    """列出任务生成的所有图片"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)
    if task.status != "done":
        return _err(f"任务尚未完成，当前状态: {task.status}", 409)

    images_dir = Path(task.output_dir) / "images"
    images = []
    if images_dir.exists():
        images = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() == ".png")

    return _ok({
        "images": images,
        "count": len(images),
        "base_url": f"/api/v1/tasks/{task_id}/images/",
    })


@api_bp.route("/tasks/<task_id>/images/<filename>", methods=["GET"])
@require_api_key
def task_image(task_id: str, filename: str):
    """获取指定图片文件"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)
    if task.status != "done":
        return _err(f"任务尚未完成，当前状态: {task.status}", 409)

    images_dir = Path(task.output_dir) / "images"
    safe_name = Path(filename).name
    img_path = images_dir / safe_name
    if not img_path.exists() or not img_path.is_relative_to(images_dir):
        return _err("图片不存在", 404)

    return send_from_directory(str(images_dir), safe_name)


@api_bp.route("/tasks/<task_id>/download", methods=["GET"])
@require_api_key
def task_download_zip(task_id: str):
    """打包下载 Markdown + 图片（ZIP 格式）"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)
    if task.status != "done":
        return _err(f"任务尚未完成，当前状态: {task.status}", 409)

    output_dir = Path(task.output_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        md_file = Path(task.result_md)
        zf.write(md_file, md_file.name)
        images_dir = output_dir / "images"
        if images_dir.exists():
            for img in images_dir.iterdir():
                zf.write(img, f"images/{img.name}")

    buf.seek(0)
    zip_name = Path(task.pdf_name).stem + "_markdown.zip"
    return send_file(buf, as_attachment=True, download_name=zip_name, mimetype="application/zip")


@api_bp.route("/tasks/<task_id>/download/markdown", methods=["GET"])
@require_api_key
def task_download_md(task_id: str):
    """直接下载 Markdown 文件"""
    task = _tasks.get(task_id)
    if not task:
        return _err("任务不存在", 404)
    if task.status != "done":
        return _err(f"任务尚未完成，当前状态: {task.status}", 409)

    return send_file(task.result_md, as_attachment=True)


# ── API 文档路由 ─────────────────────────────────────────────────────

@api_bp.route("/docs", methods=["GET"])
def api_docs():
    """返回 API 文档（JSON 格式）"""
    base = request.host_url.rstrip("/")
    docs = {
        "service": "mdfy - PDF to Markdown API",
        "version": "v1",
        "base_url": f"{base}/api/v1",
        "authentication": {
            "description": "如果服务端设置了 MDFY_API_KEY 环境变量，则需要认证",
            "methods": [
                {"type": "header", "name": "X-API-Key", "example": "X-API-Key: your-key-here"},
                {"type": "query", "name": "api_key", "example": "?api_key=your-key-here"},
            ],
        },
        "response_format": {
            "description": "所有 JSON 响应使用统一格式",
            "schema": {
                "success": "bool - 请求是否成功",
                "data": "object|null - 成功时返回的数据",
                "error": "string|null - 失败时的错误信息",
            },
        },
        "endpoints": [
            {
                "method": "GET",
                "path": "/health",
                "description": "健康检查（不需要认证）",
                "response": {"status": "ok", "timestamp": 1234567890.0},
            },
            {
                "method": "GET",
                "path": "/models",
                "description": "列出可用的 AI 模型",
                "response": {"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL},
            },
            {
                "method": "POST",
                "path": "/convert",
                "description": "上传 PDF 并启动异步转换",
                "content_type": "multipart/form-data",
                "parameters": [
                    {"name": "file", "type": "file", "required": True, "description": "PDF 文件"},
                    {"name": "model", "type": "string", "required": False,
                     "description": f"AI 模型，默认 {DEFAULT_MODEL}，可选: {', '.join(AVAILABLE_MODELS)}"},
                ],
                "response_status": 202,
                "response": {
                    "task_id": "20260329_120000_abc123def456",
                    "pdf_name": "example.pdf",
                    "model": DEFAULT_MODEL,
                    "status": "pending",
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}",
                "description": "查询任务状态",
                "response": {
                    "task_id": "...",
                    "pdf_name": "example.pdf",
                    "model": DEFAULT_MODEL,
                    "status": "pending|converting|done|error",
                    "elapsed_seconds": 12.3,
                    "error": None,
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/progress",
                "description": "SSE 实时进度流（Server-Sent Events）",
                "content_type": "text/event-stream",
                "events": [
                    {"type": "data", "description": "实时日志行"},
                    {"type": "done", "description": "转换完成，payload: {elapsed: 秒数}"},
                    {"type": "error", "description": "转换失败，payload: {error: 错误信息}"},
                ],
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/result",
                "description": "获取完整转换结果（Markdown 内容 + 图片列表）",
                "response": {
                    "task_id": "...",
                    "markdown": "# 文档标题\n\n正文内容...",
                    "images": ["page0_img1.png", "page1_img1.png"],
                    "image_count": 2,
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/result/markdown",
                "description": "仅获取 Markdown 文本（轻量）",
                "response": {"markdown": "# 文档标题\n\n正文内容..."},
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/images",
                "description": "列出所有生成的图片",
                "response": {
                    "images": ["page0_img1.png"],
                    "count": 1,
                    "base_url": "/api/v1/tasks/{task_id}/images/",
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/images/{filename}",
                "description": "获取指定图片文件（返回图片二进制数据）",
                "content_type": "image/png",
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/download",
                "description": "打包下载 Markdown + 图片（ZIP）",
                "content_type": "application/zip",
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/download/markdown",
                "description": "直接下载 Markdown 文件",
                "content_type": "text/markdown",
            },
        ],
        "examples": {
            "curl_convert": (
                f'curl -X POST {base}/api/v1/convert \\\n'
                '  -H "X-API-Key: your-key" \\\n'
                '  -F "file=@document.pdf" \\\n'
                '  -F "model=qwen3.5-plus"'
            ),
            "curl_status": (
                f'curl {base}/api/v1/tasks/TASK_ID \\\n'
                '  -H "X-API-Key: your-key"'
            ),
            "curl_result": (
                f'curl {base}/api/v1/tasks/TASK_ID/result \\\n'
                '  -H "X-API-Key: your-key"'
            ),
            "curl_download": (
                f'curl -OJ {base}/api/v1/tasks/TASK_ID/download \\\n'
                '  -H "X-API-Key: your-key"'
            ),
            "python": (
                'import requests\n\n'
                '# 1. 上传并开始转换\n'
                f'resp = requests.post("{base}/api/v1/convert",\n'
                '    headers={"X-API-Key": "your-key"},\n'
                '    files={"file": open("doc.pdf", "rb")},\n'
                '    data={"model": "qwen3.5-plus"},\n'
                ')\n'
                'task_id = resp.json()["data"]["task_id"]\n\n'
                '# 2. 轮询等待完成\n'
                'import time\n'
                'while True:\n'
                f'    status = requests.get(f"{base}/api/v1/tasks/{{task_id}}",\n'
                '        headers={"X-API-Key": "your-key"}).json()\n'
                '    if status["data"]["status"] == "done":\n'
                '        break\n'
                '    elif status["data"]["status"] == "error":\n'
                '        raise Exception(status["data"]["error"])\n'
                '    time.sleep(2)\n\n'
                '# 3. 获取结果\n'
                f'result = requests.get(f"{base}/api/v1/tasks/{{task_id}}/result",\n'
                '    headers={"X-API-Key": "your-key"}).json()\n'
                'print(result["data"]["markdown"])\n'
            ),
        },
        "status_codes": {
            "200": "请求成功",
            "202": "任务已创建（异步处理中）",
            "400": "请求参数错误",
            "401": "认证失败",
            "404": "资源不存在",
            "409": "任务尚未完成",
        },
    }
    return jsonify(docs)

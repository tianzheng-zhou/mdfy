"""
PDF → Markdown — AI 增强转换 Demo 网站
Flask 后端：文件上传、后台转换、SSE 进度推送、结果预览 & 下载
"""

import io
import os
import socket
import sys
import uuid
import time
import zipfile
import threading
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory, Response,
)

from mdfy import pdf_to_markdown_ai, AVAILABLE_MODELS, DEFAULT_MODEL


# ── Flask 配置 ──────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── 任务管理 ────────────────────────────────────────────────────────

tasks: dict[str, "TaskInfo"] = {}


class LogCapture(io.TextIOBase):
    """线程内捕获 stdout，同时转发到真实 stdout"""

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


class TaskInfo:
    def __init__(self, pdf_name: str, model: str):
        self.pdf_name = pdf_name
        self.model = model
        self.status = "pending"        # pending → converting → done / error
        self.log = LogCapture()
        self.result_md: str | None = None
        self.output_dir: str | None = None
        self.error: str | None = None
        self.start_time = time.time()
        self.end_time: float | None = None


# ── 路由 ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """接收 PDF 上传，启动后台转换"""
    file = request.files.get("pdf")
    model = request.form.get("model", DEFAULT_MODEL)

    if not file or not file.filename:
        return jsonify({"error": "请选择 PDF 文件"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "仅支持 PDF 格式"}), 400
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL

    task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:12]
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    pdf_path = task_dir / safe_name
    file.save(str(pdf_path))

    task = TaskInfo(pdf_name=safe_name, model=model)
    tasks[task_id] = task

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
    return jsonify({"task_id": task_id})


@app.route("/progress/<task_id>")
def progress(task_id: str):
    """SSE 端点：实时推送转换日志"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404

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
                    yield f"event: done\ndata: {{\"elapsed\": {elapsed:.1f}}}\n\n"
                else:
                    yield f"event: error\ndata: {task.error}\n\n"
                break
            time.sleep(0.5)

    return Response(_stream(), mimetype="text/event-stream")


@app.route("/result/<task_id>")
def result(task_id: str):
    """返回转换结果：Markdown 内容 + 图片列表"""
    task = tasks.get(task_id)
    if not task or task.status != "done":
        return jsonify({"error": "结果不可用"}), 404

    md_path = task.result_md
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    images_dir = Path(task.output_dir) / "images"
    images = []
    if images_dir.exists():
        images = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() == ".png")

    return jsonify({
        "markdown": content,
        "images": images,
        "task_id": task_id,
    })


@app.route("/image/<task_id>/<filename>")
def serve_image(task_id: str, filename: str):
    """提供转换生成的图片"""
    task = tasks.get(task_id)
    if not task or task.status != "done":
        return "Not found", 404

    images_dir = Path(task.output_dir) / "images"
    safe_name = Path(filename).name
    img_path = images_dir / safe_name
    if not img_path.exists() or not img_path.is_relative_to(images_dir):
        return "Not found", 404

    return send_from_directory(str(images_dir), safe_name)


@app.route("/download/<task_id>")
def download_zip(task_id: str):
    """打包下载 Markdown + 图片"""
    task = tasks.get(task_id)
    if not task or task.status != "done":
        return "Not found", 404

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


@app.route("/download_md/<task_id>")
def download_md(task_id: str):
    """直接下载 Markdown 文件"""
    task = tasks.get(task_id)
    if not task or task.status != "done":
        return "Not found", 404
    return send_file(task.result_md, as_attachment=True)


def get_lan_ip() -> str:
    """尽量获取当前机器在局域网中的可访问 IP。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


# ── 启动 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "23504"))

    print(f"[*] 本机访问: http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"[*] 局域网访问: http://{get_lan_ip()}:{port}")

    app.run(host=host, port=port, debug=False, threaded=True)

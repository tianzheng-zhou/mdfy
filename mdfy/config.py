"""全局常量与模型配置。纯视觉版：无 pipeline/vision 模式区分。"""

# 模型名以 "gemini/" 前缀开头 → 走本地 Gemini 聚合代理
# 其他（裸名）→ 走阿里云 DashScope（OpenAI 兼容端点）
AVAILABLE_MODELS = [
    "qwen3.5-flash",
    "qwen3.5-plus",
    "qwen3.6-plus",
    "gemini/gemini-3.1-pro-high",
]
DEFAULT_MODEL = "qwen3.5-plus"


def is_gemini_model(model_name: str) -> bool:
    """判断一个模型名是否走 Gemini 代理。"""
    return isinstance(model_name, str) and model_name.startswith("gemini/")

# 页面渲染 DPI；纯视觉模式下拉高可提升小字/公式 OCR
RENDER_DPI = 220

# 发送给视觉模型的图像尺寸上限（长边像素）
OCR_IMAGE_MAX_SIDE = 2200        # 主 OCR 调用
DETECTION_IMAGE_MAX_SIDE = 1600  # 图片检测阶段
VERIFY_CROP_MAX_SIDE = 800       # 裁切校验阶段送回的单张裁切图

# 图像编码阈值：超过此字节数后压缩降质量
MODEL_IMAGE_MAX_BYTES = 7_500_000
MIN_MODEL_IMAGE_SIDE = 640

# 并行工作线程数（图片检测阶段）
# qwen3.5 系列限流极宽松：30,000 RPM / 5-10M TPM；每页约 2 次调用，3-8s 一次
FIGURE_DETECT_WORKERS = 20

# AI 裁切验证最大轮次
MAX_CROP_VERIFY_ROUNDS = 2

# 拼接时前页末尾送给模型的字符数
STITCH_PREV_TAIL_CHARS = 600

# 跨页滚动大纲保留的最近标题数（防止长文档上下文膨胀）
OUTLINE_MAX_HEADINGS = 80

# 跨页 prev_tail 长度（转换阶段传给模型）
PREV_TAIL_CHARS_CONVERT = 1200

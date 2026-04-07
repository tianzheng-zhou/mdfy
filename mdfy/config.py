"""全局常量与模型配置。"""

AVAILABLE_MODELS = ["qwen3.5-flash", "qwen3.5-plus", "qwen3.6-plus"]
DEFAULT_MODEL = "qwen3.6-plus"
AVAILABLE_MODES = ["pipeline", "vision"]
DEFAULT_MODE = "vision"
MODEL_IMAGE_MAX_BYTES = 7_500_000
DETECTION_IMAGE_MAX_SIDE = 1600
OCR_IMAGE_MAX_SIDE = 2200
MIN_MODEL_IMAGE_SIDE = 640

# 并行工作线程数（图片检测阶段）
# qwen3.5-plus / qwen3.5-flash 限流极宽松：30,000 RPM / 5-10M TPM
# 每页约 2 次 API 调用（detect + verify），单次 3-8s，实际并发远低于限流上限
FIGURE_DETECT_WORKERS = 20

# AI 裁切验证最大轮次
MAX_CROP_VERIFY_ROUNDS = 2

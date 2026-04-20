"""OpenAI / DashScope 客户端初始化。"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_client():
    """返回一个指向阿里云 DashScope OpenAI 兼容端点的 OpenAI 客户端。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows:  set DASHSCOPE_API_KEY=sk-xxxxx\n"
            "  Linux:    export DASHSCOPE_API_KEY=sk-xxxxx\n"
            "或在项目根目录的 .env 文件中写入 DASHSCOPE_API_KEY=sk-xxxxx"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

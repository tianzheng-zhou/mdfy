"""OpenAI / DashScope 客户端初始化。"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_client():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows:  set DASHSCOPE_API_KEY=sk-xxxxx\n"
            "  Linux:    export DASHSCOPE_API_KEY=sk-xxxxx"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

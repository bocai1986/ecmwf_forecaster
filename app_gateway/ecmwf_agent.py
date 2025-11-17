import time
import asyncio
import json

from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os
import sys

# 自动把项目根目录加入 Python 搜索路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app_gateway.ecmwf_mcp_client import call_mcp_forecast_full


# from app_gateway.ecmwf_mcp_client import call_mcp_forecast_full

# =========================
# 1. Ollama (OpenAI 兼容) + 计时
# =========================
t0 = time.time()
print("[init] 准备创建 AsyncOpenAI 客户端...")
ollama_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://10.0.74.56:11434/v1",
    timeout=150.0,
)
t1 = time.time()
print(f"[init] AsyncOpenAI 客户端创建完成，用时 {t1 - t0:.3f} 秒")
print("[init] 准备创建 OpenAIChatCompletionsModel...")

chat_model = OpenAIChatCompletionsModel(
    # model="qwen3:8b",
    model="gpt-oss:latest",
    openai_client=ollama_client,
)
t2 = time.time()
print(f"[init] OpenAIChatCompletionsModel 创建完成，用时 {t2 - t1:.3f} 秒")
print(f"[init] 客户端 + 模型 初始化总耗时 {t2 - t0:.3f} 秒")


# =========================
# 2. 不让 agent 再调 MCP，只负责“说话”
# =========================
ecmwf_agent = Agent(
    name="ECMWF Forecaster",
    instructions=(
        "你是一名数值天气预报员助手。"
        "我会给你一段 ECMWF 24 小时预报的结构化 JSON，字段通常有 synoptic(形势)、precip(降水)、winds(大风)、conclusion(结论)。"
        "如果 JSON 里有 error，你要直接说明错误，不要编造预报。"
        "如果 JSON 是正常的，就按：形势 → 降水 → 大风 → 一句话总结 的顺序输出中文汇报。"
    ),
    model=chat_model,
)
print(f"[init] agent 创建完成，用时 {time.time() - t2:.3f} 秒")


if __name__ == "__main__":
    manifest = "/data/ecmwf/upper_airfields/2025-11-03/2025110212/products/manifest.json"

    # 1) 先同步调用 MCP，把慢的活干完
    t_mcp_start = time.time()
    mcp_payload = call_mcp_forecast_full(
        manifest_path=manifest,
        model_path="/data/Qwen/Qwen3-VL-8B-Instruct",
        gpu_id=2,
        max_tokens=1024,
        timeout=300,
    )
    t_mcp_end = time.time()
    print(f"[timing] MCP done in {t_mcp_end - t_mcp_start:.2f}s")

    # 2) 保证有 event loop，再跑 Runner.run_sync
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # 3) 让 agent 只做“格式化”
    t_llm_start = time.time()
    result = Runner.run_sync(
        ecmwf_agent,
        "下面是 ECMWF 24h 的结构化结果，请直接按要求输出中文汇报：\n"
        + json.dumps(mcp_payload, ensure_ascii=False),
    )
    t_llm_end = time.time()

    print("=== 最终输出 ===")
    print(result.final_output)
    print(f"[timing] LLM format done in {t_llm_end - t_llm_start:.2f}s")
    print(f"[timing] 总耗时 {(t_llm_end - t0):.2f}s")

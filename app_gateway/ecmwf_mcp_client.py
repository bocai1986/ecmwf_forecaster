# # app_gateway/ecmwf_mcp_client.py
# import asyncio
# from typing import Any, Dict, Optional

# from fastmcp import Client

# # MCP 服务地址
# MCP_ENDPOINT = "http://127.0.0.1:8001/mcp"

# # 默认给你留的模型/参数，可以按你实际的服务改
# DEFAULT_MODEL_PATH = "/data/Qwen/Qwen3-VL-8B-Instruct"
# DEFAULT_GPU_ID = 2
# DEFAULT_MAX_TOKENS = 1024


# async def _call_mcp_tool_async(
#     tool_name: str,
#     payload: Dict[str, Any],
#     endpoint: str = MCP_ENDPOINT,
# ) -> Dict[str, Any]:
#     """
#     真正的异步调用函数。
#     用 fastmcp.Client 去连你的 http://127.0.0.1:8001/mcp，
#     这个 Client 会按 MCP 要求走 text/event-stream，所以不会再出现你刚才那个
#     “Client must accept text/event-stream” 的问题。
#     """
#     client = Client(endpoint)
#     async with client:
#         result = await client.call_tool(tool_name, payload)
#     # fastmcp 的返回是 CallToolResult，真正的数据在 .data 里
#     return result.data


# def call_mcp_forecast_full(
#     manifest_path: str,
#     model_path: str = DEFAULT_MODEL_PATH,
#     gpu_id: int = DEFAULT_GPU_ID,
#     max_tokens: int = DEFAULT_MAX_TOKENS,
#     endpoint: str = MCP_ENDPOINT,
# ) -> Dict[str, Any]:
#     """
#     给同步的 Agent 用的包装函数。

#     注意：我们这里做了一个“如果已经在事件循环里，就开一个新的任务跑”的防护，
#     避免以后你把这个函数放到一个已经在跑 asyncio 的上下文里时报错。
#     """
#     payload = {
#         "manifest_path": manifest_path,
#         "model_path": model_path,
#         "gpu_id": gpu_id,
#         "max_tokens": max_tokens,
#     }

#     try:
#         # 如果当前没有事件循环，就走最简单的 asyncio.run
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         # 没有 loop，说明现在是普通同步环境，直接跑
#         return asyncio.run(_call_mcp_tool_async("forecast_full", payload, endpoint=endpoint))
#     else:
#         # 已经有 loop 在跑（有些框架/SDK 会这样），不能再 asyncio.run
#         # 我们在当前 loop 上开一个新的 future，然后阻塞拿结果
#         # 注意：这个分支要看你实际的运行环境，如果是在线程池里，一般也是 OK 的
#         fut = asyncio.run_coroutine_threadsafe(
#             _call_mcp_tool_async("forecast_full", payload, endpoint=endpoint),
#             loop,
#         )
#         return fut.result(timeout=120)


# app_gateway/ecmwf_mcp_client.py
import asyncio
import time
from typing import Any, Dict

from fastmcp import Client

MCP_ENDPOINT = "http://127.0.0.1:8001/mcp"
DEFAULT_MODEL_PATH = "/data/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_GPU_ID = 2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 300  # 秒，vLLM 冷启动+推理就给它足一点


async def _call_mcp_tool_async(
    tool_name: str,
    payload: Dict[str, Any],
    endpoint: str = MCP_ENDPOINT,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    真正的异步调用：连接 MCP -> 调用工具 -> 等待结果
    用 asyncio.wait_for 包一层，确保我们这边的超时时间够长
    """
    start = time.time()
    client = Client(endpoint)
    async with client:
        # fastmcp 的 call_tool 本身是一个协程，这里我们包一层显式超时
        result = await asyncio.wait_for(
            client.call_tool(tool_name, payload),
            timeout=timeout,
        )
    elapsed = time.time() - start
    print(f"[MCP CLIENT] tool={tool_name} done in {elapsed:.2f}s")
    return result.data


def call_mcp_forecast_full(
    manifest_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    gpu_id: int = DEFAULT_GPU_ID,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    endpoint: str = MCP_ENDPOINT,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    给同步的 agent 用的入口，自动补全 MCP 需要的几个参数
    """
    payload = {
        "manifest_path": manifest_path,
        "model_path": model_path,
        "gpu_id": gpu_id,
        "max_tokens": max_tokens,
    }

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 普通同步环境
        return asyncio.run(
            _call_mcp_tool_async(
                "forecast_full",
                payload,
                endpoint=endpoint,
                timeout=timeout,
            )
        )
    else:
        # 已经有事件循环的环境（有些 SDK 会这样）
        fut = asyncio.run_coroutine_threadsafe(
            _call_mcp_tool_async(
                "forecast_full",
                payload,
                endpoint=endpoint,
                timeout=timeout,
            ),
            loop,
        )
        return fut.result(timeout=timeout)

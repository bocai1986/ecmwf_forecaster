# ecmwf_mcp/tools/__init__.py
"""
把现有的三个离线工具包装成 fastmcp 的 MCP 服务：
- forecast_24h_maker
- forecast_publisher
- forecast_reporter
再加一个一键的 forecast_full

运行方式：
    # STDIO 模式（给 MCP 客户端用）
    python -m ecmwf_mcp.tools stdio

    # HTTP 模式（方便 curl / 浏览器）
    python -m ecmwf_mcp.tools 
"""

from __future__ import annotations
import json
import sys
from fastmcp import FastMCP

mcp = FastMCP(name="ecmwf-forecast-tools")

# --- 按你原来的风格：包内导入→失败就退成同目录导入 ---

try:
    from .forecast_24h_maker import run as maker_run, Args as MakerArgs
except ImportError:
    from forecast_24h_maker import run as maker_run, Args as MakerArgs

try:
    from .forecast_publisher import run as publisher_run, Args as PublisherArgs
except ImportError:
    from forecast_publisher import run as publisher_run, Args as PublisherArgs

try:
    from .forecast_reporter import run as reporter_run, Args as ReporterArgs
except ImportError:
    from forecast_reporter import run as reporter_run, Args as ReporterArgs


# ========== 1) maker ==========
@mcp.tool
def forecast_24h_maker(manifest_path: str) -> dict:
    """
    读取离线生成的 manifest.json，识别 3h / 12h 视频资源。
    参数:
        manifest_path: /data/.../products/manifest.json
    """
    return maker_run(MakerArgs(manifest_path=manifest_path))


# ========== 2) publisher ==========
@mcp.tool
def forecast_publisher(maker_payload: dict | str) -> dict:
    """
    把 maker 的输出整理成发布结构。
    入参既可以是 dict，也可以是 JSON 字符串。
    """
    if isinstance(maker_payload, str):
        payload_json = maker_payload
    else:
        payload_json = json.dumps(maker_payload, ensure_ascii=False)
    return publisher_run(PublisherArgs(maker_payload_json=payload_json))


# ========== 3) reporter ==========
@mcp.tool
def forecast_reporter(
    publisher_payload: dict | str,
    model_path: str = "/data/Qwen/Qwen3-VL-8B-Instruct",
    gpu_id: int = 2,
    max_tokens: int = 1024,
) -> dict:
    """
    调本地 vLLM / Qwen3-VL 做多模态分析，生成结构化天气报告。
    """
    if isinstance(publisher_payload, str):
        payload_json = publisher_payload
    else:
        payload_json = json.dumps(publisher_payload, ensure_ascii=False)

    return reporter_run(
        ReporterArgs(
            publisher_payload_json=payload_json,
            model_path=model_path,
            gpu_id=gpu_id,
            max_tokens=max_tokens,
        )
    )


# ========== 4) 一键流 ==========
@mcp.tool
def forecast_full(
    manifest_path: str,
    model_path: str = "/data/Qwen/Qwen3-VL-8B-Instruct",
    gpu_id: int = 2,
    max_tokens: int = 1024,
) -> dict:
    """
    一键：manifest → maker → publisher → reporter
    """
    maker_out = maker_run(MakerArgs(manifest_path=manifest_path))
    publisher_out = publisher_run(
        PublisherArgs(
            maker_payload_json=json.dumps(maker_out, ensure_ascii=False)
        )
    )
    report_out = reporter_run(
        ReporterArgs(
            publisher_payload_json=json.dumps(publisher_out, ensure_ascii=False),
            model_path=model_path,
            gpu_id=gpu_id,
            max_tokens=max_tokens,
        )
    )
    return {
        "maker": maker_out,
        "publisher": publisher_out,
        "report": report_out,
    }


# ===== entrypoint =====
# if __name__ == "__main__":
#     # 默认走 stdio
#     if len(sys.argv) >= 2 and sys.argv[1] == "http":
#         host = sys.argv[2] if len(sys.argv) >= 3 else "127.0.0.1"
#         port = int(sys.argv[3]) if len(sys.argv) >= 4 else 8001
#         mcp.run(transport="http", host=host, port=port)
#     else:
#         mcp.run()

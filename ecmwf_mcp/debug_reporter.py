#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_reporter.py
一口气测试 3 个 MCP 工具：
1) forecast_24h_maker   → 读 manifest.json
2) forecast_publisher   → 整理结构
3) forecast_reporter    → 调本地 Qwen3-VL(vLLM) 生成文字预报
"""

import asyncio
import json
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8001/mcp"
# MANIFEST_PATH = "/data/ecmwf/upper_airfields/2025-11-03/2025110212/products/manifest.json"
MANIFEST_PATH = "/data/ecmwf/upper_airfields/2025-11-14/2025-11-14/products/manifest.json"

# 按你上次跑通时的路径改，如果模型不在这个目录，就换成你的 qwen3-vl 路径
QWEN_MODEL_PATH = "/data/Qwen/Qwen3-VL-8B-Instruct"
GPU_ID = 2
MAX_TOKENS = 2048


async def main():
    print(f"🔗 连接 MCP: {SERVER_URL}")
    client = Client(SERVER_URL)

    async with client:
        # 1) maker
        print("\n[1/3] 🧰 调用 forecast_24h_maker ...")
        maker_ret = await client.call_tool(
            "forecast_24h_maker",
            {"manifest_path": MANIFEST_PATH},
        )
        maker_payload = maker_ret.data
        print("✅ maker OK，识别到根目录:", maker_payload.get("root"))

        # 2) publisher
        print("\n[2/3] 🧰 调用 forecast_publisher ...")
        pub_ret = await client.call_tool(
            "forecast_publisher",
            {
                # 这个工具的定义里支持 dict 或 json 字符串，我们直接传 dict 最省事
                "maker_payload": maker_payload
            },
        )
        publisher_payload = pub_ret.data
        print("✅ publisher OK，生成了 keys:", list(publisher_payload.keys()))

        # 3) reporter
        print("\n[3/3] 🧠 调用 forecast_reporter (Qwen3-VL) ...")
        rep_ret = await client.call_tool(
            "forecast_reporter",
            {
                "publisher_payload": publisher_payload,
                "model_path": QWEN_MODEL_PATH,
                "gpu_id": GPU_ID,
                "max_tokens": MAX_TOKENS,
            },
        )

        print("\n=== 📄 预报员总结（forecast_reporter 返回） ===")
        # 看你的 reporter 怎么返回的，这里兼容两种
        if rep_ret.data:
            # 通常会是一坨字典
            print(json.dumps(rep_ret.data, ensure_ascii=False, indent=2))
        elif rep_ret.content:
            # 有的实现会把文本放 content 里
            for c in rep_ret.content:
                if hasattr(c, "text"):
                    print(c.text)
        else:
            print(rep_ret)

    print("\n✅ 全流程测试完成。")


if __name__ == "__main__":
    asyncio.run(main())

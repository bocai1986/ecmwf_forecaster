#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_mcp.py
-------------
调试 FastMCP HTTP 服务器：列出工具并调用 forecast_24h_maker。
"""

import asyncio
from fastmcp import Client

# === MCP 服务器地址 ===
SERVER_URL = "http://127.0.0.1:8001/mcp"

# === 示例参数 ===
MANIFEST_PATH = "/data/ecmwf/upper_airfields/2025-11-03/2025110212/products/manifest.json"


async def main():
    print(f"🔗 正在连接 MCP 服务器: {SERVER_URL}")

    # 创建客户端（不需要 session_id）
    client = Client(SERVER_URL)

    # 建立连接
    async with client:
        print("✅ 已连接，正在获取工具列表...\n")

        # 1️⃣ 列出所有可用工具
        tools = await client.list_tools()
        for tool in tools:
            print(f"🧰 工具: {tool.name}")
            print(f"   描述: {tool.description}")
            if tool.inputSchema:
                print(f"   参数: {tool.inputSchema}\n")

        # 2️⃣ 调用 forecast_24h_maker
        print("🚀 调用 forecast_24h_maker ...")
        result = await client.call_tool(
            "forecast_24h_maker",
            {"manifest_path": MANIFEST_PATH},
        )

        print("\n=== forecast_24h_maker 返回结果 ===")
        if hasattr(result, "data") and result.data:
            print(result.data)
        elif hasattr(result, "content"):
            for c in result.content:
                if hasattr(c, "text"):
                    print(c.text)
        else:
            print(result)

    print("\n✅ 调试结束。")


if __name__ == "__main__":
    asyncio.run(main())

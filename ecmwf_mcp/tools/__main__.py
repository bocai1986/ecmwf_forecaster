# ecmwf_mcp/tools/__main__.py
from . import mcp  # 从 __init__.py 里拿到我们创建的 FastMCP 实例
import sys

if __name__ == "__main__":
    # ✅ 默认使用 HTTP 模式
    host = "0.0.0.0"
    port = 8001

    # 如果用户传入 "stdio"，则切回 STDIO 模式（方便调试）
    if len(sys.argv) >= 2 and sys.argv[1] == "stdio":
        print("🔄 Running MCP in STDIO mode (for debug)...")
        mcp.run()
    else:
        print(f"🚀 Starting MCP HTTP server at http://{host}:{port}/mcp")
        mcp.run(transport="http", host=host, port=port)


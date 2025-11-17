# mcp/main.py
import os
import sys
import json
from pathlib import Path

# 允许从 mcp/ 下导入 tools/*
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

from tools.forecast_24h_maker import run as maker_run, Args as MakerArgs
from tools.forecast_publisher import run as publisher_run, Args as PublisherArgs
from tools.forecast_reporter import run as reporter_run, Args as ReporterArgs


def pick_default_manifest() -> str:
    """
    不传参时，自动找最近一天的 manifest.json
    约定目录：/data/ecmwf/upper_airfields/YYYY-MM-DD/YYYYMMDDHH/products/manifest.json
    """
    root = Path("/data/ecmwf/upper_airfields")
    if not root.exists():
        raise FileNotFoundError(f"找不到基础目录：{root}")

    # 找最近日期
    dates = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    for d in dates:
        # 每天下面可能有 2025110212 这种
        subs = sorted([p for p in d.iterdir() if p.is_dir() and p.name.isdigit()], reverse=True)
        for s in subs:
            manifest = s / "products" / "manifest.json"
            if manifest.exists():
                print(f"📁 自动使用 manifest: {manifest}")
                return str(manifest)
    raise FileNotFoundError("最近几天都没找到 manifest.json，请手动指定")


def main(manifest_path: str | None = None):
    # 1) 找到 manifest.json
    if manifest_path is None:
        manifest_path = pick_default_manifest()
    else:
        manifest_path = str(Path(manifest_path).resolve())

    # 2) maker：从 manifest 里挑出 3h / 12h 的那两段视频信息
    maker_out = maker_run(MakerArgs(manifest_path=manifest_path))

    # 3) publisher：补全成可发布结构（含 abs_path）
    publisher_out = publisher_run(
        PublisherArgs(maker_payload_json=json.dumps(maker_out, ensure_ascii=False))
    )

    # 4) reporter：让 Qwen3-VL 看这个结构并写一份报告
    reporter_out = reporter_run(
        ReporterArgs(
            publisher_payload_json=json.dumps(publisher_out, ensure_ascii=False),
            model_path=os.getenv("QWEN3VL_PATH", "/data/Qwen/Qwen3-VL-8B-Instruct"),
            gpu_id=2,         # ← 你说要用 GPU=2
            max_tokens=2048,
        )
    )

    # 5) 打印最终结果
    print(json.dumps(reporter_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 支持：python mcp/main.py /path/to/manifest.json
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)

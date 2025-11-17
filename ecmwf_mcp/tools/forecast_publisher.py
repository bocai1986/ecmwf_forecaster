# mcp/tools/forecast_publisher.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path


class Args(BaseModel):
    """
    两种用法二选一：
    1) manifest_path: 直接给离线生成的 manifest.json
    2) maker_payload_json: 上一个工具 forecast_24h_maker 的返回结果(JSON字符串)

    不在这里做校验，run() 里手动检查，避免 pydantic v1/v2 的装饰器差异。
    """
    manifest_path: Optional[str] = Field(default=None)
    maker_payload_json: Optional[str] = Field(default=None)


def _load_from_manifest(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        man = json.load(f)

    groups = man.get("groups") or []
    root = man.get("root")
    out: Dict[str, Any] = {
        "root": root,
        "analyze_3h": None,
        "result_12h": None,
        "kind": "ecmwf-forecast-24h-assets",
    }
    for g in groups:
        if g.get("name") == "analyze_3h":
            out["analyze_3h"] = g
        elif g.get("name") == "result_12h":
            out["result_12h"] = g
    return out


def _normalize_group_from_manifest_group(g: Dict[str, Any], root_dir: Optional[str]) -> Dict[str, Any]:
    """把 manifest 里那个 group 变成一个比较扁平的结构，方便“发布”"""
    if g is None:
        return {}

    combined_meta = g.get("combined")
    layers_meta = g.get("layers") or {}

    def _abs_or_rel(meta: Dict[str, Any]):
        if not meta:
            return None
        rel = meta.get("rel_path")
        url = meta.get("url")
        abs_path = None
        if rel and root_dir:
            abs_path = str(Path(root_dir) / rel)
        return {
            "rel_path": rel,
            "abs_path": abs_path,
            "url": url,
            "duration_s": meta.get("duration_s"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
        }

    norm = {
        "step": g.get("step"),
        "tag": g.get("tag"),
        "combined": _abs_or_rel(combined_meta) if combined_meta else None,
        "layers": {},
    }
    for lname, lmeta in layers_meta.items():
        norm["layers"][lname] = _abs_or_rel(lmeta)

    return norm


def run(args: Args) -> Dict[str, Any]:
    """
    把 manifest / maker 的结果变成“可以直接发出去”的结构。
    这里不真正推送，只是返回统一的 payload。
    """
    # 手动校验输入来源
    if not args.maker_payload_json and not args.manifest_path:
        raise ValueError("必须提供 manifest_path 或 maker_payload_json 其中之一")

    # 1. 先把原始数据 load 出来
    if args.maker_payload_json:
        raw = json.loads(args.maker_payload_json)
        root_dir = raw.get("root")
        analyze = raw.get("analyze_3h")
        result = raw.get("result_12h")
    else:
        raw = _load_from_manifest(args.manifest_path)  # type: ignore
        root_dir = raw.get("root")
        analyze = raw.get("analyze_3h")
        result = raw.get("result_12h")

    # 2. 规范化两个部分
    norm_analyze = _normalize_group_from_manifest_group(analyze, root_dir) if analyze else None
    norm_result = _normalize_group_from_manifest_group(result, root_dir) if result else None

    # 3. 统一返回
    payload = {
        "title": "ECMWF 24h 预报素材",
        "root": root_dir,
        "analyze_3h": norm_analyze,
        "result_12h": norm_result,
        "summary": {
            "has_analyze": bool(norm_analyze),
            "has_result": bool(norm_result),
        },
    }

    print("📤 forecast_publisher 已生成可发布结构")
    return payload

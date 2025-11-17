# mcp/tools/forecast_24h_maker.py
from pydantic import BaseModel
from typing import Any, Dict, Optional
import json
import os




# /data/ecmwf/upper_airfields/2025-11-03/2025110212/products/manifest.json


class Args(BaseModel):
    # manifest.json 的绝对路径或相对路径
    manifest_path: str


def _pick_group(manifest: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """从 manifest['groups'] 里挑一个指定 name 的组"""
    groups = manifest.get("groups") or []
    for g in groups:
        if g.get("name") == name:
            return g
    return None


def run(args: Args) -> Dict[str, Any]:
    """把离线产出的 manifest.json 变成“24h 预报可消费结构”"""
    path = args.manifest_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        man = json.load(f)

    root_dir = man.get("root")  # e.g. /data/ecmwf/upper_airfields/2025-11-03/2025110212

    # 1) 分别拿两个组
    analyze = _pick_group(man, "analyze_3h")
    result = _pick_group(man, "result_12h")

    # 2) 整理输出
    def _normalize_group(g: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if g is None:
            return None
        out: Dict[str, Any] = {
            "name": g.get("name"),
            "step": g.get("step"),
            "tag": g.get("tag"),
            "combined": None,
            "layers": {},
        }

        # combined 视频（分析版/结果版的总视频）
        comb = g.get("combined")
        if comb:
            # 保留相对路径和可能的 URL，方便后面的视频理解模型直接用
            out["combined"] = {
                "rel_path": comb.get("rel_path"),
                "url": comb.get("url"),
                "duration_s": comb.get("duration_s"),
                "width": comb.get("width"),
                "height": comb.get("height"),
                "fps": comb.get("fps"),
            }

        # 各层视频（850/700/500/surface）
        layers = g.get("layers") or {}
        for lname, meta in layers.items():
            out["layers"][lname] = {
                "rel_path": meta.get("rel_path"),
                "url": meta.get("url"),
                "duration_s": meta.get("duration_s"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "fps": meta.get("fps"),
            }

        return out

    result_payload = {
        "root": root_dir,
        "analyze_3h": _normalize_group(analyze),
        "result_12h": _normalize_group(result),
        # 给后续 agent 一个简单的提示：这是一套 24h 预报素材
        "kind": "ecmwf-forecast-24h-assets",
    }

    print("🎯 forecast_24h_maker 已汇总 3h 分析版与 12h 结果版素材")
    return result_payload


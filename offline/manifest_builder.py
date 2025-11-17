#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manifest_builder.py
-------------------
扫描 <root>/products 下的 MP4，生成结构化 manifest.json，供多模态模型直接使用。

约定目录（示例）：
root/
  products/
    analyze_3h/
      combined_012-036.mp4
    analyze_3h_layers/
      850hPa_012-036.mp4
      700hPa_012-036.mp4
      500hPa_012-036.mp4
      surface_phase_012-036.mp4
      surface_synoptic_012-036.mp4
    result_12h/
      combined_012-024.mp4
    result_12h_layers/
      850hPa_012-024.mp4
      700hPa_012-024.mp4
      500hPa_012-024.mp4
      surface_phase_012-024.mp4
      surface_synoptic_012-024.mp4

用法：
  python offline/manifest_builder.py /path/to/root [--out manifest.json] [--base-url https://cdn.example.com]
  # 不传 root 时，自动挑选最近就绪的根目录（/data/ecmwf/upper_airfields/YYYY-MM-DD/...）
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse
import cv2

# 与 run_offline_pipeline / video_encoder 保持一致
DATA_ROOT_DEFAULT = "/data/ecmwf/upper_airfields"

# 这里的名称是 manifest 中对每一层的 key
LAYER_ORDER = ["850hPa", "700hPa", "500hPa", "surface_phase", "surface_synoptic"]

GROUPS = {
    "analyze_3h": {
        "step": "3h",
        "tag":  "012-036",
        "combined_dir": "products/analyze_3h",
        "layers_dir":   "products/analyze_3h_layers",
    },
    "result_12h": {
        "step": "12h",
        "tag":  "012-024",
        "combined_dir": "products/result_12h",
        "layers_dir":   "products/result_12h_layers",
    },
}


# ======================= root 自动选择逻辑 =======================

def pick_latest_ready_root(base: Path) -> Path:
    """
    自动选取最近一次“就绪”的起报根目录 root：

    兼容两种常见结构：
      1) /base/YYYY-MM-DD/step3h/...              （日期目录直接就是 root）
      2) /base/YYYY-MM-DD/2025111212/step3h/...   （日期下面再套一层起报目录）

    判定为“就绪”的条件：存在任意一个 step / products 目录：
      step3h / step6h / step12h / step24h / products
    """
    base = Path(base).resolve()
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()

    for d in range(0, 5):
        date_dir = base / (today - timedelta(days=d)).strftime("%Y-%m-%d")
        if not date_dir.is_dir():
            continue

        # 情况1：日期目录本身就是 root（直接有 step 或 products）
        for step_name in ("step3h", "step6h", "step12h", "step24h", "products"):
            if (date_dir / step_name).exists():
                print(f"📂 选定 root = {date_dir}（{step_name} 存在）")
                return date_dir

        # 情况2：日期目录下面有若干子目录（起报目录）
        subs = sorted(
            [p for p in date_dir.iterdir() if p.is_dir()],
            reverse=True,
        )
        for s in subs:
            for step_name in ("step3h", "step6h", "step12h", "step24h", "products"):
                if (s / step_name).exists():
                    print(f"📂 选定 root = {s}（{step_name} 存在）")
                    return s

    raise FileNotFoundError(
        f"最近 5 天在 {base} 下未找到就绪起报目录（step3h/step6h/step12h/step24h/products 任一）"
    )


# ======================= 工具函数 =======================

def video_props(path: Path):
    """读取视频属性（fps, width, height, frames, duration_s）。失败则返回 None。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or w <= 0 or h <= 0 or n <= 0:
        return None
    duration_s = float(n) / float(fps) if fps > 0 else 0.0
    return dict(
        fps=round(fps, 3),
        width=w,
        height=h,
        frames=n,
        duration_s=round(duration_s, 3),
    )


def file_stats(path: Path):
    st = path.stat()
    return dict(
        bytes=st.st_size,
        mtime_iso=datetime.fromtimestamp(
            st.st_mtime, ZoneInfo("Asia/Shanghai")
        ).isoformat(),
    )


def to_url(rel_path: str, base_url: str | None):
    if not base_url:
        return None
    return base_url.rstrip("/") + "/" + rel_path.lstrip("/")


# ======================= 核心扫描逻辑 =======================

def scan_group(root: Path, group_name: str, cfg: dict, base_url: str | None):
    """扫描一个组（analyze_3h / result_12h）"""
    results = {
        "name": group_name,
        "step": cfg["step"],
        "tag":  cfg["tag"],
        "combined": None,
        "layers": {},
    }

    # 1) combined
    combined_name = f"combined_{cfg['tag']}.mp4"
    combined_path = root / cfg["combined_dir"] / combined_name
    if combined_path.exists():
        props = video_props(combined_path)
        if props:
            rel = str(combined_path.relative_to(root))
            results["combined"] = {
                "rel_path": rel,
                "url": to_url(rel, base_url),
                **props,
                **file_stats(combined_path),
            }
        else:
            print(f"⚠️ 无法读取 combined 属性：{combined_path}")
    else:
        print(f"ℹ️ 缺少 combined：{combined_path}")

    # 2) per-layer
    layers_dir = root / cfg["layers_dir"]

    for layer in LAYER_ORDER:
        expect_name = f"{layer}_{cfg['tag']}.mp4"
        p = layers_dir / expect_name

        # 兼容老命名：surface_012-036.mp4 → 归到 surface_phase
        if not p.exists() and layer == "surface_phase":
            cand = list(layers_dir.glob(f"surface_*{cfg['tag']}*.mp4"))
            if cand:
                p = cand[0]

        if p.exists():
            props = video_props(p)
            if props:
                rel = str(p.relative_to(root))
                results["layers"][layer] = {
                    "rel_path": rel,
                    "url": to_url(rel, base_url),
                    **props,
                    **file_stats(p),
                }
            else:
                print(f"⚠️ 无法读取层视频属性：{p}")
        else:
            print(f"ℹ️ 缺少层视频：{layers_dir}/{expect_name}（layer={layer}）")

    return results


def build_index(manifest: dict):
    """构建方便检索的索引（按 step 聚合）"""
    index = {}
    for g in manifest["groups"]:
        step = g["step"]
        index.setdefault(step, {"combined": [], "layers": {}})
        if g.get("combined"):
            index[step]["combined"].append(g["combined"])
        for layer, meta in g.get("layers", {}).items():
            index[step]["layers"].setdefault(layer, [])
            index[step]["layers"][layer].append(meta)
    return index


# ======================= main =======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        nargs="?",
        help="根目录（包含 products 的那一层）；不传则自动挑选最近就绪",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="manifest 输出路径（默认：<root>/products/manifest.json）",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="可选：为 rel_path 补前缀生成可访问 URL（如 https://cdn.example.com/ecmwf）",
    )
    ap.add_argument(
        "--data-root",
        default=DATA_ROOT_DEFAULT,
        help=f"自动挑选模式时的起点（默认：{DATA_ROOT_DEFAULT}）",
    )
    args = ap.parse_args()

    if args.root:
        root = Path(args.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"指定 root 不存在：{root}")
        print(f"📁 使用命令行指定 root：{root}")
    else:
        root = pick_latest_ready_root(Path(args.data_root).resolve())

    products_dir = root / "products"
    if not products_dir.exists():
        raise FileNotFoundError(f"未找到 products 目录：{products_dir}")

    manifest = {
        "root": str(root),
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "groups": [],
        "version": "2025-11-03",
        "note": "analyze_3h/result_12h：各自含 combined 与 per-layer 列表；index 为检索加速用",
    }

    for name, cfg in GROUPS.items():
        group_info = scan_group(root, name, cfg, args.base_url)
        manifest["groups"].append(group_info)

    manifest["index"] = build_index(manifest)

    out_path = Path(args.out).resolve() if args.out else (products_dir / "manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ 已写出 manifest：{out_path}")
    # 简短摘要
    for g in manifest["groups"]:
        c = "有" if g.get("combined") else "无"
        print(
            f" • {g['name']}（step={g['step']} tag={g['tag']}） "
            f"combined={c}  layers={list(g['layers'].keys())}"
        )


if __name__ == "__main__":
    main()

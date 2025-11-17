#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline/run_offline_pipeline.py
-------------------------------
每日离线调度脚本：下载 ECMWF → 绘图 → 合成视频 → 生成 manifest.json

依赖：
- offline/ecmwf_upper_surface_oneclick.py
- offline/video_encoder.py
- offline/manifest_builder.py   （使用其中的 pick_latest_ready_root 选择就绪目录）

调度（示例）：
0 21 * * * /home/maqingbo/miniconda3/envs/draw_image/bin/python \
  /home/maqingbo/ecmwf_forecaster/offline/run_offline_pipeline.py \
  >> /home/maqingbo/logs/offline_pipeline.log 2>&1
"""
import os
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util


# === 你机器上的 Python 解释器（conda 环境） ===
PYTHON = "/home/maqingbo/miniconda3/envs/draw_image/bin/python"

# === 工程根路径（以本文件所在层为准，按需修改） ===
PROJECT_ROOT = "/home/maqingbo/ecmwf_forecaster/offline"

# === 和 manifest_builder.py 保持一致的起点目录 ===
# （manifest_builder.py 默认 DATA_ROOT_DEFAULT = "/data/ecmwf/upper_airfields"）
DATA_ROOT_DEFAULT = "/data/ecmwf/upper_airfields"


def run_cmd(cmd: list[str], cwd: str | None = None):
    """打印并执行命令；失败抛异常中断"""
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


# def pick_ready_root_by_manifest_logic() -> Path:
#     """
#     直接“复用” manifest_builder.py 的就绪目录选择逻辑：
#     - 这样 oneclick、video_encoder、manifest 三者对根目录判断完全一致
#     - 避免不同脚本各自实现导致 00Z/12Z 选错
#     """
#     import importlib.util

#     mb_path = Path(PROJECT_ROOT) / "manifest_builder.py"
#     if not mb_path.exists():
#         raise FileNotFoundError(f"找不到 manifest_builder.py：{mb_path}")

#     # 动态导入 manifest_builder.py
#     spec = importlib.util.spec_from_file_location("manifest_builder", str(mb_path))
#     mb = importlib.util.module_from_spec(spec)
#     assert spec and spec.loader
#     spec.loader.exec_module(mb)  # type: ignore

#     base = Path(getattr(mb, "DATA_ROOT_DEFAULT", DATA_ROOT_DEFAULT))
#     pick_func = getattr(mb, "pick_latest_ready_root", None)
#     if pick_func is None:
#         raise RuntimeError("manifest_builder.py 缺少 pick_latest_ready_root()")

#     ready_root = pick_func(base)
#     print(f"📂 通过 manifest 逻辑选定就绪根目录：{ready_root}")
#     return ready_root


def pick_ready_root_by_manifest_logic() -> Path:
    """
    选定“就绪根目录”（即包含 step3h/step6h/step12h 或 products 的那一层）：

    优先级：
    1) 按 ecmwf_images_oneclick 的约定：OUT_ROOT/YYYY-MM-DD/ 子目录中，选最近修改的那个，
       且该子目录下存在 step3h/step6h/step12h 或 products；
    2) 如果今天没有，就回退到 manifest_builder.pick_latest_ready_root(base_root)；
    3) 再不行，最后在 base_root 下做一轮递归扫描。
    """

    # ===== 1. 动态导入 manifest_builder，拿到 DATA_ROOT_DEFAULT（OUT_ROOT） =====
    mb_path = Path(PROJECT_ROOT) / "manifest_builder.py"
    if not mb_path.exists():
        raise FileNotFoundError(f"找不到 manifest_builder.py：{mb_path}")

    spec = importlib.util.spec_from_file_location("manifest_builder", str(mb_path))
    mb = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mb)  # type: ignore

    base_root = Path(getattr(mb, "DATA_ROOT_DEFAULT", DATA_ROOT_DEFAULT)).resolve()
    print(f"ℹ️ pick_ready_root 使用的 base_root = {base_root}")

    # ===== 2. 优先按 ecmwf_images_oneclick 的目录约定：OUT_ROOT / 今天日期 =====
    today_local = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    today_dir = base_root / today_local
    print(f"ℹ️ 今天的 SAVE_DIR = {today_dir}")

    candidates: list[Path] = []

    if today_dir.is_dir():
        # 先看今天日期目录下面的子目录（一般是 2025111212 这样的起报目录）
        for sub in today_dir.iterdir():
            if not sub.is_dir():
                continue
            if any(
                (sub / name).exists()
                for name in ("products", "step3h", "step6h", "step12h")
            ):
                candidates.append(sub)

        # 如果没有子目录，也允许今天目录本身就是起报目录（极端情况）
        if not candidates and any(
            (today_dir / name).exists()
            for name in ("products", "step3h", "step6h", "step12h")
        ):
            candidates.append(today_dir)

    if candidates:
        # 取最近修改时间的那一个作为 ready_root
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"📂 按今天目录选定就绪根目录：{best}")
        return best.resolve()

    print("ℹ️ 今天目录下没有发现包含 step*/products 的子目录，尝试用 manifest 的逻辑…")

    # ===== 3. 回退：用 manifest_builder 自带的 pick_latest_ready_root =====
    pick_func = getattr(mb, "pick_latest_ready_root", None)
    if pick_func is not None:
        try:
            ready_root = pick_func(base_root)
            print(f"📂 通过 manifest 逻辑选定就绪根目录：{ready_root}")
            return Path(ready_root).resolve()
        except FileNotFoundError as e:
            print(f"⚠️ manifest 的 pick_latest_ready_root 失败：{e}")
    else:
        print("⚠️ manifest_builder 中没有 pick_latest_ready_root，跳过该步骤。")

    # ===== 4. 最后兜底：在 base_root 下递归扫描含 products/step* 的目录 =====
    print(f"ℹ️ 启用兜底方案：在 {base_root} 下递归扫描包含 step*/products 的目录…")
    deep_candidates: list[Path] = []
    for p in base_root.rglob("*"):
        if not p.is_dir():
            continue
        if any(
            (p / name).exists()
            for name in ("products", "step3h", "step6h", "step12h")
        ):
            deep_candidates.append(p)

    if not deep_candidates:
        raise FileNotFoundError(
            f"在 {base_root} 下递归扫描也未找到包含 products/step3h/step6h/step12h 的目录"
        )

    best = max(deep_candidates, key=lambda p: p.stat().st_mtime)
    print(f"📂 兜底扫描选定就绪根目录：{best}")
    return best.resolve()


def main():
    # 1) 先跑一键离线（下载+绘图）
    print("\n🚀 Step 1: 下载 + 绘图 → ecmwf_upper_surface_oneclick.py")
    # run_cmd([PYTHON, f"{PROJECT_ROOT}/ecmwf_upper_surface_oneclick.py"])
    # run_cmd([PYTHON, f"{PROJECT_ROOT}/ecmwf_images_oneclick.py"])

    # 2) 用 manifest 的 pick 逻辑挑出“就绪根目录”
    ready_root = pick_ready_root_by_manifest_logic()

    # 3) 生成视频（显式传入 ready_root，保持一致）
    print("\n🎬 Step 2: 生成视频 → video_encoder.py")
    run_cmd([PYTHON, f"{PROJECT_ROOT}/video_encoder.py", str(ready_root)])

    # # 4) 生成 manifest.json（显式传入 ready_root）
    print("\n🧾 Step 3: 生成 manifest.json → manifest_builder.py")
    run_cmd([PYTHON, f"{PROJECT_ROOT}/manifest_builder.py", str(ready_root)])

    manifest_path = Path(ready_root) / "products" / "manifest.json"
    if manifest_path.exists():
        print(f"\n✅ Manifest OK: {manifest_path}")
        print(f"📦 大小：{manifest_path.stat().st_size / 1024:.1f} KB")
    else:
        print("\n⚠️ 未找到 manifest.json，可能生成失败。请检查日志。")

    print("\n🎉 全部离线任务完成！")


if __name__ == "__main__":
    main()


    # ==== Crontab 调度建议 ====
    # 每晚 21:00 自动运行：
    # sudo crontab -e
    # 0 21 * * * /home/maqingbo/miniconda3/envs/draw_image/bin/python \
    #     /home/maqingbo/ecmwf_forecaster/offline/run_offline_pipeline.py \
    #     >> /home/maqingbo/logs/offline_pipeline.log 2>&1
    # sudo crontab -l

             

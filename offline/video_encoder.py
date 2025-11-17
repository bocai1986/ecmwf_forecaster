#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 OpenCV：从 PNG 生成各层单层 MP4，并按 850→700→500→surface 顺序线性拼接为总视频
- step3h: 取 +012～+036
- step12h: 取 +012～+024

目录结构（root = 起报目录，例如 /data/ecmwf/upper_airfields/2025-11-03/2025110212）：
root/
  step3h/{850hPa,700hPa,500hPa,surface}/*.png
  step12h/{850hPa,700hPa,500hPa,surface}/*.png

输出：
  products/analyze_3h_layers/{850hPa,700hPa,500hPa,surface}_012-036.mp4
  products/analyze_3h/combined_012-036.mp4
  products/result_12h_layers/{850hPa,700hPa,500hPa,surface}_012-024.mp4
  products/result_12h/combined_012-024.mp4

用法：
  python offline/video_encoder.py                       # 自动找最近就绪目录
  python offline/video_encoder.py /path/to/root         # 指定起报目录

可选环境变量：
  DATA_ROOT=/data/ecmwf/upper_airfields    # 自动探测用的根
  FPS_OUT=6                                # 输出帧率（默认 6）
  FOURCC=mp4v                              # OpenCV fourcc（常见 mp4v/avc1/H264）
  SIZE_LOCK="1920x1080"                    # 强制输出分辨率（WxH），默认首帧尺寸
  DRAW_TIME=1                              # 是否在帧上画 "T+xxxh" 角标（默认 0=不画）
"""

import os, re, sys, glob
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import cv2
import numpy as np

# ===== 配置 =====
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/ecmwf/upper_airfields")).resolve()
FOURCC   = os.environ.get("FOURCC", "mp4v")
# FPS_OUT  = float(os.environ.get("FPS_OUT", "6"))
FPS_OUT  = float(os.environ.get("FPS_OUT", "2"))
DRAW_TIME = str(os.environ.get("DRAW_TIME", "0")) in ("1", "true", "True")

LAYER_ORDER = ["850hPa", "700hPa", "500hPa", "surface"]
STEP_JOBS = [
    # (step_dir, tag, hours_range, out_layer_dir, out_combined_dir)
    ("step3h",  "012-036", (12, 36), "products/analyze_3h_layers", "products/analyze_3h"),
    ("step12h", "012-024", (12, 24), "products/result_12h_layers", "products/result_12h"),
]

# ===== 工具 =====
def parse_size_lock():
    s = os.environ.get("SIZE_LOCK")
    if not s:
        return None
    try:
        w, h = s.lower().split("x")
        return (int(w), int(h))
    except Exception:
        raise ValueError("SIZE_LOCK 应为 'WxH'，例如 1920x1080")

# def pick_latest_ready_root(cli_arg=None) -> Path:
#     if cli_arg:
#         root = Path(cli_arg).resolve()
#         if not root.exists():
#             raise FileNotFoundError(f"指定路径不存在：{root}")
#         return root
#     today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
#     for d in range(0, 5):
#         date_root = DATA_ROOT / (today - timedelta(days=d)).strftime("%Y-%m-%d")
#         if not date_root.is_dir():
#             continue
#         subs = sorted([p for p in date_root.iterdir()
#                        if p.is_dir() and re.fullmatch(r"\d{10}", p.name)], reverse=True)
#         for s in subs:
#             # 任一 step 目录存在即可视为“就绪”
#             if (s / "step3h").exists() or (s / "step12h").exists():
#                 print(f"📁 Root = {s}（最近就绪）")
#                 return s
#     raise FileNotFoundError(f"最近5天在 {DATA_ROOT} 下未找到就绪起报目录")

def pick_latest_ready_root(cli_arg=None) -> Path:
    """
    自动选取最近一次“就绪”的起报根目录 root：

    兼容两种常见结构：
      1) /DATA_ROOT/YYYY-MM-DD/step3h/...              （日期目录直接就是 root）
      2) /DATA_ROOT/YYYY-MM-DD/2025111212/step3h/...   （日期下面再套一层起报目录）

    判定为“就绪”的条件：存在任意一个 step 目录：
      step3h / step6h / step12h / step24h
    """
    # 如果命令行已经显式指定 root，就直接用它，不再自动扫描
    if cli_arg:
        root = Path(cli_arg).resolve()
        if not root.exists():
            raise FileNotFoundError(f"指定路径不存在：{root}")
        print(f"📁 使用命令行指定 root：{root}")
        return root

    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()

    # 往前最多找 5 天
    for d in range(0, 5):
        date_root = DATA_ROOT / (today - timedelta(days=d)).strftime("%Y-%m-%d")
        if not date_root.is_dir():
            continue

        # --- 情况1：日期目录本身就是 root（直接有 step 目录） ---
        for step_name in ("step3h", "step6h", "step12h", "step24h"):
            if (date_root / step_name).exists():
                print(f"📂 选定 root = {date_root}（{step_name} 存在）")
                return date_root

        # --- 情况2：日期目录下面有若干子目录（起报目录），任意一个有 step 目录即可 ---
        subs = sorted(
            [p for p in date_root.iterdir() if p.is_dir()],
            reverse=True,
        )
        for s in subs:
            for step_name in ("step3h", "step6h", "step12h", "step24h"):
                if (s / step_name).exists():
                    print(f"📂 选定 root = {s}（{step_name} 存在）")
                    return s

    raise FileNotFoundError(f"最近5天在 {DATA_ROOT} 下未找到就绪起报目录（step3h/step6h/step12h/step24h 任一）")

# def pick_latest_ready_root(cli_arg=None) -> Path:
#     """
#     自动挑选最近 5 天内“就绪”的起报目录。

#     兼容两种结构：
#       1) 新/推荐结构：DATA_ROOT/YYYY-MM-DD/YYYYMMDDHH/step3h, step12h, products
#       2) 简化结构：    DATA_ROOT/YYYY-MM-DD/step3h, step12h, products
#     """
#     if cli_arg:
#         root = Path(cli_arg).resolve()
#         if not root.exists():
#             raise FileNotFoundError(f"指定路径不存在：{root}")
#         print(f"📁 使用显式传入的 root: {root}")
#         return root

#     today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
#     for d in range(0, 5):
#         date_root = DATA_ROOT / (today - timedelta(days=d)).strftime("%Y-%m-%d")
#         if not date_root.is_dir():
#             continue

#         # --- 1) 优先查找 YYYYMMDDHH 子目录 ---
#         subs = sorted(
#             [
#                 p for p in date_root.iterdir()
#                 if p.is_dir() and re.fullmatch(r"\d{10}", p.name)
#             ],
#             reverse=True,
#         )
#         for s in subs:
#             if (s / "step3h").exists() or (s / "step12h").exists() or (s / "products").exists():
#                 print(f"📁 Root = {s}（最近就绪，使用起报子目录）")
#                 return s

#         # --- 2) 兼容：step3h/step12h 直接在 date_root 下 ---
#         if (date_root / "step3h").exists() or (date_root / "step12h").exists() or (date_root / "products").exists():
#             print(f"📁 Root = {date_root}（最近就绪，使用日期目录本身）")
#             return date_root

#     raise FileNotFoundError(f"最近5天在 {DATA_ROOT} 下未找到就绪起报目录")


# def pick_latest_ready_root(base: Path) -> Path:
#     """
#     自动挑选“最近就绪”的起报目录。

#     策略分两步：
#       1）优先按原来的规则：base/YYYY-MM-DD/YYYYMMDDHH 下是否有 step3h/step6h/step12h/products
#       2）如果 1）找不到，则在 base 下递归扫描所有子目录，只要本目录里包含任意一个
#          ["products", "step3h", "step6h", "step12h"]，就视为“候选根目录”，
#          从中按 mtime 选最新一个。
#     """
#     base = Path(base).resolve()
#     today = datetime.now(ZoneInfo("Asia/Shanghai")).date()

#     # ---------- 第一轮：按 YYYY-MM-DD / YYYYMMDDHH 结构查找 ----------
#     for d in range(0, 5):
#         date_dir = base / (today - timedelta(days=d)).strftime("%Y-%m-%d")
#         if not date_dir.is_dir():
#             continue

#         # 1) 优先看起报子目录：YYYYMMDDHH
#         subs = sorted(
#             [
#                 p for p in date_dir.iterdir()
#                 if p.is_dir() and re.fullmatch(r"\d{10}", p.name)
#             ],
#             reverse=True,
#         )
#         for s in subs:
#             if any(
#                 (s / name).exists()
#                 for name in ("products", "step3h", "step6h", "step12h")
#             ):
#                 print(f"📁 使用最近就绪目录：{s}（按日期/起报结构匹配）")
#                 return s

#         # 2) 兼容：step3h/step6h/step12h/products 直接在 YYYY-MM-DD 下面
#         if any(
#             (date_dir / name).exists()
#             for name in ("products", "step3h", "step6h", "step12h")
#         ):
#             print(f"📁 使用最近就绪目录：{date_dir}（日期目录本身就绪）")
#             return date_dir

#     # ---------- 第二轮：全局回退扫描 ----------
#     print(f"ℹ️ 按日期结构未找到就绪目录，开始在 {base} 下递归扫描候选目录…")

#     candidates: list[Path] = []
#     for p in base.rglob("*"):
#         if not p.is_dir():
#             continue
#         if any(
#             (p / name).is_dir()
#             for name in ("products", "step3h", "step6h", "step12h")
#         ):
#             candidates.append(p)

#     if candidates:
#         # 按最近修改时间挑选最新的一个
#         best = max(candidates, key=lambda p: p.stat().st_mtime)
#         print(f"📁 使用回退就绪目录：{best}")
#         return best

#     # 真的一个都没有，只能报错
#     raise FileNotFoundError(f"最近 5 天在 {base} 下未找到就绪目录（包含 products/step3h/step6h/step12h 的任一目录）")


def extract_hour_from_name(name: str):
    """
    支持两种常见命名：
      1) ..._plus009h.png      → 009
      2) ..._006h_....png      → 006
    回退：匹配 _(\d{3})[h.] 取三位数
    """
    m = re.search(r"plus(\d{3})h", name)
    if not m:
        m = re.search(r"_(\d{3})h[_\.]", name)
    if not m:
        m = re.search(r"_(\d{3})[h\.]", name)
    return int(m.group(1)) if m else None

def collect_frames(dirp: Path, hrange):
    """
    收集目标目录里的 PNG，并按小时筛选/排序
    hrange = (h0, h1) 包含端点
    """
    paths = []
    for ext in ("*.png", "*.PNG"):
        paths.extend(glob.glob(str(dirp / ext)))
    items = []
    h0, h1 = hrange
    for p in paths:
        h = extract_hour_from_name(os.path.basename(p))
        if h is None:
            continue
        if h0 <= h <= h1:
            items.append((h, p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def overlay_time(frame, label: str):
    # 半透明黑底条 + 白字（左上角）
    overlay = frame.copy()
    h, w = frame.shape[:2]
    box_w, box_h = 240, 54
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def write_video_from_images(img_list, out_mp4: Path, fps: float, size_lock=None, fourcc="mp4v"):
    if not img_list:
        raise RuntimeError(f"没有帧可写：{out_mp4}")
    first = cv2.imread(img_list[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"无法读取首帧：{img_list[0]}")

    if size_lock is None:
        H, W = first.shape[:2]
        size = (W, H)
    else:
        size = (size_lock[0], size_lock[1])

    ensure_dir(out_mp4.parent)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
    if not vw.isOpened():
        raise RuntimeError(f"视频写入器打开失败：{out_mp4} fourcc={fourcc}")

    count = 0
    for p in img_list:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            print(f"[warn] 跳过无法读取：{p}")
            continue
        if (im.shape[1], im.shape[0]) != size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)

        if DRAW_TIME:
            h = extract_hour_from_name(os.path.basename(p))
            if h is not None:
                im = overlay_time(im, f"T+{h:03d}h")

        vw.write(im)
        count += 1

    vw.release()
    print(f"[OK] 写出：{out_mp4}  帧数={count}  fps={fps}  size={size[0]}x{size[1]}")

def concat_videos_cv2(files, out_mp4: Path, fps_out: float, size_lock=None, fourcc="mp4v"):
    """
    纯 OpenCV 线性串接：逐段读帧→（必要时重采样/重定尺寸）→统一写出
    注意：OpenCV 没有“无损拼接”，这里是重新编码。
    策略：统一用 fps_out，若源 fps 高于 fps_out，则按比例跳帧。
    """
    # 获取每段属性
    props = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{f}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or w <= 0 or h <= 0 or n <= 0:
            raise RuntimeError(f"视频无效：{f} (fps={fps}, size={w}x{h}, frames={n})")
        props.append((fps, w, h, n))
        print(f"[INFO] {Path(f).name}: fps={fps:.3f}, size={w}x{h}, frames={n}")

    # 输出尺寸
    if size_lock is None:
        size_lock = (props[0][1], props[0][2])

    ensure_dir(out_mp4.parent)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*fourcc), float(fps_out), size_lock)
    if not vw.isOpened():
        raise RuntimeError(f"视频写入器打开失败：{out_mp4} fourcc={fourcc}")

    total = 0
    for f, (in_fps, w, h, n) in zip(files, props):
        cap = cv2.VideoCapture(str(f))
        ratio = (in_fps / fps_out) if fps_out > 0 else 1.0
        next_keep = 0.0
        idx = 0
        print(f"[APPEND] {Path(f).name}  (in_fps={in_fps:.3f} → out_fps={fps_out:.3f}, ratio={ratio:.3f})")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= int(next_keep - 1e-6):
                if (frame.shape[1], frame.shape[0]) != size_lock:
                    frame = cv2.resize(frame, size_lock, interpolation=cv2.INTER_AREA)
                vw.write(frame)
                total += 1
                next_keep += ratio
            idx += 1
        cap.release()

    vw.release()
    print(f"[OK] 串接完成：{out_mp4}  总帧数={total}  fps={fps_out}  size={size_lock[0]}x{size_lock[1]}")

# ===== 任务流 =====
# def build_per_layer_videos(root: Path, step_dir: str, tag: str, hrange, out_layer_dir: str):
#     """
#     从 step_dir 下四个层目录读取 PNG，生成四个单层 mp4。
#     """
#     produced = {}
#     for layer in LAYER_ORDER:
#         src_dir = root / step_dir / layer
#         if not src_dir.exists():
#             print(f"ℹ️ 缺目录：{src_dir}，跳过该层")
#             continue
#         imgs = collect_frames(src_dir, hrange)
#         if not imgs:
#             print(f"ℹ️ {step_dir}/{layer} 在 {hrange} 无图片，跳过该层")
#             continue
#         out_mp4 = root / out_layer_dir / f"{layer}_{tag}.mp4"
#         write_video_from_images(imgs, out_mp4, fps=FPS_OUT, size_lock=parse_size_lock(), fourcc=FOURCC)
#         produced[layer] = out_mp4
#     return produced
# def build_per_layer_videos(root: Path, step_dir: str, tag: str, hrange, out_layer_dir: str):
#     """
#     从 step_dir 下四个层目录读取 PNG，生成四个单层 mp4。
#     新结构兼容：
#       - 上空:   root/step6h/{850hPa,700hPa,500hPa}/*.png
#       - 地面:   root/step6h/surface/synoptic/*.png  或 surface/phase/*.png
#       - 旧结构: root/step6h/surface/*.png
#     """
#     produced = {}
#     for layer in LAYER_ORDER:
#         # === 针对 surface 做兼容处理 ===
#         if layer == "surface":
#             # 按优先级依次尝试：synoptic → phase → 老的 surface 根目录
#             candidates = [
#                 root / step_dir / "surface" / "synoptic",
#                 root / step_dir / "surface" / "phase",
#                 root / step_dir / "surface",
#             ]
#             src_dir = None
#             for d in candidates:
#                 if d.exists():
#                     src_dir = d
#                     break
#             if src_dir is None:
#                 print(f"ℹ️ 缺目录：{root/step_dir/'surface'}（synoptic/phase 也无），跳过 surface")
#                 continue
#         else:
#             # 上空层：沿用原逻辑
#             src_dir = root / step_dir / layer

#         if not src_dir.exists():
#             print(f"ℹ️ 缺目录：{src_dir}，跳过该层")
#             continue

#         imgs = collect_frames(src_dir, hrange)
#         if not imgs:
#             print(f"ℹ️ {step_dir}/{layer} 在 {hrange} 无图片，跳过该层")
#             continue

#         out_mp4 = root / out_layer_dir / f"{layer}_{tag}.mp4"
#         write_video_from_images(
#             imgs, out_mp4,
#             fps=FPS_OUT,
#             size_lock=parse_size_lock(),
#             fourcc=FOURCC
#         )
#         produced[layer] = out_mp4

#     return produced


# def build_per_layer_videos(root: Path, step_dir: str, tag: str, hrange, out_layer_dir: str):
#     """
#     从 step_dir 下四个层目录读取 PNG，生成四个单层 mp4。

#     约定：
#       - 上空:   root/stepXXh/{850hPa,700hPa,500hPa}/*.png
#       - 地面:   root/stepXXh/surface/phase/*.png  + surface/synoptic/*.png
#                 若只有其中一种，则用现有的；若连 surface/ 都没有，则跳过 surface。
#     """
#     produced = {}
#     h0, h1 = hrange

#     for layer in LAYER_ORDER:
#         # ================= surface: phase + synoptic 一起做成一个视频 =================
#         if layer == "surface":
#             phase_dir    = root / step_dir / "surface" / "phase"
#             synoptic_dir = root / step_dir / "surface" / "synoptic"
#             legacy_dir   = root / step_dir / "surface"   # 旧结构兜底

#             # 收集候选目录（按优先级：phase、synoptic、legacy）
#             dir_list: list[tuple[int, Path]] = []
#             if phase_dir.exists():
#                 dir_list.append((0, phase_dir))      # type_id=0: phase
#             if synoptic_dir.exists():
#                 dir_list.append((1, synoptic_dir))   # type_id=1: synoptic
#             if not dir_list and legacy_dir.exists():
#                 dir_list.append((2, legacy_dir))     # type_id=2: 旧结构兜底

#             if not dir_list:
#                 print(f"ℹ️ 缺目录：{root/step_dir/'surface'}（phase/synoptic/legacy 都不存在），跳过 surface")
#                 continue

#             # 把所有地面 PNG（phase + synoptic）按 “小时 + 类型” 合在一起
#             items: list[tuple[int, int, str]] = []   # (hour, type_id, path)
#             for type_id, d in dir_list:
#                 for ext in ("*.png", "*.PNG"):
#                     for p in glob.glob(str(d / ext)):
#                         h = extract_hour_from_name(os.path.basename(p))
#                         if h is None:
#                             continue
#                         if h0 <= h <= h1:
#                             items.append((h, type_id, p))

#             if not items:
#                 print(f"ℹ️ {step_dir}/surface 在 {hrange} 无图片，跳过 surface")
#                 continue

#             # 先按 forecast 小时排序，再按类型排序（保证同一时次上，phase 与 synoptic 顺序稳定）
#             items.sort(key=lambda x: (x[0], x[1]))
#             img_list = [p for _, _, p in items]

#             out_mp4 = root / out_layer_dir / f"{layer}_{tag}.mp4"
#             write_video_from_images(
#                 img_list,
#                 out_mp4,
#                 fps=FPS_OUT,
#                 size_lock=parse_size_lock(),
#                 fourcc=FOURCC,
#             )
#             produced[layer] = out_mp4
#             continue  # ✅ surface 处理完，继续下一个 layer

#         # ================= 上空三层：沿用旧逻辑 =================
#         src_dir = root / step_dir / layer
#         if not src_dir.exists():
#             print(f"ℹ️ 缺目录：{src_dir}，跳过该层")
#             continue

#         imgs = collect_frames(src_dir, hrange)
#         if not imgs:
#             print(f"ℹ️ {step_dir}/{layer} 在 {hrange} 无图片，跳过该层")
#             continue

#         out_mp4 = root / out_layer_dir / f"{layer}_{tag}.mp4"
#         write_video_from_images(
#             imgs,
#             out_mp4,
#             fps=FPS_OUT,
#             size_lock=parse_size_lock(),
#             fourcc=FOURCC,
#         )
#         produced[layer] = out_mp4

#     return produced

def build_per_layer_videos(root: Path, step_dir: str, tag: str, hrange, out_layer_dir: str):
    """
    从 step_dir 下读取 PNG，生成各层单层 mp4。

    约定结构：
      - 上空:   root/stepXXh/{850hPa,700hPa,500hPa}/*.png
      - 地面降水:   root/stepXXh/surface/phase/*.png       → surface_phase_{tag}.mp4
      - 地面MSLP+10m风: root/stepXXh/surface/synoptic/*.png → surface_synoptic_{tag}.mp4
    """
    produced = {}

    # ===== 1. 上空三层：850 / 700 / 500 =====
    for layer in ["850hPa", "700hPa", "500hPa"]:
        src_dir = root / step_dir / layer
        if not src_dir.exists():
            print(f"ℹ️ 缺目录：{src_dir}，跳过 {layer}")
            continue

        imgs = collect_frames(src_dir, hrange)
        if not imgs:
            print(f"ℹ️ {step_dir}/{layer} 在 {hrange} 无图片，跳过该层")
            continue

        out_mp4 = root / out_layer_dir / f"{layer}_{tag}.mp4"
        write_video_from_images(
            imgs,
            out_mp4,
            fps=FPS_OUT,
            size_lock=parse_size_lock(),
            fourcc=FOURCC,
        )
        produced[layer] = out_mp4

    # ===== 2. 地面降水：surface/phase → surface_phase_{tag}.mp4 =====
    phase_dir = root / step_dir / "surface" / "phase"
    if phase_dir.exists():
        imgs_phase = collect_frames(phase_dir, hrange)
        if imgs_phase:
            out_phase = root / out_layer_dir / f"surface_phase_{tag}.mp4"
            write_video_from_images(
                imgs_phase,
                out_phase,
                fps=FPS_OUT,
                size_lock=parse_size_lock(),
                fourcc=FOURCC,
            )
            produced["surface_phase"] = out_phase
        else:
            print(f"ℹ️ {step_dir}/surface/phase 在 {hrange} 无图片，跳过 surface_phase")
    else:
        print(f"ℹ️ 缺目录：{phase_dir}，跳过 surface_phase")

    # ===== 3. 地面 MSLP+10m 风：surface/synoptic → surface_synoptic_{tag}.mp4 =====
    syn_dir = root / step_dir / "surface" / "synoptic"
    if syn_dir.exists():
        imgs_syn = collect_frames(syn_dir, hrange)
        if imgs_syn:
            out_syn = root / out_layer_dir / f"surface_synoptic_{tag}.mp4"
            write_video_from_images(
                imgs_syn,
                out_syn,
                fps=FPS_OUT,
                size_lock=parse_size_lock(),
                fourcc=FOURCC,
            )
            produced["surface_synoptic"] = out_syn
        else:
            print(f"ℹ️ {step_dir}/surface/synoptic 在 {hrange} 无图片，跳过 surface_synoptic")
    else:
        print(f"ℹ️ 缺目录：{syn_dir}，跳过 surface_synoptic")

    return produced




# def build_combined_video(root: Path, produced: dict, out_combined_dir: str, tag: str):
#     """
#     将四个层按 850→700→500→surface 顺序串接为一个视频。缺层则跳过该层。
#     """
#     files = [produced.get(layer) for layer in LAYER_ORDER if produced.get(layer)]
#     if len(files) == 0:
#         print("ℹ️ 无可串接的层，跳过合并。")
#         return
#     out_mp4 = root / out_combined_dir / f"combined_{tag}.mp4"
#     concat_videos_cv2([str(p) for p in files], out_mp4, fps_out=FPS_OUT,
#                       size_lock=parse_size_lock(), fourcc=FOURCC)

def build_combined_video(root: Path, produced: dict, out_combined_dir: str, tag: str):
    """
    将各层按顺序串接为一个视频：

      850hPa → 700hPa → 500hPa → surface_phase → surface_synoptic

    缺哪一层就跳过哪一层。
    """
    files = []

    # 1) 上空三层
    for layer in ["850hPa", "700hPa", "500hPa"]:
        p = produced.get(layer)
        if p:
            files.append(p)

    # 2) 地面降水（phase）
    p_phase = produced.get("surface_phase")
    if p_phase:
        files.append(p_phase)

    # 3) 地面 MSLP+10m 风（synoptic）
    p_syn = produced.get("surface_synoptic")
    if p_syn:
        files.append(p_syn)

    if not files:
        print("ℹ️ 无可串接的视频，跳过 combined。")
        return

    out_mp4 = root / out_combined_dir / f"combined_{tag}.mp4"
    concat_videos_cv2(
        [str(p) for p in files],
        out_mp4,
        fps_out=FPS_OUT,
        size_lock=parse_size_lock(),
        fourcc=FOURCC,
    )




def main():
    cli_root = sys.argv[1] if len(sys.argv) > 1 else None
    root = pick_latest_ready_root(cli_root)

    for step_dir, tag, hrange, out_layer_dir, out_combined_dir in STEP_JOBS:
        print(f"\n===== 处理 {step_dir}  时段 {tag}（{hrange[0]}~{hrange[1]}h）=====")
        produced = build_per_layer_videos(root, step_dir, tag, hrange, out_layer_dir)
        build_combined_video(root, produced, out_combined_dir, tag)

    print("\n🎉 全部视频生成与合并完成。")

if __name__ == "__main__":
    main()

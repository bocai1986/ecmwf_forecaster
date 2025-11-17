#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试版：只画地面两张图
1) 降水相态专图（雨 / 雪 / 夹雪 + 数值）
2) MSLP + 10m 风 + 2m 温度

说明：
- 优先使用你已经下载好的地面 GRIB
- 如果没有，就去 ECMWF OpenData 下 (3h, 0-72h)
"""

import os
import re
import time
import random
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from requests.exceptions import HTTPError
from ecmwf.opendata import Client
from scipy.ndimage import gaussian_filter

from metpy.plots import StationPlot
from metpy.units import units
import metpy.calc as mpcalc

# ========== 你可以改的配置 ==========
# 如果已经有下载好的 .grib2，就写绝对路径；否则留空让脚本自己下
EXISTING_SURFACE_PATH = "./offline/output/2025-11-14/ec_surface_2025111212_3h_0-72_0p25.grib2"   # 例如："/data/ecmwf/test_surface/2025-11-11/ec_surface_2025111100_3h_0-72_0p25.grib2"
OUT_ROOT = "./offline/output"   # 输出图片根目录
REGION = (60, 140, 10, 60)             # 东亚范围
JOBS = [(3, 72)]                       # 就这一档


# ========== 通用小工具 ==========

def _resolve_outpath(output_path: str, step_hour: int) -> str:
    # 若传的是目录，则自动拼成文件名
    if output_path.lower().endswith(".png"):
        out_png = output_path
        out_dir = os.path.dirname(out_png) or "."
    else:
        out_dir = output_path
        out_png = os.path.join(out_dir, f"EC_PrecipPhase_{int(step_hour):03d}h.png")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.abspath(out_png)


def resolve_surface_path(path_like: str) -> str:
    """
    允许传目录：如果是目录，就在里面找 .grib2
    允许传文件：如果是文件就直接用
    """
    if os.path.isfile(path_like):
        return path_like
    if os.path.isdir(path_like):
        # 找目录里所有 grib2，按名字排一下
        files = sorted(
            glob.glob(os.path.join(path_like, "*.grib2"))
        )
        if not files:
            raise FileNotFoundError(f"目录里没有找到 .grib2 文件: {path_like}")
        return files[-1]   # 取最后一个，通常是最新的
    raise FileNotFoundError(f"找不到文件或目录: {path_like}")

def extract_cycle_from_filename(filepath: str) -> datetime:
    """
    从文件名里抓 10 位起报时间（2025111100），转成 UTC datetime
    """
    m = re.search(r"(\d{10})", os.path.basename(filepath))
    if not m:
        raise ValueError(f"无法从文件名提取起报时间: {filepath}")
    return datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)


def regrid_to_0125(da: xr.DataArray,
                   lon_min=60, lon_max=140, lat_min=10, lat_max=60,
                   dlon=0.125, dlat=0.125):
    if da is None:
        return None
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    da2 = da.sortby(lon_name)
    if np.any(np.diff(da2[lat_name].values) < 0):
        da2 = da2.sortby(lat_name)
    da2 = da2.sel({lon_name: slice(lon_min, lon_max),
                   lat_name: slice(lat_min, lat_max)})
    new_lons = np.arange(lon_min, lon_max + 1e-9, dlon)
    new_lats = np.arange(lat_min, lat_max + 1e-9, dlat)
    return da2.interp({lon_name: new_lons, lat_name: new_lats}, method="linear")

# rain = np.array([0.1, 2.5, 5, 10, 20, 999])
def get_phase_bounds(interval_hours: int):
    ih = int(interval_hours)
    if ih == 3:
        snow = np.array([0.1, 0.3, 1, 3, 6, 999])
        mix  = np.array([0.1, 0.3, 1, 3, 6, 999])
        rain = np.array([0.05, 2.5, 5, 10, 20, 999])
    else:
        snow = np.array([0.1, 0.7, 2, 5, 8, 999])
        mix  = np.array([0.1, 0.7, 2, 5, 8, 999])
        rain = np.array([0.1, 5, 10, 25, 60, 999])
    return snow, mix, rain


def load_grib_fields(filepath: str) -> dict:
    datasets = cfgrib.open_datasets(filepath)
    out = {}
    for ds in datasets:
        for v in ds.data_vars:
            out[v.lower()] = ds[v]
    print("✅ 读取变量:", list(out.keys()))
    return out


# ========== 下载（可跳过） ==========

def retrieve_surface_only(
    client,
    save_dir: str,
    *,
    step_interval_hours: int,
    max_hour: int,
):
    """
    只下载地面 (msl/tp/2t/10u/10v)，返回 (filepath, steps_array, cycle_utc)
    """
    os.makedirs(save_dir, exist_ok=True)

    steps = np.arange(0, max_hour + 1, step_interval_hours, dtype=int)
    step_str = "/".join(map(str, steps))
    last_step = int(steps.max()) if steps.size else 0

    # 起报时间 → 00/12
    latest_dt = client.latest(type="fc")
    if latest_dt.tzinfo is None:
        latest_dt = latest_dt.replace(tzinfo=timezone.utc)
    h = latest_dt.hour
    if h in (0, 12):
        start_cycle = latest_dt.replace(minute=0, second=0, microsecond=0)
    else:
        start_cycle = latest_dt.replace(hour=(0 if h < 12 else 12),
                                        minute=0, second=0, microsecond=0)

    # 多试几个周期
    cycles = [start_cycle] + [start_cycle - timedelta(hours=12*i) for i in range(1, 4)]
    last_exc = None

    for cycle in cycles:
        date_str = cycle.strftime("%Y-%m-%d")
        time_str = cycle.strftime("%H")
        init_tag = cycle.strftime("%Y%m%d%H")

        target_file = os.path.join(
            save_dir,
            f"ec_surface_{init_tag}_{step_interval_hours}h_0-{last_step}_0p25.grib2"
        )

        req = dict(
            date=date_str,
            time=time_str,
            type="fc",
            stream="oper",
            step=step_str,
            param="msl/tp/2t/10u/10v",
            target=target_file,
        )

        print(f"📡 下载地面: {date_str} {time_str} UTC step={step_str}")
        try:
            client.retrieve(**req)
            print(f"✅ surface 下载成功: {target_file}")
            return target_file, steps, cycle
        except HTTPError as e:
            last_exc = e
            print(f"⚠️ HTTP {e.response.status_code} → 用上一周期 …")
            continue
        except Exception as e:
            last_exc = e
            print(f"⚠️ 其他错误: {e} → 用上一周期 …")
            continue

    raise RuntimeError(f"surface 下载失败，最后错误: {last_exc}")


# ========== 图1：降水相态专图 ==========

# def plot_precip_phase_only(
#     surface_fields: dict,
#     step_hour: int,
#     *,
#     output_dir="./output_images",
#     extent=(60, 140, 10, 60),
#     interval_hours=6,
#     accum_alignment="trailing",  # "trailing"|"leading"|"center"
# ):
#     """
#     单时效降水相态专图：
#     - 只画 雪 / 雨夹雪 / 雨 三层分色
#     - 降水用“时段增量”(tp_this - tp_prev)，支持 trailing/leading/center
#     - 三个场都统一重采样到 0.125°，确保 shape 一致
#     - 右侧三条独立色标
#     - 有海岸线、国界、经纬网
#     """
#     import os, time
#     import numpy as np
#     import xarray as xr
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import ListedColormap, BoundaryNorm
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature

#     t0_all = time.perf_counter()
#     os.makedirs(output_dir, exist_ok=True)

#     # ---------- helpers ----------
#     def pick_first(d: dict, keys):
#         for k in keys:
#             if k in d and d[k] is not None:
#                 return d[k]
#         return None

#     def _to_2d(da: xr.DataArray | None):
#         if da is None:
#             return None
#         arr = da
#         for dims in [("latitude", "longitude"), ("lat", "lon"), ("y", "x")]:
#             if all(dim in arr.dims for dim in dims):
#                 arr = arr.transpose(*dims)
#                 break
#         arr = arr.squeeze()
#         while hasattr(arr, "ndim") and arr.ndim > 2:
#             arr = arr.isel({arr.dims[0]: 0}).squeeze()
#         return arr.values

#     def _regrid(da):
#         # 你项目里已有的 0.125° 插值函数，直接调用
#         return regrid_to_0125(da) if da is not None else None

#     def _steps_hours(da: xr.DataArray | None):
#         if da is None or "step" not in da.coords:
#             return np.array([], dtype=int)
#         step = da.coords["step"].values
#         if np.issubdtype(step.dtype, np.timedelta64):
#             return step.astype("timedelta64[h]").astype(int)
#         return np.asarray(step, dtype=int)

#     def _sel_step(da: xr.DataArray | None, h: int):
#         if da is None or "step" not in da.coords:
#             return None
#         avail = _steps_hours(da)
#         if avail.size == 0:
#             return None
#         nearest = int(avail[np.argmin(np.abs(avail - int(h)))])
#         key = (
#             nearest
#             if not np.issubdtype(da.coords["step"].values.dtype, np.timedelta64)
#             else np.timedelta64(nearest, "h")
#         )
#         try:
#             return da.sel(step=key)
#         except Exception:
#             return None

#     def _valid_time_str(da_like, h):
#         try:
#             base = getattr(da_like, "time", None)
#             if base is not None:
#                 v = base.values
#                 if np.ndim(v) == 0:
#                     tstr = np.datetime_as_string(v, unit="m")
#                 else:
#                     tstr = np.datetime_as_string(v[-1], unit="m")
#                 return f"{tstr} (+{int(h):03d}h)"
#         except Exception:
#             pass
#         return f"+{int(h):03d}h"

#     # ---------- fetch fields ----------
#     tp  = pick_first(surface_fields, ["tp"])          # m（自起报累积）
#     t2m = pick_first(surface_fields, ["t2m", "2t"])   # K
#     if tp is None:
#         raise ValueError("plot_precip_phase_only(): 需要 tp 字段")

#     # ---------- 降水“时段增量” ----------
#     iv = int(interval_hours)
#     if accum_alignment == "trailing":
#         prev_h = max(0, int(step_hour) - iv); this_h = int(step_hour)
#         if int(step_hour) == 0:  # +000h 特判
#             prev_h, this_h = 0, iv
#     elif accum_alignment == "leading":
#         prev_h = int(step_hour); this_h = int(step_hour) + iv
#     else:  # center
#         half = iv // 2
#         prev_h = max(0, int(step_hour) - half); this_h = int(step_hour) + half

#     tp_prev = _sel_step(tp, prev_h)
#     tp_this = _sel_step(tp, this_h)
#     if tp_prev is None or tp_this is None:
#         raise ValueError(f"tp 在 {prev_h}h 或 {this_h}h 缺数据")

#     # 把 time/valid_time 去掉，保证差分能做
#     def _strip_align_coords(da):
#         keep = set(["latitude", "longitude", "lat", "lon"])
#         drop = [c for c in da.coords if c not in keep]
#         return da.reset_coords(drop=True) if drop else da

#     tp_prev_c = _strip_align_coords(tp_prev)
#     tp_this_c = _strip_align_coords(tp_this)

#     tp_diff_da = tp_this_c - tp_prev_c
#     tp_diff_da.attrs = tp_this.attrs

#     # 重采样到 0.125°
#     tp_0125_da = _regrid(tp_diff_da)

#     # m → mm
#     if tp_0125_da is not None:
#         units = (getattr(tp_0125_da, "attrs", {}) or {}).get("units", "")
#         if isinstance(units, str) and units.strip().lower() in ("m", "meter", "metre"):
#             tp_0125_da = tp_0125_da * 1000.0
#             tp_0125_da.attrs["units"] = "mm"
#         tp_0125_da = tp_0125_da.where(tp_0125_da >= 0)

#     # 温度也重采样一下，用来分相态
#     if t2m is not None:
#         t2m_this = _sel_step(t2m, int(step_hour))
#         t2m_0125 = _regrid(t2m_this)
#     else:
#         t2m_0125 = None

#     # ---------- lon/lat ----------
#     # 以 tp_0125_da 为基准
#     lons = tp_0125_da["longitude"].values
#     lats = tp_0125_da["latitude"].values
#     lon2d, lat2d = np.meshgrid(lons, lats)

#     # ---------- 数值场 ----------
#     tp_mm = _to_2d(tp_0125_da)
#     t2m_c = (_to_2d(t2m_0125) - 273.15) if isinstance(t2m_0125, xr.DataArray) else None
#     valid_time = _valid_time_str(tp_0125_da, step_hour)

#     # ---------- 相态分离 ----------
#     snow_mm = mix_mm = rain_mm = None
#     if tp_mm is not None:
#         if t2m_c is not None:
#             snow_mask = (t2m_c <= 0.0)
#             mix_mask  = (t2m_c > 0.0) & (t2m_c <= 2.0)
#             rain_mask = (t2m_c > 2.0)
#             snow_mm = np.where(snow_mask, tp_mm, np.nan)
#             mix_mm  = np.where(mix_mask,  tp_mm, np.nan)
#             rain_mm = np.where(rain_mask, tp_mm, np.nan)
#         else:
#             rain_mm = tp_mm.copy()

#     # ---------- 颜色映射 ----------
#     snow_bounds, mix_bounds, rain_bounds = get_phase_bounds(interval_hours)

#     snow_cmap = ListedColormap([
#         "#CCFFFF", "#66FFFF", "#00E6B8", "#00CC66", "#006600"
#     ])
#     mix_cmap = ListedColormap([
#         "#FFFACD", "#FFD700", "#FF8C00", "#FF4500", "#FF00FF"
#     ])
#     rain_cmap = ListedColormap([
#         "#ADD8E6", "#00BFFF", "#1E90FF", "#8A2BE2", "#4B0082"
#     ])

#     snow_norm = BoundaryNorm(snow_bounds, snow_cmap.N, clip=True)
#     mix_norm  = BoundaryNorm(mix_bounds,  mix_cmap.N,  clip=True)
#     rain_norm = BoundaryNorm(rain_bounds, rain_cmap.N, clip=True)

#     # ---------- 绘图 ----------
#     fig = plt.figure(figsize=(14, 7))
#     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#     ax.set_extent(extent, crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7, edgecolor="0.35")
#     ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.6, edgecolor="0.45")
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='0.85', linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False

#     # 右侧预留面板
#     fig.subplots_adjust(right=0.86)

#     # 区域遮罩
#     x0, x1, y0, y1 = extent
#     mask = (lon2d < x0) | (lon2d > x1) | (lat2d < y0) | (lat2d > y1)

#     def _plot_layer(mm, cmap, norm):
#         if mm is None:
#             return None
#         z = mm.copy()
#         z[mask] = np.nan
#         return ax.contourf(
#             lon2d, lat2d, z,
#             levels=norm.boundaries,
#             cmap=cmap,
#             norm=norm,
#             transform=ccrs.PlateCarree(),
#             alpha=0.95,
#             antialiased=True,
#             extend="max",
#         )

#     cs_snow = _plot_layer(snow_mm, snow_cmap, snow_norm)
#     cs_mix  = _plot_layer(mix_mm,  mix_cmap,  mix_norm)
#     cs_rain = _plot_layer(rain_mm, rain_cmap, rain_norm)

#     # ---------- 右侧三竖色标 ----------
#     ax_pos = ax.get_position()
#     cbar_panel_left = 0.88
#     cbar_width  = 0.022
#     cbar_height = 0.22
#     cbar_gap    = 0.035

#     top_y = ax_pos.y1 - cbar_height - 0.01
#     mid_y = top_y - cbar_height - cbar_gap
#     bot_y = mid_y - cbar_height - cbar_gap

#     def _make_cbar(mappable, bounds, label, rect):
#         if mappable is None:
#             return
#         cax = fig.add_axes(rect)
#         ticks = [v for v in bounds[1:-1]]
#         cb = fig.colorbar(mappable, cax=cax, orientation="vertical", ticks=ticks)
#         ticklabels = [str(v) for v in ticks]
#         if len(bounds) >= 2:
#             ticklabels[-1] = f"≥{int(bounds[-2])}"
#         cb.ax.set_yticklabels(ticklabels)
#         cb.set_label(label, rotation=90, fontsize=10)
#         cb.ax.tick_params(labelsize=9)

#     if cs_snow is not None:
#         _make_cbar(cs_snow, snow_bounds, "Snow (mm)",  [cbar_panel_left, top_y, cbar_width, cbar_height])
#     if cs_mix is not None:
#         _make_cbar(cs_mix,  mix_bounds,  "Mixed (mm)", [cbar_panel_left, mid_y, cbar_width, cbar_height])
#     if cs_rain is not None:
#         _make_cbar(cs_rain, rain_bounds, "Rain (mm)",  [cbar_panel_left, bot_y, cbar_width, cbar_height])

#     ax.set_title(f"ECMWF Precipitation phase (Δ={int(interval_hours)}h)\n{valid_time}",
#                  fontsize=15, pad=10)

#     out_png = os.path.join(output_dir, f"EC_PrecipPhase_{int(step_hour):03d}h.png")
#     plt.tight_layout()
#     fig.savefig(out_png, dpi=180, bbox_inches="tight")
#     plt.close(fig)

#     print(f"✅ [Precip phase] +{int(step_hour):03d}h saved: {out_png} | ⏱️ {time.perf_counter() - t0_all:.2f}s")



# welldone
# def plot_precip_phase_only(
#     surface_fields: dict,
#     step_hour: int,
#     *,
#     output_dir="./output_images",
#     extent=(60, 140, 10, 60),
#     interval_hours=6,
#     accum_alignment="trailing",  # "trailing"|"leading"|"center"
#     label_stride: int = 4,       # 数值标注的稀疏程度
# ):
#     """
#     单时效降水相态专图（带数值）：
#     - 只画 雪 / 雨夹雪 / 雨 三层分色
#     - 降水用时段增量 (tp_this - tp_prev)
#     - 重采样到 0.125°
#     - 右侧三个竖色标
#     - 经纬网、海岸线、国界
#     - 数值稀疏标注（label_stride）
#     """
#     import os, time
#     import numpy as np
#     import xarray as xr
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import ListedColormap, BoundaryNorm
#     from matplotlib.cm import ScalarMappable
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature

#     t0_all = time.perf_counter()
#     os.makedirs(output_dir, exist_ok=True)

#     # ---------- helpers ----------
#     def pick_first(d: dict, keys):
#         for k in keys:
#             if k in d and d[k] is not None:
#                 return d[k]
#         return None

#     def _to_2d(da: xr.DataArray | None):
#         if da is None:
#             return None
#         arr = da
#         for dims in [("latitude", "longitude"), ("lat", "lon"), ("y", "x")]:
#             if all(dim in arr.dims for dim in dims):
#                 arr = arr.transpose(*dims)
#                 break
#         arr = arr.squeeze()
#         while hasattr(arr, "ndim") and arr.ndim > 2:
#             arr = arr.isel({arr.dims[0]: 0}).squeeze()
#         return arr.values

#     def _regrid(da):
#         # 用你项目里的 0.125° 插值
#         return regrid_to_0125(da) if da is not None else None

#     def _steps_hours(da: xr.DataArray | None):
#         if da is None or "step" not in da.coords:
#             return np.array([], dtype=int)
#         step = da.coords["step"].values
#         if np.issubdtype(step.dtype, np.timedelta64):
#             return step.astype("timedelta64[h]").astype(int)
#         return np.asarray(step, dtype=int)

#     def _sel_step(da: xr.DataArray | None, h: int):
#         if da is None or "step" not in da.coords:
#             return None
#         avail = _steps_hours(da)
#         if avail.size == 0:
#             return None
#         nearest = int(avail[np.argmin(np.abs(avail - int(h)))])
#         key = (
#             nearest
#             if not np.issubdtype(da.coords["step"].values.dtype, np.timedelta64)
#             else np.timedelta64(nearest, "h")
#         )
#         try:
#             return da.sel(step=key)
#         except Exception:
#             return None

#     def _valid_time_str(da_like, h):
#         try:
#             base = getattr(da_like, "time", None)
#             if base is not None:
#                 v = base.values
#                 if np.ndim(v) == 0:
#                     tstr = np.datetime_as_string(v, unit="m")
#                 else:
#                     tstr = np.datetime_as_string(v[-1], unit="m")
#                 return f"{tstr} (+{int(h):03d}h)"
#         except Exception:
#             pass
#         return f"+{int(h):03d}h"

#     # ---------- 1. 拿字段 ----------
#     tp  = pick_first(surface_fields, ["tp"])
#     t2m = pick_first(surface_fields, ["t2m", "2t"])
#     if tp is None:
#         raise ValueError("plot_precip_phase_only(): 需要 tp 字段")

#     # ---------- 2. 时段增量 ----------
#     iv = int(interval_hours)
#     if accum_alignment == "trailing":
#         prev_h = max(0, int(step_hour) - iv)
#         this_h = int(step_hour)
#         if int(step_hour) == 0:
#             prev_h, this_h = 0, iv
#     elif accum_alignment == "leading":
#         prev_h = int(step_hour)
#         this_h = int(step_hour) + iv
#     else:  # center
#         half = iv // 2
#         prev_h = max(0, int(step_hour) - half)
#         this_h = int(step_hour) + half

#     tp_prev = _sel_step(tp, prev_h)
#     tp_this = _sel_step(tp, this_h)
#     if tp_prev is None or tp_this is None:
#         raise ValueError(f"tp 在 {prev_h}h 或 {this_h}h 缺数据")

#     def _strip_align_coords(da):
#         keep = set(["latitude", "longitude", "lat", "lon"])
#         drop = [c for c in da.coords if c not in keep]
#         return da.reset_coords(drop=True) if drop else da

#     tp_prev_c = _strip_align_coords(tp_prev)
#     tp_this_c = _strip_align_coords(tp_this)

#     tp_diff_da = tp_this_c - tp_prev_c
#     tp_diff_da.attrs = tp_this.attrs

#     tp_0125_da = _regrid(tp_diff_da)

#     # m -> mm
#     if tp_0125_da is not None:
#         units = (getattr(tp_0125_da, "attrs", {}) or {}).get("units", "")
#         if isinstance(units, str) and units.strip().lower() in ("m", "meter", "metre"):
#             tp_0125_da = tp_0125_da * 1000.0
#             tp_0125_da.attrs["units"] = "mm"
#         tp_0125_da = tp_0125_da.where(tp_0125_da >= 0)

#     # 温度
#     if t2m is not None:
#         t2m_this = _sel_step(t2m, int(step_hour))
#         t2m_0125 = _regrid(t2m_this)
#     else:
#         t2m_0125 = None

#     # ---------- 3. lon/lat & 数值 ----------
#     lons = tp_0125_da["longitude"].values
#     lats = tp_0125_da["latitude"].values
#     lon2d, lat2d = np.meshgrid(lons, lats)

#     tp_mm = _to_2d(tp_0125_da)
#     t2m_c = (_to_2d(t2m_0125) - 273.15) if isinstance(t2m_0125, xr.DataArray) else None
#     valid_time = _valid_time_str(tp_0125_da, step_hour)

#     # ---------- 4. 相态分离 ----------
#     snow_mm = mix_mm = rain_mm = None
#     if tp_mm is not None:
#         if t2m_c is not None:
#             snow_mask = (t2m_c <= 0.0)
#             mix_mask  = (t2m_c > 0.0) & (t2m_c <= 2.0)
#             rain_mask = (t2m_c > 2.0)
#             snow_mm = np.where(snow_mask, tp_mm, np.nan)
#             mix_mm  = np.where(mix_mask,  tp_mm, np.nan)
#             rain_mm = np.where(rain_mask, tp_mm, np.nan)
#         else:
#             rain_mm = tp_mm.copy()

#     # ---------- 5. 配色 ----------
#     snow_bounds, mix_bounds, rain_bounds = get_phase_bounds(interval_hours)

#     snow_cmap = ListedColormap(["#CCFFFF", "#66FFFF", "#00E6B8", "#00CC66", "#006600"])
#     mix_cmap  = ListedColormap(["#FFFACD", "#FFD700", "#FF8C00", "#FF4500", "#FF00FF"])
#     rain_cmap = ListedColormap(["#ADD8E6", "#00BFFF", "#1E90FF", "#8A2BE2", "#4B0082"])

#     snow_norm = BoundaryNorm(snow_bounds, snow_cmap.N, clip=True)
#     mix_norm  = BoundaryNorm(mix_bounds,  mix_cmap.N,  clip=True)
#     rain_norm = BoundaryNorm(rain_bounds, rain_cmap.N, clip=True)

#     # ---------- 6. 绘图 ----------
#     fig = plt.figure(figsize=(14, 7))
#     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#     ax.set_extent(extent, crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7, edgecolor="0.35")
#     ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.6, edgecolor="0.45")
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='0.85', linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False

#     # 右侧预留
#     fig.subplots_adjust(right=0.86)

#     x0, x1, y0, y1 = extent
#     mask = (lon2d < x0) | (lon2d > x1) | (lat2d < y0) | (lat2d > y1)

#     def _plot_layer(mm, cmap, norm):
#         if mm is None:
#             return None
#         z = mm.copy()
#         z[mask] = np.nan
#         return ax.contourf(
#             lon2d, lat2d, z,
#             levels=norm.boundaries,
#             cmap=cmap,
#             norm=norm,
#             transform=ccrs.PlateCarree(),
#             alpha=0.95,
#             antialiased=True,
#             extend="max",
#         )

#     cs_snow = _plot_layer(snow_mm, snow_cmap, snow_norm)
#     cs_mix  = _plot_layer(mix_mm,  mix_cmap,  mix_norm)
#     cs_rain = _plot_layer(rain_mm, rain_cmap, rain_norm)

#     # ---------- 7. 数值叠加（稀疏，整数+小数混排） ----------
#     def _fmt_val(v: float) -> str | None:
#         # 过小或无效的不标
#         if np.isnan(v) or v < 0.05:
#             return None
#         # >=10 取整
#         if v >= 10:
#             return f"{v:.0f}"
#         # [1,10) 近似整数就取整，否则一位小数
#         if v >= 1:
#             r = round(v)
#             if abs(v - r) < 0.05:      # 接近整数的“粘连区”，你也可以改成 0.1
#                 return f"{r:d}"
#             return f"{v:.1f}"
#         # <1 一位小数
#         return f"{v:.1f}"

#     if tp_mm is not None:
#         ny, nx = tp_mm.shape
#         for iy in range(0, ny, label_stride):
#             for ix in range(0, nx, label_stride):
#                 val = tp_mm[iy, ix]
#                 label = _fmt_val(val)
#                 if not label:
#                     continue

#                 lon = lon2d[iy, ix]
#                 lat = lat2d[iy, ix]
#                 ax.text(
#                     lon, lat, label,
#                     ha="center", va="center",
#                     fontsize=6, color="#101010",
#                     bbox=dict(facecolor=(1, 1, 1, 0.35), edgecolor="none", pad=0.4),
#                     transform=ccrs.PlateCarree(),
#                     zorder=6,
#                 )



#     # ---------- 8. 右侧三竖色标 ----------
#     ax_pos = ax.get_position()
#     cbar_panel_left = 0.88
#     cbar_width  = 0.022
#     cbar_height = 0.22
#     cbar_gap    = 0.035

#     top_y = ax_pos.y1 - cbar_height - 0.01
#     mid_y = top_y - cbar_height - cbar_gap
#     bot_y = mid_y - cbar_height - cbar_gap

#     def _make_cbar(norm, cmap, bounds, label, rect):
#         cax = fig.add_axes(rect)
#         sm = ScalarMappable(norm=norm, cmap=cmap)
#         ticks = [v for v in bounds[1:-1]]
#         cb = fig.colorbar(sm, cax=cax, orientation="vertical", ticks=ticks)
#         ticklabels = [str(v) for v in ticks]
#         if len(bounds) >= 2:
#             ticklabels[-1] = f"≥{int(bounds[-2])}"
#         cb.ax.set_yticklabels(ticklabels)
#         cb.set_label(label, rotation=90, fontsize=10)
#         cb.ax.tick_params(labelsize=9)

#     if cs_snow is not None:
#         _make_cbar(snow_norm, snow_cmap, snow_bounds, "Snow (mm)",  [cbar_panel_left, top_y, cbar_width, cbar_height])
#     if cs_mix is not None:
#         _make_cbar(mix_norm,  mix_cmap,  mix_bounds, "Mixed (mm)", [cbar_panel_left, mid_y, cbar_width, cbar_height])
#     if cs_rain is not None:
#         _make_cbar(rain_norm, rain_cmap, rain_bounds, "Rain (mm)",  [cbar_panel_left, bot_y, cbar_width, cbar_height])

#     # ====== 自动生成时间信息 ======
#     # 1. 尝试获取起报时间
#     init_tag = None
#     for name in ["msl", "sp", "tp"]:
#         da = surface_fields.get(name)
#         if da is not None:
#             if "time" in da.coords:
#                 tval = np.array(da["time"].values).ravel()[0]
#                 init_tag = np.datetime_as_string(tval, unit="h").replace("-", "").replace("T", "").replace(":", "")
#                 break

#     if init_tag is None:
#         init_tag = datetime.utcnow().strftime("%Y%m%d%H")

#     # 2. 生成红色的“时段字符串”，形如 1112–1212(000–024)
#     #   例：起报 2025-11-11 12UTC，当前步长 24 → 1112–1212(000–024)
#     base_dt = datetime.strptime(init_tag, "%Y%m%d%H")
#     start_dt = base_dt
#     end_dt = base_dt + timedelta(hours=int(step_hour))
#     period_str = f"{start_dt.strftime('%m%d')}-{end_dt.strftime('%m%d')}({start_dt.strftime('%H')}-{end_dt.strftime('%H')})"

#     # ====== 主标题 ======
#     ax.set_title(f"ECMWF Precip phase  Δ{interval_hours}h  +{step_hour}h", fontsize=12)
#     # ====== 左蓝右红 ======
#     # ====== 左蓝右红（居中微调版） ======
#     fig.text(
#         0.25, 0.975,            # 稍靠中，离顶0.975
#         init_tag,
#         fontsize=12,
#         color="blue",
#         ha="center",
#         va="top"
#     )
#     fig.text(
#         0.75, 0.975,            # 对称位置
#         period_str,
#         fontsize=12,
#         color="red",
#         ha="center",
#         va="top"
#     )
#     # fig.text(0.5 - 0.15, 0.97, init_tag, fontsize=12, color="blue", ha="center", va="top")
#     # fig.text(0.5 + 0.15, 0.97, period_str, fontsize=12, color="red", ha="center", va="top")


#     out_png = os.path.join(output_dir, f"EC_PrecipPhase_{int(step_hour):03d}h.png")
#     plt.tight_layout()
#     fig.savefig(out_png, dpi=180, bbox_inches="tight")
#     plt.close(fig)

#     print(f"✅ precip phase saved: {out_png} | ⏱️ {time.perf_counter() - t0_all:.2f}s")


def plot_surface_one_time(
    surface_fields: dict,
    step_hour: int,
    *,
    output_dir="./output_images",
    extent=(60, 140, 10, 60),
    interval_hours=6,
    accum_alignment="trailing",  # "trailing"|"leading"|"center"
    label_stride: int = 4,       # 降水数值标注稀疏程度
):
    """
    单时效地面降水相态专图：
    - 只画 雪 / 雨夹雪 / 雨 三层分色
    - 降水用时段增量 (tp_this - tp_prev)
    - 三个场统一重采样到 0.125°
    - 支持数值稀疏标注（label_stride）
    依赖：regrid_to_0125(xarray.DataArray), get_phase_bounds(interval_hours)
    """

    t0_all = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    # ---------- helpers ----------
    def pick_first(d: dict, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _to_2d(da: xr.DataArray | None):
        if da is None:
            return None
        arr = da
        for dims in [("latitude", "longitude"), ("lat", "lon"), ("y", "x")]:
            if all(dim in arr.dims for dim in dims):
                arr = arr.transpose(*dims)
                break
        arr = arr.squeeze()
        while hasattr(arr, "ndim") and arr.ndim > 2:
            arr = arr.isel({arr.dims[0]: 0}).squeeze()
        return arr.values

    def _get_lonlat(da_like):
        for ln, lt in [("longitude", "latitude"), ("lon", "lat"), ("x", "y")]:
            if hasattr(da_like, lt) and hasattr(da_like, ln):
                lon1d = getattr(da_like, ln).values
                lat1d = getattr(da_like, lt).values
                lon2d, lat2d = np.meshgrid(lon1d, lat1d)
                return lon2d, lat2d
        raise RuntimeError("Cannot locate lon/lat on surface fields.")

    def _valid_time_str(da_like, h):
        try:
            base = getattr(da_like, "time", None)
            if base is not None:
                v = base.values
                if np.ndim(v) == 0:
                    tstr = np.datetime_as_string(v, unit="m")
                else:
                    tstr = np.datetime_as_string(v[-1], unit="m")
                return f"{tstr} (+{int(h):03d}h)"
        except Exception:
            pass
        return f"+{int(h):03d}h"

    def _regrid(da):
        return regrid_to_0125(da) if da is not None else None

    def _steps_hours(da: xr.DataArray | None):
        if da is None or "step" not in da.coords:
            return np.array([], dtype=int)
        step = da.coords["step"].values
        if np.issubdtype(step.dtype, np.timedelta64):
            return step.astype("timedelta64[h]").astype(int)
        return np.asarray(step, dtype=int)

    def _sel_step(da: xr.DataArray | None, h: int):
        if da is None or "step" not in da.coords:
            return None
        avail = _steps_hours(da)
        if avail.size == 0:
            return None
        nearest = int(avail[np.argmin(np.abs(avail - int(h)))]
                      )
        key = (
            nearest
            if not np.issubdtype(da.coords["step"].values.dtype, np.timedelta64)
            else np.timedelta64(nearest, "h")
        )
        try:
            return da.sel(step=key)
        except Exception:
            return None

    # ---------- fetch fields ----------
    msl = pick_first(surface_fields, ["msl", "sp"])   # 仅用于取经纬度/时间，不再绘制
    tp  = pick_first(surface_fields, ["tp"])          # m（自起报累积）
    t2m = pick_first(surface_fields, ["t2m", "2t"])   # K
    if msl is None:
        raise ValueError("plot_surface_one_time(): 'msl' (or 'sp') is required in surface_fields.")
    if tp is None:
        raise ValueError("plot_surface_one_time(): 'tp' is required for precipitation.")

    # ---------- 当前时效 ----------
    msl_this = _sel_step(msl, int(step_hour))
    t2m_this = _sel_step(t2m, int(step_hour)) if t2m is not None else None
    if msl_this is None:
        raise ValueError(f"plot_surface_one_time(): no MSL/SP for step={step_hour}")

    # ---------- 降水“时段增量” → 重采样 0.125° ----------
    tp_0125_da = None
    iv = int(interval_hours)

    if accum_alignment == "trailing":
        prev_h = max(0, int(step_hour) - iv)
        this_h = int(step_hour)
        if int(step_hour) == 0:  # +000h 前向差分，避免首帧无底色
            prev_h, this_h = 0, iv
    elif accum_alignment == "leading":
        prev_h = int(step_hour)
        this_h = int(step_hour) + iv
    else:  # center
        half = iv // 2
        prev_h = max(0, int(step_hour) - half)
        this_h = int(step_hour) + half

    tp_prev = _sel_step(tp, prev_h)
    tp_this = _sel_step(tp, this_h)

    if tp_prev is not None and tp_this is not None:
        # 去掉会触发对齐的非经纬度坐标（如 time/valid_time），只保留 lat/lon
        def _strip_align_coords(da):
            keep = set(["latitude", "longitude", "lat", "lon"])
            drop = [c for c in da.coords if c not in keep]
            return da.reset_coords(drop=True) if drop else da

        tp_prev_c = _strip_align_coords(tp_prev)
        tp_this_c = _strip_align_coords(tp_this)

        # 做差分（仍是 DataArray）
        tp_diff_da = tp_this_c - tp_prev_c
        tp_diff_da.attrs = tp_this.attrs  # 保留单位信息等

        # 统一重采样到 0.125°
        tp_0125_da = _regrid(tp_diff_da)

        if tp_0125_da is not None:
            # m → mm
            units = (getattr(tp_0125_da, "attrs", {}) or {}).get("units", "")
            if isinstance(units, str) and units.strip().lower() in ("m", "meter", "metre"):
                tp_0125_da = tp_0125_da * 1000.0
                tp_0125_da.attrs["units"] = "mm"

            # 清理与诊断
            tp_0125_da = tp_0125_da.where(tp_0125_da >= 0)
            _arr = tp_0125_da.values
            try:
                print(
                    f"ℹ️ tp(Δ{iv}h) stats — "
                    f"min:{np.nanmin(_arr):.3f}, "
                    f"p50:{np.nanpercentile(_arr,50):.3f}, "
                    f"p90:{np.nanpercentile(_arr,90):.3f}, "
                    f"max:{np.nanmax(_arr):.3f}"
                )
            except Exception:
                pass

    # ---------- MSL/T2M 统一重采样到 0.125° ----------
    msl_0125 = _regrid(msl_this)
    t2m_0125 = _regrid(t2m_this) if t2m_this is not None else None

    # 将降水对齐到 MSL 网格
    if tp_0125_da is not None and isinstance(msl_0125, xr.DataArray):
        try:
            if (
                set(tp_0125_da.dims) != set(msl_0125.dims)
                or any(
                    tp_0125_da.sizes.get(d, -1) != msl_0125.sizes.get(d, -2)
                    for d in msl_0125.dims
                )
            ):
                tp_0125_da = tp_0125_da.interp_like(msl_0125, method="nearest")
        except Exception:
            pass

    # ---------- lon/lat（以 msl_0125 为基准） & 方向统一 ----------
    msl_for_xy = msl_0125 if isinstance(msl_0125, xr.DataArray) else msl_this
    lon2d, lat2d = _get_lonlat(msl_for_xy)

    if lat2d[0, 0] > lat2d[-1, 0]:
        lat2d = np.flipud(lat2d)
        lon2d = np.flipud(lon2d)
        if isinstance(msl_0125, xr.DataArray):
            msl_0125 = msl_0125.sortby("latitude")
        if isinstance(t2m_0125, xr.DataArray):
            t2m_0125 = t2m_0125.sortby("latitude")
        if isinstance(tp_0125_da, xr.DataArray):
            tp_0125_da = tp_0125_da.sortby("latitude")
    if lon2d[0, 0] > lon2d[0, -1]:
        lon2d = np.fliplr(lon2d)
        lat2d = np.fliplr(lat2d)
        if isinstance(msl_0125, xr.DataArray):
            msl_0125 = msl_0125.sortby("longitude")
        if isinstance(t2m_0125, xr.DataArray):
            t2m_0125 = t2m_0125.sortby("longitude")
        if isinstance(tp_0125_da, xr.DataArray):
            tp_0125_da = tp_0125_da.sortby("longitude")

    # ---------- 数值场 ----------
    t2m_c = (_to_2d(t2m_0125) - 273.15) if isinstance(t2m_0125, xr.DataArray) else None
    tp_mm = _to_2d(tp_0125_da) if isinstance(tp_0125_da, xr.DataArray) else None
    valid_time = _valid_time_str(
        msl_0125 if isinstance(msl_0125, xr.DataArray) else msl_this, step_hour
    )

    # ---------- 相态分离 ----------
    snow_mm = mix_mm = rain_mm = None
    if tp_mm is not None:
        if t2m_c is not None:
            snow_mask = t2m_c <= 0.0
            mix_mask = (t2m_c > 0.0) & (t2m_c <= 2.0)
            rain_mask = t2m_c > 2.0
            snow_mm = np.where(snow_mask, tp_mm, np.nan)
            mix_mm = np.where(mix_mask, tp_mm, np.nan)
            rain_mm = np.where(rain_mask, tp_mm, np.nan)
        else:
            rain_mm = tp_mm.copy()

    # ---------- 相态颜色映射 ----------
    snow_bounds, mix_bounds, rain_bounds = get_phase_bounds(interval_hours)

    snow_cmap = ListedColormap(
        [
            "#CCFFFF",  # 极浅蓝（极弱雪）
            "#66FFFF",  # 浅蓝（小雪）
            "#00E6B8",  # 青绿（中雪）
            "#00CC66",  # 亮绿（大雪）
            "#006600",  # 深绿（特大雪）
        ]
    )

    mix_cmap = ListedColormap(
        [
            "#FFFACD",  # 柠檬浅黄（极弱夹雪）
            "#FFD700",  # 金黄（小）
            "#FF8C00",  # 深橙（中）
            "#FF4500",  # 橙红（大）
            "#FF00FF",  # 品红（特大）
        ]
    )

    rain_cmap = ListedColormap(
        [
            "#ADD8E6",  # 浅蓝（毛毛雨）
            "#00BFFF",  # 亮天蓝（小到中雨）
            "#1E90FF",  # 道奇蓝（中到大雨）
            "#8A2BE2",  # 蓝紫（暴雨）
            "#4B0082",  # 靛蓝（特大暴雨）
        ]
    )

    snow_norm = BoundaryNorm(snow_bounds, snow_cmap.N, clip=True)
    mix_norm = BoundaryNorm(mix_bounds, mix_cmap.N, clip=True)
    rain_norm = BoundaryNorm(rain_bounds, rain_cmap.N, clip=True)

    # ---------- 绘图 ----------
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="0.35"
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"), linewidth=0.6, edgecolor="0.45"
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="0.85", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # 收缩主图：预留右侧色标
    fig.subplots_adjust(right=0.86)

    # 区域遮罩
    x0, x1, y0, y1 = extent
    mask = (lon2d < x0) | (lon2d > x1) | (lat2d < y0) | (lat2d > y1)

    def _plot_layer(mm, cmap, norm):
        if mm is None:
            return None
        z = mm.copy()
        z[mask] = np.nan
        return ax.contourf(
            lon2d,
            lat2d,
            z,
            levels=norm.boundaries,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            alpha=0.95,
            antialiased=True,
            extend="max",
        )

    cs_snow = _plot_layer(snow_mm, snow_cmap, snow_norm)
    cs_mix = _plot_layer(mix_mm, mix_cmap, mix_norm)
    cs_rain = _plot_layer(rain_mm, rain_cmap, rain_norm)

    # ---------- 右侧三竖色标 ----------
    ax_pos = ax.get_position()
    cbar_panel_left = 0.88
    cbar_width = 0.022
    cbar_height = 0.22
    cbar_gap = 0.035

    top_y = ax_pos.y1 - cbar_height - 0.01
    mid_y = top_y - cbar_height - cbar_gap
    bot_y = mid_y - cbar_height - cbar_gap

    def _make_cbar(mappable, bounds, label, rect):
        if mappable is None:
            return
        cax = fig.add_axes(rect)
        ticks = [v for v in bounds[1:-1]]
        cb = fig.colorbar(mappable, cax=cax, orientation="vertical", ticks=ticks)
        ticklabels = [str(v) for v in ticks]
        if len(bounds) >= 2:
            ticklabels[-1] = f"≥{int(bounds[-2])}"
        cb.ax.set_yticklabels(ticklabels)
        cb.set_label(label, rotation=90, fontsize=10)
        cb.ax.tick_params(labelsize=9)

    if cs_snow is not None:
        _make_cbar(
            cs_snow, snow_bounds, "Snow (mm)",
            [cbar_panel_left, top_y, cbar_width, cbar_height]
        )
    if cs_mix is not None:
        _make_cbar(
            cs_mix, mix_bounds, "Mixed (mm)",
            [cbar_panel_left, mid_y, cbar_width, cbar_height]
        )
    if cs_rain is not None:
        _make_cbar(
            cs_rain, rain_bounds, "Rain (mm)",
            [cbar_panel_left, bot_y, cbar_width, cbar_height]
        )

    # ---------- 7. 数值叠加（稀疏，整数+小数混排） ----------
    def _fmt_val(v: float) -> str | None:
        # 过小或无效的不标
        if np.isnan(v) or v < 0.05:
            return None
        # >=10 取整
        if v >= 10:
            return f"{v:.0f}"
        # [1,10) 近似整数就取整，否则一位小数
        if v >= 1:
            r = round(v)
            if abs(v - r) < 0.05:  # 接近整数的“粘连区”
                return f"{r:d}"
            return f"{v:.1f}"
        # <1 一位小数
        return f"{v:.1f}"

    if tp_mm is not None:
        ny, nx = tp_mm.shape
        for iy in range(0, ny, label_stride):
            for ix in range(0, nx, label_stride):
                val = tp_mm[iy, ix]
                label = _fmt_val(val)
                if not label:
                    continue

                lon = lon2d[iy, ix]
                lat = lat2d[iy, ix]
                ax.text(
                    lon,
                    lat,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#101010",
                    bbox=dict(
                        facecolor=(1, 1, 1, 0.35),
                        edgecolor="none",
                        pad=0.4,
                    ),
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )

    # ---------- 标题 & 保存 ----------
    ax.set_title(
        f"EC Surface • Precipitation Phase & Amount (Δ={int(interval_hours)}h)\n"
        f"{valid_time}",
        fontsize=15,
        pad=10,
    )
    safe_time = (
        valid_time.replace(":", "").replace(" ", "_")
        if isinstance(valid_time, str)
        else f"step{int(step_hour):03d}"
    )
    out_png = os.path.join(
        output_dir, f"EC_SurfacePrecip_{int(step_hour):03d}h_{safe_time}.png"
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(
        f"✅ [SurfacePrecip] +{int(step_hour):03d}h saved: {out_png} | "
        f"⏱️ {time.perf_counter() - t0_all:.2f}s"
    )











# ========== 图2：MSLP + 10m 风 + 2m 温度 ==========
from scipy.ndimage import gaussian_filter


# --- 像素级抽稀：每个 min_px×min_px 只留优先级最高的一个 ---
def thin_by_pixels(ax, lon, lat, priority, min_px=18):
    import numpy as np
    xy = ax.transData.transform(np.column_stack([lon.ravel(), lat.ravel()]))
    gx = np.floor(xy[:, 0] / float(min_px)).astype(int)
    gy = np.floor(xy[:, 1] / float(min_px)).astype(int)
    keep = {}
    for i, key in enumerate(zip(gx, gy)):
        p = float(priority.ravel()[i])
        if key not in keep or p > keep[key][0]:
            keep[key] = (p, i)
    return np.array([v[1] for v in keep.values()], dtype=int)

# --- 自绘风羽（北半球右侧60°出杠；hcnt:0不画，1~4横，5旗） ---
# def draw_barbs_custom(ax, lon, lat, u, v, hcnt, *, red_min=3,
#                       L=0.55, tick_len=0.22, gap=0.12, start=0.10,
#                       color_lo="#5C2318", color_hi="red"):
#     import numpy as np
#     from matplotlib.collections import LineCollection, PolyCollection
#     spd = np.hypot(u, v)
#     nanm = np.isnan(lon) | np.isnan(lat) | np.isnan(u) | np.isnan(v) | np.isnan(spd) | (hcnt <= 0)
#     if (~nanm).sum() == 0: return

#     x = lon[~nanm].ravel(); y = lat[~nanm].ravel()
#     uu = u[~nanm].ravel();  vv = v[~nanm].ravel()
#     k  = hcnt[~nanm].ravel().astype(int)

#     # 杆朝来风方向
#     sp = np.hypot(uu, vv); sp[sp == 0] = 1.0
#     dx = -uu / sp; dy = -vv / sp
#     # 右侧60°的横杠方向
#     c, s = np.cos(np.deg2rad(60.0)), np.sin(np.deg2rad(60.0))
#     tx =  c*dx + s*dy
#     ty = -s*dx + c*dy

#     stems_lo = []; stems_hi = []
#     segs_lo  = []; segs_hi  = []
#     polys_hi = []

#     for i in range(x.size):
#         n = int(k[i])
#         show_flag = (n >= 5); n = min(n, 4)

#         # 杆
#         stem = [(x[i], y[i]), (x[i] + L*dx[i], y[i] + L*dy[i])]
#         (stems_hi if k[i] >= red_min else stems_lo).append(stem)

#         # 横杠
#         s0 = start * L
#         for j in range(n):
#             bx = x[i] + (s0 + j*gap*L) * dx[i]
#             by = y[i] + (s0 + j*gap*L) * dy[i]
#             ex = bx + tick_len*L * tx[i]
#             ey = by + tick_len*L * ty[i]
#             seg = [(bx, by), (ex, ey)]
#             (segs_hi if k[i] >= red_min else segs_lo).append(seg)

#         # 旗
#         if show_flag:
#             tipx, tipy = x[i], y[i]
#             p1 = (tipx, tipy)
#             p2 = (tipx + tick_len*L*tx[i], tipy + tick_len*L*ty[i])
#             p3 = (tipx + 0.55*tick_len*L*tx[i] + 0.55*tick_len*L*dx[i],
#                   tipy + 0.55*tick_len*L*ty[i] + 0.55*tick_len*L*dy[i])
#             polys_hi.append([p1, p2, p3])

#     # 组装绘制
#     from cartopy.crs import PlateCarree
#     def add_lc(segs, color, lw, z):
#         if not segs: return
#         lc = LineCollection(segs, colors=color, linewidths=lw,
#                             transform=PlateCarree(), zorder=z, clip_on=False)
#         ax.add_collection(lc)
#     def add_pc(polys, color, z):
#         if not polys: return
#         pc = PolyCollection(polys, facecolors=color, edgecolors=color,
#                             transform=PlateCarree(), zorder=z, clip_on=False)
#         ax.add_collection(pc)

#     add_lc(stems_lo, color_lo, 0.5, 5)
#     add_lc(segs_lo,  color_lo, 0.5, 5)
#     add_lc(stems_hi, color_hi, 0.7, 6)
#     add_lc(segs_hi,  color_hi, 0.7, 6)
#     add_pc(polys_hi, color_hi, 6)

def draw_barbs_custom(ax, lon, lat, u, v, hcnt, *,
                      red_min=3,
                      L=0.55, tick_len=0.22, gap=0.12, start=0.10,
                      color_lo="#5C2318", color_hi="red",
                      lw_lo=0.7, lw_hi=0.9,
                      scale=1.0):
    """
    自绘风羽（北半球右侧出杠，60°），可通过 scale 参数整体放大
    hcnt: 0=不画, 1~4=横杠数, 5=一旗
    """

    import numpy as np
    from matplotlib.collections import LineCollection, PolyCollection
    from cartopy.crs import PlateCarree

    # ===== 🔹 这里加上统一放大部分（放大比例影响所有尺寸） =====
    L        *= scale
    tick_len *= scale
    gap      *= scale
    start    *= scale
    lw_lo    *= (0.9 * scale)
    lw_hi    *= (0.9 * scale)
    # ============================================================

    spd = np.hypot(u, v)
    nanm = np.isnan(lon) | np.isnan(lat) | np.isnan(u) | np.isnan(v) | np.isnan(spd) | (hcnt <= 0)
    if (~nanm).sum() == 0:
        return

    x = lon[~nanm].ravel(); y = lat[~nanm].ravel()
    uu = u[~nanm].ravel();  vv = v[~nanm].ravel()
    k  = hcnt[~nanm].ravel().astype(int)

    # 杆朝来风方向
    sp = np.hypot(uu, vv); sp[sp == 0] = 1.0
    dx = -uu / sp; dy = -vv / sp
    # 右侧 60° 的横杠方向
    c, s = np.cos(np.deg2rad(60.0)), np.sin(np.deg2rad(60.0))
    tx =  c*dx + s*dy
    ty = -s*dx + c*dy

    stems_lo = []; stems_hi = []
    segs_lo  = []; segs_hi  = []
    polys_hi = []

    for i in range(x.size):
        n = int(k[i])
        show_flag = (n >= 5)
        n = min(n, 4)

        # 杆
        stem = [(x[i], y[i]), (x[i] + L*dx[i], y[i] + L*dy[i])]
        (stems_hi if k[i] >= red_min else stems_lo).append(stem)

        # 横杠
        s0 = start * L
        for j in range(n):
            bx = x[i] + (s0 + j*gap*L) * dx[i]
            by = y[i] + (s0 + j*gap*L) * dy[i]
            ex = bx + tick_len*L * tx[i]
            ey = by + tick_len*L * ty[i]
            seg = [(bx, by), (ex, ey)]
            (segs_hi if k[i] >= red_min else segs_lo).append(seg)

        # 旗
        if show_flag:
            tipx, tipy = x[i], y[i]
            p1 = (tipx, tipy)
            p2 = (tipx + tick_len*L*tx[i], tipy + tick_len*L*ty[i])
            p3 = (tipx + 0.55*tick_len*L*tx[i] + 0.55*tick_len*L*dx[i],
                  tipy + 0.55*tick_len*L*ty[i] + 0.55*tick_len*L*dy[i])
            polys_hi.append([p1, p2, p3])

    def add_lc(segs, color, lw, z):
        if not segs: return
        lc = LineCollection(segs, colors=color, linewidths=lw,
                            transform=PlateCarree(), zorder=z, clip_on=False)
        ax.add_collection(lc)

    def add_pc(polys, color, z):
        if not polys: return
        pc = PolyCollection(polys, facecolors=color, edgecolors=color,
                            transform=PlateCarree(), zorder=z, clip_on=False)
        ax.add_collection(pc)

    add_lc(stems_lo, color_lo, lw_lo, 5)
    add_lc(segs_lo,  color_lo, lw_lo, 5)
    add_lc(stems_hi, color_hi, lw_hi, 6)
    add_lc(segs_hi,  color_hi, lw_hi, 6)
    add_pc(polys_hi, color_hi, 6)


# def plot_surface_synoptic(
#     surface_fields: dict,
#     step_hour: int,
#     *,
#     output_path: str,
#     extent=(60, 140, 10, 60),
#     temp_contour_interval: int = 4,
#     wind_stride: int = 10,
# ):
#     """
#     MSLP + 10m风专图
#     - msl/sp 选一个
#     - 10u/ u10 选一个；10v/ v10 选一个
#     - 有经纬度、海岸线、国界
#     """
#     # 1) 安全地取变量，不能用 DataArray or DataArray
#     msl = surface_fields.get("msl")
#     if msl is None:
#         msl = surface_fields.get("sp")
#     if msl is None:
#         raise ValueError("plot_surface_synoptic: 必须有 msl 或 sp")

#     u10 = surface_fields.get("10u")
#     if u10 is None:
#         u10 = surface_fields.get("u10")

#     v10 = surface_fields.get("10v")
#     if v10 is None:
#         v10 = surface_fields.get("v10")

#     t2m = surface_fields.get("2t")
#     if t2m is None:
#         t2m = surface_fields.get("t2m")

#     # 小工具：按最接近 step 取一帧
#     def _sel(da, h):
#         steps = da.coords["step"].values
#         if np.issubdtype(steps.dtype, np.timedelta64):
#             steps_h = (steps / np.timedelta64(1, "h")).astype(int)
#         else:
#             steps_h = steps.astype(int)
#         nearest = int(steps_h[np.argmin(np.abs(steps_h - int(h)))])
#         if np.issubdtype(steps.dtype, np.timedelta64):
#             key = np.timedelta64(nearest, "h")
#         else:
#             key = nearest
#         return da.sel(step=key)

#     msl_s = _sel(msl, step_hour)
#     if t2m is not None:
#         t2m_s = _sel(t2m, step_hour)
#     else:
#         t2m_s = None
#     if u10 is not None and v10 is not None:
#         u10_s = _sel(u10, step_hour)
#         v10_s = _sel(v10, step_hour)
#     else:
#         u10_s = v10_s = None

#     # 2) 重采样到 0.125°（你已有 regrid_to_0125）
#     msl_r = regrid_to_0125(msl_s)
#     if t2m_s is not None:
#         t2m_r = regrid_to_0125(t2m_s)
#     else:
#         t2m_r = None
#     if u10_s is not None and v10_s is not None:
#         u10_r = regrid_to_0125(u10_s)
#         v10_r = regrid_to_0125(v10_s)
#     else:
#         u10_r = v10_r = None

#     # 3) 数组化
#     lons = msl_r["longitude"].values
#     lats = msl_r["latitude"].values
#     lon2d, lat2d = np.meshgrid(lons, lats)

#     msl_hpa = (msl_r.values / 100.0).astype(np.float32)
#     # 稍微平滑一下等压线好看
#     msl_hpa = gaussian_filter(msl_hpa, sigma=1.0)

#     if t2m_r is not None:
#         t2m_c = t2m_r.values - 273.15
#     else:
#         t2m_c = None

#     # 4) 画图
#     fig = plt.figure(figsize=(14, 7))
#     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#     ax.set_extent(extent, crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="0.35")
#     ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, edgecolor="0.45")
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="0.85", linestyle="--")
#     gl.top_labels = False
#     gl.right_labels = False

#     # 区域遮罩
#     x0, x1, y0, y1 = extent
#     mask = (lon2d < x0) | (lon2d > x1) | (lat2d < y0) | (lat2d > y1)

#     # 5) 温度等值线（间隔 4℃）
#     # if t2m_c is not None:
#     #     t2m_plot = t2m_c.copy()
#     #     t2m_plot[mask] = np.nan
#     #     tmin = np.nanmin(t2m_plot)
#     #     tmax = np.nanmax(t2m_plot)
#     #     start = int(np.floor(tmin / temp_contour_interval) * temp_contour_interval)
#     #     stop = int(np.ceil(tmax / temp_contour_interval) * temp_contour_interval) + temp_contour_interval
#     #     t_levels = np.arange(start, stop, temp_contour_interval, dtype=int)
#     #     cs_t = ax.contour(
#     #         lon2d,
#     #         lat2d,
#     #         t2m_plot,
#     #         levels=t_levels,
#     #         colors="#A43F00",
#     #         linewidths=0.6,
#     #         linestyles="--",
#     #         transform=ccrs.PlateCarree(),
#     #     )
#     #     ax.clabel(cs_t, fmt="%d℃", fontsize=6)

#     # 6) MSLP 等压线（5hPa 一条）
#     msl_plot = msl_hpa.copy()
#     msl_plot[mask] = np.nan
#     msl_min = np.nanmin(msl_plot)
#     msl_max = np.nanmax(msl_plot)
#     msl_levels = np.arange(np.floor(msl_min / 5) * 5, np.ceil(msl_max / 5) * 5 + 1, 5)
#     # 在 contour 前加一行
#     msl_plot_smooth = gaussian_filter(msl_plot, sigma=1.0)   # sigma=1~2 越大越平滑
#     cs_p = ax.contour(
#         lon2d,
#         lat2d,
#         msl_plot_smooth,        # 用平滑后的数据
#         levels=msl_levels,
#         colors="#1E63FF",
#         linewidths=1.0,
#         transform=ccrs.PlateCarree(),
#     )
#     ax.clabel(cs_p, fmt="%.0f", fontsize=7)
    
#     # 7) 10m 风（风羽版 barb）
#     # if u10_r is not None and v10_r is not None:
#     #     # --- 数据与稀疏 ---
#     #     u_plot = u10_r.values.copy(); v_plot = v10_r.values.copy()
#     #     u_plot[mask] = np.nan;       v_plot[mask] = np.nan

#     #     # --- 稀疏采样 ---
#     #     stride = wind_stride if 'wind_stride' in locals() else 6
#     #     lon_s = lon2d[::stride, ::stride]
#     #     lat_s = lat2d[::stride, ::stride]
#     #     u_s   = u10_r.values.copy()[::stride, ::stride]
#     #     v_s   = v10_r.values.copy()[::stride, ::stride]
#     #     u_s[mask[::stride, ::stride]] = np.nan
#     #     v_s[mask[::stride, ::stride]] = np.nan

#     #     # --- 风速与稳定化 ---
#     #     spd   = np.hypot(u_s, v_s)
#     #     nanm  = np.isnan(u_s) | np.isnan(v_s) | np.isnan(spd)
#     #     spd_r = np.round(spd, 2)
#     #     spd_r = np.where(nanm, np.nan, spd_r - 1e-6)

#     #     # --- m/s → 横杠/旗个数（整数 0/1/2/3/4/5）---
#     #     hcnt = np.zeros(spd.shape, dtype=int)
#     #     hcnt[(spd_r >= 0.3)  & (spd_r < 3.4)]   = 1
#     #     hcnt[(spd_r >= 3.4)  & (spd_r < 8.0)]   = 2
#     #     hcnt[(spd_r >= 8.0)  & (spd_r < 10.8)]  = 3
#     #     hcnt[(spd_r >= 10.8) & (spd_r < 17.2)]  = 4
#     #     hcnt[(spd_r >= 17.2)]                   = 5
#     #     hcnt[nanm] = 0

#     #     # 有效点
#     #     valid = hcnt > 0
#     #     lon_v, lat_v = lon_s[valid], lat_s[valid]
#     #     u_v, v_v     = u_s[valid],   v_s[valid]
#     #     spd_v, hc_v  = spd[valid],   hcnt[valid]

#     #     # 强风优先（横杠数主导，其次风速）
#     #     priority = hc_v.astype(float) * 100.0 + spd_v
#     #     keep = thin_by_pixels(ax, lon_v, lat_v, priority, min_px=18)  # 16~20 视图幅调整

#     #     # 保留抽稀后的点
#     #     lon_k = lon_v.ravel()[keep]; lat_k = lat_v.ravel()[keep]
#     #     u_k   = u_v.ravel()[keep];   v_k   = v_v.ravel()[keep]
#     #     hc_k  = hc_v.ravel()[keep]

#     #     # 画！(红色阈值：≥3 横；若要 ≥2 横红，把 red_min=2)
#     #     # draw_barbs_custom(ax, lon_k, lat_k, u_k, v_k, hc_k, red_min=3,
#     #     #                 L=0.60, tick_len=0.24, gap=0.12, start=0.10)
#     #     draw_barbs_custom(
#     #                         ax,
#     #                         lon_k, lat_k, u_k, v_k, hc_k,
#     #                         red_min=3,           # ≥3 横红
#     #                         L=1.0,               # 杆长度，原 0.55 → 0.9
#     #                         tick_len=0.30,       # 横杠长度，原 0.22 → 0.32
#     #                         gap=0.14,            # 横杠间距，略加大以免太密
#     #                         start=0.08,          # 横杠距杆尖距离，略收一点更紧凑
#     #                          scale=1.2           # 🌟 整体放大约 1.8 倍
#     #                     )

#     # 7) 10m 风（箭头版 quiver）
#     if u10_r is not None and v10_r is not None:
#         u_plot = u10_r.values.copy()
#         v_plot = v10_r.values.copy()
#         u_plot[mask] = np.nan
#         v_plot[mask] = np.nan

#         # ---- 稀疏抽点 ----
#         stride = wind_stride if 'wind_stride' in locals() else 6
#         lon_s = lon2d[::stride, ::stride]
#         lat_s = lat2d[::stride, ::stride]
#         u_s   = u_plot[::stride, ::stride]
#         v_s   = v_plot[::stride, ::stride]

#         # 展平
#         lon_1d = lon_s.ravel()
#         lat_1d = lat_s.ravel()
#         u_1d = u_s.ravel()
#         v_1d = v_s.ravel()

#         valid = ~np.isnan(u_1d) & ~np.isnan(v_1d)
#         lon_1d = lon_1d[valid]
#         lat_1d = lat_1d[valid]
#         u_1d   = u_1d[valid]
#         v_1d   = v_1d[valid]

#         # ---- 风速 ----
#         speed = np.sqrt(u_1d**2 + v_1d**2)

#         # ---- 分色：≥8 m/s 红色，其他棕色 ----
#         colors = np.where(speed >= 8.0, "red", "#5C2318")   # 可调

#         # ---- 画箭头 quiver ----
#         ax.quiver(
#             lon_1d, lat_1d, u_1d, v_1d,
#             color=colors,
#             angles='xy',
#             scale_units='xy',
#             scale=1/0.12,       # 箭头长度缩放，可调（越大箭头越短）
#             width=0.0022,       # 线宽
#             headwidth=4.0,      # 箭头头部宽度
#             headlength=5.5,     # 箭头头部长度
#             headaxislength=4.8, # 箭头头部轴长
#             minlength=0.1,
#             pivot="middle",     # 箭头从中点出发（非常清爽）
#             transform=ccrs.PlateCarree(),
#             zorder=6,
#         )

#     ax.set_title(f"ECMWF Surface  MSLP + 10m Wind +{step_hour}h", fontsize=12)
#     plt.tight_layout()
#     fig.savefig(output_path, dpi=160, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✅ surface synoptic saved: {output_path}")

def plot_surface_synoptic(
    surface_fields: dict,
    step_hour: int,
    *,
    output_path: str | None,
    extent=(60, 140, 10, 60),
    temp_contour_interval: int = 4,
    wind_stride: int = 10,
):
    """
    地面天气图：MSLP 等压线 + 10m 风箭头（≥8 m/s 红色）
    - MSLP 自动从 msl/sp 中选择
    - 风从 10u/u10、10v/v10 选取
    - 风用 quiver 绘制，更清晰
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # ------------------------- 工具函数 -------------------------
    def _pick_first(d: dict, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _sel_nearest_step(da, h):
        """按最接近 step 选择一帧"""
        steps = da.coords["step"].values
        if np.issubdtype(steps.dtype, np.timedelta64):
            steps_h = (steps / np.timedelta64(1, "h")).astype(int)
        else:
            steps_h = steps.astype(int)

        nearest = int(steps_h[np.argmin(np.abs(steps_h - int(h)))])
        key = np.timedelta64(nearest, "h") if np.issubdtype(steps.dtype, np.timedelta64) else nearest
        return da.sel(step=key)

    def _valid_time_str(da_like, h):
        try:
            base = da_like.time.values
            if np.ndim(base) == 0:
                tstr = np.datetime_as_string(base, unit="m")
            else:
                tstr = np.datetime_as_string(base[0], unit="m")
            return f"{tstr} (+{int(h):03d}h)"
        except:
            return f"+{int(h):03d}h"

    # ------------------------- 1. 取字段 -------------------------
    msl = _pick_first(surface_fields, ["msl", "sp"])
    if msl is None:
        raise ValueError("必须提供 msl 或 sp")

    u10 = _pick_first(surface_fields, ["10u", "u10"])
    v10 = _pick_first(surface_fields, ["10v", "v10"])
    t2m = _pick_first(surface_fields, ["2t", "t2m"])

    # ------------------------- 2. 选时次 -------------------------
    msl_s = _sel_nearest_step(msl, step_hour)
    t2m_s = _sel_nearest_step(t2m, step_hour) if t2m is not None else None
    u10_s = _sel_nearest_step(u10, step_hour) if u10 is not None else None
    v10_s = _sel_nearest_step(v10, step_hour) if v10 is not None else None

    # ------------------------- 3. 重采样到 0.125° -------------------------
    msl_r = regrid_to_0125(msl_s)
    t2m_r = regrid_to_0125(t2m_s) if t2m_s is not None else None
    u10_r = regrid_to_0125(u10_s) if u10_s is not None else None
    v10_r = regrid_to_0125(v10_s) if v10_s is not None else None

    valid_time = _valid_time_str(msl_r, step_hour)

    # ------------------------- 4. 数组化 -------------------------
    lons = msl_r["longitude"].values
    lats = msl_r["latitude"].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    msl_hpa = gaussian_filter(msl_r.values / 100.0, sigma=1.0)  # 平滑等压线

    # 遮罩
    x0, x1, y0, y1 = extent
    mask = (lon2d < x0) | (lon2d > x1) | (lat2d < y0) | (lat2d > y1)

    # ------------------------- 5. 绘图 -------------------------
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="0.35")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, edgecolor="0.45")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="0.85", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # ------------------------- 6. 等压线 -------------------------
    msl_plot = msl_hpa.copy()
    msl_plot[mask] = np.nan

    msl_min = np.nanmin(msl_plot)
    msl_max = np.nanmax(msl_plot)
    msl_levels = np.arange(
        np.floor(msl_min / 5) * 5,
        np.ceil(msl_max / 5) * 5 + 1,
        5
    )
    cs = ax.contour(
        lon2d, lat2d, msl_plot,
        levels=msl_levels,
        colors="#1E63FF",
        linewidths=1.0,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, fontsize=7, fmt="%.0f")

    # ------------------------- 7. 风场 quiver -------------------------
    if u10_r is not None and v10_r is not None:
        u_plot = u10_r.values.copy()
        v_plot = v10_r.values.copy()
        u_plot[mask] = np.nan
        v_plot[mask] = np.nan

        stride = wind_stride
        lon_s = lon2d[::stride, ::stride]
        lat_s = lat2d[::stride, ::stride]
        u_s = u_plot[::stride, ::stride]
        v_s = v_plot[::stride, ::stride]

        lon_1d = lon_s.ravel()
        lat_1d = lat_s.ravel()
        u_1d = u_s.ravel()
        v_1d = v_s.ravel()

        valid = ~np.isnan(u_1d) & ~np.isnan(v_1d)
        lon_1d = lon_1d[valid]
        lat_1d = lat_1d[valid]
        u_1d = u_1d[valid]
        v_1d = v_1d[valid]

        speed = np.sqrt(u_1d ** 2 + v_1d ** 2)
        colors = np.where(speed >= 8.0, "red", "#5C2318")

        ax.quiver(
            lon_1d, lat_1d, u_1d, v_1d,
            color=colors,
            angles="xy",
            scale_units="xy",
            scale=1/0.12,
            width=0.0022,
            headwidth=4.0,
            headlength=5.5,
            headaxislength=4.8,
            pivot="middle",
            minlength=0.1,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    # ------------------------- 8. 统一标题 & 保存 -------------------------
    ax.set_title(
        f"EC Surface • MSLP & 10m Wind\n{valid_time}",
        fontsize=15, pad=10
    )

    if output_path is None or output_path == "":
        safe_time = valid_time.replace(":", "").replace(" ", "_")
        output_path = f"./output_images/EC_SurfaceSynop_{step_hour:03d}h_{safe_time}.png"

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ surface synoptic saved: {output_path}")



# ========== 主流程 ==========


# if __name__ == "__main__":
#     # 仅画一组：+3h 的降水相态图 & 地面综合图
#     TARGET_STEP_H = 3          # 想看别的就改这里，比如 6/12
#     interval = 3               # 你这次的数据就是 3 小时间隔

#     today_local = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
#     SAVE_DIR = os.path.join(OUT_ROOT, today_local)
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # 1) 拿到 grib2 数据
#     if EXISTING_SURFACE_PATH:
#         # 可以是文件也可以是目录
#         surface_path = resolve_surface_path(EXISTING_SURFACE_PATH)
#         print(f"📁 使用已有文件: {surface_path}")
#         cycle_utc = extract_cycle_from_filename(surface_path)
#     else:
#         # 真找不到就去拉一份
#         client = Client(source="ecmwf")
#         surface_path, _, cycle_utc = retrieve_surface_only(
#             client,
#             SAVE_DIR,
#             step_interval_hours=interval,
#             max_hour=72,
#         )

#     init_tag = cycle_utc.strftime("%Y%m%d%H")

#     # 2) 读字段
#     surface_fields = load_grib_fields(surface_path)

#     # 3) 输出目录
#     out_dir = os.path.join(SAVE_DIR, init_tag, f"step{interval}h", "surface")
#     os.makedirs(out_dir, exist_ok=True)

#     # 4) 先检查这个文件里是不是真有 +3h
#     # anchor = surface_fields.get("msl") or surface_fields.get("tp")
#     # 原来这句会报错
#     # anchor = surface_fields.get("msl") or surface_fields.get("tp")

#     msl_da = surface_fields.get("msl")
#     tp_da  = surface_fields.get("tp")

#     if msl_da is not None:
#         anchor = msl_da
#     elif tp_da is not None:
#         anchor = tp_da
#     else:
#         raise ValueError("msl 和 tp 都不存在，无法确定可用的 step 列表")
#     # if anchor is None:
#     #     raise ValueError("msl/tp 都没有，没法确定有哪些时效")

#     steps_avail = anchor.coords["step"].values
#     if np.issubdtype(steps_avail.dtype, np.timedelta64):
#         steps_h = (steps_avail / np.timedelta64(1, "h")).astype(int)
#     else:
#         steps_h = steps_avail.astype(int)

#     if TARGET_STEP_H not in steps_h:
#         raise ValueError(f"这个 GRIB 里没有 +{TARGET_STEP_H}h 这个时效，实际有的：{steps_h}")

#     # 5) 画降水相态专图
#     precip_png = os.path.join(out_dir, f"EC_PrecipPhase_{TARGET_STEP_H:03d}h.png")
#     plot_precip_phase_with_values(
#         surface_fields,
#         TARGET_STEP_H,
#         output_path=precip_png,
#         extent=REGION,
#         interval_hours=interval,
#         label_stride=6,
#     )

#     # 6) 画 MSLP + 10m 风 + 2mT 图
#     syn_png = os.path.join(out_dir, f"EC_Surface_{TARGET_STEP_H:03d}h.png")
#     plot_surface_synoptic(
#         surface_fields,
#         TARGET_STEP_H,
#         output_path=syn_png,
#         extent=REGION,
#     )

#     print("\n🎉 OK，已经生成：")
#     print("   ", precip_png)
#     print("   ", syn_png)
# if __name__ == "__main__":
#     # 你的已有 GRIB 路径
#     # EXISTING_SURFACE_PATH = "./offline/output/ec_surface_2025111100_3h_0-72_0p25.grib2"  # 换成你自己的

#     REGION = (60, 140, 10, 60)
#     OUT_DIR = "./offline/output/test_plots"
#     os.makedirs(OUT_DIR, exist_ok=True)

#     # 1) 读取字段
#     surface_path = resolve_surface_path(EXISTING_SURFACE_PATH)
#     surface_fields = load_grib_fields(surface_path)

#     # 2) 画一张地面综合：+3h
#     syn_png_3 = os.path.join(OUT_DIR, "EC_Surface_003h.png")
#     plot_surface_synoptic(
#         surface_fields,
#         3,
#         output_path=syn_png_3,
#         extent=REGION,
#     )

#     # 3) 画一张降水相态：+6h （因为 +6h 可以用 +3h 做差）
#     precip_png_6 = os.path.join(OUT_DIR, "EC_PrecipPhase_006h.png")
#     plot_precip_phase_with_values(
#         surface_fields,
#         6,
#         output_path=precip_png_6,
#         extent=REGION,
#         interval_hours=3,   # 6h - 3h
#         label_stride=4,
#     )

#     # （可选）也把 +6h 的地面综合画一下
#     syn_png_6 = os.path.join(OUT_DIR, "EC_Surface_006h.png")
#     plot_surface_synoptic(
#         surface_fields,
#         6,
#         output_path=syn_png_6,
#         extent=REGION,
#     )

#     print("✅ done, 去 offline/output/test_plots 看图")
# if __name__ == "__main__":
#     # 你的现成 GRIB，可以是文件也可以是目录
#     # EXISTING_SURFACE_PATH = "./offline/output"   # 换成你自己的
#     REGION = (60, 140, 10, 60)
#     OUT_DIR = "./offline/output/test_plots"
#     os.makedirs(OUT_DIR, exist_ok=True)

#     # 1) 找到真正的 grib2
#     surface_path = resolve_surface_path(EXISTING_SURFACE_PATH)
#     surface_fields = load_grib_fields(surface_path)

#     # 2) 用 msl 或 tp 来取 step 列表
#     anchor = surface_fields.get("msl")
#     if anchor is None:
#         anchor = surface_fields.get("sp")
#     if anchor is None:
#         anchor = surface_fields.get("tp")
#     if anchor is None:
#         raise ValueError("这个 GRIB 里连 msl/sp/tp 都没有，没法确定 step")

#     steps_raw = anchor.coords["step"].values
#     if np.issubdtype(steps_raw.dtype, np.timedelta64):
#         steps_h = (steps_raw / np.timedelta64(1, "h")).astype(int)
#     else:
#         steps_h = steps_raw.astype(int)

#     # 排序去重
#     steps_h = np.unique(steps_h)
#     print("📦 文件里实际有的 step(h):", steps_h)

#     # 选第一个非 0 的时效
#     valid_steps = [h for h in steps_h if h > 0]
#     if not valid_steps:
#         raise ValueError("文件里没有 >0 的预报时效")
#     first_h = valid_steps[0]          # 比如可能是 6 而不是 3
#     interval_guess = first_h          # 基本能反推出间隔，比如 6
#     second_h = first_h + interval_guess

#     print(f"✅ 将画这两张：surface @ +{first_h}h, precip @ +{second_h}h")

#     # 3) 地面综合（第一张）
#     syn_png_1 = os.path.join(OUT_DIR, f"EC_Surface_{first_h:03d}h.png")
#     plot_surface_synoptic(
#         surface_fields,
#         first_h,
#         output_path=syn_png_1,
#         extent=REGION,
#     )

#     # 4) 降水相态（第二张，用第一张做差）
#     precip_png_2 = os.path.join(OUT_DIR, f"EC_PrecipPhase_{second_h:03d}h.png")
#     plot_precip_phase_with_values(
#         surface_fields,
#         second_h,
#         output_path=precip_png_2,
#         extent=REGION,
#         interval_hours=interval_guess,
#         label_stride=4,
#     )

#     # （可选）把第二张的地面综合也画一下，方便对比
#     syn_png_2 = os.path.join(OUT_DIR, f"EC_Surface_{second_h:03d}h.png")
#     plot_surface_synoptic(
#         surface_fields,
#         second_h,
#         output_path=syn_png_2,
#         extent=REGION,
#     )

#     print("🎉 done")

if __name__ == "__main__":
    # REGION = (60, 140, 10, 60)
    OUT_DIR = "./offline/output/test_plots"
    # EXISTING_SURFACE_PATH = "./offline/output"   # 目录或 .grib2
    JOBS = [(3, 72)]  # 可以扩展成 [(3,72),(6,240)]
    os.makedirs(OUT_DIR, exist_ok=True)

    # client = Client(source="ecmwf")

    for interval, maxh in JOBS:
        print(f"\n🚀 TEST JOB: interval={interval}h, maxh={maxh}h")

        # 1) 拿到 grib 文件
        if EXISTING_SURFACE_PATH:
            surface_path = resolve_surface_path(EXISTING_SURFACE_PATH)
            print(f"📁 使用已有文件: {surface_path}")
            cycle_utc = extract_cycle_from_filename(surface_path)
        else:
            surface_path, _, cycle_utc = retrieve_surface_only(
                client,
                OUT_DIR,
                step_interval_hours=interval,
                max_hour=maxh,
            )

        init_tag = cycle_utc.strftime("%Y%m%d%H")
        print(f"🕓 起报时间: {init_tag}")

        # 2) 读变量
        surface_fields = load_grib_fields(surface_path)
        print(f"✅ 读取变量: {list(surface_fields.keys())}")

        # 3) 看文件里实际有哪些 step
        anchor = surface_fields.get("msl")
        if anchor is None:
            anchor = surface_fields.get("tp")
        if anchor is None:
            raise ValueError("msl/tp 都没有，无法确定 step 列表")

        steps_raw = anchor.coords["step"].values
        if np.issubdtype(steps_raw.dtype, np.timedelta64):
            steps_h = (steps_raw / np.timedelta64(1, "h")).astype(int)
        else:
            steps_h = steps_raw.astype(int)
        steps_h = np.unique(steps_h)
        print(f"📦 文件里实际有的 step(h): {steps_h}")

        # 为了快速定位“有没有上一帧”，做个 set
        steps_set = set(steps_h.tolist())

        # 4) 真正循环画
        for h in steps_h:
            if h == 0:
                continue  # +0h 不画

            # 4.1 地面综合图
            # syn_png = os.path.join(OUT_DIR, f"EC_Surface_{h:03d}h.png")
            # plot_surface_synoptic(
            #     surface_fields,
            #     int(h),
            #     output_path=syn_png,
            #     extent=REGION,
            # )

            # 4.2 降水相态：要保证有 h - interval 这一帧才能画
            # prev_h = h - interval
            # if prev_h >= 0 and prev_h in steps_set:
            plot_surface_one_time(surface_fields, int(h), output_dir=OUT_DIR, extent=REGION, interval_hours=interval, label_stride=10)
            # else:
            #     # 没有上一帧就不画降水，不算错误
            #     print(f"ℹ️ +{h}h 没找到上一帧 (+{prev_h}h)，跳过降水相态图")

    print("\n🎉 全部 JOB 完成，去输出目录看图吧！")

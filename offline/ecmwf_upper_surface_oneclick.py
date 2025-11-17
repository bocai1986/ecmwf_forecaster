#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECMWF 一键脚本：下载(0.25°) + 批量绘图
- 起报：基于 client.latest(type='fc')，并映射为 00/12Z（00→00,06→00,12→12,18→12）
- 下载：为每个间隔生成一份合并 GRIB
    * 上空(PL, levelist=500/700/850, param=t/u/v/z)
    * 地面(SFC, param=msl/tp/2t)
  间隔配置：
    - 3h → 0–72h
    - 6/12/24h → 0–240h
- 绘图：
    * 上空：500/700/850 hPa，Temp(°C, dashed) + Height(dam, blue) + Wind
    * 地面：MSLP(蓝等压线) + 降水相态着色（依据 2m 温度 + 时段增量降水）
- 输出结构：
  <OUT_ROOT>/<本地日期YYYY-MM-DD>/<起报YYYYMMDDHH>/
    ├─ step3h
    │   ├─ 500hPa|700hPa|850hPa/*.png
    │   └─ surface/*.png
    ├─ step6h
    ├─ step12h
    └─ step24h
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from requests.exceptions import HTTPError
from ecmwf.opendata import Client
import time
import random


# ========= 时间 & 起报 =========
def decide_ec_start_from_latest(client) -> datetime:
    latest_dt = client.latest(type='fc')
    if latest_dt.tzinfo is None:
        latest_dt = latest_dt.replace(tzinfo=timezone.utc)
    h = latest_dt.hour
    if h in (0, 12):
        chosen = latest_dt.replace(minute=0, second=0, microsecond=0)
    elif h == 6:
        chosen = latest_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif h == 18:
        chosen = latest_dt.replace(hour=12, minute=0, second=0, microsecond=0)
    else:
        chosen = latest_dt.replace(hour=(0 if h < 12 else 12), minute=0, second=0, microsecond=0)
    return chosen.astimezone(timezone.utc)

def build_step_string(step_interval_hours: int, max_hour: int):
    steps = np.arange(0, max_hour + 1, int(step_interval_hours), dtype=int)
    return "/".join(map(str, steps)), steps

def _cycle_candidates_from(chosen_utc: datetime, depth=6):
    t = chosen_utc
    for _ in range(depth):
        yield t
        t -= timedelta(hours=12)

def _param_to_wire(params):
    # 支持 list 或 "/" 字符串
    if isinstance(params, (list, tuple)):
        return "/".join(params)
    return str(params)


def _sleep_backoff(attempt, retry_after=None, base=3, cap=300):
    """指数退避+抖动。attempt从1开始；优先尊重服务器的Retry-After(秒)"""
    if retry_after:
        try:
            wait = float(retry_after)
            if wait > 0:
                time.sleep(min(wait, cap))
                return
        except Exception:
            pass
    # 退避：base * 2^(attempt-1) ，并加一点随机抖动，最大不超过cap秒
    wait = min(base * (2 ** (attempt - 1)), cap)
    wait = wait * (0.8 + 0.4 * random.random())  # 0.8x ~ 1.2x 抖动
    time.sleep(wait)

def retrieve_combined(
    client,
    save_dir: str,
    *,
    step_interval_hours: int,
    max_hour: int,
    params,
    level_list=None,
    init_from_latest=True,
    anchor_cycle_utc=None,   # ← 新增：用它来固定起报（通常传上层的 cycle_utc）
    max_attempts_per_cycle=4,
    backoff_base=3,
    backoff_cap=180,
):
    import os, time, random
    from datetime import datetime, timedelta, timezone
    from requests.exceptions import HTTPError
    os.makedirs(save_dir, exist_ok=True)

    # 生成 step 串
    step_str, steps_array = build_step_string(step_interval_hours, max_hour)
    last_step = int(steps_array.max()) if steps_array.size else 0

    # 归一到 00/12Z
    def _to_00_12(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        h = dt.hour
        if h in (0, 12):
            return dt.replace(minute=0, second=0, microsecond=0)
        return dt.replace(hour=(0 if h < 12 else 12), minute=0, second=0, microsecond=0)

    # 起报候选序列
    if anchor_cycle_utc is not None:
        start_guess = _to_00_12(anchor_cycle_utc)
    else:
        start_guess = _to_00_12(decide_ec_start_from_latest(client) if init_from_latest
                                 else datetime.now(timezone.utc))

    cycles = [start_guess] + [start_guess - timedelta(hours=12*i) for i in range(1,6)]

    wire_param = "/".join(params) if isinstance(params, (list, tuple)) else str(params)
    levelist_str = None if level_list is None else "/".join(map(str, level_list))

    def _sleep_backoff(att, retry_after=None):
        import math
        if retry_after:
            try:
                wait = float(retry_after)
                time.sleep(min(wait, backoff_cap)); return
            except Exception:
                pass
        wait = min(backoff_base * (2 ** (att - 1)), backoff_cap)
        wait *= (0.8 + 0.4 * random.random())
        time.sleep(wait)

    last_exc = None
    for cycle in cycles:
        date_str = cycle.strftime("%Y-%m-%d")
        time_str = cycle.strftime("%H")
        init_tag = cycle.strftime("%Y%m%d%H")
        fam = "upper" if levelist_str else "surface"

        target_file = os.path.join(
            save_dir, f"ec_{fam}_{init_tag}_{step_interval_hours}h_0-{last_step}_0p25.grib2"
        )

        req = dict(
            date=date_str, time=time_str,
            type="fc", stream="oper",
            param=wire_param, step=step_str,
            target=target_file,
        )
        if levelist_str:
            req["levelist"] = levelist_str

        print(f"\n📡 尝试下载 [{fam}] {date_str} {time_str} UTC | step={step_str} | param={wire_param}"
              + (f" | levelist={levelist_str}" if levelist_str else ""))

        # —— 重要：不再手动传 use_index —— #
        for attempt in range(1, max_attempts_per_cycle + 1):
            try:
                client.retrieve(**req)     # 交给库自行决定 use_index
                print(f"✅ 成功 → {target_file}")
                return target_file, steps_array, cycle
            except HTTPError as e:
                last_exc = e
                status = getattr(e.response, "status_code", None)
                # 404/403：通常是该周期/索引未就绪；直接换周期更快
                if status in (404, 403):
                    print(f"⚠️ HTTP {status}（周期可能未就绪/无数据）→ 回退 -12h")
                    break  # 跳出重试，换更早起报
                # 429/5xx：指数退避后重试
                if status in (429, 500, 502, 503, 504):
                    ra = None
                    try: ra = e.response.headers.get("Retry-After")
                    except Exception: pass
                    print(f"⚠️ HTTP {status}（attempt {attempt}/{max_attempts_per_cycle}）→ 退避后重试…")
                    _sleep_backoff(attempt, ra)
                    continue
                print(f"❌ HTTP {status} 非可重试错误：{e} → 回退 -12h")
                break
            except Exception as e2:
                last_exc = e2
                print(f"❌ 其他错误（attempt {attempt}/{max_attempts_per_cycle}）：{e2} → 退避后重试…")
                _sleep_backoff(attempt)
                continue

        # 本周期失败，尝试更早周期
        print(f"↩️ 起报 {date_str} {time_str} UTC 失败，改用更早的周期（最后错误：{last_exc}）")

    raise RuntimeError(f"所有候选起报均失败。最后错误：{last_exc}")




# ========= 读取 & 公共工具 =========
def load_grib_fields(filepath: str) -> dict:
    """读取合并 GRIB，返回 {var.lower(): xr.DataArray}, 并将 z→gh(gpm)"""
    datasets = cfgrib.open_datasets(filepath)
    all_vars = {}
    for ds in datasets:
        for var in ds.data_vars:
            all_vars[var.lower()] = ds[var]
    if "gh" not in all_vars and "z" in all_vars:
        gh = all_vars["z"] / 9.80665
        gh.attrs["long_name"] = "geopotential height"
        gh.attrs["units"] = "gpm"
        all_vars["gh"] = gh
    print(f"✅ 加载 {os.path.basename(filepath)} 变量: {list(all_vars.keys())}")
    return all_vars

def _step_values_hours(da: xr.DataArray) -> np.ndarray:
    if "step" not in da.coords:
        return np.array([0], dtype=int)
    vals = da.coords["step"].values
    if np.issubdtype(vals.dtype, np.timedelta64):
        return (vals / np.timedelta64(1, "h")).astype(int)
    return vals.astype(int)

def _coerce_step_key(da: xr.DataArray, hours: int):
    if "step" not in da.coords:
        return 0
    vals = da.coords["step"].values
    if np.issubdtype(vals.dtype, np.timedelta64):
        return np.timedelta64(int(hours), "h")
    return int(hours)

def _valid_time_str(base_time, step_hours: int) -> str:
    bt_py = np.datetime64(base_time).astype("datetime64[s]").astype(datetime)
    vt_py = bt_py + timedelta(hours=int(step_hours))
    return vt_py.strftime("%Y-%m-%d %H:%M UTC")

def _fmt_dur(s: float) -> str:
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ========= 上空绘图 =========
def plot_t_gh_uv_one_time(fields, level_hpa, step_hour,
                          *, lon_min=60, lon_max=140, lat_min=10, lat_max=60,
                          barb_stride=8, output_path="out.png", title_prefix=""):
    for key in ("t", "gh", "u", "v"):
        if key not in fields:
            raise ValueError(f"缺少变量: {key}")
    t_key  = _coerce_step_key(fields["t"],  step_hour)
    gh_key = _coerce_step_key(fields["gh"], step_hour)
    u_key  = _coerce_step_key(fields["u"],  step_hour)
    v_key  = _coerce_step_key(fields["v"],  step_hour)

    tK = fields["t"].sel(isobaricInhPa=level_hpa, step=t_key)
    gh = fields["gh"].sel(isobaricInhPa=level_hpa, step=gh_key)
    u  = fields["u"].sel(isobaricInhPa=level_hpa, step=u_key)
    v  = fields["v"].sel(isobaricInhPa=level_hpa, step=v_key)

    base_time = tK.coords["time"].values if "time" in tK.coords else gh.coords["time"].values
    valid_str = _valid_time_str(base_time, step_hour)

    lats = tK.latitude.values; lons = tK.longitude.values
    if lons[0] > lons[-1]:
        tK=tK.isel(longitude=slice(None,None,-1)); gh=gh.isel(longitude=slice(None,None,-1))
        u=u.isel(longitude=slice(None,None,-1)); v=v.isel(longitude=slice(None,None,-1))
        lons = lons[::-1]
    if lats[0] > lats[-1]:
        tK=tK.isel(latitude=slice(None,None,-1)); gh=gh.isel(latitude=slice(None,None,-1))
        u=u.isel(latitude=slice(None,None,-1)); v=v.isel(latitude=slice(None,None,-1))
        lats = lats[::-1]

    lon_mask = (lons>=lon_min)&(lons<=lon_max); lat_mask=(lats>=lat_min)&(lats<=lat_max)
    lons_roi = lons[lon_mask]; lats_roi = lats[lat_mask]
    lon2d, lat2d = np.meshgrid(lons_roi, lats_roi)

    tC = (tK.values[np.ix_(lat_mask,lon_mask)] - 273.15)
    gh_dam = (gh.values[np.ix_(lat_mask,lon_mask)] / 10.0)
    uu = u.values[np.ix_(lat_mask,lon_mask)]; vv = v.values[np.ix_(lat_mask,lon_mask)]

    HEIGHT_BLUE="#1f66ff"; HEIGHT_EMPH="#7e2f8e"; TEMP_RED="#d62728"; WIND_BLACK="k"
    BORDER_GRAY="#666666"; GRID_GRAY="#cfcfcf"

    fig = plt.figure(figsize=(12,8), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min,lon_max,lat_min,lat_max], crs=ccrs.PlateCarree())
    ax.coastlines("110m", linewidth=0.7, color=BORDER_GRAY)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.5, edgecolor=BORDER_GRAY)
    gl=ax.gridlines(draw_labels=True, linewidth=0.5, color=GRID_GRAY, linestyle='--')
    gl.top_labels=False; gl.right_labels=False

    if level_hpa==500: h_levels=np.arange(500,602,4)
    elif level_hpa==700: h_levels=np.arange(300,341,4)
    elif level_hpa==850: h_levels=np.arange(120,169,4)
    else:
        h_levels=np.arange(np.floor(np.nanmin(gh_dam)/4)*4, np.ceil(np.nanmax(gh_dam)/4)+1, 4)

    cs_h=ax.contour(lon2d,lat2d,gh_dam,levels=h_levels, colors=HEIGHT_BLUE, linewidths=1.0, transform=ccrs.PlateCarree())
    ax.clabel(cs_h, fmt="%d", fontsize=9)

    if level_hpa==500 and (np.nanmin(gh_dam)<=588<=np.nanmax(gh_dam)):
        cs_588=ax.contour(lon2d,lat2d,gh_dam,levels=[588], colors=HEIGHT_EMPH, linewidths=2.0, transform=ccrs.PlateCarree())
        ax.clabel(cs_588, fmt="%d", fontsize=10)

    t_levels=np.arange(-60,41,4)
    cs_t=ax.contour(lon2d,lat2d,tC,levels=t_levels, colors=TEMP_RED, linestyles='--', linewidths=1.0, transform=ccrs.PlateCarree())
    ax.clabel(cs_t, fmt="%d°C", fontsize=8)

    ax.barbs(lon2d[::8,::8], lat2d[::8,::8], uu[::8,::8], vv[::8,::8], color=WIND_BLACK, length=5, transform=ccrs.PlateCarree())

    ax.set_title(f"{level_hpa} hPa  Temp(°C,dashed) • Height(dam,blue) • Wind(m/s)\n"
                 f"{title_prefix}  Valid: {valid_str} (+{step_hour:03d}h)", fontsize=14)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close(fig)
    print(f"✅ 保存: {output_path}")



def plot_upper_from_file(filepath, output_root, levels=(500,700,850), init_tag=None,
                         step_interval_hours=3, max_hour=72, region=(60,140,10,60)):
    fields = load_grib_fields(filepath)
    steps_avail = _step_values_hours(fields["t"])
    base_time = fields["t"].coords["time"].values
    base_dt = np.datetime64(base_time).astype("datetime64[h]").astype(datetime)
    base_tag = base_dt.strftime("%Y%m%d%H") if init_tag is None else init_tag
    title_prefix = f"ECMWF Init: {base_dt.strftime('%Y-%m-%d %H:%M UTC')}"
    desired = np.arange(0, max_hour + 1, int(step_interval_hours), dtype=int)
    steps_to_plot = np.intersect1d(steps_avail, desired)
    print(f"🧭 上空-可用(h): {steps_avail} | 期望(h): {desired} | 实际(h): {steps_to_plot}")

    out_dir = os.path.join(output_root, base_tag, f"step{int(step_interval_hours)}h")

    # —— 计时器 —— #
    t_total_start = time.perf_counter()
    img_count = 0

    for lvl in levels:
        lvl_dir = os.path.join(out_dir, f"{lvl}hPa")
        os.makedirs(lvl_dir, exist_ok=True)
        for h in steps_to_plot:
            out_png = os.path.join(lvl_dir, f"ECMWF_{lvl}hPa_{base_tag}_plus{h:03d}h.png")

            t0 = time.perf_counter()
            plot_t_gh_uv_one_time(
                fields, lvl, h,
                lon_min=region[0], lon_max=region[1], lat_min=region[2], lat_max=region[3],
                output_path=out_png, title_prefix=title_prefix
            )
            dt_sec = time.perf_counter() - t0
            img_count += 1
            print(f"⏱️ Upper | {lvl} hPa | +{h:03d}h | {dt_sec:.2f}s")

    total_sec = time.perf_counter() - t_total_start
    try:
        print(f"✅ 上空批量绘图完成：{img_count} 张 | 总耗时 {total_sec:.2f}s "
              f"（≈ 每张 {total_sec/img_count:.2f}s） | {_fmt_dur(total_sec)}")
    except Exception:
        print(f"✅ 上空批量绘图完成：{img_count} 张 | 总耗时 {total_sec:.2f}s")


# ========= 地面绘图 =========


# 修正版
def build_target_grid(lon_min=60, lon_max=140, lat_min=10, lat_max=60, dlon=0.125, dlat=0.125):
    # 注意：Cartopy/你之前的绘图习惯通常需要 lat 升序
    new_lons = np.arange(lon_min, lon_max + 1e-9, dlon)
    new_lats = np.arange(lat_min, lat_max + 1e-9, dlat)
    return new_lons, new_lats

def regrid_to_0125(da: xr.DataArray, lon_min=60, lon_max=140, lat_min=10, lat_max=60, dlon=0.125, dlat=0.125):
    """
    将任意经纬网格插值到 0.125° 等间距网格（东亚裁切）。
    仅线性插值（tp 这种累计量只是*空间*插值，不改变*时间*语义）。
    """
    if da is None:
        return None
    # 取出变量的纬经坐标名（cfgrib 常用 'latitude', 'longitude'）
    lat_name = 'latitude' if 'latitude' in da.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in da.coords else 'lon'

    # 先按升序统一
    da2 = da.sortby(lon_name)
    if np.any(np.diff(da2[lat_name].values) < 0):
        da2 = da2.sortby(lat_name)

    # 裁切（即便下载时已 area 裁切，这里再保底裁一次）
    da2 = da2.sel({lon_name: slice(lon_min, lon_max),
                   lat_name: slice(lat_min, lat_max)})

    # 目标 0.125° 网格
    new_lons, new_lats = build_target_grid(lon_min, lon_max, lat_min, lat_max, dlon, dlat)

    # 插值
    da0125 = da2.interp({lon_name: new_lons, lat_name: new_lats}, method="linear")
    return da0125


def get_phase_bounds(interval_hours: int):
    """
    返回 (snow_bounds, mix_bounds, rain_bounds)
    各自都是长度=6 的数组： [0.1, ..., 999]  → 5 个色段
    """
    ih = int(interval_hours)

    if ih == 3:
        snow = np.array([0.1, 0.3, 1, 3, 6, 999])
        mix  = np.array([0.1, 0.3, 1, 3, 6, 999])
        rain = np.array([0.1, 2.5, 5, 10, 20, 999])

    elif ih == 6:
        # 参考你 6h 样例（右侧雨色标最高 ~60）
        snow = np.array([0.1, 0.7, 2, 5, 8, 999])
        mix  = np.array([0.1, 0.7, 2, 5, 8, 999])
        rain = np.array([0.1, 5, 10, 25, 60, 999])

    elif ih == 12:
        # 参考 12h 样例（雪到 ~10，雨到 ~40）
        snow = np.array([0.1, 1, 3, 6, 10, 999])
        mix  = np.array([0.1, 1, 3, 6, 10, 999])
        rain = np.array([0.1, 5, 10, 20, 40, 999])

    elif ih == 24:
        # 参考 24h 样例（雨到 ~100）
        snow = np.array([0.1, 1, 3, 10, 20, 999])
        mix  = np.array([0.1, 1, 3, 10, 20, 999])
        rain = np.array([0.1, 10, 20, 50, 100, 999])

    else:
        # 其它间隔回退到 6h 档（比较稳妥）
        snow = np.array([0.1, 0.7, 2, 5, 8, 999])
        mix  = np.array([0.1, 0.7, 2, 5, 8, 999])
        rain = np.array([0.1, 5, 10, 25, 60, 999])

    return snow, mix, rain


def plot_surface_one_time(surface_fields: dict, step_hour: int, *,
                          output_dir="./output_images",
                          extent=(60, 140, 10, 60),
                          interval_hours=6,
                          accum_alignment="trailing"):  # "trailing"|"leading"|"center"
    """
    单时效地面天气图（MSLP 等压线 + 雨/夹雪/雪相态着色；降水用“时段增量”）
    - 三个场统一重采样到 0.125° 后再转 numpy，shape 一致
    - trailing 下对 +000h 做前向差分(0→Δh)特判，避免首帧无底色
    - 降低分级阈值以显示弱降水（从 0.1 mm 起）
    - 打印单张耗时与降水统计
    依赖：regrid_to_0125(xarray.DataArray)
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
        nearest = int(avail[np.argmin(np.abs(avail - int(h)))])
        key = nearest if not np.issubdtype(da.coords["step"].values.dtype, np.timedelta64) else np.timedelta64(nearest, "h")
        try:
            return da.sel(step=key)
        except Exception:
            return None

    # ---------- fetch fields ----------
    msl = pick_first(surface_fields, ["msl", "sp"])   # Pa
    tp  = pick_first(surface_fields, ["tp"])          # m（自起报累积）
    t2m = pick_first(surface_fields, ["t2m", "2t"])   # K
    if msl is None:
        raise ValueError("plot_surface_one_time(): 'msl' (or 'sp') is required in surface_fields.")

    # ---------- 当前时效 ----------
    msl_this = _sel_step(msl, int(step_hour))
    t2m_this = _sel_step(t2m, int(step_hour)) if t2m is not None else None
    if msl_this is None:
        raise ValueError(f"plot_surface_one_time(): no MSL/SP for step={step_hour}")

    # ---------- 降水“时段增量” → 重采样 0.125° ----------
    tp_0125_da = None
    if tp is not None:
        iv = int(interval_hours)
        if accum_alignment == "trailing":
            prev_h = max(0, int(step_hour) - iv); this_h = int(step_hour)
            if int(step_hour) == 0:  # +000h 前向差分，避免首帧无底色
                prev_h, this_h = 0, iv
        elif accum_alignment == "leading":
            prev_h = int(step_hour); this_h = int(step_hour) + iv
        else:  # center
            half = iv // 2
            prev_h = max(0, int(step_hour) - half); this_h = int(step_hour) + half

        tp_prev = _sel_step(tp, prev_h)
        tp_this = _sel_step(tp, this_h)

        if tp_prev is not None and tp_this is not None:
            # 关键：去掉会触发对齐的非经纬度坐标（如 time/valid_time），只保留 lat/lon
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
                    print(f"ℹ️ tp(Δ{iv}h) stats — min:{np.nanmin(_arr):.3f}, "
                        f"p50:{np.nanpercentile(_arr,50):.3f}, p90:{np.nanpercentile(_arr,90):.3f}, "
                        f"max:{np.nanmax(_arr):.3f}")
                except Exception:
                    pass

    # ---------- MSL/T2M 统一重采样到 0.125° ----------
    msl_0125 = _regrid(msl_this)
    t2m_0125 = _regrid(t2m_this) if t2m_this is not None else None

    # 将降水对齐到 MSL 网格
    if tp_0125_da is not None and isinstance(msl_0125, xr.DataArray):
        try:
            if set(tp_0125_da.dims) != set(msl_0125.dims) or any(tp_0125_da.sizes.get(d, -1) != msl_0125.sizes.get(d, -2) for d in msl_0125.dims):
                tp_0125_da = tp_0125_da.interp_like(msl_0125, method="nearest")
        except Exception:
            pass

    # ---------- lon/lat（以 msl_0125 为基准） & 方向统一 ----------
    msl_for_xy = msl_0125 if isinstance(msl_0125, xr.DataArray) else msl_this
    lons2d, lats2d = _get_lonlat(msl_for_xy)

    if lats2d[0, 0] > lats2d[-1, 0]:
        lats2d = np.flipud(lats2d); lons2d = np.flipud(lons2d)
        if isinstance(msl_0125, xr.DataArray): msl_0125 = msl_0125.sortby("latitude")
        if isinstance(t2m_0125, xr.DataArray): t2m_0125 = t2m_0125.sortby("latitude")
        if isinstance(tp_0125_da, xr.DataArray): tp_0125_da = tp_0125_da.sortby("latitude")
    if lons2d[0, 0] > lons2d[0, -1]:
        lons2d = np.fliplr(lons2d); lats2d = np.fliplr(lats2d)
        if isinstance(msl_0125, xr.DataArray): msl_0125 = msl_0125.sortby("longitude")
        if isinstance(t2m_0125, xr.DataArray): t2m_0125 = t2m_0125.sortby("longitude")
        if isinstance(tp_0125_da, xr.DataArray): tp_0125_da = tp_0125_da.sortby("longitude")

    # ---------- 数值场 ----------
    msl_hpa = _to_2d(msl_0125) / 100.0
    t2m_c   = (_to_2d(t2m_0125) - 273.15) if isinstance(t2m_0125, xr.DataArray) else None
    tp_mm   = _to_2d(tp_0125_da) if isinstance(tp_0125_da, xr.DataArray) else None
    valid_time = _valid_time_str(msl_0125 if isinstance(msl_0125, xr.DataArray) else msl_this, step_hour)

    # ---------- 相态分离 ----------
    snow_mm = mix_mm = rain_mm = None
    if tp_mm is not None:
        if t2m_c is not None:
            snow_mask = (t2m_c <= 0.0)
            mix_mask  = (t2m_c > 0.0) & (t2m_c <= 2.0)
            rain_mask = (t2m_c > 2.0)
            snow_mm = np.where(snow_mask, tp_mm, np.nan)
            mix_mm  = np.where(mix_mask,  tp_mm, np.nan)
            rain_mm = np.where(rain_mask, tp_mm, np.nan)
        else:
            rain_mm = tp_mm.copy()

    # ---------- 等压线等级 ----------
    msl_min, msl_max = np.nanmin(msl_hpa), np.nanmax(msl_hpa)
    # msl_levels = np.arange(np.floor(msl_min / 2.5) * 2.5,
    #                        np.ceil(msl_max / 2.5) * 2.5 + 0.1, 2.5)
    msl_levels = np.arange(np.floor(msl_min / 5) * 5,
                           np.ceil(msl_max / 5) * 5 + 0.1, 5)

    # ---------- 颜色映射（降低阈值以显示弱降水） ----------
    # snow_bounds = np.array([0.1, 0.3, 1, 3, 6, 999])
    # mix_bounds  = np.array([0.1, 0.3, 1, 3, 6, 999])
    # rain_bounds = np.array([0.1, 2.5, 5, 10, 20, 999])

    # snow_cmap = ListedColormap(["#66FFFF", "#00E6B8", "#00CC66", "#006600"])
    # mix_cmap  = ListedColormap(["#FFD700", "#FF8C00", "#FF4500", "#FF00FF"])
    # rain_cmap = ListedColormap(["#00BFFF", "#8A2BE2", "#800080", "#4B0082"])
    snow_bounds, mix_bounds, rain_bounds = get_phase_bounds(interval_hours)


    snow_cmap = ListedColormap([
    "#CCFFFF",  # 极浅蓝（极弱雪）
    "#66FFFF",  # 浅蓝（小雪）
    "#00E6B8",  # 青绿（中雪）
    "#00CC66",  # 亮绿（大雪）
    "#006600",  # 深绿（特大雪）
        ])

    mix_cmap = ListedColormap([
        "#FFFACD",  # 柠檬浅黄（极弱夹雪）
        "#FFD700",  # 金黄（小）
        "#FF8C00",  # 深橙（中）
        "#FF4500",  # 橙红（大）
        "#FF00FF",  # 品红（特大）
        ])

    rain_cmap = ListedColormap([
        "#ADD8E6",  # 浅蓝（毛毛雨）
        "#00BFFF",  # 亮天蓝（小到中雨）
        "#1E90FF",  # 道奇蓝（中到大雨）
        "#8A2BE2",  # 蓝紫（暴雨）
        "#4B0082",  # 靛蓝（特大暴雨）
        ])

    snow_norm = BoundaryNorm(snow_bounds, snow_cmap.N, clip=True)
    mix_norm  = BoundaryNorm(mix_bounds,  mix_cmap.N,  clip=True)
    rain_norm = BoundaryNorm(rain_bounds, rain_cmap.N, clip=True)

    # ---------- 绘图（收缩主图，右侧独立色标区） ----------
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7, edgecolor="0.35")
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.6, edgecolor="0.45")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='0.85', linestyle='--')
    gl.top_labels = False; gl.right_labels = False

    # 收缩主图：预留右侧面板（~14%宽度）
    fig.subplots_adjust(right=0.86)

    # 区域遮罩
    x0, x1, y0, y1 = extent
    mask = (lons2d < x0) | (lons2d > x1) | (lats2d < y0) | (lats2d > y1)

    def _plot_layer(mm, cmap, norm):
        if mm is None:
            return None
        z = mm.copy()
        z[mask] = np.nan
        return ax.contourf(
            lons2d, lats2d, z,
            levels=norm.boundaries, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            alpha=0.95, antialiased=True, extend="max"
        )

    cs_snow = _plot_layer(snow_mm, snow_cmap, snow_norm)
    cs_mix  = _plot_layer(mix_mm,  mix_cmap,  mix_norm)
    cs_rain = _plot_layer(rain_mm, rain_cmap, rain_norm)

    # MSLP 等压线
    cs = ax.contour(
        lons2d, lats2d, msl_hpa, levels=msl_levels,
        colors="#1E63FF", linewidths=1.1, transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f")

    # 右侧三竖色标（不与主图重叠）
    ax_pos = ax.get_position()
    cbar_panel_left = 0.88
    cbar_width  = 0.022
    cbar_height = 0.22
    cbar_gap    = 0.035

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
        _make_cbar(cs_snow, snow_bounds, "Snow (mm)",  [cbar_panel_left, top_y, cbar_width, cbar_height])
    if cs_mix is not None:
        _make_cbar(cs_mix,  mix_bounds,  "Mixed (mm)", [cbar_panel_left, mid_y, cbar_width, cbar_height])
    if cs_rain is not None:
        _make_cbar(cs_rain, rain_bounds, "Rain (mm)",  [cbar_panel_left, bot_y, cbar_width, cbar_height])

    # 标题 & 保存
    ax.set_title(f"EC Surface • MSLP & Precipitation (Δ={int(interval_hours)}h)\n{valid_time}",
                 fontsize=15, pad=10)
    safe_time = valid_time.replace(":", "").replace(" ", "_") if isinstance(valid_time, str) else f"step{int(step_hour):03d}"
    out_png = os.path.join(output_dir, f"EC_Surface_{int(step_hour):03d}h_{safe_time}.png")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ [Surface] +{int(step_hour):03d}h saved: {out_png} | ⏱️ {time.perf_counter() - t0_all:.2f}s")


def plot_surface_from_file(surface_fields: dict,
                           output_root: str,
                           init_tag: str,
                           *,
                           step_interval_hours: int = 6,
                           max_hour: int = 240,
                           extent=(60, 140, 10, 60),
                           skip_zero: bool = True,
                           accum_alignment: str = "trailing",
                           step_filter: list[int] | None = None):
    """
    批量绘制地面天气图（MSLP 等压线 + 雨/夹雪/雪 相态分色，时段增量）。
    参数：
      - surface_fields: load_grib_fields(filepath) 返回的字典
      - output_root:   输出根目录
      - init_tag:      起报标记（如 "2025103012"），用于组织输出路径
      - step_interval_hours: 3/6/12/24 等
      - max_hour:      72（3h）或 240（6/12/24h）等
      - extent:        (lon_min, lon_max, lat_min, lat_max)
      - skip_zero:     是否跳过 +000h（默认 True）
      - accum_alignment: "trailing"|"leading"|"center"，传递给单张函数
      - step_filter:   若提供，仅绘制此列表中的步长（单位：小时）
    """

    # --- 小工具：兼容你项目里的 pick_first ---
    def _pick_first(d: dict, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    anchor = _pick_first(surface_fields, ["msl", "sp", "tp", "t2m", "2t"])
    if anchor is None:
        raise ValueError("plot_surface_from_file(): surface_fields 缺少 msl/sp/tp/t2m/2t 任一可用锚点。")

    # 可用步长（小时）
    steps_avail = _step_values_hours(anchor).astype(int)

    # 目标步长集合
    if step_filter is not None and len(step_filter) > 0:
        desired = np.unique(np.asarray(step_filter, dtype=int))
    else:
        desired = np.arange(0, int(max_hour) + 1, int(step_interval_hours), dtype=int)

    if skip_zero:
        desired = desired[desired > 0]

    # 交集：确保绘制的步长既在文件中、也在目标集合中
    steps_to_plot = np.intersect1d(steps_avail, desired).astype(int)

    print(f"🧭 地面-可用(h): {steps_avail}")
    print(f"🎯 地面-期望(h): {desired}")
    print(f"🗂 地面-实际绘制(h): {steps_to_plot}")

    # 输出目录：…/<init_tag>/step{interval}h/surface/
    out_dir = os.path.join(output_root, init_tag, f"step{int(step_interval_hours)}h", "surface")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    # —— 批量计时 —— #
    t_total_start = time.perf_counter()
    img_count = 0

    for h in steps_to_plot:
        t0 = time.perf_counter()
        plot_surface_one_time(
            surface_fields, h,
            output_dir=out_dir,
            extent=extent,
            interval_hours=int(step_interval_hours),
            accum_alignment=accum_alignment
        )
        dt = time.perf_counter() - t0
        img_count += 1
        print(f"⏱️ Surface | +{h:03d}h | {dt:.2f}s")

    total_sec = time.perf_counter() - t_total_start
    per_img = total_sec / max(img_count, 1)
    print(f"✅ 地面批量绘图完成：{img_count} 张 | 总耗时 {total_sec:.2f}s | ≈ 每张 {per_img:.2f}s")

import re
# 2) 从文件名提取 init_tag（如 2025103012）
def get_init_tag_from_path(filepath: str, default="unknown"):
    m = re.search(r"(\d{10})", os.path.basename(filepath))
    return m.group(1) if m else default

# 3)（可选）从文件名推断间隔与最大时效（形如 “…_6h_0-240_…”）
def guess_interval_and_maxh(filepath: str, fallback=(6, 240)):
    m = re.search(r"_(\d+)h_0-(\d+)_", os.path.basename(filepath))
    if m:
        return int(m.group(1)), int(m.group(2))
    return fallback




# ========= 主流程 =========
if __name__ == "__main__":
    REGION = (60, 140, 10, 60)      # 绘图裁剪范围
    LEVELS = (500, 700, 850)        # 上空等压面
    OUT_ROOT = "/data/ecmwf/upper_airfields"  # 结果根目录

    today_local = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    SAVE_DIR = os.path.join(OUT_ROOT, today_local)

    JOBS = [(3,72), (6,240), (12,240), (24,240)]  # (间隔, 最大时效)

    client = Client(source="ecmwf")

    for interval, maxh in JOBS:
        print(f"\n🚀 开始任务: interval={interval}h, maxh={maxh}h")

        # 1) 上层
        upper_path, steps_array, cycle_utc = retrieve_combined(
            client, SAVE_DIR,
            step_interval_hours=interval, max_hour=maxh,
            params=["gh","u","v","t"],
            level_list=(500,700,850),
            init_from_latest=True,
            max_attempts_per_cycle=6, backoff_base=5, backoff_cap=300
        )
        init_tag = cycle_utc.strftime("%Y%m%d%H")

        # —— 建议：上层与地面之间稍微歇一下，降低429概率 —— #
        import time, random
        time.sleep(random.uniform(5, 10))

        # 2) 地面（锚定同一轮起报）
        surface_path, _, _ = retrieve_combined(
            client, SAVE_DIR,
            step_interval_hours=interval, max_hour=maxh,
            params=["msl","tp","2t"],
            level_list=None,
            init_from_latest=False,
            anchor_cycle_utc=cycle_utc
        )

        # 3) 上空绘图
        plot_upper_from_file(
            filepath=upper_path, output_root=SAVE_DIR, levels=LEVELS,
            init_tag=init_tag, step_interval_hours=interval, max_hour=maxh, region=REGION
        )

        # 4) 地面绘图
        surface_fields = load_grib_fields(surface_path)
        plot_surface_from_file(
            surface_fields, output_root=SAVE_DIR, init_tag=init_tag,
            step_interval_hours=interval, max_hour=maxh, extent=REGION, 
            skip_zero=True,                # 跳过 +000h，避免“空降水”图
            accum_alignment="trailing",    # 时段增量：tp(h)-tp(h-Δ)
        )

        # —— 建议：每个 JOB 之间冷却 60–120s，避免被 OpenData 限流 —— #
        time.sleep(random.uniform(60, 120))

    print("\n🎉 全部下载 + 上空/地面绘图完成！")



# offline/num_algs/precip_stats.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import xarray as xr

from .base import BaseStatAlgo, register_algo


def _guess_lat_lon_names(da: xr.DataArray) -> tuple[str, str]:
    """
    尝试在 DataArray 中猜测纬度/经度维度名称。
    """
    lat_candidates = ["latitude", "lat", "Latitude", "nav_lat"]
    lon_candidates = ["longitude", "lon", "Longitude", "nav_lon"]

    lat_name = next((n for n in lat_candidates if n in da.dims), None)
    lon_name = next((n for n in lon_candidates if n in da.dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot infer lat/lon dims from {da.dims}. "
            "Please rename dims or customize this helper."
        )

    return lat_name, lon_name


def _get_step_hours(da: xr.DataArray) -> np.ndarray:
    """
    将 step 坐标（通常是 pandas.TimedeltaIndex）转换为整数小时数组。
    """
    if "step" not in da.dims:
        raise ValueError("Precipitation DataArray must have 'step' dimension.")

    step_coord = da["step"].values

    # 常见情况：pandas.Timedelta 或 numpy.timedelta64
    step_hours = []
    for v in step_coord:
        # 兼容 pandas Timedelta / numpy timedelta64 / 纯数字（小时）
        if "timedelta64" in str(type(v)).lower():
            h = int(v / np.timedelta64(1, "h"))
        else:
            # 如果本身就是整数（小时），直接用
            h = int(v)
        step_hours.append(h)
    return np.asarray(step_hours, dtype=int)


def _build_acc_fields(
    tp: xr.DataArray,
    acc_windows_h: List[int],
) -> Dict[int, xr.DataArray]:
    """
    根据累计降水 'tp' 构建不同窗口的累计量（单位 mm）。

    约定：
    - tp: cumulative total precipitation (m)
    - 维度至少包含 'step'、lat、lon
    - 返回：
        {window_h: acc_all}
      其中 acc_all dims: ('step_end', lat, lon)，单位 mm
    """
    step_hours = _get_step_hours(tp)
    hour_to_index = {int(h): i for i, h in enumerate(step_hours)}

    lat_name, lon_name = _guess_lat_lon_names(tp)

    acc_dict: Dict[int, xr.DataArray] = {}

    for win in acc_windows_h:
        acc_list = []
        step_end_vals = []

        for h_end, idx_end in hour_to_index.items():
            h_start = h_end - win
            if h_start not in hour_to_index:
                continue
            idx_start = hour_to_index[h_start]

            # 累计量差分：单位 m -> mm
            acc = (tp.isel(step=idx_end) - tp.isel(step=idx_start)) * 1000.0
            acc_list.append(acc)
            step_end_vals.append(h_end)

        if not acc_list:
            # 某些窗口可能在当前 step 设置下无法计算（比如数据只有 24h，却要求 48h）
            continue

        # 拼成一个新的 DataArray
        acc_all = xr.concat(acc_list, dim="step_end")
        acc_all = acc_all.assign_coords(step_end=("step_end", np.asarray(step_end_vals, dtype=int)))
        acc_all.attrs["long_name"] = f"{win}-hour accumulated precipitation"
        acc_all.attrs["units"] = "mm"

        acc_dict[win] = acc_all

    return acc_dict


def _basic_stats(acc_all: xr.DataArray, thresholds_mm: List[float]) -> Dict[str, Any]:
    """
    对一个窗口的累计降水（step_end, lat, lon）做基础统计。
    统计维度：在空间 + 时间上全域统计（即所有 step_end 合并）。
    """
    # 展平所有维度
    values = acc_all.values  # shape: (n_time, n_lat, n_lon)
    flat = values.reshape(-1)

    # 去掉 NaN
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {
            "max_mm": None,
            "mean_mm": None,
            "pct_points_ge": {str(th): None for th in thresholds_mm},
        }

    max_mm = float(np.nanmax(flat))
    mean_mm = float(np.nanmean(flat))

    total = flat.size
    pct_points_ge: Dict[str, float] = {}
    for th in thresholds_mm:
        pct = float(np.count_nonzero(flat >= th) / total)
        pct_points_ge[str(th)] = pct

    return {
        "max_mm": max_mm,
        "mean_mm": mean_mm,
        "pct_points_ge": pct_points_ge,
    }


def _subset_region_box(
    da: xr.DataArray,
    region_cfg: Dict[str, Any],
) -> Optional[xr.DataArray]:
    """
    旧的 box 版本，保留着以防你以后要用。
    """
    lat_name, lon_name = _guess_lat_lon_names(da)

    lon_min = region_cfg["lon_min"]
    lon_max = region_cfg["lon_max"]
    lat_min = region_cfg["lat_min"]
    lat_max = region_cfg["lat_max"]

    lat = da[lat_name]
    lon = da[lon_name]

    lat_desc = bool(lat[0] > lat[-1])
    if lat_desc:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    lon_desc = bool(lon[0] > lon[-1])
    if lon_desc:
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    sub = da.sel({lat_name: lat_slice, lon_name: lon_slice})
    if sub.size == 0:
        return None
    return sub


def _subset_region(
    da: xr.DataArray,
    ds_full: xr.Dataset,
    region_cfg: Dict[str, Any],
) -> Optional[xr.DataArray]:
    """
    新的统一入口：
    - 如果 region_cfg 里有 mask_var + mask_value → 用掩膜选省份
    - 否则 → 回退到 lon/lat box 方式
    """
    mask_var = region_cfg.get("mask_var")
    mask_value = region_cfg.get("mask_value")

    if mask_var is not None and mask_value is not None:
        if mask_var not in ds_full:
            raise KeyError(
                f"Region config uses mask_var='{mask_var}', "
                f"but it is not found in dataset."
            )
        mask = ds_full[mask_var]
        # da 和 mask 共享同一组坐标（要求你在 merge 的时候对齐好）
        sub = da.where(mask == mask_value)
        if sub.size == 0:
            return None
        return sub

    # 否则仍然走 box 逻辑（兼容老配置）
    return _subset_region_box(da, region_cfg)


@register_algo
class PrecipStats(BaseStatAlgo):
    """
    降水统计算法：

    - 从累计降水（tp, m）计算不同时间窗口（3/6/12/24h）的累计量（mm）
    - 对全域 + 若干区域 box 进行：
        - max_mm
        - mean_mm
        - 各阈值以上的点数占比

    config 示例（JSON / YAML）：

    {
      "var_name": "tp",
      "acc_windows_h": [3, 6, 12, 24],
      "thresholds_mm": [10, 25, 50, 100],
      "regions": [
        {
          "id": "north_china",
          "name": "华北",
          "lon_min": 110,
          "lon_max": 125,
          "lat_min": 35,
          "lat_max": 42
        },
        {
          "id": "yangtze",
          "name": "长江中下游",
          "lon_min": 110,
          "lon_max": 122,
          "lat_min": 27,
          "lat_max": 33
        }
      ]
    }
    """

    name = "precip_stats"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.var_name: str = self.config.get("var_name", "tp")
        self.acc_windows_h: List[int] = self.config.get(
            "acc_windows_h", [3, 6, 12, 24]
        )
        self.thresholds_mm: List[float] = self.config.get(
            "thresholds_mm", [10.0, 25.0, 50.0, 100.0]
        )
        self.regions: List[Dict[str, Any]] = self.config.get("regions", [])

    def run(self, ds: xr.Dataset) -> Dict[str, Any]:
        if self.var_name not in ds:
            raise KeyError(
                f"PrecipStats: variable '{self.var_name}' not found in dataset."
            )

        tp = ds[self.var_name]

        acc_dict = _build_acc_fields(tp, self.acc_windows_h)

        result: Dict[str, Any] = {
            "var_name": self.var_name,
            "unit": "mm",
            "acc_windows_h": self.acc_windows_h,
            "thresholds_mm": self.thresholds_mm,
            "global": {},
            "regions": {},
        }

        # 全域统计
        for win, acc_all in acc_dict.items():
            result["global"][str(win)] = _basic_stats(
                acc_all, self.thresholds_mm
            )

        # 区域统计
        if self.regions:
            for region_cfg in self.regions:
                rid = region_cfg["id"]
                rname = region_cfg.get("name", rid)

                region_stat: Dict[str, Any] = {
                    "name": rname,
                }

                for win, acc_all in acc_dict.items():
                    sub = _subset_region(acc_all, ds, region_cfg)
                    if sub is None or sub.size == 0:
                        region_stat[str(win)] = {
                            "max_mm": None,
                            "mean_mm": None,
                            "pct_points_ge": {
                                str(th): None for th in self.thresholds_mm
                            },
                        }
                    else:
                        region_stat[str(win)] = _basic_stats(sub, self.thresholds_mm)

                result["regions"][rid] = region_stat

                # for win, acc_all in acc_dict.items():
                #     sub = _subset_region_box(acc_all, region_cfg)
                #     if sub is None or sub.size == 0:
                #         region_stat[str(win)] = {
                #             "max_mm": None,
                #             "mean_mm": None,
                #             "pct_points_ge": {
                #                 str(th): None for th in self.thresholds_mm
                #             },
                #         }
                #     else:
                #         region_stat[str(win)] = _basic_stats(
                #             sub, self.thresholds_mm
                #         )

                # result["regions"][rid] = region_stat

        return result

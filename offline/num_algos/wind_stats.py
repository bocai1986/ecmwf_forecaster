# offline/num_algs/wind_stats.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import xarray as xr

from .base import BaseStatAlgo, register_algo
# from .precip_stats import _guess_lat_lon_names, _subset_region_box  # 复用工具函数
from .precip_stats import _subset_region  # 复用统一的区域选择逻辑



def _select_wind_component(
    ds: xr.Dataset,
    candidates: List[str],
    comp_name: str,
) -> xr.DataArray:
    """
    在多个候选变量名中，找到第一个存在于 ds 的，并返回对应 DataArray。

    comp_name 仅用于报错信息，例如 "u" 或 "v"。
    """
    for name in candidates:
        if name in ds:
            return ds[name]

    raise KeyError(
        f"WindStats: cannot find {comp_name}-component in dataset. "
        f"Tried: {candidates}"
    )


def _basic_stats_ws(ws: xr.DataArray, thresholds_ms: List[float]) -> Dict[str, Any]:
    """
    对风速 ws(step, lat, lon) 做基础统计。
    统计维度：将时间 + 空间所有点展开。
    """
    values = ws.values  # shape: (n_time, n_lat, n_lon) 或类似
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        return {
            "max_ms": None,
            "mean_ms": None,
            "pct_points_ge": {str(th): None for th in thresholds_ms},
        }

    max_ms = float(np.nanmax(flat))
    mean_ms = float(np.nanmean(flat))

    total = flat.size
    pct_points_ge: Dict[str, float] = {}
    for th in thresholds_ms:
        pct = float(np.count_nonzero(flat >= th) / total)
        pct_points_ge[str(th)] = pct

    return {
        "max_ms": max_ms,
        "mean_ms": mean_ms,
        "pct_points_ge": pct_points_ge,
    }


@register_algo
class WindStats(BaseStatAlgo):
    """
    10 米风（或其它层风）的基础统计算法。

    功能：
    - 根据 u/v 分量计算风速 ws（m/s）
    - 在全域和各区域 box 上计算：
        - max_ms
        - mean_ms
        - 各阈值以上的点数占比

    config 示例：

    {
      "u_candidates": ["10u", "u10", "U10M"],
      "v_candidates": ["10v", "v10", "V10M"],
      "thresholds_ms": [10, 17.2, 24.5],
      "regions": [
        {
          "id": "north_china",
          "name": "华北",
          "lon_min": 110,
          "lon_max": 125,
          "lat_min": 35,
          "lat_max": 42
        }
      ]
    }

    - thresholds_ms：你可以按需求自定义，例如：
        10 m/s   ≈ 大风蓝色预警附近
        17.2 m/s ≈ 8 级风（大风）
        24.5 m/s ≈ 10 级风
    """

    name = "wind_stats"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # u/v 变量名候选，默认按 ECMWF 常见命名
        self.u_candidates: List[str] = self.config.get(
            "u_candidates", ["10u", "u10", "U10M"]
        )
        self.v_candidates: List[str] = self.config.get(
            "v_candidates", ["10v", "v10", "V10M"]
        )

        # 大风阈值（m/s）
        self.thresholds_ms: List[float] = self.config.get(
            "thresholds_ms", [10.0, 17.2, 24.5]
        )

        # 区域配置，与 PrecipStats 保持完全一致格式
        self.regions: List[Dict[str, Any]] = self.config.get("regions", [])

    def run(self, ds: xr.Dataset) -> Dict[str, Any]:
        # 选择 u/v 分量
        u = _select_wind_component(ds, self.u_candidates, "u")
        v = _select_wind_component(ds, self.v_candidates, "v")

        # 计算风速 ws = sqrt(u^2 + v^2)
        ws = np.hypot(u, v)
        ws = xr.DataArray(
            ws,
            coords=u.coords,
            dims=u.dims,
            name="ws",
        )
        ws.attrs["long_name"] = "wind_speed"
        ws.attrs["units"] = "m s-1"

        result: Dict[str, Any] = {
            "u_var_candidates": self.u_candidates,
            "v_var_candidates": self.v_candidates,
            "unit": "m s-1",
            "thresholds_ms": self.thresholds_ms,
            "global": {},
            "regions": {},
        }

        # —— 全域统计（与降水 PrecipStats 的 _basic_stats 一样，时间+空间一起展开）——
        result["global"] = _basic_stats_ws(ws, self.thresholds_ms)

        # —— 区域统计 —— 
        # if self.regions:
        #     for region_cfg in self.regions:
        #         rid = region_cfg["id"]
        #         rname = region_cfg.get("name", rid)

        #         sub = _subset_region_box(ws, region_cfg)
        #         if sub is None or sub.size == 0:
        #             result["regions"][rid] = {
        #                 "name": rname,
        #                 "max_ms": None,
        #                 "mean_ms": None,
        #                 "pct_points_ge": {
        #                     str(th): None for th in self.thresholds_ms
        #                 },
        #             }
        #         else:
        #             region_stats = _basic_stats_ws(sub, self.thresholds_ms)
        #             region_stats["name"] = rname
        #             result["regions"][rid] = region_stats
        # —— 区域统计 —— 
        if self.regions:
            for region_cfg in self.regions:
                rid = region_cfg["id"]
                rname = region_cfg.get("name", rid)

                sub = _subset_region(ws, ds, region_cfg)
                if sub is None or sub.size == 0:
                    result["regions"][rid] = {
                        "name": rname,
                        "max_ms": None,
                        "mean_ms": None,
                        "pct_points_ge": {
                            str(th): None for th in self.thresholds_ms
                        },
                    }
                else:
                    region_stats = _basic_stats_ws(sub, self.thresholds_ms)
                    region_stats["name"] = rname
                    result["regions"][rid] = region_stats

        return result

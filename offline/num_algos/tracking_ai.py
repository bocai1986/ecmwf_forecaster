"""Rainband trajectory and velocity recognition algorithm.

This module implements a lightweight "Alert Engine" style tracker that scans
periodic precipitation fields, identifies the centroid of the dominant
rainband, and derives motion vectors between consecutive detections.

为了方便快速验证，本文件也提供了一个 ``python -m`` 自检脚本：

.. code-block:: bash

   python offline/num_algos/tracking_ai.py

该脚本会构造一个合成的 6 小时间隔降水样例，运行算法，并把 summary/
track_points/segments 打印出来，方便确认算法可以正确跑通。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr

try:  # pragma: no cover - import guard for standalone demo
    from .base import BaseStatAlgo, register_algo
    from .precip_stats import _get_step_hours, _guess_lat_lon_names, _subset_region
except ImportError:  # pragma: no cover
    # 允许 ``python offline/num_algos/tracking_ai.py`` 直接运行
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from offline.num_algos.base import BaseStatAlgo, register_algo  # type: ignore
    from offline.num_algos.precip_stats import (  # type: ignore
        _get_step_hours,
        _guess_lat_lon_names,
        _subset_region,
    )

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG_LAT = 111.32


@dataclass
class _PeriodField:
    start_hour: float
    end_hour: float
    field: xr.DataArray


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km."""

    phi1, phi2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_KM * c)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return bearing (0-360 degrees) from point1 to point2."""

    phi1, phi2 = np.radians([lat1, lat2])
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    brng = np.degrees(np.arctan2(y, x))
    return float((brng + 360.0) % 360.0)


def _estimate_cell_area(lat_vals: np.ndarray, lon_vals: np.ndarray) -> Optional[np.ndarray]:
    """Approximate per-grid-cell area (km^2) for a regular lat/lon grid."""

    lat_vals = np.asarray(lat_vals, dtype=float)
    lon_vals = np.asarray(lon_vals, dtype=float)
    if lat_vals.size < 2 or lon_vals.size < 2:
        return None

    dlat = float(np.nanmean(np.abs(np.diff(lat_vals))))
    dlon = float(np.nanmean(np.abs(np.diff(lon_vals))))
    if dlat <= 0 or dlon <= 0:
        return None

    lat_grid = lat_vals[:, None]
    km_lat = KM_PER_DEG_LAT * dlat
    km_lon = KM_PER_DEG_LAT * np.cos(np.radians(lat_grid)) * dlon
    return km_lat * km_lon


@register_algo
class RainbandTrackingAlgo(BaseStatAlgo):
    """Detect rainband centroids and derive motion vectors."""

    name = "tracking_ai"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.var_name: str = self.config.get("var_name", "tp")
        self.threshold_mm: float = float(self.config.get("threshold_mm", 5.0))
        self.min_pixels: int = int(self.config.get("min_pixels", 4))
        self.region_cfg: Optional[Dict[str, Any]] = self.config.get("region")
        self.input_is_incremental: bool = bool(self.config.get("input_is_incremental", False))
        self.scale_to_mm: Optional[float] = (
            float(self.config.get("scale_to_mm")) if self.config.get("scale_to_mm") is not None else None
        )
        self.default_interval_h: float = float(self.config.get("default_interval_h", 6.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _derive_scale(self, da: xr.DataArray) -> float:
        if self.scale_to_mm is not None:
            return self.scale_to_mm

        units = (getattr(da, "attrs", {}) or {}).get("units", "")
        units = units.lower()
        if "m" == units or units in {"meter", "meters"}:
            return 1000.0
        if "mm" in units:
            return 1.0
        # 默认假设数据是米（ECMWF tp 常见）
        return 1000.0

    def _build_period_fields(self, tp: xr.DataArray, step_hours: Sequence[int]) -> List[_PeriodField]:
        scale = self._derive_scale(tp)
        periods: List[_PeriodField] = []

        if not self.input_is_incremental:
            if len(step_hours) < 2:
                raise ValueError("tracking_ai: cumulative precipitation requires at least 2 steps.")
            for idx in range(1, len(step_hours)):
                delta = (tp.isel(step=idx) - tp.isel(step=idx - 1)) * scale
                delta = delta.astype(float)
                delta = delta.where(np.isfinite(delta), 0.0)
                delta.attrs["units"] = "mm"
                periods.append(
                    _PeriodField(
                        start_hour=float(step_hours[idx - 1]),
                        end_hour=float(step_hours[idx]),
                        field=delta,
                    )
                )
            return periods

        # 输入已经是时段降水
        inferred_interval = None
        if len(step_hours) >= 2:
            inferred_interval = float(step_hours[1] - step_hours[0])
        interval = float(self.config.get("increment_interval_h", inferred_interval or self.default_interval_h))

        for idx, hour in enumerate(step_hours):
            start_hour = float(hour - interval)
            delta = tp.isel(step=idx) * scale
            delta = delta.astype(float)
            delta = delta.where(np.isfinite(delta), 0.0)
            delta.attrs["units"] = "mm"
            periods.append(_PeriodField(start_hour=start_hour, end_hour=float(hour), field=delta))
        return periods

    def _apply_region(self, da: xr.DataArray, ds_full: xr.Dataset) -> Optional[xr.DataArray]:
        if not self.region_cfg:
            return da
        return _subset_region(da, ds_full, self.region_cfg)

    def _extract_rainband(self, da: xr.DataArray, lat_name: str, lon_name: str) -> Optional[Dict[str, Any]]:
        da = da.transpose(lat_name, lon_name, ...)
        arr = da.values
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        mask = arr >= self.threshold_mm
        if mask.sum() < self.min_pixels:
            return None

        arr_mask = np.where(mask, arr, np.nan)
        total_intensity = np.nansum(arr_mask)
        if not np.isfinite(total_intensity) or total_intensity <= 0:
            return None

        lat_vals = da[lat_name].values
        lon_vals = da[lon_name].values
        lat_grid = lat_vals[:, None]
        lon_grid = lon_vals[None, :]

        centroid_lat = float(np.nansum(arr_mask * lat_grid) / total_intensity)
        centroid_lon = float(np.nansum(arr_mask * lon_grid) / total_intensity)

        try:
            max_idx = np.nanargmax(arr_mask)
            max_pos = np.unravel_index(int(max_idx), arr_mask.shape)
            peak_lat = float(lat_vals[max_pos[0]])
            peak_lon = float(lon_vals[max_pos[1]])
        except (ValueError, IndexError):
            peak_lat = centroid_lat
            peak_lon = centroid_lon

        lat_used = lat_vals[np.any(mask, axis=1)]
        lon_used = lon_vals[np.any(mask, axis=0)]
        lat_min = float(np.nanmin(lat_used)) if lat_used.size else float(np.nanmin(lat_vals))
        lat_max = float(np.nanmax(lat_used)) if lat_used.size else float(np.nanmax(lat_vals))
        lon_min = float(np.nanmin(lon_used)) if lon_used.size else float(np.nanmin(lon_vals))
        lon_max = float(np.nanmax(lon_used)) if lon_used.size else float(np.nanmax(lon_vals))

        coverage_pct = float(mask.sum() / mask.size)
        max_mm = float(np.nanmax(arr_mask))
        mean_mm = float(np.nanmean(arr_mask))

        cell_area = _estimate_cell_area(lat_vals, lon_vals)
        area_km2 = float(np.nansum(cell_area * mask)) if cell_area is not None else None

        return {
            "centroid": {"lat": centroid_lat, "lon": centroid_lon},
            "peak": {"lat": peak_lat, "lon": peak_lon, "max_mm": max_mm},
            "mean_mm": mean_mm,
            "total_intensity_mm": float(total_intensity),
            "covered_pct": coverage_pct,
            "area_km2": area_km2,
            "extent": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        }

    def _build_segments(self, track_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        if len(track_points) < 2:
            return segments

        for prev, curr in zip(track_points[:-1], track_points[1:]):
            dt = curr["center_hour"] - prev["center_hour"]
            if dt <= 0:
                continue
            p_cent = prev.get("centroid") or {}
            c_cent = curr.get("centroid") or {}
            if "lat" not in p_cent or "lon" not in p_cent:
                continue
            if "lat" not in c_cent or "lon" not in c_cent:
                continue
            lat1 = float(p_cent["lat"])
            lon1 = float(p_cent["lon"])
            lat2 = float(c_cent["lat"])
            lon2 = float(c_cent["lon"])

            dist = _haversine_distance(lat1, lon1, lat2, lon2)
            speed = dist / dt if dt > 0 else 0.0
            bearing = _bearing_deg(lat1, lon1, lat2, lon2)
            segments.append(
                {
                    "start_hour": prev["center_hour"],
                    "end_hour": curr["center_hour"],
                    "distance_km": dist,
                    "speed_kmh": speed,
                    "bearing_deg": bearing,
                }
            )
        return segments

    def _build_summary(
        self, track_points: List[Dict[str, Any]], segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "detections": len(track_points),
            "segments": len(segments),
            "threshold_mm": self.threshold_mm,
            "min_pixels": self.min_pixels,
        }
        if not track_points:
            summary.update({"mean_speed_kmh": None, "max_speed_kmh": None, "total_distance_km": None})
            return summary

        areas = [tp.get("area_km2") for tp in track_points if isinstance(tp.get("area_km2"), (int, float))]
        if areas:
            summary["mean_area_km2"] = float(np.mean(areas))
        extent_vals = {
            "lat_min": min(tp["extent"]["lat_min"] for tp in track_points if "extent" in tp),
            "lat_max": max(tp["extent"]["lat_max"] for tp in track_points if "extent" in tp),
            "lon_min": min(tp["extent"]["lon_min"] for tp in track_points if "extent" in tp),
            "lon_max": max(tp["extent"]["lon_max"] for tp in track_points if "extent" in tp),
        }
        summary["overall_extent"] = extent_vals

        if segments:
            speeds = [seg["speed_kmh"] for seg in segments]
            summary["mean_speed_kmh"] = float(np.mean(speeds))
            summary["max_speed_kmh"] = float(np.max(speeds))
            total_distance = sum(seg["distance_km"] for seg in segments)
            summary["total_distance_km"] = float(total_distance)
        else:
            summary.update({"mean_speed_kmh": None, "max_speed_kmh": None, "total_distance_km": None})

        return summary

    # ------------------------------------------------------------------
    # BaseStatAlgo implementation
    # ------------------------------------------------------------------
    def run(self, ds: xr.Dataset) -> Dict[str, Any]:
        if self.var_name not in ds:
            raise KeyError(f"tracking_ai: variable '{self.var_name}' not found in dataset")

        tp = ds[self.var_name]
        if "step" not in tp.dims:
            raise ValueError("tracking_ai: precipitation variable must have 'step' dimension")

        lat_name, lon_name = _guess_lat_lon_names(tp)
        step_hours = _get_step_hours(tp)
        period_fields = self._build_period_fields(tp, step_hours)

        track_points: List[Dict[str, Any]] = []
        for period in period_fields:
            da = period.field
            da = da.where(da >= 0, 0)
            da_region = self._apply_region(da, ds)
            if da_region is None:
                continue
            rainband = self._extract_rainband(da_region, lat_name, lon_name)
            if not rainband:
                continue
            rainband.update(
                {
                    "start_hour": period.start_hour,
                    "end_hour": period.end_hour,
                    "center_hour": (period.start_hour + period.end_hour) / 2.0,
                }
            )
            track_points.append(rainband)

        segments = self._build_segments(track_points)
        summary = self._build_summary(track_points, segments)

        return {
            "track_points": track_points,
            "segments": segments,
            "summary": summary,
            "config_used": {
                "var_name": self.var_name,
                "threshold_mm": self.threshold_mm,
                "min_pixels": self.min_pixels,
                "region": self.region_cfg,
                "input_is_incremental": self.input_is_incremental,
            },
        }


def _build_demo_dataset() -> xr.Dataset:
    """Construct a tiny precipitation dataset for manual smoke tests."""

    steps = xr.DataArray([6, 12, 18], dims=("step",), attrs={"units": "hours"})
    lat = xr.DataArray(np.linspace(20.0, 23.0, 4), dims=("latitude",))
    lon = xr.DataArray(np.linspace(110.0, 113.0, 4), dims=("longitude",))

    # shape: (step, lat, lon)
    data = np.zeros((3, 4, 4), dtype=float)
    # 构造一个简单雨带：逐步往东南移动
    data[0, 1:3, 1:3] = 8.0
    data[1, 1:3, 2:4] = 12.0
    data[2, 0:2, 2:4] = 15.0

    da = xr.DataArray(
        data,
        coords={"step": steps, "latitude": lat, "longitude": lon},
        dims=("step", "latitude", "longitude"),
        name="tp",
    )
    da.attrs["units"] = "mm"

    return xr.Dataset({"tp": da})


def _demo_run() -> None:
    ds = _build_demo_dataset()
    algo = RainbandTrackingAlgo(
        {
            "threshold_mm": 5.0,
            "min_pixels": 4,
            "input_is_incremental": True,
        }
    )
    result = algo.run(ds)
    print("[tracking_ai] Demo summary:")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("[tracking_ai] Track points:")
    print(json.dumps(result["track_points"], ensure_ascii=False, indent=2))
    print("[tracking_ai] Segments:")
    print(json.dumps(result["segments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo_run()

# offline/stats_runner.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import xarray as xr

# 核心调度
from .num_algos.base import build_algos, run_all

# 这两行很重要：触发 @register_algo，把算法注册进 ALGO_REGISTRY
from .num_algos import precip_stats  # noqa: F401
from .num_algos import wind_stats    # noqa: F401


def _open_dataset(path: str, province_mask_path: Optional[str] = None) -> xr.Dataset:
    """
    打开预报文件，并可选地 merge 省份掩膜到同一个 Dataset 里。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    # 用 cfgrib 打开 GRIB2
    # 那个 heightAboveGround 的 DatasetBuildError 会在 cfgrib 内部处理掉，
    # 我们只要拿到 latitude / longitude / tp / 10u / 10v 就可以。
    ds_fc = xr.open_dataset(p, engine="cfgrib")

    if province_mask_path is not None:
        pm_path = Path(province_mask_path)
        if not pm_path.exists():
            raise FileNotFoundError(f"Province mask file not found: {pm_path}")

        ds_mask = xr.open_dataset(pm_path)

        # 这里要求：省份掩膜在同一经纬网格（latitude, longitude）
        # compat="override" 避免 attrs / 小差异报错
        ds = xr.merge([ds_fc, ds_mask], compat="override")
    else:
        ds = ds_fc

    return ds


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")

    if cfg_path.suffix.lower() == ".json":
        return json.loads(text)

    if cfg_path.suffix.lower() in [".yml", ".yaml"]:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config. Install via 'pip install pyyaml'."
            ) from exc
        return yaml.safe_load(text)

    # 默认尝试 JSON
    return json.loads(text)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run numerical statistics algorithms on forecast dataset."
    )

    parser.add_argument(
        "input",
        help="Input forecast file (GRIB2/NetCDF etc.).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="stats.json",
        help="Output JSON file path. Default: stats.json",
    )
    parser.add_argument(
        "--algs",
        nargs="+",
        default=None,
        help="List of algorithm names to run. Default: all registered.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to config file (JSON or YAML).",
    )
    parser.add_argument(
        "--province-mask",
        default=None,
        help="Path to province mask NetCDF (with variable 'province_id').",
    )

    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    # 约定 config 里 algos/algorithms 节点存每个算法自己的配置
    algo_cfg = cfg.get("algos", cfg.get("algorithms", {}))

    # ✅ 在这里把 GRIB2 + 省份掩膜 merge 成一个 ds
    ds = _open_dataset(args.input, province_mask_path=args.province_mask)

    # 构建算法实例列表
    algos = build_algos(config=algo_cfg, names=args.algs)

    # 运行所有算法
    stats = run_all(ds, algos)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[stats_runner] Stats written to: {out_path}")


if __name__ == "__main__":
    main()

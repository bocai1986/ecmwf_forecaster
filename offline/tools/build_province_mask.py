# offline/tools/build_province_mask.py
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from rasterio import features

warnings.filterwarnings("ignore", category=UserWarning)


def build_province_mask(ec_grib_path: str,
                        shp_path: str,
                        out_path: str,
                        province_field: str | None = None) -> None:
    print(f"读取 ECMWF GRIB2 网格: {ec_grib_path}")
    ds = xr.open_dataset(ec_grib_path, engine="cfgrib")

    # 这里假定是规则经纬网，变量名为 latitude/longitude（EC 常规输出）
    if "latitude" not in ds or "longitude" not in ds:
        raise KeyError("数据集中未找到 'latitude' 或 'longitude'，请检查 GRIB 打开方式。")

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    print("纬度方向：", lat[0], "→", lat[-1], "（一般是北到南递减）")
    print("网格大小: ", lat.shape[0], "×", lon.shape[0])

    # 构建仿射变换（affine transform）用于 rasterize
    res_lon = float(abs(lon[1] - lon[0]))
    res_lat = float(abs(lat[1] - lat[0]))

    # 注意 EC 纬度通常是从北向南递减，所以 y 缩放用 -res_lat
    transform = Affine.translation(lon[0] - res_lon / 2.0,
                                   lat[0] + res_lat / 2.0) * Affine.scale(
        res_lon, -res_lat
    )

    height = lat.shape[0]
    width = lon.shape[0]

    print(f"读取省界 Shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs("EPSG:4326")

    print("Shapefile 字段名: ", list(gdf.columns))

    # 自动猜一下省名字段
    if province_field is None:
        for cand in ["NAME_CHN", "NAME", "NAME_1", "NL_NAME_1"]:
            if cand in gdf.columns:
                province_field = cand
                break
    if province_field is None:
        raise ValueError("无法自动识别省名字段，请用 --field-name 手动指定。")

    print(f"使用字段 '{province_field}' 作为省份名称。")

    province_ids = np.zeros((height, width), dtype=np.int16)

    unique_names = gdf[province_field].unique()
    name_to_id = {name: i + 1 for i, name in enumerate(unique_names)}

    print("开始 rasterize 每个省到网格上...")
    for name, pid in name_to_id.items():
        geom = gdf[gdf[province_field] == name].geometry
        if geom.empty:
            continue

        shapes = (
            (poly, pid)
            for poly in geom
            if poly is not None and not poly.is_empty
        )

        mask = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int16,
        )

        province_ids = np.maximum(province_ids, mask)

    # 保存为 NetCDF，与 ECMWF 网格对齐
    ds_mask = xr.Dataset(
        {"province_id": (("latitude", "longitude"), province_ids)},
        coords={"latitude": lat, "longitude": lon},
    )

    print(f"写出到: {out_path}")
    ds_mask.to_netcdf(out_path)

    # print("省份编码表：")
    # for name, pid in name_to_id.items():
    #     print(f"{pid:2d}  {name}")
    print("省份编码表：")
    for name, pid in name_to_id.items():
        try:
            fixed = name.encode("latin1").decode("utf-8")  # 或 utf-8/gbk 自己试
        except Exception:
            fixed = name
        print(f"{pid:2d}  {fixed}")



def main():
    parser = argparse.ArgumentParser(
        description="构建与 ECMWF 网格对齐的省份掩膜 province_mask_*.nc"
    )
    parser.add_argument(
        "--ec-grib",
        required=True,
        help="任意一份 ECMWF GRIB2 文件路径，用来提供经纬网格。",
    )
    parser.add_argument(
        "--shp",
        required=True,
        help="中国省界 Shapefile 文件路径（*.shp）。",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出 NetCDF 掩膜文件路径，例如 offline/data/province_mask_0125.nc",
    )
    parser.add_argument(
        "--field-name",
        default=None,
        help="省名字段名（例如 NAME_CHN / NAME_1 等），不填则自动猜测。",
    )

    args = parser.parse_args()

    build_province_mask(
        ec_grib_path=args.ec_grib,
        shp_path=args.shp,
        out_path=args.out,
        province_field=args.field_name,
    )


if __name__ == "__main__":
    main()

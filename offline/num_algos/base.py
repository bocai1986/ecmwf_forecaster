# offline/num_algs/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Optional

import xarray as xr


class BaseStatAlgo(ABC):
    """
    所有数值统计算法的基类。

    约定：
    - name: 算法在 stats.json 中的 key
    - run(ds): 输入一个 xarray.Dataset，返回纯 Python dict（可 JSON 序列化）
    """

    # 子类必须覆盖
    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # config 由外部传入，便于控制阈值、区域等参数
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def run(self, ds: xr.Dataset) -> Dict[str, Any]:
        """
        执行统计，返回 dict（可直接 dump 成 JSON）。
        """
        raise NotImplementedError


# 全局算法注册表： name -> AlgoClass
ALGO_REGISTRY: Dict[str, Type[BaseStatAlgo]] = {}


def register_algo(cls: Type[BaseStatAlgo]) -> Type[BaseStatAlgo]:
    """
    装饰器：将算法类注册到 ALGO_REGISTRY 中。

    示例：
    @register_algo
    class PrecipStats(BaseStatAlgo):
        name = "precip_stats"
        ...
    """
    if not issubclass(cls, BaseStatAlgo):
        raise TypeError(f"{cls} is not a subclass of BaseStatAlgo")

    if not getattr(cls, "name", None):
        raise ValueError(f"{cls} must define a non-empty 'name' attribute")

    if cls.name in ALGO_REGISTRY:
        # 允许重复导入时覆盖，但打印个提示（可按需改成报错）
        # print(f"[BaseStatAlgo] Algo '{cls.name}' already registered, overriding.")
        pass

    ALGO_REGISTRY[cls.name] = cls
    return cls


def build_algos(
    config: Optional[Dict[str, Any]] = None,
    names: Optional[List[str]] = None,
) -> List[BaseStatAlgo]:
    """
    根据可选的 names 列表 + config 构建算法实例列表。

    config 结构约定为：
    {
      "precip_stats": { ...该算法的配置... },
      "wind_extreme": { ... }
    }
    """
    config = config or {}
    selected_names = names or list(ALGO_REGISTRY.keys())

    algos: List[BaseStatAlgo] = []
    for name in selected_names:
        if name not in ALGO_REGISTRY:
            raise KeyError(f"Algo '{name}' not found in ALGO_REGISTRY")

        algo_cls = ALGO_REGISTRY[name]
        algo_cfg = config.get(name, {})
        algos.append(algo_cls(algo_cfg))

    return algos


def run_all(ds: xr.Dataset, algos: List[BaseStatAlgo]) -> Dict[str, Any]:
    """
    依次运行所有算法，将结果合并为一个 dict：
    {
      "precip_stats": { ... },
      "wind_extreme": { ... }
    }
    """
    all_stats: Dict[str, Any] = {}
    for algo in algos:
        all_stats[algo.name] = algo.run(ds)
    return all_stats

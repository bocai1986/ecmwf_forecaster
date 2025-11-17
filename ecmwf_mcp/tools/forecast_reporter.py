# mcp/tools/forecast_reporter.py
"""
用本地 vLLM + Qwen3-VL 把离线产出的 24h 素材转成结构化天气报告。
重点：在模块加载阶段就初始化 vLLM，这样同一个进程里的多次工具调用不会重复加载模型。
"""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
import json
import os

from vllm import LLM
from vllm.sampling_params import SamplingParams

# -----------------------------------------------------------------------------
# 1) 顶层配置
# -----------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_QWEN_PATH = os.getenv("QWEN3VL_PATH", "/data/Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_GPU_ID = int(os.getenv("QWEN3VL_GPU", "2"))
DEFAULT_MAX_TOKENS = 1024

# 先设置 GPU，再初始化（一定要在 LLM() 之前）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(DEFAULT_GPU_ID))
print(f"🧠 [forecast_reporter] init vLLM at import time on GPU {DEFAULT_GPU_ID}, model={DEFAULT_QWEN_PATH}")

# 👉 顶层一次性初始化
_LLM = LLM(model=DEFAULT_QWEN_PATH)
_SAMPLING_PARAMS = SamplingParams(
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=0.2,
    top_p=0.9,
)

# -----------------------------------------------------------------------------
# 2) Pydantic 入参
# -----------------------------------------------------------------------------
class Args(BaseModel):
    publisher_payload_json: Optional[str] = Field(default=None)
    manifest_path: Optional[str] = Field(default=None)
    # 下面这些能传就传，但即使不传我们也已经在顶层初始化了
    model_path: str = Field(default=DEFAULT_QWEN_PATH)
    max_tokens: int = DEFAULT_MAX_TOKENS
    gpu_id: int = DEFAULT_GPU_ID


# -----------------------------------------------------------------------------
# 3) 辅助函数
# -----------------------------------------------------------------------------
def _load_publisher_like_from_manifest(manifest_path: str) -> Dict[str, Any]:
    from tools.forecast_publisher import run as pub_run, Args as PubArgs
    return pub_run(PubArgs(manifest_path=manifest_path))


# def _build_system_prompt() -> str:
#     return (
#         "你是一名数值预报员助手，需要根据 ECMWF 产出的高空场和地面场的素材，"
#         "写一份给预报员看的要点稿。\n"
#         "输出必须是 JSON，字段固定为：synoptic, precip, winds, advisory, conclusion, missing。\n"
#         "先讲形势，再讲降水，再讲风，最后给1~2句预报意见，缺资料要写在 missing 里。\n"
#     )


# def _build_user_prompt_from_payload(payload: Dict[str, Any]) -> str:
#     root = payload.get("root", "")
#     analyze = payload.get("analyze_3h") or {}
#     result = payload.get("result_12h") or {}

#     lines = []
#     lines.append("以下是本次 ECMWF 24 小时预报可用的本地素材（已转视频）：")
#     if root:
#         lines.append(f"- 根目录: {root}")

#     if analyze:
#         lines.append(f"- 分析版：步长 {analyze.get('step')}，时段 {analyze.get('tag')}")
#         if analyze.get("combined"):
#             lines.append(f"  • 综合视频: {analyze['combined'].get('abs_path')}")
#         for lname, meta in (analyze.get("layers") or {}).items():
#             lines.append(f"  • {lname}: {meta.get('abs_path')}")

#     if result:
#         lines.append(f"- 结果版：步长 {result.get('step')}，时段 {result.get('tag')}")
#         if result.get("combined"):
#             lines.append(f"  • 综合视频: {result['combined'].get('abs_path')}")
#         for lname, meta in (result.get("layers") or {}).items():
#             lines.append(f"  • {lname}: {meta.get('abs_path')}")

#     lines.append("")
#     lines.append("请基于这些素材写出 JSON：")
#     lines.append(
#         '{\n'
#         '  "synoptic": "...",\n'
#         '  "precip": "...",\n'
#         '  "winds": "...",\n'
#         '  "advisory": "...",\n'
#         '  "conclusion": "...",\n'
#         '  "missing": ""\n'
#         '}'
#     )
#     return "\n".join(lines)

def _build_system_prompt() -> str:
    return (
        "你是一名数值预报员助手，负责根据 ECMWF 的高空场与地面场素材，"
        "自动生成一份专业、面向预报员的天气要点稿。\n"
        "输出必须是 JSON，字段固定为：synoptic, precip, winds, advisory, conclusion, missing。\n"
        "写作逻辑必须遵循：先讲大尺度形势，再讲降水，再讲风，然后提出预报建议，最后给1～2句总括性判断；"
        "缺失的资料必须列在 missing。\n"
        "写作风格参考中国气象局短期天气公报：结构清晰、定量描述、客观中性，"
        "避免冗长叙述，不要使用口语化表达。\n"
        "【写作要点】\n"
        "1) synoptic：描述主要环流形势、槽脊演变、冷暖空气活动、主要天气系统（槽、脊、切变、低涡、副高、倒槽等）、关键时间节点。\n"
        "2) precip：说明主要降水区域、阶段、量级（小雨/中雨/大雨/暴雨/雪/雨夹雪）、落区变化。\n"
        "3) winds：交代大风落区、风向风力、冷空气强度及影响海区情况（如适用）。\n"
        "4) advisory：提出对预报操作有用的提示，例如需关注的落区偏差、强对流可能性、降温幅度、次生灾害风险。\n"
        "5) conclusion：1～2 句总括性判断，例如“未来三天冷空气影响明显，中东部将出现大范围降温及大风天气”。\n"
        "6) missing：列出缺失的关键资料（如某层风场未提供、某时效缺帧等）。\n"
        "请确保所有描述基于提供的 ECMWF 素材推断，不可胡乱编造。"
    )

def _build_user_prompt_from_payload(payload: Dict[str, Any]) -> str:
    root = payload.get("root", "")
    analyze = payload.get("analyze_3h") or {}
    result = payload.get("result_12h") or {}

    lines = []
    lines.append("以下是本次 ECMWF 预报转码后的素材视频（上空 + 地面）：")
    if root:
        lines.append(f"- 根目录: {root}")

    # ----- 分析版（3h） -----
    if analyze:
        lines.append(f"\n【分析版（step={analyze.get('step')}，时段={analyze.get('tag')}）】")

        if analyze.get("combined"):
            lines.append(f"  • 综合视频 (850→700→500→地面)：{analyze['combined'].get('abs_path')}")

        # 按层汇报
        layers = analyze.get("layers") or {}
        if "850hPa" in layers:
            lines.append(f"  • 850hPa 高空场：{layers['850hPa'].get('abs_path')}")
        if "700hPa" in layers:
            lines.append(f"  • 700hPa 高空场：{layers['700hPa'].get('abs_path')}")
        if "500hPa" in layers:
            lines.append(f"  • 500hPa 高空场：{layers['500hPa'].get('abs_path')}")

        # 重点：地面部分区分 phase / synoptic
        if "surface_phase" in layers:
            lines.append(f"  • 地面降水相态图：{layers['surface_phase'].get('abs_path')}")
        if "surface_synoptic" in layers:
            lines.append(f"  • 地面MSLP+10m 风场图：{layers['surface_synoptic'].get('abs_path')}")

    # ----- 结果版（12h） -----
    if result:
        lines.append(f"\n【结果版（step={result.get('step')}，时段={result.get('tag')}）】")

        if result.get("combined"):
            lines.append(f"  • 综合视频 (850→700→500→地面)：{result['combined'].get('abs_path')}")

        layers = result.get("layers") or {}
        if "850hPa" in layers:
            lines.append(f"  • 850hPa 高空场：{layers['850hPa'].get('abs_path')}")
        if "700hPa" in layers:
            lines.append(f"  • 700hPa 高空场：{layers['700hPa'].get('abs_path')}")
        if "500hPa" in layers:
            lines.append(f"  • 500hPa 高空场：{layers['500hPa'].get('abs_path')}")

        if "surface_phase" in layers:
            lines.append(f"  • 地面降水相态图：{layers['surface_phase'].get('abs_path')}")
        if "surface_synoptic" in layers:
            lines.append(f"  • 地面MSLP+10m 风场图：{layers['surface_synoptic'].get('abs_path')}")

    # ----- 输出格式提示 -----
    lines.append("\n请基于这些素材写出 JSON：")
    lines.append(
        '{\n'
        '  "synoptic": "...",\n'
        '  "precip": "...",\n'
        '  "winds": "...",\n'
        '  "advisory": "...",\n'
        '  "conclusion": "...",\n'
        '  "missing": ""\n'
        '}'
    )

    return "\n".join(lines)




# -----------------------------------------------------------------------------
# 4) 主入口
# -----------------------------------------------------------------------------
def run(args: Args) -> Dict[str, Any]:
    # 1) 拿素材
    if args.publisher_payload_json:
        payload = json.loads(args.publisher_payload_json)
    elif args.manifest_path:
        payload = _load_publisher_like_from_manifest(args.manifest_path)
    else:
        raise ValueError("必须提供 publisher_payload_json 或 manifest_path")

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt_from_payload(payload)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 2) 直接用顶层已经初始化好的 vLLM
    #    如果你需要按调用动态改 max_tokens，也可以在这里重新 new SamplingParams
    sampling_params = _SAMPLING_PARAMS
    if args.max_tokens != _SAMPLING_PARAMS.max_tokens:
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.2,
            top_p=0.9,
        )

    outputs = _LLM.chat(messages=messages, sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {
            "synoptic": text,
            "precip": "",
            "winds": "",
            "advisory": "",
            "conclusion": "",
            "missing": "LLM未按JSON输出，已保留原文在synoptic。",
        }

    print("📤 forecast_reporter 已完成推理。")
    return {
        "gpu_used": args.gpu_id,
        "model": args.model_path,
        "raw_text": text,
        "report": parsed,
        "source_payload": payload,
    }

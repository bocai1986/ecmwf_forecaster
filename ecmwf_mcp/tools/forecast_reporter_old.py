# mcp/tools/forecast_reporter.py
"""
用本地 vLLM + Qwen3-VL 把离线产出的 24h 素材（视频路径）转成一段结构化天气报告。

用法示例（和前两个工具衔接）：
    from tools.forecast_publisher import run as pub_run, Args as PubArgs
    from tools.forecast_reporter import run as rep_run, Args as RepArgs

    pub_payload = pub_run(PubArgs(manifest_path=".../products/manifest.json"))
    report = rep_run(RepArgs(publisher_payload_json=json.dumps(pub_payload)))
    print(report["text"])
"""


"""
用本地 vLLM + Qwen3-VL 把离线产出的 24h 素材（视频路径）转成结构化天气报告。
"""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
from pathlib import Path
import json
import os

from vllm import LLM
from vllm.sampling_params import SamplingParams

# 默认模型路径，可用环境变量 QWEN3VL_PATH 覆盖
DEFAULT_QWEN_PATH = os.getenv("QWEN3VL_PATH", "/data/Qwen/Qwen3-VL-8B-Instruct")


class Args(BaseModel):
    publisher_payload_json: Optional[str] = Field(default=None)
    manifest_path: Optional[str] = Field(default=None)
    model_path: str = Field(default=DEFAULT_QWEN_PATH)
    max_tokens: int = 1024
    gpu_id: int = 2   # 默认使用 GPU:2


def _load_publisher_like_from_manifest(manifest_path: str) -> Dict[str, Any]:
    from tools.forecast_publisher import run as pub_run, Args as PubArgs
    return pub_run(PubArgs(manifest_path=manifest_path))



# def _build_system_prompt() -> str:
#     return (
#         "你是一名数值预报员助手，素材来自 ECMWF 预报（上空 500/700/850hPa 和地面要素，含多个预报时次）。\n"
#         "你的目标是：在不臆测、不过度数值化的前提下，把图像/视频里的天气信号整理成一份给预报员看的要点稿。\n"
#         "\n"
#         "# 总体原则\n"
#         "1. 仅关注内陆区域的信号，海上或远洋信号可略写或不写。\n"
#         "2. 优先描述已经在图上能看到的结构、中心、范围和演变方向，避免凭空推测数值大小。\n"
#         "3. 若素材缺层次、缺时次或路径不存在，要在结果里标注“缺失”。\n"
#         "\n"
#         "# 任务分解\n"
#         "1) 逐时次识别（每个预报时次，只写内陆）：\n"
#         "   - 500 hPa：说明槽/脊/切变位置，是否有高压脊或低槽深入内陆；标出明显的 H/L/W/C 标志；若能看到 0℃线或冷中心/暖中心，可简单说明。\n"
#         "   - 700 hPa：给出主风向和风速等级；识别是否存在 ≥16 m/s 的急流区，说明核心方位、是否呈带状分布；若能看到辐合/切变线也要写；若有明显的高湿区（如 ≥9 g/kg）要指出与地面天气的潜在配合。\n"
#         "   - 地面/10 m 风（内陆）：参照图例或以 ≥8 / ≥12 / ≥17 m/s 的分级口径，描述强风核心或风带的位置、形态（条带/扇形/通道）、大致走向和主风向；若能看出与 700 hPa 急流入口、切变线、湿舌或地形通道化有耦合，要写成说明句；量化困难时用“弱/中/强+核心区域/走向”。\n"
#         "\n"
#         "2) 跨时次动态（仍以内陆为主）：\n"
#         "   - 把 500/700 的主要系统（槽、脊、切变、急流带）和地面/10 m 强风带联系起来，给出它们在多个时次之间的移动路径、强度变化和控制范围的变化。\n"
#         "   - 指出哪个时段是内陆风力最强、抬升或水汽最配合的时段。\n"
#         "\n"
#         "3) 动态总评（写在最后）：\n"
#         "   - 先说主控环流的格局和演变方向（比如：西风槽东移、南支扰动北抬、低涡东移等）。\n"
#         "   - 再说内陆 10 m 风的总体演变：最强在哪个时段、风带从哪里推进到哪里、是否有地形通道导致的局地增强。\n"
#         "   - 若 12 小时结果版给出了更清晰的降水/落区，就用它来校正前面的判断，并写明“以 12h 结果版为准”。\n"
#         "\n"
#         "# 输出格式\n"
#         "请最终输出 JSON，字段如下：\n"
#         "{\n"
#         '  "synoptic": "高空和地面主要系统的描述，按时次或按层次展开",\n'
#         '  "winds": "内陆10m风、强风带、与700急流/地形的耦合情况，说明最强时段和移动路径",\n'
#         '  "precip": "若素材能看出降水/湿区/锋面配合，则写出落区和强弱；看不清要说清楚原因",\n'
#         '  "missing": "素材缺失的层次/时次/文件列表，没有就写空字符串",\n'
#         '  "conclusion": "一句话总结，告诉预报员24小时内最值得看的点"\n'
#         "}\n"
#         "如果素材中出现的时次只覆盖 +12~+24 或 +12~+36，也要在 conclusion 里说明时效范围。\n"
#     )

def _build_system_prompt() -> str:
    return (
        "你是一名数值预报员助手，需要根据 ECMWF 产出的高空场和地面场的**视频/图片序列**，"
        "给出一份可直接给预报员看的要素化描述。\n"
        "写作要求：\n"
        "1. 先讲形势，再讲天气，再讲影响，最后给1~2句预报意见。\n"
        "2. 形势重点写：500 hPa 槽脊/短波、700 hPa 急流和水汽带、地面气压场配置；要说明它们在这24小时是东移、加深还是减弱。\n"
        "3. 降水要单独一段写，必须包含：落区（地理+方位）、时段（例如+12~+24h）、强度等级（弱/中等/较强，可参考图上色阶）、成因（槽前抬升/急流入口/低涡切变等）。\n"
        "4. 风要素写10 m或地面风的**最强带/核心/影响区**，说明它和700 hPa急流或地形是否耦合。\n"
        "5. 预报意见要明确，不要出现“以……为准”这种模糊表述；如果你是依据结果版(12h)去修正分析版(3h)，请说成“以结果版所示的××落区为主要降水范围，可结合实况再行订正”。\n"
        "6. 如果某一层或某一时段的视频/图片缺失，请在 `missing` 字段里写明“缺少××资料，降水位置存在不确定性”。\n"
        "7. 全部输出必须是一个 JSON 对象，字段固定为：synoptic, precip, winds, advisory, conclusion, missing。\n"
        "8. 语言风格保持业务口径，少用口语。"
    )





# def _build_user_prompt_from_payload(payload: Dict[str, Any]) -> str:
#     """
#     根据 manifest/publisher payload 构造用户提示，列出视频路径，并要求模型生成结构化天气分析。
#     """
#     root = payload.get("root", "")
#     analyze = payload.get("analyze_3h") or {}
#     result = payload.get("result_12h") or {}

#     lines = []
#     lines.append("以下是 ECMWF 24 小时预报的图像与视频素材，请按说明完成分析：")
#     lines.append(f"根目录：{root}\n")

#     # === step3h 分析版 ===
#     if analyze:
#         lines.append(f"【分析版】（步长 {analyze.get('step')}，时段 {analyze.get('tag')}）")
#         if analyze.get("combined"):
#             lines.append(f"  ├─ 总视频：{analyze['combined'].get('abs_path', '')}")
#         for lname, meta in (analyze.get("layers") or {}).items():
#             lines.append(f"  ├─ {lname}: {meta.get('abs_path', '')}")
#     else:
#         lines.append("【分析版】暂无数据")

#     # === step12h 结果版 ===
#     if result:
#         lines.append(f"\n【结果版】（步长 {result.get('step')}，时段 {result.get('tag')}）")
#         if result.get("combined"):
#             lines.append(f"  ├─ 总视频：{result['combined'].get('abs_path', '')}")
#         for lname, meta in (result.get("layers") or {}).items():
#             lines.append(f"  ├─ {lname}: {meta.get('abs_path', '')}")
#     else:
#         lines.append("【结果版】暂无数据")

#     # === 指令部分 ===
#     lines.append("\n请基于这些素材（仅关注内陆区域）完成以下任务：")
#     lines.append(
#         "1. 识别各时次的 500/700/10m（地面）信号，包括槽脊、切变、急流、高湿区、强风带、主风向等。\n"
#         "2. 综合分析 500/700 与地面风的时空演变，指出强风与环流系统的对应关系。\n"
#         "3. 结合 12h 结果版，判断主要降水或落区位置与强度。\n"
#         "4. 输出结构化报告（JSON 格式）："
#     )

#     lines.append(
#         '{\n'
#         '  "synoptic": "高空和地面系统演变的综合描述",\n'
#         '  "winds": "内陆10m风与700急流/地形的耦合及其时空变化",\n'
#         '  "precip": "降水/湿区/锋面等落区描述",\n'
#         '  "missing": "缺失素材说明（若无写空字符串）",\n'
#         '  "conclusion": "一句话总结，指出24小时内的主要天气特征"\n'
#         '}'
#     )

#     return "\n".join(lines)

# from typing import Dict, Any

def _build_user_prompt_from_payload(payload: Dict[str, Any]) -> str:
    root = payload.get("root", "")
    analyze = payload.get("analyze_3h") or {}
    result = payload.get("result_12h") or {}

    lines = []
    lines.append("以下是本次 ECMWF 24 小时预报可用的本地素材（已转成视频）：")
    if root:
        lines.append(f"- 根目录: {root}")

    if analyze:
        lines.append(f"- 分析版（高频，{analyze.get('step')}，{analyze.get('tag')}）：")
        if analyze.get("combined"):
            lines.append(f"  • 综合视频: {analyze['combined'].get('abs_path')}")
        layers = analyze.get("layers") or {}
        for lname, meta in layers.items():
            lines.append(f"  • {lname} 分层视频: {meta.get('abs_path')}")

    if result:
        lines.append(f"- 结果版（低频，{result.get('step')}，{result.get('tag')}）：")
        if result.get("combined"):
            lines.append(f"  • 综合视频: {result['combined'].get('abs_path')}")
        layers = result.get("layers") or {}
        for lname, meta in layers.items():
            lines.append(f"  • {lname} 分层视频: {meta.get('abs_path')}")

    lines.append("")
    lines.append("请结合上述素材，输出下面这个 JSON 模板（字段名不要改）：")
    lines.append(
        '{\n'
        '  "synoptic": "整体环流形势和各层演变，比如500槽东移、700急流位置、地面低压高压的配合",\n'
        '  "precip": "分3要素写：①哪里下（地理+方位）；②什么时候下（如+12~+24h）；③下多大（弱/中/较强或图上色阶）。若分析版与结果版位置略有差异，请说“以结果版所示区域为主要降水范围，可结合实况再订正”。不要写成“以…为准”。",\n'
        '  "winds": "说明地面/10m风的最强时段、核心区和走向，以及与700hPa急流或地形的耦合关系。",\n'
        '  "advisory": "给预报员的直接意见，比如：重点关注××一带的降水增强时段；江南一线阵风可达×级；如晨间资料不支持东移幅度，则北界需下调。",\n'
        '  "conclusion": "一句话业务总结，把最重要的天气现象+时间+范围说清楚。禁止出现“以…为准”这种模糊表述。",\n'
        '  "missing": "如果发现上面文件路径有缺失或无法判断的要素，在这里写明；否则写空串。"\n'
        '}'
    )

    return "\n".join(lines)



def run(args: Args) -> Dict[str, Any]:
    # 1) 准备输入数据
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

    # 2) 固定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"🧠 Using GPU ID: {args.gpu_id}")

    # 3) 初始化 vLLM 模型
    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.2,
        top_p=0.9,
    )

    # 4) 生成结果
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"synoptic": text, "precip": "", "impact": "", "conclusion": ""}

    print("📤 forecast_reporter 已完成推理。")
    return {
        "gpu_used": args.gpu_id,
        "model": args.model_path,
        "raw_text": text,
        "report": parsed,
        "source_payload": payload,
    }
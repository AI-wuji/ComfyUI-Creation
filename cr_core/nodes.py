"""Creative Reasoning API 节点。

双模型设计：
  * 多模态文本模型（text_model）：始终参与。负责「元指令」「反推提示词」的处理，
    以及在「不选择生成模型」时直接产出文本（写小说、扩写等全靠元指令驱动）。
  * 生成模型（gen_model）：图像 / 视频 / 音频模型，可选。留空 = 纯文本输出。
    一旦选中，就把文本模型整理好的最终提示词交给它做文生图 / 图生视频 / 音乐 / 语音克隆。

图像可按语义接入首帧 / 尾帧 / 多参考图，视频与音频同样支持动态多路输入，
所有接入都会真实写进请求体（role 字段 / image 数组 / content 分段），不是摆设。
服务端点写死在 config.py，不在界面上出现。

反推是节点默认就具备的能力：开一个开关即可，具体「反什么、按什么格式输出」
完全由元指令决定（元指令里带 json 关键词就按 JSON 输出），节点不内置任何反推提示词。
"""

from __future__ import annotations

import json
import logging
import re
from fractions import Fraction
from pathlib import Path

import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, InputImpl, Types

from . import capabilities as caps
from . import catalog, config, media
from .http_client import CRError, is_done, poll_task, request_json

logger = logging.getLogger("CreativeReasoning")

# gen_model 下拉菜单里表示「不选择生成模型 = 纯文本输出」的占位项
GEN_NONE_LABEL = "（纯文本：只输出文本，不调用生成模型）"

# 界面占位/说明文案一律用全角括号包裹，永不可能是合法模型 ID
_PLACEHOLDER_MODEL_RE = re.compile(r"^[（(].*[）)]$", re.S)
_NONE_WORDS = {"none", "null", "<none>", "n/a", "na", "-", "--"}


def is_placeholder_model(value: str) -> bool:
    """是否为「界面占位文案」而非真实模型 ID。

    gen_model 下拉的「（纯文本…）」以及将来任何说明性选项都会落到这里。
    它绝不能进入请求体的 model 字段，否则服务端会返回
    `invalid type: string "…", expected a borrowed string`（serde 拒绝转义串）。
    """
    text = (value or "").strip()
    if not text:
        return True
    if text == GEN_NONE_LABEL:
        return True
    if text.lower() in _NONE_WORDS:
        return True
    return bool(_PLACEHOLDER_MODEL_RE.match(text))

# 首尾帧 / 多参模式。值 -> UI 文案
ROLE_MODE_KEYS = [
    "all_reference",
    "auto",
    "first_last",
    "first_only",
    "last_only",
    "first_then_reference",
]
ROLE_MODE_LABELS = {
    "auto": "① 智能｜1图=首帧，2图=首+尾帧，多图=首+尾+其余作参考",
    "first_last": "② 首尾帧｜前两张作首/尾帧，其余忽略⚠",
    "first_only": "③ 仅首帧｜第1张作首帧，其余忽略⚠",
    "last_only": "④ 仅尾帧｜第1张作尾帧，其余忽略⚠",
    "all_reference": "⑤ 多参｜全部作参考图，不设首尾帧",
    "first_then_reference": "⑥ 首帧+多参｜第1张作首帧，其余全部作参考图",
}
ROLE_MODE_CHOICES = [ROLE_MODE_LABELS[k] for k in ROLE_MODE_KEYS]
_ROLE_MODE_BY_LABEL = {v: k for k, v in ROLE_MODE_LABELS.items()}

# 只保留上游真实支持的档位
ASPECT_RATIOS = ["", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9"]
RESOLUTIONS = ["", "480p", "720p", "1080p"]

# 图像模型（如 Doubao-Seedream）只认 size="WxH"，不认 aspect_ratio。
# 这里把「画面比例 + 分辨率档位」组合成绝对像素 size。
# 任一为空就退到默认 size（2048x2048），两个都空则不下发（模型自决）。
SIZE_PRESETS: dict[tuple[str, str], str] = {
    ("16:9",  "480p"):  "854x480",
    ("16:9",  "720p"):  "1280x720",
    ("16:9",  "1080p"): "1920x1080",
    ("16:9",  ""):      "2048x1152",
    ("9:16",  "480p"):  "480x854",
    ("9:16",  "720p"):  "720x1280",
    ("9:16",  "1080p"): "1080x1920",
    ("9:16",  ""):      "1152x2048",
    ("1:1",   "480p"):  "640x640",
    ("1:1",   "720p"):  "1024x1024",
    ("1:1",   "1080p"): "1440x1440",
    ("1:1",   ""):      "2048x2048",
    ("4:3",   "480p"):  "640x480",
    ("4:3",   "720p"):  "960x720",
    ("4:3",   "1080p"): "1440x1080",
    ("4:3",   ""):      "2048x1536",
    ("3:4",   ""):      "1536x2048",
    ("21:9",  ""):      "2560x1080",
    ("",      "720p"):  "1280x720",
    ("",      "1080p"): "1920x1080",
}


def _image_size(aspect_ratio: str, resolution: str) -> str:
    """把 (aspect_ratio, resolution) 映射成 image 接口的 size 字符串。"""
    ar = (aspect_ratio or "").strip()
    res = (resolution or "").strip()
    if res.upper() in {"1K", "2K", "4K"}:
        long_edge = {"1K": 1024, "2K": 2048, "4K": 4096}[res.upper()]
        try:
            width_ratio, height_ratio = (float(part) for part in ar.split(":"))
            if width_ratio <= 0 or height_ratio <= 0:
                raise ValueError
        except (ValueError, TypeError):
            width_ratio, height_ratio = 1.0, 1.0
        if width_ratio >= height_ratio:
            width = long_edge
            height = max(8, round((long_edge * height_ratio / width_ratio) / 8) * 8)
        else:
            height = long_edge
            width = max(8, round((long_edge * width_ratio / height_ratio) / 8) * 8)
        return f"{width}x{height}"
    if not ar and not res:
        return ""
    if (ar, res) in SIZE_PRESETS:
        return SIZE_PRESETS[(ar, res)]
    # 任一填了另一个没填：按比例匹配默认档位
    if res and ar:
        return SIZE_PRESETS.get((ar, res), SIZE_PRESETS.get((ar, ""), "2048x2048"))
    if res:
        return SIZE_PRESETS.get(("", res), "1280x720")
    return SIZE_PRESETS.get((ar, ""), "2048x2048")


def _raw_base64_data_uri(uri: str) -> str:
    """将 data URI 转为裸 base64，兼容只接受裸编码的上游 provider。"""
    value = (uri or "").strip()
    if value.lower().startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _is_image_upstream_failure(exc: Exception) -> bool:
    """判断是否为可尝试另一种图像编码的上游失败。

    该函数只在 ``_run_image`` 的图像请求异常分支中调用，但仍要求错误信封
    明确指向图像参数或编码格式；单纯的 provider 故障不应重复发送请求。
    """
    msg = str(exc).lower().replace("\\_", "_")
    upstream = (
        "upstream_request_failed" in msg
        or "upstream provider request failed" in msg
        or "upstream_error" in msg
    )
    # 没有参数/编码线索时通常是 provider 过载或模型路由故障，
    # 改编码没有意义，也会额外增加一次请求。
    image_hint = (
        "param=image" in msg
        or "出错参数：image" in msg
        or "data uri" in msg
        or "data-uri" in msg
        or "base64" in msg
    )
    return upstream and image_hint


def _is_upstream_failure(exc: Exception) -> bool:
    """识别服务端报告的 provider/路由故障（与具体媒体参数无关）。"""
    # 某些日志/网关会把下划线转义成 ``\\_``，两种形式都应视为同一错误码。
    msg = str(exc).lower().replace("\\_", "_")
    return (
        "upstream_request_failed" in msg
        or "upstream provider request failed" in msg
        or "upstream_error" in msg
    )

# 内部默认
DEFAULT_REVERSE_TEMPERATURE = 0.7
DEFAULT_POLL_INTERVAL = 3.0
# "auto" = 按预算自适应（PNG -> JPEG -> 降边长）。
# 实测 2048px PNG 照片的 data URI 可达 3~4.3MB，是上游 400/413 的高发原因；
# 同一张图 q90 JPEG@1536 只要 0.2~0.7MB，多图场景差别更大（N 张线性叠加）。
DEFAULT_UPLOAD_FORMAT = "auto"
DEFAULT_UPLOAD_MAX_EDGE = 2048
DEFAULT_UPLOAD_BUDGET_MB = media.DEFAULT_IMAGE_BUDGET_MB
DEFAULT_VIDEO_MAX_MB = 48.0
DEFAULT_AUDIO_MAX_MB = 16.0

# 整个请求体（所有参考图之和）超过这个量级就明确提醒，因为网关普遍在 4~10MB 设限
UPLOAD_TOTAL_WARN_MB = 4.0

# 多图参考（image 数组）的探路时限。平台文档写明「数组支持取决于所选模型」；
# 实测（2026-09-10，同一 key/模型/两张真实参考图）该路径极不稳定：
# 同一请求可能 31s 返回 504 upstream timeout，也可能 55s / 122s 才成功；
# 而单图参考 7 次全部稳定成功（26~52s）。因此多图单独限时探测，失败即降级为第一张，
# 既不把整个 prompt 拖成几分钟，也保证用户始终拿到结果。
MULTI_IMAGE_PROBE_TIMEOUT = 120.0

# 元指令 / 提示词里出现这些词，反推结果就按 JSON 下发
REVERSE_JSON_HINTS = ("json", "json 对象", "结构化", "字段", "schema", "键名")

MAX_IMAGE_SLOTS = 16
MAX_VIDEO_SLOTS = 8
MAX_AUDIO_SLOTS = 8


def _sorted_slots(grow: dict | None) -> list:
    """Autogrow 传入 {image_0: IMAGE, image_2: IMAGE}，按下标顺序取出非空值。"""
    if not grow:
        return []

    def key(name: str):
        tail = name.rsplit("_", 1)[-1]
        return (0, int(tail)) if tail.isdigit() else (1, name)

    return [grow[k] for k in sorted(grow, key=key) if grow[k] is not None]


def assign_roles(count: int, mode: str) -> tuple[list[str], int]:
    """按模式给有序图片分配作用。

    返回 (roles, used)：roles 长度等于真正参与请求的图片数，
    used 之外的图片会被丢弃，由调用方据 count - used 发出警告。
    """
    if count <= 0:
        return [], 0
    if mode == "all_reference":
        return [caps.ROLE_REFERENCE_IMAGE] * count, count
    if mode == "first_only":
        return [caps.ROLE_FIRST_FRAME], 1
    if mode == "last_only":
        return [caps.ROLE_LAST_FRAME], 1
    if mode == "first_last":
        used = min(count, 2)
        return [caps.ROLE_FIRST_FRAME, caps.ROLE_LAST_FRAME][:used], used
    if mode == "first_then_reference":
        return [caps.ROLE_FIRST_FRAME] + [caps.ROLE_REFERENCE_IMAGE] * (count - 1), count
    # auto
    if count == 1:
        return [caps.ROLE_FIRST_FRAME], 1
    if count == 2:
        return [caps.ROLE_FIRST_FRAME, caps.ROLE_LAST_FRAME], 2
    return (
        [caps.ROLE_FIRST_FRAME, caps.ROLE_LAST_FRAME]
        + [caps.ROLE_REFERENCE_IMAGE] * (count - 2),
        count,
    )


def _chat_text(payload: dict) -> str:
    """从对话响应提取纯文本，兼容 content 为字符串或分段数组。"""
    chunks: list[str] = []
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or choice.get("delta") or {}
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                chunks.append(content)
        elif isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") in (None, "text", "output_text")
            ]
            joined = "".join(parts)
            if joined.strip():
                chunks.append(joined)
        if not chunks and isinstance(msg.get("reasoning_content"), str):
            chunks.append(msg["reasoning_content"])
    if chunks:
        return "\n\n---\n\n".join(chunks) if len(chunks) > 1 else chunks[0]
    # Responses API 形态兜底
    for item in payload.get("output") or []:
        if isinstance(item, dict):
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "\n".join(chunks)


class CreativeReasoningAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        # 每次 /object_info 请求都会重新执行此函数，因此模型列表始终取最新缓存：
        # 目录新增模型后刷新浏览器即可看到，无需重启 ComfyUI。
        text_names = catalog.text_model_names()
        gen_names = catalog.gen_model_names()
        cat_note = "、".join(
            f"{m['name']}（{m['category']}）" for m in catalog.get_models()[:8]
        ) or "（目录暂不可用，已用内置列表）"

        return io.Schema(
            node_id="CreativeReasoningAPI",
            display_name="Creative Reasoning API",
            category="Creative Reasoning",
            description=(
                "统一调用 Creative Reasoning 平台的文本 / 图像 / 视频 / 音频模型，采用双模型设计。\n"
                "① 多模态文本模型（必选）：负责处理「元指令」「反推提示词」，和不选生成模型时的纯文本输出（写小说、扩写等）。\n"
                "② 生成模型（可选）：图像 / 视频 / 音频模型；留空即为纯文本输出。\n"
                "模型下拉菜单直接来自线上目录，刷新浏览器即见新模型。\n"
                "图像 / 视频 / 音频输入槽随连接自动增长；图像可指定首帧、尾帧或多参考图，"
                "这些语义会真实写入请求的 role 字段。\n"
                "开启「反推提示词」开关后，会先用文本模型从素材反推出提示词，再用于正式生成；"
                "反什么、按什么格式输出，全写在元指令里（含 json 关键词即按 JSON 输出）。\n"
                f"当前可用：{cat_note}"
            ),
            inputs=[
                io.Combo.Input(
                    "text_model",
                    display_name="多模态文本模型",
                    options=text_names,
                    default=text_names[0],
                    tooltip=(
                        "必选。负责：元指令作为 system、反推提示词（始终走此模型）、以及不选生成模型时的文本生成。\n"
                        "建议选支持视觉的多模态文本模型，这样反推 / 纯文本也能吃图、视频、音频素材。"
                    ),
                ),
                io.Combo.Input(
                    "gen_model",
                    display_name="生成模型（图像/视频/音频）",
                    options=[GEN_NONE_LABEL] + gen_names,
                    default=GEN_NONE_LABEL,
                    tooltip=(
                        "可选。选图像/视频/音频模型时，用文本模型整理好的最终提示词去生成媒体；"
                        "留空（默认）= 纯文本输出，不调用任何生成模型。\n"
                        "目录里没有的上游模型名填在最下面的「手动模型 ID」，优先于此下拉。"
                    ),
                ),
                io.String.Input(
                    "meta_instruction",
                    display_name="元指令",
                    multiline=True,
                    default="",
                    optional=True,
                    placeholder="系统级指令：设定角色、输出格式、硬性约束……",
                    tooltip=(
                        "纯文本模式：作为对话的 system 消息。\n"
                        "生成模式：前置到生成提示词；若开启反推，则元指令就是反推指令"
                        "（反推什么、按什么格式输出，全写在这里）。\n"
                        "节点不内置任何提示词——反推完全由元指令驱动。"
                    ),
                ),
                io.String.Input(
                    "prompt",
                    display_name="提示词",
                    multiline=True,
                    default="",
                    optional=True,
                    placeholder="本次要生成的具体内容。开启反推后，这里作为「补充要求」参与反推。",
                    tooltip="本轮的具体请求内容。",
                ),
                io.Autogrow.Input(
                    "images",
                    display_name="图像输入",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image", tooltip="接上即自动新增下一个图像槽位。"),
                        prefix="image_",
                        min=0,
                        max=MAX_IMAGE_SLOTS,
                    ),
                    tooltip=f"动态图像输入，最多 {MAX_IMAGE_SLOTS} 个槽位；按顺序参与首尾帧/参考图分配。单个槽位若是图像批次会按帧全部上传。",
                ),
                io.Autogrow.Input(
                    "videos",
                    display_name="视频输入",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Video.Input("video", tooltip="接上即自动新增下一个视频槽位。"),
                        prefix="video_",
                        min=0,
                        max=MAX_VIDEO_SLOTS,
                    ),
                    tooltip=f"动态视频输入，最多 {MAX_VIDEO_SLOTS} 个槽位。视频接口中作为 reference_video 生效。",
                ),
                io.Autogrow.Input(
                    "audios",
                    display_name="音频输入",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio", tooltip="接上即自动新增下一个音频槽位。"),
                        prefix="audio_",
                        min=0,
                        max=MAX_AUDIO_SLOTS,
                    ),
                    tooltip=(
                        f"动态音频输入，最多 {MAX_AUDIO_SLOTS} 个槽位，编码为 WAV 后上传。"
                        "视频接口中作为 reference_audio 生效；音频生成接口中作为参考音频（语音克隆）。"
                    ),
                ),
                io.Combo.Input(
                    "image_role_mode",
                    display_name="图像作用",
                    options=ROLE_MODE_CHOICES,
                    default=ROLE_MODE_LABELS["all_reference"],
                    optional=True,
                    tooltip=(
                        "决定图像输入如何参与请求，选项里已写明每种模式的行为。\n"
                        "带 ⚠ 的模式会丢弃多余图片（数量会在 info 里提醒）。\n"
                        "role 语义（首帧/尾帧/参考图）只有视频接口会下发；"
                        "图像与对话接口按这里选中的图片顺序作为多参输入。"
                    ),
                ),
                io.String.Input(
                    "api_key",
                    display_name="API Key",
                    default="",
                    optional=True,
                    placeholder="留空则读环境变量 CR_API_KEY 或插件目录 config.json",
                    tooltip=(
                        "留空更安全：优先读环境变量 CR_API_KEY，再读插件目录 config.json / api_key.txt。"
                        "填在这里会被保存进工作流文件，分享前记得清空。"
                    ),
                ),
                # --- 反推：只一个开关，模型自动选、格式由元指令决定 ---
                io.Boolean.Input(
                    "reverse_prompt",
                    display_name="反推提示词",
                    default=False,
                    optional=True,
                    tooltip=(
                        "开启后先调一次「多模态文本模型」，让它根据图像与视频素材反推出提示词，再用反推结果去正式生成。\n"
                        "节点不内置任何提示词：**反推什么、按什么格式输出，完全由你的「元指令」决定**；"
                        "「提示词」作为补充要求一并传给反推模型。\n"
                        "元指令里带 json 关键词时，反推结果会按 JSON 结构返回。\n"
                        "反推固定走上面选的「多模态文本模型」。"
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFF,
                    control_after_generate=True, optional=True,
                    tooltip="0 表示不下发 seed（每次结果都不同）。",
                ),
                # --- 以下是生成模型相关参数，多数模型留空即可 ---
                io.Combo.Input(
                    "aspect_ratio", display_name="画面比例",
                    options=ASPECT_RATIOS, default="", optional=True,
                    tooltip="图像模型会与「分辨率档位」组合成 size 字符串（如 16:9+720p → 1280x720）；视频模型直接下发 aspect_ratio 字段。留空不下发。", advanced=True,
                ),
                io.Combo.Input(
                    "resolution", display_name="分辨率档位",
                    options=RESOLUTIONS, default="", optional=True,
                    tooltip="图像模型与「画面比例」组合成 size（1280x720 / 1920x1080 等）；视频模型直接下发 resolution 字段。留空不下发。", advanced=True,
                ),
                io.Int.Input(
                    "duration", display_name="时长（秒）", default=0, min=0, max=60, optional=True,
                    tooltip="仅视频模型；0 表示不下发，用模型默认（通常 5 或 10 秒）。", advanced=True,
                ),
                io.String.Input(
                    "negative_prompt", display_name="负向提示词", default="", optional=True,
                    tooltip="图像/视频/音频模型可用；留空不下发。", advanced=True,
                ),
                io.Int.Input(
                    "timeout", default=600, min=10, max=3600, optional=True,
                    tooltip="单次请求 / 任务轮询的总超时（秒）。视频/音频任务请留足。", advanced=True,
                ),
                io.Int.Input(
                    "max_retries", default=2, min=0, max=6, optional=True,
                    tooltip="限流与网络错误的重试次数（指数退避）。", advanced=True,
                ),
                io.String.Input(
                    "model_override", display_name="手动模型 ID", default="", optional=True,
                    placeholder="留空即用上面的生成模型下拉",
                    tooltip="目录里没有的上游模型名（如 doubao-seedream-5.0、seedance-2.0）填这里，优先于生成模型下拉。",
                    advanced=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.Image.Output(display_name="image"),
                io.Video.Output(display_name="video"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="info"),
                io.String.Output(display_name="prompt_used"),
            ],
        )

    # ------------------------------------------------------------------
    @classmethod
    async def execute(
        cls,
        text_model: str,
        gen_model: str = GEN_NONE_LABEL,
        meta_instruction: str = "",
        prompt: str = "",
        images: io.Autogrow.Type = None,
        videos: io.Autogrow.Type = None,
        audios: io.Autogrow.Type = None,
        image_role_mode: str = ROLE_MODE_LABELS["all_reference"],
        api_key: str = "",
        reverse_prompt: bool = False,
        seed: int = 0,
        aspect_ratio: str = "",
        resolution: str = "",
        duration: int = 0,
        negative_prompt: str = "",
        timeout: int = 600,
        max_retries: int = 2,
        model_override: str = "",
        temperature: float = 0.7,
    ) -> io.NodeOutput:
        text_model_id = (text_model or "").strip()
        if not text_model_id:
            raise CRError("未选择多模态文本模型（text_model）。")

        gen_sel = (gen_model or "").strip()
        if is_placeholder_model(gen_sel):
            # 下拉里的「（纯文本…）」= 不选生成模型
            gen_sel = ""

        override_sel = (model_override or "").strip()
        dropped_override = ""
        if override_sel and is_placeholder_model(override_sel):
            # 历史前端 bug 会把占位文案写进「手动模型 ID」；此处自动忽略并给出提示，
            # 否则它会被当成 model 发给服务端，得到难以理解的 400。
            dropped_override = override_sel
            override_sel = ""

        gen_model_id = override_sel or gen_sel

        key = config.resolve_api_key(api_key)

        # 接口类别：gen 为空 -> 纯文本；否则按 gen 模型决定
        gen_category = catalog.category_of(gen_model_id) if gen_model_id else None

        meta = (meta_instruction or "").strip()
        text_prompt = (prompt or "").strip()
        role_mode = _ROLE_MODE_BY_LABEL.get(image_role_mode, "auto")

        image_tensors = _sorted_slots(images)
        video_objects = _sorted_slots(videos)
        audio_objects = _sorted_slots(audios)

        image_uris: list[str] = []
        for tensor in image_tensors:
            image_uris.extend(
                media.image_tensor_to_data_uris(
                    tensor,
                    # Seedance 类视频 provider 对 PNG data URI 的兼容性较差，
                    # 视频参考图固定使用 JPEG；图像/文本路径继续使用原策略。
                    fmt="jpeg" if gen_category == "video" else DEFAULT_UPLOAD_FORMAT,
                    max_edge=DEFAULT_UPLOAD_MAX_EDGE,
                    max_mb=DEFAULT_UPLOAD_BUDGET_MB,
                )
            )
        video_uris = [
            media.video_to_data_uri(v, max_mb=DEFAULT_VIDEO_MAX_MB) for v in video_objects
        ]
        audio_uris = [
            media.audio_to_data_uri(a, max_mb=DEFAULT_AUDIO_MAX_MB) for a in audio_objects
        ]

        if not meta and not text_prompt and not image_uris and not video_uris and not audio_uris:
            raise CRError("元指令、提示词、图像、视频、音频至少要提供一项。")

        # 上行体积：网关普遍在 4~10MB 设限，超限会以 400/413 或上游失败的形式炸掉，
        # 且报错信息完全看不出原因，所以在这里主动量出来、写进日志和 info。
        image_upload_mb = sum(media.data_uri_mb(u) for u in image_uris)
        logger.info(
            "[CreativeReasoning] gen=%s | model=%s | text=%s | 图 %d | 视频 %d | 音频 %d | 图片上行 %.2fMB",
            gen_category or "text(纯文本)", gen_model_id or text_model_id, text_model_id,
            len(image_uris), len(video_uris), len(audio_uris), image_upload_mb,
        )

        common = {
            "api_key": key,
            "timeout": float(timeout),
            "max_retries": int(max_retries),
        }
        info: dict = {
            "text_model": text_model_id,
            "gen_model": gen_model_id or "",
            "mode": "pure_text" if not gen_model_id else f"generation:{gen_category}",
            "gen_category": gen_category or "text",
            "images_sent": len(image_uris),
            "videos_sent": len(video_uris),
            "audios_sent": len(audio_uris),
            "image_upload_mb": round(image_upload_mb, 2),
            "api_key_source": "节点控件" if (api_key or "").strip() else config.key_source_hint(),
        }
        if image_upload_mb > UPLOAD_TOTAL_WARN_MB:
            info.setdefault("warnings", []).append(
                f"本次参考图共 {image_upload_mb:.1f}MB（base64 后），已接近网关常见的 4~10MB 上限。"
                "若报「上游请求失败」或 413，请减少参考图数量或降低分辨率。"
            )
        if dropped_override:
            logger.warning(
                "[CreativeReasoning] 「手动模型 ID」是界面占位文案，已自动忽略：%s", dropped_override
            )
            info.setdefault("warnings", []).append(
                f"「手动模型 ID」里是界面占位文案（{dropped_override}），已自动忽略。"
                "这是旧版前端脚本的历史 bug：请刷新浏览器页面，"
                "或右键节点 →「Creative Reasoning：修复旧版控件错位」以彻底清除。"
            )

        # --- 反推提示词：元指令既指导反推，也继续约束最终文本/图像/视频生成 ---
        user_text = text_prompt
        gen_meta = meta
        if reverse_prompt:
            if not (image_uris or video_uris or text_prompt or meta):
                raise CRError("开启反推后，至少需要元指令、提示词、图像、视频中的一项作为反推依据。")
            rev_model, derived = await cls._reverse_prompt(
                model_id=text_model_id,
                category="text",
                image_uris=image_uris,
                video_uris=video_uris,
                hint=text_prompt,
                instruction=meta,
                **common,
            )
            info["reverse_model"] = rev_model
            info["reverse_prompt_result"] = derived
            if not meta:
                info.setdefault("warnings", []).append(
                    "反推已开启但元指令为空：节点不内置任何反推提示词，"
                    "建议在元指令里写明反推什么、按什么格式输出。"
                )
            user_text = derived

        # 与已验证可用的融合节点保持完全一致：元指令和用户提示词原样合并，
        # 由生成模型自行理解；不额外插入文本扩写请求或约束文本。
        full_prompt = "\n\n".join(p for p in (gen_meta, user_text) if p)
        if gen_category == "video":
            # Seedance 2.0 通过提示词中的素材标记绑定上传资源；仅上传 content
            # 而不引用 @图片N/@视频N/@音频N 会被部分 provider 判为无效请求。
            refs = []
            refs.extend(f"@图片{i}" for i in range(1, len(image_uris) + 1))
            refs.extend(f"@视频{i}" for i in range(1, len(video_uris) + 1))
            refs.extend(f"@音频{i}" for i in range(1, len(audio_uris) + 1))
            if refs and not any(token in full_prompt for token in refs):
                full_prompt = (full_prompt + "\n\n参考素材：" + "、".join(refs)).strip()

        # --- 接入能力检查：把"接了但不生效"的情况明确告知 ---
        roles, used = assign_roles(len(image_uris), role_mode)
        dropped = max(0, len(image_uris) - used)
        image_uris = image_uris[:used]
        if gen_category == "video":
            info["image_roles"] = roles
        else:
            roles = []
        warnings = caps.build_warnings(
            model_id=gen_model_id or text_model_id,
            category=gen_category or "text",
            n_images=len(image_uris),
            n_videos=len(video_uris),
            n_audios=len(audio_uris),
            roles=roles,
            dropped_images=dropped,
        )
        if warnings:
            info["warnings"] = warnings
            for w in warnings:
                logger.warning("[CreativeReasoning] %s", w)

        # --- 分发：纯文本 / 图像 / 视频 / 音频 ---
        if not gen_model_id or gen_category == "text":
            # 纯文本：用文本模型直接对话（含多模态素材）
            run_model = text_model_id if not gen_model_id else gen_model_id
            result = await cls._run_text(
                run_model, gen_meta, user_text, image_uris, video_uris, audio_uris, info,
                seed=seed, **common,
                temperature=float(temperature),
            )
        elif gen_category == "image":
            result = await cls._run_image(
                gen_model_id, full_prompt, image_uris, info,
                aspect_ratio=aspect_ratio, resolution=resolution,
                seed=seed, negative_prompt=negative_prompt, **common,
            )
        elif gen_category == "video":
            result = await cls._run_video(
                gen_model_id, full_prompt, image_uris, video_uris, audio_uris, info,
                aspect_ratio=aspect_ratio, resolution=resolution,
                duration=duration, seed=seed, negative_prompt=negative_prompt,
                roles=roles, **common,
            )
        elif gen_category == "audio":
            result = await cls._run_audio(
                gen_model_id, full_prompt, audio_uris, info,
                seed=seed, negative_prompt=negative_prompt, **common,
            )
        else:
            result = await cls._run_text(
                text_model_id, gen_meta, user_text, image_uris, video_uris, audio_uris, info,
                seed=seed, **common,
                temperature=float(temperature),
            )

        # 统一补上第 6 个输出：本次真正用于生成的提示词
        text, image_out, video_out, audio_out, info_str = result
        # 未真正产出的输出用占位对象兜底，避免下游图像/视频/音频节点收到 None 崩溃
        if image_out is None:
            image_out = cls._placeholder_image()
            info.setdefault("warnings", []).append(
                "本次未生成图像（文本 / 视频 / 音频模式下 image 输出为占位，下游图像节点安全跳过）。"
            )
        if video_out is None:
            video_out = cls._placeholder_video()
            info.setdefault("warnings", []).append(
                "本次未生成视频（图像 / 文本 / 音频模式下 video 输出为占位，下游视频节点安全跳过）。"
            )
        if audio_out is None:
            audio_out = cls._placeholder_audio()
            info.setdefault("warnings", []).append(
                "本次未生成音频（文本 / 图像 / 视频模式下 audio 输出为占位，下游音频节点安全跳过）。"
            )
        return io.NodeOutput(text, image_out, video_out, audio_out, info_str, full_prompt)

    # ------------------------------------------------------------------
    @classmethod
    async def _reverse_prompt(
        cls, *, model_id, category, image_uris, video_uris, hint,
        instruction, api_key, timeout, max_retries,
    ) -> tuple[str, str]:
        """反推第一段：拿素材换一段提示词。固定走多模态文本模型（model_id 已是文本模型）。"""
        target = model_id if category == "text" else cls._default_text_model()
        if not target:
            raise CRError("目录里没有可用的文本模型来做反推，请在「多模态文本模型」里选一个支持视觉的模型。")

        parts: list[dict] = []
        if hint:
            parts.append({"type": "text", "text": f"补充要求：{hint}"})
        for uri in image_uris:
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        for uri in video_uris:
            parts.append({"type": "video_url", "video_url": {"url": uri}})

        if not parts:
            raise CRError("反推需要至少一张图片或一段视频。")

        messages = []
        system = (instruction or "").strip()
        if system:
            messages.append({"role": "system", "content": system})
        if len(parts) == 1 and parts[0]["type"] == "text":
            messages.append({"role": "user", "content": parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": parts})

        body = {
            "model": target,
            "messages": messages,
            "stream": False,
            "temperature": DEFAULT_REVERSE_TEMPERATURE,
        }
        blob = f"{system} {hint}".lower()
        if any(h in blob for h in REVERSE_JSON_HINTS):
            body["response_format"] = {"type": "json_object"}

        logger.info("[CreativeReasoning] 反推提示词：模型 %s，素材 %d", target, len(parts))
        try:
            payload = await request_json(
                "POST", "chat/completions", api_key, body=body,
                timeout=timeout, max_retries=max_retries,
            )
        except CRError:
            raise
        derived = _chat_text(payload).strip()
        if not derived:
            raise CRError("反推未返回文本内容，请检查反推模型是否支持图像输入。")
        return target, derived

    @staticmethod
    def _default_text_model() -> str:
        for m in catalog.get_models():
            if m.get("category") == "text":
                return str(m["name"])
        return ""

    # ------------------------------------------------------------------
    @classmethod
    async def _run_text(
        cls, model_id, meta, prompt, image_uris, video_uris, audio_uris, info,
        *, seed, api_key, timeout, max_retries,
        temperature=0.7,
    ):
        if is_placeholder_model(model_id):
            fallback = cls._default_text_model()
            if not fallback:
                raise CRError(f"模型名无效（界面占位文案）：{model_id}")
            info.setdefault("warnings", []).append(
                f"传入的模型名是界面占位文案（{model_id}），已自动改用默认文本模型 {fallback}。"
            )
            logger.warning("[CreativeReasoning] 占位文案当模型名，已回落：%s → %s", model_id, fallback)
            model_id = fallback

        messages = []
        if meta:
            messages.append({"role": "system", "content": meta})

        parts = []
        if prompt:
            parts.append({"type": "text", "text": prompt})
        for uri in image_uris:
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        for uri in video_uris:
            parts.append({"type": "video_url", "video_url": {"url": uri}})
        for uri in audio_uris:
            parts.append({"type": "input_audio", "input_audio": {"url": uri, "format": "wav"}})

        if not parts:
            parts.append({"type": "text", "text": meta or ""})

        # 纯文本时用字符串 content，兼容面最广
        if len(parts) == 1 and parts[0]["type"] == "text":
            messages.append({"role": "user", "content": parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": parts})

        body = {"model": model_id, "messages": messages, "stream": False}
        if float(temperature) >= 0:
            body["temperature"] = max(0.0, min(2.0, float(temperature)))
        if int(seed) > 0:
            body["seed"] = int(seed)

        try:
            payload = await request_json(
                "POST", "chat/completions", api_key, body=body,
                timeout=timeout, max_retries=max_retries,
            )
        except CRError as exc:
            if not _is_upstream_failure(exc):
                raise
            fallback_text = "\n\n".join(p for p in (meta, prompt) if p).strip()
            fallback_messages = []
            if len(parts) > 1 or (parts and parts[0].get("type") != "text"):
                fallback_parts = list(parts)
                if fallback_text:
                    fallback_parts.insert(0, {"type": "text", "text": fallback_text})
                fallback_messages.append({"role": "user", "content": fallback_parts})
            else:
                fallback_messages.append({"role": "user", "content": fallback_text})
            fallback_body = {"model": model_id, "messages": fallback_messages, "stream": False}
            info.setdefault("warnings", []).append(
                "文本模型首次请求被上游拒绝，已用兼容格式重试（移除扩展参数与 system 角色）。"
            )
            logger.warning("[CreativeReasoning] 文本请求兼容性降级重试：%s", str(exc).splitlines()[0])
            payload = await request_json(
                "POST", "chat/completions", api_key, body=fallback_body,
                timeout=timeout, max_retries=0,
            )

        text = _chat_text(payload)
        info["usage"] = payload.get("usage") or {}
        info["finish_reason"] = (
            (payload.get("choices") or [{}])[0].get("finish_reason") if payload.get("choices") else None
        )
        if not text.strip():
            info.setdefault("warnings", []).append(
                "接口未返回文本内容，请检查模型是否支持该输入组合。"
            )

        # 少数对话模型会在消息里回图，能解就解
        image_out = await media.images_from_payload(payload, timeout=timeout, max_retries=max_retries)
        if image_out is not None:
            info["saved_images"] = [Path(p).name for p in media.save_images(image_out)]

        return (text, image_out, None, None, cls._info(info))

    # ------------------------------------------------------------------
    @classmethod
    async def _post_image_multi(
        cls, path, body, image_uris, info, *, api_key, timeout, max_retries,
    ):
        """多图参考：先按用户意图把 N 张图作为 image 数组下发；上游失败即降级为第 1 张。

        为什么不把数组当稳定路径：
        - 平台文档明确「数组支持取决于所选模型」；
        - 实测同一 key/模型/两张真实参考图，数组请求会在 31s(504) / 55s(成功) / 122s(成功)
          之间随机摇摆，而单图路径 7 次全部稳定成功。

        降级保证用户始终拿到结果，同时把「降级了、为什么降级」写进 info 与日志，
        绝不静默改变语义。
        """
        try:
            return await request_json(
                "POST", path, api_key, body=body,
                timeout=min(float(timeout), MULTI_IMAGE_PROBE_TIMEOUT), max_retries=0,
            )
        except CRError as exc:
            head = str(exc).splitlines()[0]
            if "401" in head or "403" in head:
                raise  # 鉴权类问题换成单图也一样，没必要多跑一次
            logger.warning(
                "[CreativeReasoning] 多图数组（%d 张）在上游失败，自动降级为第 1 张重跑：%s",
                len(image_uris), head,
            )
            info["multi_image_degraded"] = True
            info["images_sent"] = 1
            info.setdefault("warnings", []).append(
                f"多图参考（{len(image_uris)} 张）在上游失败，已自动降级为第 1 张参考图重跑；"
                f"原因：{head}。该模型对 image 数组的支持不稳定（平台文档：数组支持取决于所选模型）。"
                "若必须多图同时生效，请换支持数组的模型，或先把多张图在本地合成一张。"
            )
            body["image"] = image_uris[0]
            return await request_json(
                "POST", path, api_key, body=body, timeout=timeout, max_retries=max_retries
            )

    # ------------------------------------------------------------------
    @classmethod
    async def _run_image(
        cls, model_id, full_prompt, image_uris, info,
        *, aspect_ratio, resolution, seed, negative_prompt,
        api_key, timeout, max_retries,
    ):
        if not full_prompt:
            raise CRError("图像模型需要提示词（元指令或提示词至少填一个）。")

        body: dict = {"model": model_id, "prompt": full_prompt}
        size = _image_size(aspect_ratio, resolution)
        if size:
            body["size"] = size
        if negative_prompt.strip():
            body["negative_prompt"] = negative_prompt.strip()
        if int(seed) > 0:
            body["seed"] = int(seed)

        # 有参考图 -> 走编辑接口（JSON 传图，不是 multipart）；多图以数组下发 = 多参
        multi = len(image_uris) > 1
        if image_uris:
            body["image"] = image_uris if multi else image_uris[0]
            path = "images/edits"
        else:
            path = "images/generations"

        info["endpoint"] = path
        info["images_sent"] = len(image_uris)
        if size:
            info["size"] = size

        if multi:
            info["multi_image"] = True
            payload = await cls._post_image_multi(
                path, body, image_uris, info,
                api_key=api_key, timeout=timeout, max_retries=max_retries,
            )
        else:
            try:
                payload = await request_json(
                    "POST", path, api_key, body=body, timeout=timeout, max_retries=max_retries
                )
            except CRError as exc:
                # 部分 provider 接受文档声明的 data URI，但其适配器只接受裸 base64。
                # 仅在明确的 image 上游失败时回退，invalid_body 等真实参数错误仍原样抛出。
                if not image_uris or not _is_image_upstream_failure(exc):
                    raise
                fallback_body = dict(body)
                fallback_body["image"] = _raw_base64_data_uri(image_uris[0])
                logger.warning(
                    "[CreativeReasoning] 图像 data URI 被上游拒绝，改用裸 base64 重试：%s",
                    str(exc).splitlines()[0],
                )
                info["image_encoding_fallback"] = "raw_base64"
                info.setdefault("warnings", []).append(
                    "上游拒绝了 data URI 图像编码，已自动改用裸 base64 重试。"
                )
                payload = await request_json(
                    "POST", path, api_key, body=fallback_body,
                    # 原始编码已经按 max_retries 尝试过；回退只再发一次，
                    # 避免同一任务因两种编码各自重试而等待过久或重复计费。
                    timeout=timeout, max_retries=0,
                )

        if not is_done(payload) and payload.get("id"):
            info["task_id"] = payload["id"]
            # 文档：图像任务（含 images/edits 提交的）统一挂在 images/generations 下轮询。
            # poll_task 内 404 时仍会回退到 images/edits，兼容后端把任务挂在别处的情况。
            payload = await poll_task(
                "images/generations", payload["id"], api_key,
                poll_interval=DEFAULT_POLL_INTERVAL, timeout=timeout, max_retries=max_retries,
                poll_url=payload.get("poll_url"),
                on_progress=lambda st, w: logger.info("[CreativeReasoning] 图像任务 %s，已等待 %.0fs", st, w),
            )

        image_out = await media.images_from_payload(payload, timeout=timeout, max_retries=max_retries)
        if image_out is None:
            raise CRError("接口未返回可解析的图像结果。原始响应见 info 输出。" + cls._brief(payload))

        info["usage"] = payload.get("usage") or {}
        info["image_count"] = int(image_out.shape[0])
        info["image_size"] = f"{int(image_out.shape[2])}x{int(image_out.shape[1])}"
        info["saved_images"] = [Path(p).name for p in media.save_images(image_out)]

        text = _chat_text(payload) or ""
        return (text, image_out, None, None, cls._info(info))

    # ------------------------------------------------------------------
    @classmethod
    async def _run_video(
        cls, model_id, full_prompt, image_uris, video_uris, audio_uris, info,
        *, aspect_ratio, resolution, duration, seed, negative_prompt,
        roles, api_key, timeout, max_retries,
    ):
        content: list[dict] = []
        if full_prompt:
            content.append({"type": "text", "text": full_prompt})

        # 首尾帧 / 多参考图：role 是视频接口的原生字段，这里真正写入
        for uri, role in zip(image_uris, roles):
            content.append({"type": "image_url", "image_url": {"url": uri}, "role": role})
        for uri in video_uris:
            content.append(
                {"type": "video_url", "video_url": {"url": uri}, "role": caps.ROLE_REFERENCE_VIDEO}
            )
        for uri in audio_uris:
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"url": uri, "format": "wav"},
                    "role": caps.ROLE_REFERENCE_AUDIO,
                }
            )

        if not content:
            raise CRError("视频模型需要提示词或参考素材。")

        body: dict = {"model": model_id, "content": content}
        if image_uris and roles and all(role == caps.ROLE_REFERENCE_IMAGE for role in roles):
            info.setdefault("warnings", []).append(
                "当前选择为多参模式，已将图片全部标记为 reference_image；但 i2v 类模型在仅接入一张图片时可能仍按首帧处理。"
            )
        if aspect_ratio.strip():
            body["aspect_ratio"] = aspect_ratio.strip()
        if resolution.strip():
            body["resolution"] = resolution.strip()
        if int(duration) > 0:
            body["duration"] = int(duration)
        if int(seed) > 0:
            body["seed"] = int(seed)
        if negative_prompt.strip():
            body["negative_prompt"] = negative_prompt.strip()

        try:
            payload = await request_json(
                "POST", "videos/generations", api_key, body=body,
                timeout=timeout, max_retries=max_retries,
            )
        except CRError as exc:
            # 上游适配器有时拒绝扩展字段或 role。用最小兼容体重试一次，
            # 保留模型、提示词和所有媒体，避免把失败误判成模型不可用。
            if _is_upstream_failure(exc):
                minimal_content = []
                for item in content:
                    if item.get("type") == "text":
                        minimal_content.append(item)
                    elif item.get("type") == "image_url":
                        minimal_content.append({"type": "image_url", "image_url": item["image_url"]})
                    else:
                        minimal_content.append(item)
                minimal_body = {"model": model_id, "content": minimal_content}
                info.setdefault("warnings", []).append("视频请求首次被上游拒绝，已切换兼容格式重试。")
                logger.warning("[CreativeReasoning] 视频请求兼容格式重试：%s", str(exc).splitlines()[0])
                try:
                    payload = await request_json(
                        "POST", "videos/generations", api_key, body=minimal_body,
                        timeout=timeout, max_retries=0,
                    )
                    body = minimal_body
                except CRError:
                    payload = None
                # 不再尝试顶层 image/prompt 旧格式：该统一端点明确要求 content，
                # 服务端会返回 missing field `content`，而且会把大段 base64
                # 拼进错误位置，造成误导性日志与额外一次请求。
            else:
                payload = None
            if payload is not None:
                pass
            elif len(image_uris) <= 1 or not _is_upstream_failure(exc):
                raise
            # 视频模型经常只支持单个首帧/参考图。平台文档虽允许 content 数组，
            # 但不支持的 role 在部分上游适配器中会直接变成 400 upstream_error。
            # 多图提交失败时保留第一张重试；单图、鉴权错误和真正的参数错误原样抛出。
            if len(image_uris) <= 1 or not _is_upstream_failure(exc):
                raise
            first_image = next(
                (item for item in content if item.get("type") == "image_url"), None
            )
            if first_image is None:
                raise
            # 删除多余图片但保留第一张在原 content 中的位置，避免视频/音频素材
            # 因重排而触发上游对首帧或参考素材顺序的校验。
            fallback_content = []
            kept_image = False
            for item in content:
                if item.get("type") != "image_url":
                    fallback_content.append(item)
                elif not kept_image:
                    fallback_content.append(item)
                    kept_image = True
            fallback_body = dict(body)
            fallback_body["content"] = fallback_content
            info["video_image_degraded"] = True
            info["images_sent"] = 1
            info.setdefault("warnings", []).append(
                f"视频模型无法处理 {len(image_uris)} 张参考图，已自动改用第 1 张图重试。"
                "如需首尾帧或多参考图，请确认所选视频模型支持对应 role。"
            )
            logger.warning(
                "[CreativeReasoning] 视频多图提交失败，自动降级为第 1 张重试：%s",
                str(exc).splitlines()[0],
            )
            payload = await request_json(
                "POST", "videos/generations", api_key, body=fallback_body,
                timeout=timeout, max_retries=0,
            )
        info["endpoint"] = "videos/generations"

        # 提交请求可能在任务已创建后超时，绝不自动重复提交，只轮询
        if not is_done(payload):
            task_id = payload.get("id")
            info["task_id"] = task_id
            payload = await poll_task(
                "videos/generations", task_id, api_key,
                poll_interval=DEFAULT_POLL_INTERVAL, timeout=timeout, max_retries=max_retries,
                poll_url=payload.get("poll_url"),
                on_progress=lambda st, w: logger.info("[CreativeReasoning] 视频任务 %s，已等待 %.0fs", st, w) if (int(w) % 15 < 4 or w <= 6) else None,
            )

        video_out, path = await media.video_from_payload(
            payload, timeout=timeout, max_retries=max_retries
        )
        if video_out is None:
            raise CRError("接口未返回可用的视频结果。原始响应见 info 输出。" + cls._brief(payload))

        info["usage"] = payload.get("usage") or {}
        info["saved_video"] = Path(path).name if path else ""
        text = _chat_text(payload) or ""
        return (text, None, video_out, None, cls._info(info))

    # ------------------------------------------------------------------
    @classmethod
    async def _run_audio(
        cls, model_id, full_prompt, audio_uris, info,
        *, seed, negative_prompt, api_key, timeout, max_retries,
    ):
        """音频生成：音乐生成或语音克隆。提示词 + 元指令驱动，参考音频用于克隆。

        接口形态与图像/视频一致：POST audio/generations，异步任务轮询。
        body 用 image 数组同款方式把参考音频作为数组下发（语音克隆时）。

        注意：截至当前，平台公开的统一接口列表里**没有音频端点**
        （只有 chat/completions、responses、embeddings、rerank、
        images/generations、images/edits、videos/generations）。
        所以这条路径只有在「手动模型 ID」强行指定音频模型时才会走到，
        届时多半返回 404 —— 这里把它翻译成可执行的中文提示，而不是让人对着 404 发懵。
        """
        if not full_prompt and not audio_uris:
            raise CRError("音频模型需要提示词（元指令或提示词至少填一个），语音克隆还需提供参考音频。")

        body: dict = {"model": model_id, "prompt": full_prompt}
        if audio_uris:
            # 参考音频（语音克隆 / 音乐参考），数组下发
            body["audio"] = audio_uris if len(audio_uris) > 1 else audio_uris[0]
        if negative_prompt.strip():
            body["negative_prompt"] = negative_prompt.strip()
        if int(seed) > 0:
            body["seed"] = int(seed)

        try:
            payload = await request_json(
                "POST", "audio/generations", api_key, body=body,
                timeout=timeout, max_retries=max_retries,
            )
        except CRError as exc:
            if "HTTP 404" in str(exc) or "HTTP 405" in str(exc):
                raise CRError(
                    f"平台当前没有对外开放音频生成端点，无法调用模型 {model_id}。"
                    "「音频」类模型暂时只能用「手动模型 ID」走上游自有接口；"
                    "本节点的 audio 输出在有音频结果时会正常透传。\n原始错误：" + str(exc)
                ) from exc
            raise
        info["endpoint"] = "audio/generations"

        if not is_done(payload) and payload.get("id"):
            info["task_id"] = payload["id"]
            payload = await poll_task(
                "audio/generations", payload["id"], api_key,
                poll_interval=DEFAULT_POLL_INTERVAL, timeout=timeout, max_retries=max_retries,
                poll_url=payload.get("poll_url"),
                on_progress=lambda st, w: logger.info("[CreativeReasoning] 音频任务 %s，已等待 %.0fs", st, w),
            )

        audio_out = await media.audio_from_payload(payload, timeout=timeout, max_retries=max_retries)
        if audio_out is None:
            raise CRError("接口未返回可用的音频结果。原始响应见 info 输出。" + cls._brief(payload))

        info["usage"] = payload.get("usage") or {}
        info["sample_rate"] = int(audio_out["sample_rate"])
        info["saved_audio"] = [Path(p).name for p in media.save_audio(audio_out)]

        text = _chat_text(payload) or ""
        return (text, None, None, audio_out, cls._info(info))

    # ------------------------------------------------------------------
    @staticmethod
    def _brief(payload: dict) -> str:
        try:
            raw = json.dumps(payload, ensure_ascii=False)[:400]
        except (TypeError, ValueError):
            raw = str(payload)[:400]
        return config.scrub(f"\n响应摘要：{raw}")

    @staticmethod
    def _info(info: dict) -> str:
        try:
            return config.scrub(json.dumps(info, ensure_ascii=False, indent=1))
        except (TypeError, ValueError):
            return config.scrub(str(info))

    # ------------------------------------------------------------------
    # 下游 SaveVideo / PreviewVideo / SaveImage / SaveAudio 等节点收到 None 会直接崩
    #（video.get_dimensions() / images[0].shape / 音频 waveform 访问），导致整个 prompt
    # 失败、连本该生成的图像也出不来。所以没真正产出时返回合法占位对象，让图能正常出。
    @staticmethod
    def _placeholder_video():
        """1 帧 16x16 黑视频占位，复用与内置 CreateVideo 节点完全相同的构造路径。"""
        frames = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
        return InputImpl.VideoFromComponents(
            Types.VideoComponents(images=frames, audio=None, frame_rate=Fraction(1))
        )

    @staticmethod
    def _placeholder_image():
        """1x1 黑图占位。"""
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    @staticmethod
    def _placeholder_audio():
        """1 样本静音单声道占位 AUDIO 字典，避免下游音频节点收到 None 崩溃。"""
        return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}

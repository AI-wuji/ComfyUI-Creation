"""独立 API、文本、图像和视频节点。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
from functools import wraps
from urllib.parse import urlsplit

from comfy_api.latest import io

from . import media, nodes
from .api import client

log = logging.getLogger(__name__)

WUJI_LLM = io.Custom("WUJI_LLM")
V2_PRESETS = {
    "创意推理": "https://api.creative-reasoning.com/unified/v1",
    "自定义（OpenAI兼容）": "",
    "OpenAI": "https://api.openai.com/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "通义千问（兼容接口）": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "智谱（兼容接口）": "https://open.bigmodel.cn/api/paas/v4",
    "Gemini（兼容接口）": "https://generativelanguage.googleapis.com/v1beta/openai",
}

PLAN_INSTRUCTION = """
请先完成用户要求，再严格只返回一个 JSON 对象，不要使用 Markdown 代码围栏或添加 JSON 以外的说明。
JSON 格式：{"image_prompts":["完整、可独立用于生成一张图的提示词"],"text_output":"图像之外的独立文本；没有则为空字符串"}。
image_prompts 中每一项必须是完整且独立的生图提示词，不得写“同上”。
多行列表关闭时，image_prompts 必须恰好包含一项，所有画面要求合并成一张图的提示词；开启时，按元指令为每张图分别生成一项。
诗歌、标题、说明等非画面文字放入 text_output；不要写进图像提示词，除非用户明确要求它们出现在画面中。
""".strip()
VIDEO_PLAN_INSTRUCTION = """
你负责把用户的元指令、提示词和参考素材扩写为可直接提交给视频生成模型的完整提示词。
先执行元指令中要求的分析、创作与扩写，再只输出最终视频提示词，不要照抄处理要求，不要输出分析过程、标题或代码围栏。
保持用户指定的人物、物体、风格、动作、数量、时长和其他硬性要求，在未限定处补足场景、镜头运动与动作节奏。
保留素材的 @图片N、@视频N、@音频N 引用和用途，编号不能改变；不要虚构未提供的素材。
如果素材或模型能力不足以确定细节，不要编造事实。
""".strip()
_VARIATION_ANGLES = (
    "从一个意外但合理的细节切入",
    "采用克制、留白、含蓄的表达",
    "突出动作、变化和明确的因果推进",
    "采用鲜明反差，避免常见套路",
    "强化具体感官细节和可感知质地",
    "改变叙述视角，呈现不常见的观察位置",
    "以层层递进的方式组织内容",
    "采用简洁有力、节奏明快的表达",
    "突出温暖、亲近且自然的情绪",
    "营造冷静、疏离但准确的观察感",
    "使用更具象征性的意象和联想",
    "从结果回看原因，采用回环式结构",
    "优先选择非对称构图和不居中的视觉重心",
    "采用低机位、近距离和强透视的视觉表达",
    "采用远景、环境叙事和大面积空间关系",
    "使用不同于常见范式的色彩与光线组合",
)
_VARIATION_STRUCTURES = (
    "以一个有辨识度的开场建立语境，再自然展开",
    "先呈现核心，再补足细节与背景",
    "通过两种相反元素的交替建立节奏",
    "避免中心化处理，使用偏置和不均衡的节奏",
    "采用清晰的起承转合，但不使用套话",
    "用一个具体场景承载主要信息",
    "采用碎片化细节组合成整体印象",
    "让情绪逐步变化，而非从头到尾保持同一强度",
    "从环境或周边元素间接突出主体",
    "用简洁构成保留一个突出的记忆点",
    "通过前后景关系增强纵深和层次",
)


_VISUAL_DIRECTIONS = (
    ("近景细节为主", "远景环境为主", "高机位俯视", "低机位仰视", "平视侧面观察", "前景遮挡形成纵深", "偏心构图保留空间", "几何线条引导视线"),
    ("柔和散射光", "侧逆光突出轮廓", "局部光源与暗部对照", "明亮通透的光线", "冷暖光线交织", "斜向光影分割空间", "反射光塑造层次", "低对比度细腻光影"),
    ("低饱和统一配色", "主体与背景形成色彩对照", "冷色主调配少量暖色", "暖色主调配少量冷色", "相近色渐变", "局部高饱和色作为视觉重点", "明暗色块形成节奏", "自然色彩与材质细节相呼应"),
    ("通过主体动作突出重点", "通过环境细节叙事", "通过空间尺度形成张力", "用前后景呼应主体", "以材质对照增强辨识度", "通过轮廓与留白突出主体", "强调细微表情或姿态", "用画面边缘细节建立故事感"),
)
_VIDEO_DIRECTIONS = (
    ("缓慢推进揭示细节", "缓慢拉远展示环境", "平稳横移跟随主体", "小幅环绕呈现空间", "固定镜头让主体运动", "由低向高轻移展示层次", "从前景移向主体", "沿主体运动方向跟拍"),
    ("动作从轻微逐步增强", "以连贯细小动作形成自然节奏", "短暂静止后发生明确动作", "主体与环境形成错落运动", "动作先快后缓自然收束", "由环境变化引出主体动作", "以一个完整动作贯穿片段", "通过前后景运动差形成纵深"),
    _VISUAL_DIRECTIONS[1],
    ("开场先展现环境再突出主体", "从主体细节展开全貌", "以空间变化推进画面", "围绕一个视觉重点持续展开", "通过前后景交换视觉重点", "从动作起势到完成呈现完整过程", "通过光影变化组织时间感", "用前后呼应的画面自然结束"),
)


def _derived_seed(seed, purpose):
    digest = hashlib.sha256(f"creation-v2:{int(seed)}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2147483647 + 1


def _variation(seed, *, kind="text"):
    # Mix every seed bit: nearby seeds must not walk through the same short cycle.
    rng = random.Random(_derived_seed(seed, kind))
    if kind == "text":
        directions = [*rng.sample(_VARIATION_ANGLES[:12], 2),
                      *rng.sample(_VARIATION_STRUCTURES, 2)]
    else:
        pools = _VIDEO_DIRECTIONS if kind == "video" else _VISUAL_DIRECTIONS
        directions = [rng.choice(pool) for pool in pools]
    return ("本轮创作变化方向（仅在用户未限定的维度采用）：" + "；".join(directions) + "。"
            "在切入点、布局或动作上作实质变化，不只替换几个形容词。"
            "用户指定的主体、事实、数量、格式、风格、参考素材与其他明确要求始终优先；"
            "与要求冲突的方向直接忽略。不要在结果中解释这些内部方向。")


def _selection(value):
    value = str(value or "").strip()
    return "" if value in ("未选择", "未启用", "None") else value


def _config(value):
    if isinstance(value, dict) and value.get("wuji_api_version") != 2:
        raise ValueError("当前动力源不是创造API，请将创造API的动力源输出连接到此节点。")
    if not isinstance(value, dict) or value.get("wuji_api_version") != 2:
        raise ValueError("请连接创造API输出的动力源。")
    return value


def _text_messages(meta, prompt, images=None, videos=None, audios=None):
    content = [{"type": "text", "text": str(prompt or "").strip()}]
    content.extend({"type": "image_url", "image_url": {"url": uri}} for uri in images or [])
    content.extend({"type": "video_url", "video_url": {"url": uri}} for uri in videos or [])
    content.extend({"type": "input_audio", "input_audio": {"url": uri, "format": "wav"}} for uri in audios or [])
    if not str(prompt or "").strip() and len(content) == 1:
        content[0]["text"] = "请根据元指令完成任务。"
    return [
        {"role": "system", "content": str(meta or "").strip()},
        {"role": "user", "content": content[0]["text"] if len(content) == 1 else content},
    ]


def _original_prompt(meta, prompt, kind):
    for value in (prompt, meta):
        clean = client.generation_text(str(value or ''))
        if clean:
            return clean
    return '根据提供的参考素材生成' + ('图像。' if kind == 'image' else '连贯的视频。')


async def _expand_media_prompt(config, messages, fallback):
    try:
        text = await asyncio.to_thread(client.chat, {**config, '_creation_text_fallback': fallback},
            messages, temperature=config.get('temperature', 1.1), max_tokens=config.get('max_tokens', 4096))
    except client.WujiApiError:
        log.warning('创造API：提示词扩写暂不可用，使用原始输入继续媒体生成')
        return fallback, True
    clean = client.generation_text(text)
    return (clean, clean == fallback) if clean else (fallback, True)


def _extract_plan(raw, fallback='', multiline=False):
    text = client.generation_text(str(raw or ''))
    text = re.sub(r'^```(?:json)?\s*\n(.*?)\n?```$', r'\1', text, flags=re.I | re.S).strip()
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Accept an otherwise valid JSON plan wrapped in a short explanation.
        decoder = json.JSONDecoder()
        for match in re.finditer(r'\{', text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and 'image_prompts' in candidate:
                payload = candidate
                break
    if isinstance(payload, dict):
        prompts = payload.get("image_prompts")
        if isinstance(prompts, str):
            prompts = [prompts]
        clean = [client.generation_text(item) for item in prompts if isinstance(item, str)] if isinstance(prompts, list) else []
        clean = [item for item in clean if item]
        extra = payload.get('text_output') or ''
        extra = client.generation_text(extra) if isinstance(extra, str) else ''
        if clean:
            return clean, extra
        return [client.generation_text('', fallback) or '根据参考素材生成图像。'], extra
    # Plain final prompts are usable too. Broken structured output and thought
    # text fall back to the original request, never the full ambiguous reply.
    if text and not text.startswith(('{', '[', '```')):
        prompts = re.split(r'\n\s*\n', text) if multiline else [text]
        return [part.strip() for part in prompts if part.strip()], ''
    return [client.generation_text('', fallback) or '根据参考素材生成图像。'], ''


def _task_config(config):
    return {**config, "_wuji_omit_seed": True}


async def _image_batch(config, prompts, reference_uris, image_size, seed, concurrency):
    """Drain every accepted wave before retrying explicitly rejected items once."""
    outputs = [None] * len(prompts)
    limit = max(1, min(8, int(concurrency)))
    downgraded = False

    async def generate(index):
        image_seed = seed if index == 0 else _derived_seed(seed, f"image-item-{index}")
        try:
            payload = await asyncio.to_thread(client.generate_image, config, prompts[index],
                images=reference_uris, size=image_size, extra={"seed": image_seed})
        except client.WujiApiError as exc:
            if client.concurrency_rejected(exc):
                return None, exc
            raise
        payload = await client.poll_media(config, payload, "image")
        output = await media.images_from_payload(payload, timeout=config.get("timeout", 600), max_retries=0)
        if output is None:
            raise client.WujiApiError("创造API未返回可解析的图像。")
        return output, None

    start = 0
    while start < len(prompts):
        indices = list(range(start, min(start + limit, len(prompts))))
        results = await asyncio.gather(*(generate(index) for index in indices), return_exceptions=True)
        rejected, failures = [], []
        for index, result in zip(indices, results):
            if isinstance(result, BaseException):
                failures.append((index, result))
            elif result[1] is not None:
                (rejected if limit > 1 else failures).append((index, result[1]))
            else:
                outputs[index] = result[0]
        if failures:
            if len(indices) == 1:
                raise failures[0][1]
            completed = "、".join(str(i + 1) for i, value in enumerate(outputs) if value is not None) or "无"
            reason = "；".join(f"第{i + 1}张：{exc}" for i, exc in failures)
            raise client.WujiApiError(f"批量图像生成未全部完成；已完成项：{completed}；{reason}；未重新提交已受理任务")
        if rejected:
            limit = 1
            downgraded = True
            for index, error in rejected:
                if error.retry_after:
                    await asyncio.sleep(error.retry_after)
                outputs[index], second_rejection = await generate(index)
                if second_rejection:
                    raise second_rejection
        start += len(indices)
    return outputs, downgraded


def _custom_image_size(custom_width=0, custom_height=0, alignment=0):
    width, height, multiple = (int(value or 0) for value in (custom_width, custom_height, alignment))
    if min(width, height, multiple) < 0:
        raise ValueError("自定义宽度、高度和对齐倍数不能为负数。")
    if not (width or height or multiple):
        return None
    width = width or height or 1024
    height = height or width
    if multiple:
        width = (width + multiple - 1) // multiple * multiple
        height = (height + multiple - 1) // multiple * multiple
    return f"{width}x{height}"


def _image_base_size(aspect_ratio, resolution, custom_width=0, custom_height=0, alignment=0):
    """The p presets use the short edge; automatic defaults to landscape 16:9."""
    custom = _custom_image_size(custom_width, custom_height, alignment)
    if custom:
        return custom
    if resolution not in {"480p", "720p", "1080p"}:
        return nodes._image_size(aspect_ratio, resolution)
    ratio = "16:9" if aspect_ratio in (None, "", "自动") else aspect_ratio
    width_ratio, height_ratio = (int(part) for part in ratio.split(":"))
    short_edge = int(resolution[:-1])
    if width_ratio >= height_ratio:
        return f"{round(short_edge * width_ratio / height_ratio)}x{short_edge}"
    return f"{short_edge}x{round(short_edge * height_ratio / width_ratio)}"


def _fit_custom_image_size(width, height, alignment):
    """Project onto the model's valid grid, retaining user divisibility as well."""
    step = math.lcm(16, int(alignment or 1))
    min_pixels, max_pixels, max_edge = 655360, 8294400, 3840
    aligned_width = (width + step - 1) // step * step
    aligned_height = (height + step - 1) // step * step
    if (max(aligned_width, aligned_height) <= max_edge
            and min_pixels <= aligned_width * aligned_height <= max_pixels
            and max(aligned_width, aligned_height) <= 3 * min(aligned_width, aligned_height)):
        return f"{aligned_width}x{aligned_height}"

    # Search at most 240 widths. Each width has a contiguous valid height range;
    # the nearest height minimizes log-size error, so two candidates suffice.
    best = None
    for candidate_width in range(step, max_edge + 1, step):
        lower = max((min_pixels + candidate_width - 1) // candidate_width,
                    (candidate_width + 2) // 3, step)
        upper = min(max_pixels // candidate_width, candidate_width * 3, max_edge)
        lower = (lower + step - 1) // step * step
        upper = upper // step * step
        if lower > upper:
            continue
        for candidate_height in {max(lower, min(upper, height // step * step)),
                                 max(lower, min(upper, (height + step - 1) // step * step))}:
            error = (math.log(candidate_width / width) ** 2
                     + math.log(candidate_height / height) ** 2)
            candidate = (error, candidate_width, candidate_height)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError(f"对齐倍数 {alignment} 与此模型的尺寸限制无法同时满足；请减小对齐倍数或设为 0。")
    return f"{best[1]}x{best[2]}"


def _image_request_size(config, aspect_ratio, resolution, custom_width=0, custom_height=0, alignment=0):
    """Fit GPT Image 2.5 sizes before submission; other models keep their presets."""
    size = _image_base_size(aspect_ratio, resolution, custom_width, custom_height, alignment)
    model = (config.get("remote") or {}).get("image") or config.get("image_model") or ""
    if model not in {"gpt-image-2.5-flare", "gpt-image-2.5-sunburst"} or not size:
        return size

    # Official Image API constraints: 16px steps, <=3840px per edge,
    # 655,360..8,294,400 pixels, and at most 3:1 in either orientation.
    width, height = map(int, size.split("x"))
    if custom_width or custom_height or alignment:
        return _fit_custom_image_size(width, height, alignment)
    if max(width, height) > 3 * min(width, height):
        raise ValueError("此图像模型的长短边比例不能超过 3:1，请调整画面比例。")
    min_pixels, max_pixels = 655360, 8294400
    scale = min(1.0, 3840 / max(width, height), math.sqrt(max_pixels / (width * height)))
    if width * height < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        align = math.ceil
    else:
        align = math.floor
    width = max(16, align(width * scale / 16) * 16)
    height = max(16, align(height * scale / 16) * 16)
    return f"{width}x{height}"


def _accept_chinese_inputs(method):
    """Keep internal/API callers compatible while schema IDs are Chinese in V1/V3."""
    aliases = {
        "元指令": "meta_instruction", "文本内容": "prompt", "提示词": "prompt",
        "参考图": "images", "图像输入": "images", "视频输入": "videos",
        "音频输入": "audios", "图像作用": "image_role_mode",
        "画面比例": "aspect_ratio", "分辨率": "resolution", "时长（秒）": "duration",
        "随机种子": "seed", "温度覆盖": "temperature",
        "自定义宽度": "custom_width", "自定义高度": "custom_height", "对齐倍数": "alignment",
    }

    @wraps(method)
    async def execute(cls, *args, **kwargs):
        for chinese, internal in aliases.items():
            if chinese in kwargs:
                kwargs[internal] = kwargs.pop(chinese)
        return await method(cls, *args, **kwargs)

    return execute


class CreationAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreationAPI", display_name="创造API", category="创造",
            description="API 模式动力源，默认使用创意推理服务。",
            inputs=[
                io.Combo.Input("API服务商", options=list(V2_PRESETS), default="创意推理"),
                io.String.Input("API地址", default=V2_PRESETS["创意推理"]),
                io.String.Input("API密钥", default=""),
                io.String.Input("API文本视觉模型", default=""),
                io.String.Input("API图像模型", default="未启用"),
                io.String.Input("API视频模型", default="未启用"),
                io.Float.Input("生成温度", default=1.1, min=0, max=2, step=0.05, optional=True),
                io.Int.Input("最大长度", default=16384, min=128, max=131072, optional=True),
                io.Int.Input("请求超时", default=600, min=10, max=3600, optional=True),
                io.Int.Input("随机种子", default=1, min=1, max=2147483647, control_after_generate=True, optional=True),
            ], outputs=[WUJI_LLM.Output(display_name="动力源")],
        )

    @classmethod
    async def execute(cls, API服务商="创意推理", API地址="", API密钥="", API文本视觉模型="",
                      API图像模型="未启用", API视频模型="未启用", 生成温度=1.1,
                      最大长度=16384, 请求超时=600, 随机种子=1):
        base = str(API地址 or "").strip() or V2_PRESETS.get(API服务商, "")
        client._url(base, "chat/completions")
        text, image, video = map(_selection, (API文本视觉模型, API图像模型, API视频模型))
        if not (text or image or video):
            raise ValueError("至少选择一个文本、图像或视频模型。")
        seed = max(1, int(随机种子 or 1))
        config = {
            "wuji_api_version": 2, "_wuji_omit_seed": True,
            "_wuji_async_image_tasks": urlsplit(base).hostname == "api.creative-reasoning.com",
            "image_edit_transport": "json" if urlsplit(base).hostname == "api.creative-reasoning.com" else "multipart",
            "enable_thinking": False,
            "provider": "openai_compatible", "text_enabled": bool(text),
            "vision_enabled": bool(text), "image_enabled": bool(image),
            "video_enabled": bool(video), "text_model": text,
            "image_model": image, "video_model": video,
            "api_key": API密钥, "base_url": base,
            "temperature": float(生成温度), "max_tokens": int(最大长度),
            "seed": seed, "timeout": int(请求超时), "max_retries": 0,
            "remote": {"base_url": base, "api_key": API密钥, "text_vision": text,
                       "image": image, "video": video},
        }
        return io.NodeOutput(config)


class CreationText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreationText", display_name="创造文本扩写", category="创造",
            inputs=[WUJI_LLM.Input("动力源", display_name="动力源"),
                    io.String.Input("元指令", display_name="元指令", multiline=True, default="", optional=True),
                    io.String.Input("文本内容", display_name="文本内容", multiline=True, default="", optional=True),
                    io.Float.Input("温度覆盖", display_name="温度覆盖", default=-1, min=-1, max=2, optional=True),
                    io.Int.Input("随机种子", display_name="随机种子", default=1, min=1, max=2147483647, control_after_generate=True, optional=True)],
            outputs=[io.String.Output(display_name="文本"), io.String.Output(display_name="信息")],
        )

    @classmethod
    @_accept_chinese_inputs
    async def execute(cls, 动力源=None, meta_instruction="", prompt="", temperature=-1, seed=0):
        config = _config(动力源)
        if not config.get("text_enabled"):
            raise ValueError("创造API尚未选择文本/视觉模型。")
        seed = max(1, int(seed or config.get("seed", 1)))
        meta = "\n\n".join(part for part in (str(meta_instruction or "").strip(), _variation(seed)) if part)
        temp = float(temperature) if float(temperature) >= 0 else config.get("temperature", 1.1)
        text_config = {**config, '_creation_text_fallback': str(prompt or meta_instruction or '').strip()}
        text = await asyncio.to_thread(client.chat, text_config, _text_messages(meta, prompt),
                                       temperature=temp, max_tokens=config.get("max_tokens", 4096))
        return io.NodeOutput(text, "")


class CreationImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreationImage", display_name="创造图像生成", category="创造",
            inputs=[
                WUJI_LLM.Input("动力源", display_name="动力源"),
                io.String.Input("元指令", display_name="元指令", multiline=True, default="", optional=True),
                io.String.Input("提示词", display_name="提示词", multiline=True, default="", optional=True),
                io.Autogrow.Input("参考图", display_name="参考图", optional=True,
                    template=io.Autogrow.TemplatePrefix(input=io.Image.Input("参考图"), prefix="参考图_", min=0, max=16)),
                io.Combo.Input("画面比例", display_name="画面比例", options=["自动", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9"], default="自动", optional=True,
                    tooltip="自定义宽度、高度、对齐倍数任一非零时，本项不参与尺寸计算。"),
                io.Combo.Input("分辨率", display_name="分辨率", options=["480p", "720p", "1080p", "1K", "2K", "4K"], default="1K", optional=True,
                    tooltip="自定义宽度、高度、对齐倍数任一非零时，本项不参与尺寸计算。"),
                io.Int.Input("自定义宽度", default=0, min=0, max=65536, step=1, optional=True,
                    tooltip="像素。三项均为 0 时沿用原比例和分辨率；任一非零则启用自定义。仅填一边时补成方图。"),
                io.Int.Input("自定义高度", default=0, min=0, max=65536, step=1, optional=True,
                    tooltip="像素。自定义启用后忽略画面比例和分辨率预设；仅填一边时补成方图。"),
                io.Int.Input("对齐倍数", default=0, min=0, max=65536, step=1, optional=True,
                    tooltip="非零时宽高向上取整到该数的倍数；0 不增加对齐，仍保留模型原有尺寸要求。仅填本项时从 1024×1024 开始。"),
                io.Int.Input("随机种子", display_name="随机种子", default=1, min=1, max=2147483647, control_after_generate=True, optional=True),
                io.Boolean.Input("多行列表", default=False, optional=True, label_on="开启", label_off="关闭"),
                io.Int.Input("并发数量", default=3, min=1, max=8, optional=True),
            ],
            outputs=[io.String.Output(display_name="文本"), io.Image.Output(display_name="图像"), io.String.Output(display_name="信息")],
        )

    @classmethod
    @_accept_chinese_inputs
    async def execute(cls, 动力源=None, meta_instruction="", prompt="", images=None,
                      aspect_ratio="自动", resolution="1K", 多行列表=False, seed=0, 并发数量=3,
                      custom_width=0, custom_height=0, alignment=0):
        config = _task_config(_config(动力源))
        seed = max(1, int(seed or config.get("seed", 1)))
        if not config.get("text_enabled"):
            raise ValueError("图像提示词规划需要选择文本/视觉模型。")
        if not config.get("image_enabled"):
            raise ValueError("创造API尚未选择图像模型。")
        ratio = "" if aspect_ratio in (None, "", "自动") else str(aspect_ratio)
        custom_size = _custom_image_size(custom_width, custom_height, alignment)
        requested_size = _image_base_size(ratio, resolution, custom_width, custom_height, alignment)
        image_size = _image_request_size(config, ratio, resolution, custom_width, custom_height, alignment)
        reference_uris = []
        for tensor in nodes._sorted_slots(images):
            reference_uris.extend(media.image_tensor_to_data_uris(
                tensor, fmt="auto", max_edge=2048, max_mb=media.DEFAULT_IMAGE_BUDGET_MB))

        list_instruction = "多行列表已开启：每张图分别输出一项。" if 多行列表 else "多行列表已关闭：只输出一项完整的图像提示词。"
        planner_meta = "\n\n".join(part.strip() for part in (meta_instruction, PLAN_INSTRUCTION, list_instruction, _variation(seed, kind="image")) if part and part.strip())
        fallback = _original_prompt(meta_instruction, prompt, 'image')
        planned, expansion_fallback = await _expand_media_prompt(config,
            _text_messages(planner_meta, prompt, reference_uris), fallback)
        prompts, extra_text = _extract_plan(planned, fallback, 多行列表)
        if not 多行列表:
            prompts = ["\n\n".join(prompts)]

        image_batches, downgraded = await _image_batch(config, prompts, reference_uris, image_size, seed, 并发数量)

        batch = media._pad_and_stack(image_batches)
        info = "\n\n".join(f"图像提示词 {index}:\n{value}" for index, value in enumerate(prompts, 1))
        if extra_text:
            info += f"\n\n额外文本：\n{extra_text}"
        if custom_size:
            info += (f"\n\n自定义尺寸输入：宽 {custom_width}，高 {custom_height}，对齐倍数 {alignment}"
                     f"；补全及自定义对齐后：{custom_size}；画面比例和分辨率预设已忽略。"
                     f"\n图像提交尺寸：{image_size}")
        else:
            info += f"\n\n图像提交尺寸：{image_size}（分辨率档位：{resolution}）"
        if image_size != requested_size:
            info += "；已按所选模型的边长、总像素和尺寸步长限制调整。"
        if downgraded:
            info += "\n\n服务端明确拒绝并发，未受理项及本批剩余项已改为串行。"
        if expansion_fallback or (prompts == [fallback] and planned != fallback):
            info += '\n\n扩写结果不可用，已采用原始输入继续生成；额外文本以实际返回为准。'
        return io.NodeOutput(extra_text, batch, info)


class CreationVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreationVideo", display_name="创造视频生成", category="创造",
            inputs=[
                WUJI_LLM.Input("动力源", display_name="动力源"),
                io.String.Input("元指令", display_name="元指令", multiline=True, default="", optional=True),
                io.String.Input("提示词", display_name="提示词", multiline=True, default="", optional=True),
                io.Autogrow.Input("图像输入", display_name="图像输入", optional=True,
                    template=io.Autogrow.TemplatePrefix(input=io.Image.Input("图像"), prefix="图像_", min=0, max=16)),
                io.Autogrow.Input("视频输入", display_name="视频输入", optional=True,
                    template=io.Autogrow.TemplatePrefix(input=io.Video.Input("视频"), prefix="视频_", min=0, max=8)),
                io.Autogrow.Input("音频输入", display_name="音频输入", optional=True,
                    template=io.Autogrow.TemplatePrefix(input=io.Audio.Input("音频"), prefix="音频_", min=0, max=8)),
                io.Combo.Input("图像作用", display_name="图像作用", options=nodes.ROLE_MODE_CHOICES,
                    default=nodes.ROLE_MODE_LABELS["all_reference"], optional=True),
                io.Combo.Input("画面比例", display_name="画面比例", options=["自动", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9"], default="自动", optional=True),
                io.Combo.Input("分辨率", display_name="分辨率", options=["自动", "480p", "720p", "1080p"], default="自动", optional=True),
                io.Int.Input("时长（秒）", display_name="时长（秒）", default=0, min=0, max=60, optional=True),
                io.Int.Input("随机种子", display_name="随机种子", default=1, min=1, max=2147483647, control_after_generate=True, optional=True),
            ],
            outputs=[io.String.Output(display_name="文本"), io.Video.Output(display_name="视频"), io.String.Output(display_name="信息")],
        )

    @classmethod
    @_accept_chinese_inputs
    async def execute(cls, 动力源=None, meta_instruction="", prompt="", images=None, videos=None,
                      audios=None, image_role_mode=nodes.ROLE_MODE_LABELS["all_reference"],
                      aspect_ratio="自动", resolution="自动", duration=0, seed=0):
        config = _task_config(_config(动力源))
        if not config.get("video_enabled"):
            raise ValueError("创造API尚未选择视频模型。")
        if not config.get("text_enabled"):
            raise ValueError("视频提示词扩写需要在创造API中选择文本/视觉模型。")
        image_uris = []
        for tensor in nodes._sorted_slots(images):
            image_uris.extend(media.image_tensor_to_data_uris(
                tensor, fmt="jpeg", max_edge=2048, max_mb=media.DEFAULT_IMAGE_BUDGET_MB))
        mode = nodes._ROLE_MODE_BY_LABEL.get(image_role_mode, "all_reference")
        roles, used = nodes.assign_roles(len(image_uris), mode)
        image_uris = image_uris[:used]
        video_uris = [media.video_to_data_uri(value) for value in nodes._sorted_slots(videos)]
        audio_uris = [media.audio_to_data_uri(value) for value in nodes._sorted_slots(audios)]
        refs = [f"@{label}{index}" for label, items in (("图片", image_uris), ("视频", video_uris), ("音频", audio_uris))
                for index in range(1, len(items) + 1)]
        if not (str(meta_instruction or "").strip() or str(prompt or "").strip() or refs):
            raise client.WujiApiError("视频模型需要提示词或参考素材。")
        extra = {}
        if aspect_ratio and aspect_ratio != "自动":
            extra["aspect_ratio"] = aspect_ratio
        if resolution and resolution != "自动":
            extra["resolution"] = resolution
        seed = max(1, int(seed or config.get("seed", 1)))
        extra["seed"] = seed
        context = [str(prompt or "").strip()]
        if refs:
            context.append("参考素材：" + "、".join(refs))
        if roles:
            role_labels = {"first_frame": "首帧", "last_frame": "尾帧", "reference_image": "参考图"}
            context.append("图像用途：" + "；".join(f"@图片{index}：{role_labels.get(role, role)}" for index, role in enumerate(roles, 1)))
        if duration:
            context.append(f"视频时长：{duration} 秒")
        for name, label in (("aspect_ratio", "画面比例"), ("resolution", "分辨率")):
            if name in extra:
                context.append(f"{label}：{extra[name]}")
        planner_meta = "\n\n".join(value for value in (
            str(meta_instruction or "").strip(), VIDEO_PLAN_INSTRUCTION, _variation(seed, kind="video")) if value)
        fallback = _original_prompt(meta_instruction, prompt, 'video')
        full_prompt, expansion_fallback = await _expand_media_prompt(config,
            _text_messages(planner_meta, "\n\n".join(value for value in context if value), image_uris, video_uris, audio_uris), fallback)
        missing_refs = [ref for ref in refs if ref not in full_prompt]
        if missing_refs:
            full_prompt += "\n\n参考素材：" + "、".join(missing_refs)
        payload = await asyncio.to_thread(client.generate_video, config, full_prompt,
            images=image_uris, videos=video_uris, audios=audio_uris,
            image_roles=roles, reference_roles=True, duration=duration, extra=extra)
        payload = await client.poll_media(config, payload, "video")
        output, _ = await media.video_from_payload(payload, timeout=config.get("timeout", 600), max_retries=0)
        if output is None:
            raise client.WujiApiError("创造API未返回可解析的视频。")
        return io.NodeOutput(full_prompt, output, json.dumps({"source": "creation_api", "kind": "video", "image_roles": roles,
            "text_model": config.get("text_model"), "final_prompt": full_prompt, "seed": seed,
            "expansion_fallback": expansion_fallback}, ensure_ascii=False))

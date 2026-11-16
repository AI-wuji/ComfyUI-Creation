"""接入能力推断与警告。

重要事实：平台模型目录**不提供任何能力元数据**（只有 category / display_name / description），
视频接口文档也明确写了"只有所选模型支持的作用类型会生效"——即不支持的 role 会被**静默丢弃**，
不会报错。所以这里做不到"精确探测"，只能做保守推断 + 明确告知。

设计原则：宁可多提醒，绝不假装生效。所有推断结果都会写进节点的 info 输出，
用户对"这次到底有没有用上"始终可见。
"""

from __future__ import annotations

ROLE_FIRST_FRAME = "first_frame"
ROLE_LAST_FRAME = "last_frame"
ROLE_REFERENCE_IMAGE = "reference_image"
ROLE_REFERENCE_VIDEO = "reference_video"
ROLE_REFERENCE_AUDIO = "reference_audio"

ALL_ROLES = (
    ROLE_FIRST_FRAME,
    ROLE_LAST_FRAME,
    ROLE_REFERENCE_IMAGE,
    ROLE_REFERENCE_VIDEO,
    ROLE_REFERENCE_AUDIO,
)

# 模型名关键词 -> 明确支持的作用。留空表示没有可靠情报，一律走类别推断。
# 平台一旦在目录里补上能力字段，这里应改为读取真实数据。
MODEL_ROLE_OVERRIDES: dict[str, tuple[str, ...]] = {}


def supported_roles(model_id: str, category: str) -> tuple[str, ...]:
    """返回该模型大概率支持的作用集合。"""
    lowered = (model_id or "").lower()
    for keyword, roles in MODEL_ROLE_OVERRIDES.items():
        if keyword in lowered:
            return roles

    if category == "video":
        # 视频接口的 content 数组原生定义了全部五种作用
        return ALL_ROLES
    if category == "image":
        # 图像接口没有 role 概念，多图以 image 数组形式参与（多参）
        return ()
    # 对话接口只有 content 分段，没有 role 语义
    return ()


def accepts_modality(category: str, modality: str) -> bool:
    """该类别的接口是否真会把这种模态放进请求体。"""
    table = {
        "text": {"image": True, "video": True, "audio": True},
        "image": {"image": True, "video": False, "audio": False},
        "video": {"image": True, "video": True, "audio": True},
        "audio": {"image": False, "video": False, "audio": True},
    }
    return table.get(category, {}).get(modality, False)


def build_warnings(
    *,
    model_id: str,
    category: str,
    n_images: int,
    n_videos: int,
    n_audios: int,
    roles: list[str],
    dropped_images: int,
) -> list[str]:
    """列出"接了但不会生效"的项，避免用户以为生效了。"""
    warnings: list[str] = []
    roles = roles or []

    if dropped_images > 0:
        warnings.append(
            f"当前首尾帧模式只使用前若干张，有 {dropped_images} 张图片未参与请求；"
            "如需全部生效请改用「全部作为参考图」或「首帧+其余参考图」。"
        )

    if n_videos and not accepts_modality(category, "video"):
        warnings.append(f"{category} 类接口不支持视频输入，已忽略 {n_videos} 个视频。")
    if n_audios and not accepts_modality(category, "audio"):
        warnings.append(f"{category} 类接口不支持音频输入，已忽略 {n_audios} 段音频。")
    if n_images and not accepts_modality(category, "image"):
        warnings.append(f"{category} 类接口不支持图像输入，已忽略 {n_images} 张图片。")

    if category in ("text", "image") and (ROLE_FIRST_FRAME in roles or ROLE_LAST_FRAME in roles):
        warnings.append(
            "首尾帧 role 仅对视频接口有意义，本次请求不会下发 role 字段"
            "（图片按顺序作为普通多模态输入）。"
        )

    if category == "audio":
        warnings.append(
            "平台公开的统一接口列表里暂时没有音频端点（audio 输出为预留能力，"
            "目录中也没有音频模型）；本模式会请求 audio/generations，"
            "若上游未开放将返回 404。接到能产出音频的模型后，audio 输出会正常透传。"
        )
        if n_images:
            warnings.append(f"音频类接口不支持图像输入，已忽略 {n_images} 张图片。")
        if n_videos:
            warnings.append(f"音频类接口不支持视频输入，已忽略 {n_videos} 段视频。")
        if n_audios > 1:
            warnings.append(
                f"已把 {n_audios} 段参考音频作为数组下发（语音克隆 / 音乐参考）。"
            )
        elif n_audios == 1 and not (n_images or n_videos):
            warnings.append("已把 1 段参考音频用于语音克隆 / 音乐参考。")

    if category == "video" and roles:
        supported = set(supported_roles(model_id, category))
        if supported:
            unknown = [r for r in set(roles) if r not in supported]
            if unknown:
                warnings.append(
                    f"模型 {model_id} 可能不支持作用 {'、'.join(unknown)}，"
                    "平台会静默忽略不支持的作用而不报错。"
                )

    if category == "image" and n_images > 1:
        warnings.append(
            f"已把 {n_images} 张图作为 image 数组下发（多参）。"
            "平台文档写明「数组支持取决于所选模型」，实测该路径上游不稳定"
            "（可能 30 秒超时、也可能 2 分钟才返回）。若失败，节点会自动降级为"
            "第 1 张参考图重跑，并在 info 里写明降级原因——不会静默改变语义。"
        )
    return warnings

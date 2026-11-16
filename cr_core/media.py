"""媒体转换：ComfyUI 张量/视频对象 <-> 接口所需的 data URI，以及结果还原。"""

from __future__ import annotations

import base64
import io
import logging
import time
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config
from .http_client import CRError, download_bytes

logger = logging.getLogger("CreativeReasoning")

IMAGE_MIME = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg", "webp": "image/webp"}

# 单张上行图的 base64 预算（MB）。超过即降级编码，避免撞上网关的
# 413 PayloadTooLarge / 上游 provider 的体积上限。
# 实测：2048px 的 PNG 照片可到 3~4.3MB，而同一张图 q90 JPEG@1536 只要 0.2~0.7MB。
DEFAULT_IMAGE_BUDGET_MB = 1.5
_PNG_IS_LOSSLESS_MIN_EDGE = 512
VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


# --- 上行：张量 -> data URI ------------------------------------------------
def _tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    arr = frame.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, "RGBA")
    if arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], "L").convert("RGB")
    return Image.fromarray(arr[..., :3], "RGB")


def _downscale(img: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return img
    longest = max(img.size)
    if longest <= max_edge:
        return img
    scale = max_edge / float(longest)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


def _encode_pil(img: Image.Image, pil_fmt: str, quality: int) -> tuple[str, str]:
    """把 PIL 图编码成 (data URI, 体积MB)。"""
    if pil_fmt in ("JPEG", "WEBP") and img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    save_kwargs = {"quality": int(quality)} if pil_fmt in ("JPEG", "WEBP") else {}
    img.save(buf, format=pil_fmt, **save_kwargs)
    payload = buf.getvalue()
    key = pil_fmt.lower()
    if key not in IMAGE_MIME:  # WEBP 之类，兜底
        key = "png"
    b64 = base64.b64encode(payload).decode("ascii")
    uri = f"data:{IMAGE_MIME[key]};base64,{b64}"
    return uri, len(uri) / (1024 * 1024)


def data_uri_mb(uri: str) -> float:
    """估算 data URI 的传输体积（MB），用于在 info/日志里如实告诉用户发了多大。"""
    return len(uri or "") / (1024 * 1024)


def _encode_with_budget(img: Image.Image, fmt: str, quality: int, max_mb: float) -> str:
    """在体积预算内编码一张图。

    顺序：PNG（无损，线稿/遮罩最优）→ JPEG(q)（照片可缩小近一个数量级）→ 逐级降边长。
    只有 fmt="auto" 会走这条降级链；显式指定 fmt 时行为与旧版完全一致。
    """
    if max_mb <= 0:
        pil_fmt = "JPEG" if fmt in ("jpeg", "jpg") else fmt.upper()
        return _encode_pil(img, pil_fmt, quality)[0]

    best = _encode_pil(img, "PNG", quality)
    if best[1] <= max_mb:
        return best[0]

    # PNG 超预算：照片类换 JPEG 通常直接达标；线稿类继续靠降边长压
    cand = _encode_pil(img, "JPEG", quality)
    if cand[1] < best[1]:
        best = cand

    edge = max(img.size)
    while best[1] > max_mb and edge > _PNG_IS_LOSSLESS_MIN_EDGE:
        edge = max(_PNG_IS_LOSSLESS_MIN_EDGE, int(edge * 0.8))
        shrunk = _downscale(img, edge)
        for pil_fmt in ("JPEG", "PNG"):
            cand = _encode_pil(shrunk, pil_fmt, quality)
            if cand[1] < best[1]:
                best = cand
            if best[1] <= max_mb:
                return best[0]
        if max(shrunk.size) <= _PNG_IS_LOSSLESS_MIN_EDGE:
            break
    return best[0]


def image_tensor_to_data_uris(
    images: torch.Tensor,
    *,
    fmt: str = "png",
    max_edge: int = 2048,
    quality: int = 92,
    max_mb: float = 0.0,
) -> list[str]:
    """一个 IMAGE 输入可能是批次，逐帧转为 data URI。

    fmt="auto" 时按 ``max_mb`` 预算自适应选择 PNG/JPEG 与边长；
    其余取值（png/jpeg/webp）行为与旧版一致。
    """
    if images is None:
        return []
    if images.ndim == 3:
        images = images.unsqueeze(0)

    fmt = (fmt or "png").lower()
    auto = fmt == "auto"
    if not auto and fmt not in IMAGE_MIME:
        fmt = "png"
    pil_fmt = "JPEG" if fmt in ("jpeg", "jpg") else fmt.upper()

    uris: list[str] = []
    for i in range(images.shape[0]):
        img = _downscale(_tensor_to_pil(images[i]), max_edge)
        if auto:
            uris.append(_encode_with_budget(img, "auto", int(quality), float(max_mb)))
        else:
            uris.append(_encode_pil(img, pil_fmt, int(quality))[0])
    return uris


def _read_video_bytes(video) -> tuple[bytes, str]:
    source = video.get_stream_source()
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        mime = VIDEO_MIME.get(path.suffix.lower(), "video/mp4")
        return data, mime
    source.seek(0)
    return source.read(), "video/mp4"


def video_to_data_uri(video, *, max_mb: float = 48.0) -> str:
    """VIDEO 对象 -> data URI。base64 会放大约 33%，因此对原始体积设上限。"""
    if video is None:
        return ""
    try:
        data, mime = _read_video_bytes(video)
    except Exception as exc:  # noqa: BLE001 - 解码失败要给出可读原因
        raise CRError(config.scrub(f"读取视频输入失败：{exc}")) from exc

    size_mb = len(data) / (1024 * 1024)
    if max_mb > 0 and size_mb > max_mb:
        raise CRError(
            f"视频输入约 {size_mb:.1f}MB，超过上限 {max_mb:.0f}MB。"
            "请先裁剪时长或降低分辨率，或调大 video_max_mb。"
        )
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _normalize_waveform(waveform) -> tuple[np.ndarray, int]:
    """AUDIO 输入归一到 [C, T] 的 int16 数组。ComfyUI 的约定是 [B, C, T]。"""
    if waveform is None:
        raise CRError("音频输入缺少 waveform。")
    if isinstance(waveform, np.ndarray):
        arr = waveform.astype(np.float32, copy=False)
    else:
        arr = waveform.detach().cpu().float().numpy()

    if arr.ndim == 3:
        arr = arr[0]  # 只取第一批
    elif arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise CRError(f"音频 waveform 维度不支持：{arr.ndim}（期望 [B, C, T]）")

    if arr.shape[0] > arr.shape[1]:
        # 少量节点会输出 [T, C]，按样本数更多的那一维判定时长轴
        arr = arr.T

    channels = min(int(arr.shape[0]), 8)
    arr = np.clip(arr[:channels], -1.0, 1.0)
    return (arr * 32767.0).astype("<i2"), channels


def audio_to_data_uri(audio, *, max_mb: float = 16.0) -> str:
    """AUDIO 对象 -> WAV data URI。接口只吃 URL/base64，所以必须编码成文件容器。"""
    if audio is None:
        return ""
    if not isinstance(audio, dict):
        raise CRError("音频输入结构异常，期望 {'waveform': ..., 'sample_rate': ...}。")

    try:
        sample_rate = int(audio.get("sample_rate") or 44100)
        if sample_rate <= 0:
            sample_rate = 44100
        pcm, channels = _normalize_waveform(audio.get("waveform"))
    except CRError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CRError(config.scrub(f"解析音频输入失败：{exc}")) from exc

    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.T.reshape(-1).tobytes())

    data = buf.getvalue()
    size_mb = len(data) / (1024 * 1024)
    if max_mb > 0 and size_mb > max_mb:
        raise CRError(
            f"音频输入编码后约 {size_mb:.1f}MB，超过上限 {max_mb:.0f}MB。"
            "请先裁剪音频长度，或调大 audio_max_mb。"
        )
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


# --- 下行：结果 -> ComfyUI 类型 -------------------------------------------
def _decode_image_bytes(data: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _pad_and_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """尺寸不一致时补齐到最大宽高再拼批，避免 cat 报错。"""
    if len(tensors) == 1:
        return tensors[0]
    h = max(t.shape[1] for t in tensors)
    w = max(t.shape[2] for t in tensors)
    fixed = []
    for t in tensors:
        if t.shape[1] == h and t.shape[2] == w:
            fixed.append(t)
            continue
        canvas = torch.zeros((t.shape[0], h, w, t.shape[3]), dtype=t.dtype)
        canvas[:, : t.shape[1], : t.shape[2], :] = t
        fixed.append(canvas)
    return torch.cat(fixed, dim=0)


def _iter_media_refs(payload: dict) -> list[dict]:
    data = payload.get("data")
    if isinstance(data, dict):
        data = [data]
    return [d for d in (data or []) if isinstance(d, dict)]


def _pick(item: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        val = item.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            inner = val.get("url") or val.get("uri")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


async def images_from_payload(payload: dict, *, timeout: float, max_retries: int) -> torch.Tensor | None:
    """解析图像结果，兼容 b64_json / url 两种形态。"""
    tensors: list[torch.Tensor] = []
    for item in _iter_media_refs(payload):
        b64 = _pick(item, ("b64_json", "b64", "image_base64"))
        raw: bytes | None = None
        if b64:
            try:
                raw = base64.b64decode(b64.split(",", 1)[-1])
            except (ValueError, TypeError):
                raw = None
        if raw is None:
            url = _pick(item, ("url", "image_url", "uri"))
            if not url:
                continue
            if url.startswith("data:"):
                try:
                    raw = base64.b64decode(url.split(",", 1)[1])
                except (ValueError, IndexError):
                    continue
            else:
                raw = await download_bytes(url, timeout=timeout, max_retries=max_retries)
        try:
            tensors.append(_decode_image_bytes(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CreativeReasoning] 图像结果解码失败: %s", config.scrub(exc))
    if not tensors:
        return None
    return _pad_and_stack(tensors)


def _output_dir(subfolder: str) -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory())
    except Exception:  # noqa: BLE001 - 非 ComfyUI 环境下退化到插件目录
        root = config.PLUGIN_ROOT / "output"
    target = root / subfolder
    target.mkdir(parents=True, exist_ok=True)
    return target


async def video_from_payload(payload: dict, *, timeout: float, max_retries: int):
    """解析视频结果并落盘（资源 URL 会过期，拿到即存），返回 VIDEO 对象。"""
    for item in _iter_media_refs(payload):
        url = _pick(item, ("url", "video_url", "uri", "download_url"))
        raw: bytes | None = None
        if url and url.startswith("data:"):
            try:
                raw = base64.b64decode(url.split(",", 1)[1])
            except (ValueError, IndexError):
                raw = None
        elif url:
            raw = await download_bytes(url, timeout=timeout, max_retries=max_retries)
        else:
            b64 = _pick(item, ("b64_json", "b64", "video_base64"))
            if b64:
                try:
                    raw = base64.b64decode(b64.split(",", 1)[-1])
                except (ValueError, TypeError):
                    raw = None
        if not raw:
            continue

        name = f"cr_{int(time.time())}_{abs(hash(url or len(raw))) % 100000}.mp4"
        path = _output_dir("creative_reasoning") / name
        path.write_bytes(raw)
        try:
            from comfy_api.latest._input_impl import VideoFromFile

            return VideoFromFile(str(path)), str(path)
        except Exception:  # noqa: BLE001
            try:
                from comfy_api.input_impl import VideoFromFile  # 旧路径兜底

                return VideoFromFile(str(path)), str(path)
            except Exception as exc:  # noqa: BLE001
                raise CRError(config.scrub(f"视频已保存到 {path.name}，但构建 VIDEO 输出失败：{exc}")) from exc
    return None, ""


def save_images(images: torch.Tensor, prefix: str = "cr_image") -> list[str]:
    paths = []
    target = _output_dir("creative_reasoning")
    stamp = int(time.time())
    for i in range(images.shape[0]):
        path = target / f"{prefix}_{stamp}_{i:03d}.png"
        _tensor_to_pil(images[i]).save(path, format="PNG")
        paths.append(str(path))
    return paths


# --- 下行：音频结果 -> ComfyUI AUDIO 结构 -----------------------------------
def _decode_audio_bytes(data: bytes, name: str = "") -> dict | None:
    """把音频字节解码成 ComfyUI 的 AUDIO 字典 {waveform: Tensor[1,C,T], sample_rate}。

    优先用标准库 wave 解 WAV；其它格式（mp3/m4a/flac 等）尝试 soundfile（ComfyUI 自带）。
    返回 None 表示无法解码（交由调用方决定是报错还是跳过）。
    """
    if not data:
        return None
    lowered = (name or "").lower()
    is_wav = lowered.endswith(".wav") or data[:4] == b"RIFF" and data[8:12] == b"WAVE"

    if is_wav:
        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                n_channels = wf.getnchannels()
                samp_w = wf.getsampwidth()
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            if samp_w == 2:
                arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            elif samp_w == 1:
                arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif samp_w == 4:
                arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                return None
            if n_channels > 1:
                arr = arr.reshape(-1, n_channels).T  # [C, T]
            else:
                arr = arr.reshape(1, -1)
            tensor = torch.from_numpy(arr).float()
            return {"waveform": tensor.unsqueeze(0), "sample_rate": int(sr)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CreativeReasoning] WAV 解码失败，尝试其它解码器: %s", config.scrub(exc))

    # 非 WAV：尝试 soundfile（ComfyUI 便携包通常自带 libsndfile）
    try:
        import soundfile as sf

        data_np, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        # soundfile 返回 [frames, channels] -> 转 [channels, frames]
        arr = np.ascontiguousarray(data_np.T, dtype=np.float32)
        tensor = torch.from_numpy(arr).float()
        return {"waveform": tensor.unsqueeze(0), "sample_rate": int(sr)}
    except ImportError:
        logger.warning("[CreativeReasoning] 音频非 WAV 且 soundfile 不可用，无法解码（仅 WAV 受支持）。")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[CreativeReasoning] 音频解码失败: %s", config.scrub(exc))
        return None


async def audio_from_payload(payload: dict, *, timeout: float, max_retries: int) -> dict | None:
    """解析音频结果（兼容 b64_json / url / data URI），返回 AUDIO 字典。"""
    for item in _iter_media_refs(payload):
        b64 = _pick(item, ("b64_json", "b64", "audio_base64"))
        raw: bytes | None = None
        if b64:
            try:
                raw = base64.b64decode(b64.split(",", 1)[-1])
            except (ValueError, TypeError):
                raw = None
        if raw is None:
            url = _pick(item, ("url", "audio_url", "uri", "download_url"))
            if not url:
                continue
            if url.startswith("data:"):
                try:
                    raw = base64.b64decode(url.split(",", 1)[1])
                except (ValueError, IndexError):
                    continue
            else:
                raw = await download_bytes(url, timeout=timeout, max_retries=max_retries)
        if not raw:
            continue
        audio = _decode_audio_bytes(raw)
        if audio is not None:
            return audio
    return None


def save_audio(audio: dict, prefix: str = "cr_audio") -> list[str]:
    """把 AUDIO 字典落盘为 WAV，返回文件路径列表（便于用户找到文件）。"""
    if not isinstance(audio, dict) or "waveform" not in audio:
        return []
    wf_tensor = audio["waveform"]
    if isinstance(wf_tensor, torch.Tensor):
        arr = wf_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(wf_tensor)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T  # 保证 [C, T]
    sr = int(audio.get("sample_rate") or 44100)
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    target = _output_dir("creative_reasoning")
    path = target / f"{prefix}_{int(time.time())}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(max(1, int(arr.shape[0])))
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.reshape(-1).tobytes())
    return [str(path)]

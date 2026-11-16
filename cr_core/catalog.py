"""模型目录：动态拉取 + 磁盘缓存 + 后台刷新。

下拉菜单的选项由 define_schema() 每次被调用时读取本模块的缓存生成，
因此浏览器刷新页面即可看到最新模型，无需重启 ComfyUI。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

from . import config

logger = logging.getLogger("CreativeReasoning")

# 缓存有效期（秒）。过期后下一次读取会触发后台刷新，不阻塞 UI。
TTL = 600
FETCH_TIMEOUT = 8

_CACHE_FILE = config.CACHE_DIR / "models.json"

# 兜底种子列表：仅在既无缓存又拉取失败时使用，保证节点始终可加载。
# 必须与线上目录一致 —— 之前这里虚构过一个 "creative-reasoning-audio-1.0"，
# 目录不可用时它会出现在下拉里，用户一选就 404。线上目录当前只有下面 5 个模型。
_SEED: list[dict] = [
    {"name": "creative-reasoning-1.5", "category": "text", "display_name": "Creative Reasoning 1.5"},
    {"name": "creative-reasoning-1.5-flash", "category": "text", "display_name": "Creative Reasoning 1.5 Flash"},
    {"name": "creative-reasoning-1.0", "category": "text", "display_name": "Creative Reasoning 1.0"},
    {"name": "creative-reasoning-image-1.5", "category": "image", "display_name": "Creative Reasoning Image 1.5"},
    {"name": "creative-reasoning-video-1.0", "category": "video", "display_name": "Creative Reasoning Video 1.0"},
]

# 生成模型类别（图像 / 视频 / 音频）；text 单独走「多模态文本模型」下拉
_GEN_CATEGORIES = ("image", "video", "audio")

_lock = threading.Lock()
_state: dict = {"models": [], "fetched_at": 0.0, "loaded": False, "last_error": ""}
_refreshing = threading.Event()


# --- 归一化 ----------------------------------------------------------------
_TEXT_HINTS = ("text", "llm", "chat", "language", "embedding", "rerank")
_IMAGE_HINTS = ("image", "vision", "图")
_VIDEO_HINTS = ("video", "视频")
_AUDIO_HINTS = ("audio", "music", "tts", "voice", "song", "clone", "语音", "音乐", "音频")


def _guess_category(name: str, raw_category: str) -> str:
    cat = (raw_category or "").strip().lower()
    if cat in ("text", "image", "video", "audio"):
        return cat
    haystack = f"{cat} {name}".lower()
    for hint in _VIDEO_HINTS:
        if hint in haystack:
            return "video"
    for hint in _IMAGE_HINTS:
        if hint in haystack:
            return "image"
    for hint in _AUDIO_HINTS:
        if hint in haystack:
            return "audio"
    for hint in _TEXT_HINTS:
        if hint in haystack:
            return "text"
    # 未知类别按文本处理（chat/completions 兼容面最广）
    return "text"


def _normalize(payload) -> list[dict]:
    """兼容两种返回：目录数组，或 OpenAI 风格 {"data": [...]}。"""
    items = payload
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("models") or []
    if not isinstance(items, list):
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            name, meta = item, {}
        elif isinstance(item, dict):
            name = item.get("name") or item.get("id") or item.get("model") or ""
            meta = item.get("metadata") or {}
        else:
            continue
        name = str(name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        raw_cat = meta.get("category") or item.get("category") if isinstance(item, dict) else ""
        out.append(
            {
                "name": name,
                "category": _guess_category(name, str(raw_cat or "")),
                "display_name": str(meta.get("display_name") or name),
                "description": str(meta.get("description") or item.get("description") or "")[:1000]
                if isinstance(item, dict) else "",
            }
        )
    return out


# --- 磁盘缓存 --------------------------------------------------------------
def _load_disk() -> tuple[list[dict], float]:
    try:
        raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        models = _normalize(raw.get("models"))
        return models, float(raw.get("fetched_at") or 0.0)
    except (OSError, ValueError, AttributeError):
        return [], 0.0


def _save_disk(models: list[dict], fetched_at: float) -> None:
    try:
        config.ensure_cache_dir()
        _CACHE_FILE.write_text(
            json.dumps({"models": models, "fetched_at": fetched_at}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("[CreativeReasoning] 模型缓存写入失败: %s", config.scrub(exc))


# --- 网络拉取 --------------------------------------------------------------
def _fetch() -> list[dict]:
    req = urllib.request.Request(
        config.catalog_url(),
        headers={"Accept": "application/json", "User-Agent": "ComfyUI-CreativeReasoning"},
    )
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    models = _normalize(payload)
    if not models:
        raise ValueError("模型目录返回为空")
    return models


def _apply(models: list[dict]) -> None:
    now = time.time()
    with _lock:
        _state["models"] = models
        _state["fetched_at"] = now
        _state["loaded"] = True
        _state["last_error"] = ""
    _save_disk(models, now)


def refresh(blocking: bool = True) -> list[dict]:
    """强制刷新。blocking=False 时在后台线程执行并立即返回当前缓存。"""
    if not blocking:
        if not _refreshing.is_set():
            _refreshing.set()
            threading.Thread(target=_background_refresh, name="cr-catalog", daemon=True).start()
        with _lock:
            return list(_state["models"])
    try:
        models = _fetch()
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        with _lock:
            _state["last_error"] = config.scrub(exc)
        logger.warning("[CreativeReasoning] 模型目录刷新失败: %s", config.scrub(exc))
        with _lock:
            return list(_state["models"])
    _apply(models)
    return models


def _background_refresh() -> None:
    try:
        refresh(blocking=True)
    finally:
        _refreshing.clear()


def _ensure_loaded() -> None:
    if _state["loaded"]:
        return
    disk, fetched_at = _load_disk()
    if disk:
        with _lock:
            _state["models"] = disk
            _state["fetched_at"] = fetched_at
            _state["loaded"] = True
        if time.time() - fetched_at > TTL:
            refresh(blocking=False)
        return
    # 首次使用且无缓存：同步拉一次，失败则用种子列表，绝不让节点加载失败
    try:
        _apply(_fetch())
    except Exception as exc:  # noqa: BLE001 - 目录不可用不能阻塞节点注册
        with _lock:
            _state["models"] = list(_SEED)
            _state["fetched_at"] = 0.0
            _state["loaded"] = True
            _state["last_error"] = config.scrub(exc)
        logger.warning("[CreativeReasoning] 模型目录不可达，暂用内置列表: %s", config.scrub(exc))


def get_models() -> list[dict]:
    """返回模型列表；缓存过期时触发后台刷新（不阻塞）。"""
    _ensure_loaded()
    with _lock:
        stale = time.time() - _state["fetched_at"] > TTL
        models = list(_state["models"])
    if stale:
        refresh(blocking=False)
    return models


def model_names() -> list[str]:
    names = [m["name"] for m in get_models()]
    return names or [m["name"] for m in _SEED]


def text_model_names() -> list[str]:
    """多模态文本模型下拉项：目录里 category == text 的模型。"""
    names = [m["name"] for m in get_models() if m.get("category") == "text"]
    return names or [m["name"] for m in _SEED if m.get("category") == "text"]


def gen_model_names() -> list[str]:
    """生成模型下拉项：目录里 category 属于 图像/视频/音频 的模型。"""
    names = [m["name"] for m in get_models() if m.get("category") in _GEN_CATEGORIES]
    return names or [m["name"] for m in _SEED if m.get("category") in _GEN_CATEGORIES]


def category_of(name: str) -> str:
    target = (name or "").strip()
    for m in get_models():
        if m["name"] == target:
            return m["category"]
    return _guess_category(target, "")


def info_of(name: str) -> dict:
    """返回模型目录中的非敏感能力信息。"""
    target = (name or "").strip()
    for model in get_models():
        if model.get("name") == target:
            return dict(model)
    return {}


def display_of(name: str) -> str:
    for m in get_models():
        if m["name"] == name:
            return m.get("display_name") or name
    return name


def status() -> dict:
    _ensure_loaded()
    with _lock:
        return {
            "count": len(_state["models"]),
            "fetched_at": _state["fetched_at"],
            "age": max(0.0, time.time() - _state["fetched_at"]) if _state["fetched_at"] else None,
            "last_error": _state["last_error"],
            "models": list(_state["models"]),
        }

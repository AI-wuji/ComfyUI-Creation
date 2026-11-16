"""端点与凭据配置。

设计约定：服务端点在此写死，不作为节点控件暴露，也不出现在任何面向用户的
文本（日志 / 报错 / 输出）中。字符串以 base64 存放仅为避免被直接检索到，
这是「不显露」而非加密，请勿当作安全边界。
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PLUGIN_ROOT / "cache"

# --- 写死的端点（不对外显露） ---------------------------------------------
_API_BASE_B64 = "aHR0cHM6Ly9hcGkuY3JlYXRpdmUtcmVhc29uaW5nLmNvbS91bmlmaWVkL3Yx"
_CATALOG_B64 = "aHR0cHM6Ly9jcmVhdGl2ZS1yZWFzb25pbmcuY29tL2FwaS92MS9jYXRhbG9nL21vZGVscw=="
_HOSTS_B64 = (
    "YXBpLmNyZWF0aXZlLXJlYXNvbmluZy5jb20=",
    "Y3JlYXRpdmUtcmVhc29uaW5nLmNvbQ==",
)

_PLACEHOLDER = "<endpoint>"

# 环境变量名按优先级排列
ENV_KEYS = ("CR_API_KEY", "CREATIVE_REASONING_API_KEY")

# 密钥文件候选
_KEY_FILES = ("config.json", "api_key.txt")


def _d(value: str) -> str:
    return base64.b64decode(value).decode("utf-8")


def api_base() -> str:
    """统一接口根地址（内部使用）。"""
    return _d(_API_BASE_B64)


def catalog_url() -> str:
    """公开模型目录地址（内部使用，无需鉴权）。"""
    return _d(_CATALOG_B64)


def endpoint(path: str) -> str:
    return f"{api_base()}/{path.lstrip('/')}"


# --- 报错脱敏 --------------------------------------------------------------
def _host_patterns() -> list[re.Pattern]:
    pats = []
    for enc in _HOSTS_B64:
        host = _d(enc)
        pats.append(re.compile(r"https?://" + re.escape(host) + r"[^\s'\"<>)\]]*", re.I))
        pats.append(re.compile(re.escape(host), re.I))
    return pats


_SCRUB_PATTERNS = _host_patterns()
_KEY_PATTERN = re.compile(r"(?i)\b(bearer\s+)?(sk-|cr-)[A-Za-z0-9_\-]{8,}")


def scrub(text) -> str:
    """移除文本中的端点与密钥痕迹，用于任何回传给用户的字符串。"""
    if text is None:
        return ""
    out = text if isinstance(text, str) else str(text)
    for pat in _SCRUB_PATTERNS:
        out = pat.sub(_PLACEHOLDER, out)
    out = _KEY_PATTERN.sub("<api_key>", out)
    return out


# --- 密钥解析 --------------------------------------------------------------
def _read_key_files() -> str:
    for name in _KEY_FILES:
        path = PLUGIN_ROOT / name
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw:
            continue
        if name.endswith(".json"):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                for field in ("api_key", "apiKey", "key", "token"):
                    val = data.get(field)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        else:
            return raw.splitlines()[0].strip()
    return ""


def resolve_api_key(explicit: str | None = None) -> str:
    """解析顺序：节点控件 -> 环境变量 -> 插件目录密钥文件。"""
    if explicit and explicit.strip():
        return explicit.strip()
    for env in ENV_KEYS:
        val = os.environ.get(env, "")
        if val.strip():
            return val.strip()
    return _read_key_files()


def key_source_hint() -> str:
    """给用户看的密钥来源提示（不含密钥本体）。"""
    for env in ENV_KEYS:
        if os.environ.get(env, "").strip():
            return f"环境变量 {env}"
    if _read_key_files():
        return "插件目录 config.json / api_key.txt"
    return "未配置"


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR

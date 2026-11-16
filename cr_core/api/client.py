"""OpenAI-compatible chat/images and Creative Reasoning-style video transport.

Media endpoint/payload conventions are explicit, not universal provider APIs.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import threading
import time
from typing import Any
from urllib.parse import quote, urlsplit, urljoin

import requests
from .thinking import api_options, final_messages, generation_text, strict_final_text

log = logging.getLogger(__name__)
_CHAT_UNSUPPORTED = {}
_CHAT_CAPABILITY_LOCK = threading.Lock()


class WujiApiError(RuntimeError):
    def __init__(self, message, *, status_code=None, error_code="", retry_after=0.0,
                 concurrency_hint=False, task_accepted=False):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after = retry_after
        self.concurrency_hint = concurrency_hint
        self.task_accepted = task_accepted


def concurrency_rejected(exc):
    """Only an explicit pre-acceptance concurrency rejection permits resubmission."""
    codes = {"concurrency_limit", "concurrency_limit_exceeded", "concurrency_exceeded",
             "too_many_concurrent_requests", "concurrent_request_limit", "concurrency_not_supported"}
    return (isinstance(exc, WujiApiError) and not exc.task_accepted
            and exc.status_code in {400, 409, 422, 429}
            and (exc.error_code in codes or exc.concurrency_hint))


def _base(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise WujiApiError("创造API地址必须以 http:// 或 https:// 开头")
    parsed = urlsplit(base)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise WujiApiError("API地址不能包含用户名、密钥、查询参数或片段")
    return base


def _url(base_url: str, path: str) -> str:
    base = _base(base_url)
    # A supplied path is a complete API prefix (/v4, /compatible-mode/v1, ...).
    # Only a bare origin receives the conventional /v1 default.
    for suffix in ("/chat/completions", "/models"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if not urlsplit(base).path.strip("/"):
        base += "/v1"
    return f"{base}/{path.lstrip('/')}"


def _headers(api_key: str) -> dict[str, str]:
    key = str(api_key or "").strip()
    return {**({"Authorization": f"Bearer {key}"} if key else {}), "Content-Type": "application/json"}


def _multipart_headers(api_key: str) -> dict[str, str]:
    """Return auth headers without Content-Type so requests can add a boundary."""
    key = str(api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _image_file(value: str, index: int) -> tuple[str, tuple[str, bytes, str]]:
    """Convert an image data URI/raw base64 value to a multipart file tuple."""
    raw = str(value or "").strip()
    mime = "image/jpeg"
    encoded = raw
    if raw.lower().startswith("data:") and "," in raw:
        header, encoded = raw.split(",", 1)
        if ";" in header:
            mime = header[5:].split(";", 1)[0] or mime
        elif header[5:]:
            mime = header[5:]
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WujiApiError("创造API参考图像 base64 内容无效") from exc
    extension = mime.rsplit("/", 1)[-1].lower()
    if extension == "jpeg":
        extension = "jpg"
    field = "image" if index == 1 else f"image{index}"
    return field, (f"reference_{index}.{extension}", content, mime)


def _image_compatibility_error(exc: Exception) -> bool:
    """Only retry an edit request when the gateway rejected its shape."""
    message = str(exc).lower()
    return any(code in message for code in ("http 400", "http 415", "http 422"))


def _upstream_failure(exc: Exception) -> bool:
    """Recognize Creative-style errors that are safe to retry in a smaller shape."""
    message = str(exc).lower().replace("\\_", "_")
    return any(marker in message for marker in (
        "upstream_request_failed",
        "upstream provider request failed",
        "upstream_error",
    ))


def _image_encoding_failure(exc: Exception) -> bool:
    message = str(exc).lower().replace("\\_", "_")
    return _upstream_failure(exc) and any(marker in message for marker in (
        "param=image", "出错参数：image", "data uri", "data-uri", "base64",
    ))


def _public_detail(value, *api_keys: str) -> str:
    """Redact before truncating, including credentials repeated in task errors."""
    text = str(value or "")
    for key in api_keys:
        if key:
            text = text.replace(str(key), "[密钥已隐藏]")
    text = re.sub(r"data:[^\s,;]+(?:;[^,\s]*)?,[^\s\"'<>;；]+", "[媒体数据已隐藏]", text, flags=re.I)
    text = re.sub(r"https?://[^\s\"'<>;；]+", "[地址已隐藏]", text, flags=re.I)
    text = re.sub(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [密钥已隐藏]", text, flags=re.I)
    text = re.sub(r"\b(?:sk-|cr-)[A-Za-z0-9_-]{8,}", "[密钥已隐藏]", text, flags=re.I)
    return text[:1200]


def _payload_error(payload) -> str:
    """Only read structured error fields, never dump prompts/media/results."""
    values = []
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for field in ("type", "code", "param", "message", "detail"):
                value = error.get(field)
                if isinstance(value, (str, int, float)) and value not in ("", 0, False) and str(value).strip():
                    values.append(f"{field}={value}")
        elif isinstance(error, str) and error.strip():
            values.append(error.strip())
        for field in ("message", "detail", "code", "error_message"):
            value = payload.get(field)
            if isinstance(value, (str, int, float)) and value not in ("", 0, False) and str(value).strip():
                values.append(f"{field}={value}")
    return "；".join(values)


def _response_error(response: requests.Response, api_key: str) -> str:
    """Extract a short provider error without ever echoing credentials."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    # Do not echo arbitrary HTML/plain-text bodies. Besides being noisy, those
    # bodies occasionally contain authorization material from a proxy. The
    # structured JSON error envelope is sufficient for provider diagnostics.
    detail = _payload_error(payload)
    if isinstance(payload, dict) and payload.get("request_id"):
        detail += f"；请求编号={payload['request_id']}"
    return _public_detail(detail, api_key)


def _http_error(response: requests.Response, api_key: str) -> WujiApiError:
    detail = _response_error(response, api_key)
    suffix = f"：{detail}" if detail else ""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    error = payload.get("error")
    error = error if isinstance(error, dict) else {}
    code = str(error.get("code") or payload.get("code") or "").lower()
    message = str(error.get("message") or payload.get("message") or "").lower()
    concurrency_hint = bool(re.search(
        r"(?:concurren(?:t|cy).{0,60}(?:limit|exceed|not support|unsupported)|"
        r"(?:too many|maximum|limit).{0,60}concurren(?:t|cy)|"
        r"并发.{0,25}(?:上限|超限|超过|限制|不支持)|不支持.{0,15}并发)", message))
    try:
        retry_after = max(0.0, min(60.0, float(response.headers.get("Retry-After", 0))))
    except (TypeError, ValueError):
        retry_after = 0.0
    accepted = any(payload.get(field) for field in ("id", "task_id", "poll_url")) or bool(response.headers.get("Location"))
    if isinstance(payload.get("data"), dict):
        accepted = accepted or any(payload["data"].get(field) for field in ("id", "task_id", "poll_url"))
    return WujiApiError(f"创造API HTTP {response.status_code}{suffix}",
                        status_code=response.status_code, error_code=code,
                        retry_after=retry_after, concurrency_hint=concurrency_hint,
                        task_accepted=bool(accepted))


def _request(method: str, url: str, api_key: str, *, body: dict | None = None, timeout: float = 600.0) -> dict:
    try:
        response = requests.request(method, url, headers=_headers(api_key), json=body, timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        raise WujiApiError(f"创造API网络请求失败：{type(exc).__name__}；未自动重复提交") from exc
    if response.status_code >= 300:
        raise _http_error(response, api_key)
    try:
        payload = response.json()
    except ValueError:
        raise WujiApiError("创造API返回了无效 JSON") from None
    if not isinstance(payload, dict):
        return {"data": payload}
    # Async acceptance can supply the canonical query path in Location.
    # _poll_url validates its origin before forwarding authorization.
    if response.status_code == 202 and not payload.get("poll_url") and response.headers.get("Location"):
        payload["poll_url"] = response.headers["Location"]
    return payload


def _request_multipart(method: str, url: str, api_key: str, *, data: dict[str, str], files: list[tuple[str, tuple[str, bytes, str]]], timeout: float = 600.0) -> dict:
    """Submit a multipart media request without overriding requests' boundary."""
    try:
        response = requests.request(
            method,
            url,
            headers=_multipart_headers(api_key),
            data=data,
            files=files,
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise WujiApiError(f"创造API网络请求失败：{type(exc).__name__}；未自动重复提交") from exc
    if response.status_code >= 300:
        raise _http_error(response, api_key)
    try:
        payload = response.json()
    except ValueError:
        raise WujiApiError("创造API返回了无效 JSON") from None
    return payload if isinstance(payload, dict) else {"data": payload}


def _chat_fallback(messages):
    for message in reversed(messages):
        if message.get('role') != 'user':
            continue
        content = message.get('content', '')
        if isinstance(content, list):
            content = '\n'.join(part.get('text', '') for part in content
                                if isinstance(part, dict) and isinstance(part.get('text'), str))
        if isinstance(content, str) and content.strip():
            return content
    return ''


def _rejected_chat_options(exc, body):
    # Retry only an explicit request validation rejection, never accepted work,
    # transport failures, authentication errors or rate/credit limits.
    if exc.task_accepted or (exc.status_code not in {400, 422}
            and not (exc.status_code is None and re.search(r'HTTP (?:400|422)\b', str(exc)))):
        return set()
    message = str(exc).lower()
    if not re.search(r'unsupported|not support|unknown (?:parameter|argument)|unrecognized|'
                     r'unexpected (?:keyword|argument|field)|not permitted|invalid (?:value|parameter)|'
                     r'must be (?:one|in)|不支持|未知参数|无效参数', message):
        return set()
    return {field for field in ('reasoning_effort', 'enable_thinking', 'thinking', 'seed', 'temperature')
            if field in body and re.search(r'(?<![a-z_])' + field + r'(?![a-z_])', message)}


def chat(config: dict[str, Any], messages: list[dict], *, temperature: float | None = None, max_tokens: int | None = None) -> str:
    remote = config.get("remote") or {}
    body: dict[str, Any] = {"model": remote.get("text_vision") or config.get("model"), "messages": final_messages(messages), "stream": False}
    if not body["model"]:
        raise WujiApiError("创造API未选择文本/视觉模型")
    model_lower = str(body["model"]).lower()
    reasoning_model = model_lower.startswith(("gpt-5", "o1", "o3", "o4"))
    if temperature is not None and not reasoning_model:
        body["temperature"] = float(temperature)
    # The OpenAI schema/SDK expose seed as an optional integer, not a
    # per-model support guarantee. Keep the existing reasoning-model sampling
    # restrictions conservative; do not invent a seed for legacy configs.
    seed = config.get("seed")
    if not config.get("_wuji_omit_seed") and not reasoning_model and isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
        body["seed"] = seed or 1
    if max_tokens is not None:
        body["max_completion_tokens" if reasoning_model else "max_tokens"] = int(max_tokens)
    base_url = remote.get('base_url') or config.get('base_url') or ''
    body.update(api_options(base_url, body['model']))
    capability_key = (base_url, str(body['model']))
    with _CHAT_CAPABILITY_LOCK:
        unsupported = set(_CHAT_UNSUPPORTED.get(capability_key, ()))
    for field in unsupported:
        body.pop(field, None)
    while True:
        try:
            payload = _request('POST', _url(base_url, 'chat/completions'),
                remote.get('api_key') or config.get('api_key'), body=body, timeout=config.get('timeout', 600))
            break
        except WujiApiError as exc:
            rejected = _rejected_chat_options(exc, body)
            if not rejected:
                raise
            log.warning('创造API：文本接口拒绝可选参数 %s，移除后继续生成', ', '.join(sorted(rejected)))
            with _CHAT_CAPABILITY_LOCK:
                _CHAT_UNSUPPORTED.setdefault(capability_key, set()).update(rejected)
            body = {key: value for key, value in body.items() if key not in rejected}
    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message", {})
    # Separate reasoning fields and token accounting never invalidate a usable
    # final response. They are deliberately never used as generation content.
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content
                          if isinstance(part, dict) and part.get('type', 'text') in ('text', 'output_text')
                          and not part.get('thought', False)
                          and str(part.get('channel', '')).lower() not in ('analysis', 'reasoning', 'thinking'))
    result = str(content or "").strip()
    fallback = config.get('_creation_text_fallback', _chat_fallback(messages))
    cleaned = generation_text(result)
    if cleaned:
        return cleaned
    log.warning('创造API：文本回复没有可分离的最终结果，使用原始输入继续，不采用思考内容')
    return generation_text('', fallback)


def _require_generation(config, kind):
    if config.get(f"{kind}_enabled") is False:
        raise WujiApiError(f"创造API未开启{'图像' if kind == 'image' else '视频'}能力")
    if config.get("provider") == "local":
        raise WujiApiError("本地主模型配置应交给支持该架构的下游执行，不能作为 API 模型提交")


def generate_image(config: dict[str, Any], prompt: str, *, images: list[str] | None = None, size: str | None = None, extra: dict | None = None) -> dict:
    _require_generation(config, "image")
    remote = config.get("remote") or {}
    model = remote.get("image") or config.get("image_model")
    if not model:
        raise WujiApiError("创造API未选择图像模型")
    if config.get("_wuji_async_image_tasks"):
        endpoint = "images/edit-tasks" if images else "images/tasks"
    else:
        endpoint = "images/edits" if images else "images/generations"
    url = _url(remote.get("base_url") or config.get("base_url"), endpoint)
    key = remote.get("api_key", config.get("api_key"))
    if images:
        if config.get("image_edit_transport") == "json":
            body: dict[str, Any] = {"model": model, "prompt": str(prompt or ""), "image": images if len(images) > 1 else images[0]}
            if size:
                body["size"] = size
            if extra:
                body.update(extra)
            try:
                return _request("POST", url, key, body=body, timeout=config.get("timeout", 600))
            except WujiApiError as first_error:
                if concurrency_rejected(first_error):
                    raise
                # Match Creative Reasoning's proven compatibility behavior:
                # provider adapters often reject data-URI syntax or image
                # arrays while accepting the same request with raw base64 or
                # one reference image.
                if len(images) > 1 and _image_compatibility_error(first_error) and not any(
                    marker in str(first_error) for marker in ("HTTP 401", "HTTP 403")
                ):
                    fallback = dict(body)
                    fallback["image"] = images[0]
                    return _request("POST", url, key, body=fallback, timeout=config.get("timeout", 600))
                if len(images) == 1 and _image_encoding_failure(first_error):
                    fallback = dict(body)
                    value = str(images[0])
                    fallback["image"] = value.split(",", 1)[1] if value.lower().startswith("data:") and "," in value else value
                    return _request("POST", url, key, body=fallback, timeout=config.get("timeout", 600))
                raise
        # The OpenAI image edit contract is multipart/form-data. Sending a
        # data URI inside JSON is rejected by several compatible gateways,
        # including the Sunburst image-to-image route.
        data: dict[str, str] = {"model": str(model), "prompt": str(prompt or "")}
        if size:
            data["size"] = str(size)
        if extra:
            data.update({str(name): str(value) for name, value in extra.items()})
        files = [_image_file(uri, index) for index, uri in enumerate(images, 1)]
        try:
            return _request_multipart("POST", url, key, data=data, files=files, timeout=config.get("timeout", 600))
        except WujiApiError as first_error:
            if concurrency_rejected(first_error):
                raise
            # Creative Reasoning-compatible gateways may implement the same
            # endpoint with the legacy JSON image contract. A validation-level
            # rejection is safe to retry with that shape; auth, routing and
            # provider failures are never submitted twice.
            if not _image_compatibility_error(first_error):
                raise
            fallback_body: dict[str, Any] = {
                "model": model,
                "prompt": str(prompt or ""),
                "image": images if len(images) > 1 else images[0],
            }
            if size:
                fallback_body["size"] = size
            if extra:
                fallback_body.update(extra)
            return _request("POST", url, key, body=fallback_body, timeout=config.get("timeout", 600))
    body: dict[str, Any] = {"model": model, "prompt": str(prompt or "")}
    if size:
        body["size"] = size
    if extra:
        body.update(extra)
    return _request("POST", url, key, body=body, timeout=config.get("timeout", 600))


def generate_video(config: dict[str, Any], prompt: str, *, images: list[str] | None = None, videos: list[str] | None = None, audios: list[str] | None = None, duration: int | None = None, extra: dict | None = None, image_roles: list[str] | None = None, reference_roles: bool = False) -> dict:
    _require_generation(config, "video")
    remote = config.get("remote") or {}
    model = remote.get("video") or config.get("video_model")
    if not model:
        raise WujiApiError("创造API未选择视频模型")
    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    if image_roles is not None and len(image_roles) != len(images or []):
        raise WujiApiError("视频参考图与作用数量不匹配")
    for index, uri in enumerate(images or []):
        part = {"type": "image_url", "image_url": {"url": uri}}
        if image_roles is not None:
            part["role"] = image_roles[index]
        content.append(part)
    for uri in videos or []:
        content.append({"type": "video_url", "video_url": {"url": uri}, **({"role": "reference_video"} if reference_roles else {})})
    for uri in audios or []:
        content.append({"type": "input_audio", "input_audio": {"url": uri, "format": "wav"}, **({"role": "reference_audio"} if reference_roles else {})})
    body: dict[str, Any] = {"model": model, "content": content}
    if duration and int(duration) > 0:
        body["duration"] = int(duration)
    if extra:
        body.update(extra)
    url = _url(remote.get("base_url") or config.get("base_url"), "videos/generations")
    key = remote.get("api_key", config.get("api_key"))
    try:
        return _request("POST", url, key, body=body, timeout=config.get("timeout", 600))
    except WujiApiError as first_error:
        if not _upstream_failure(first_error):
            raise

        # Creative Reasoning's video endpoint accepts the same content items
        # but some provider adapters reject image role extensions.
        # Retry once with the minimal content contract before changing the
        # user's reference-image meaning.
        minimal_content = []
        for item in content:
            if item.get("type") == "text":
                minimal_content.append(item)
            elif item.get("type") == "image_url":
                minimal_content.append({"type": "image_url", "image_url": item["image_url"]})
            else:
                minimal_content.append(item)
        minimal_body = {**body, "content": minimal_content}
        try:
            return _request("POST", url, key, body=minimal_body, timeout=config.get("timeout", 600))
        except WujiApiError as minimal_error:
            # Multi-image video models frequently support only one reference
            # image. Keep the first image and retry only for that explicit
            # upstream failure; auth/model/validation errors are propagated.
            if len(images or []) <= 1:
                raise minimal_error
            kept = False
            fallback_content = []
            for item in minimal_content:
                if item.get("type") != "image_url":
                    fallback_content.append(item)
                elif not kept:
                    fallback_content.append(item)
                    kept = True
            fallback_body = {**minimal_body, "content": fallback_content}
            return _request("POST", url, key, body=fallback_body, timeout=config.get("timeout", 600))


def _poll_url(base_url: str, task: dict, resource: str) -> str:
    task_id = task.get("id") or task.get("task_id")
    target = task.get("poll_url") or _url(base_url, f"{resource}/{quote(str(task_id), safe='')}")
    target = urljoin(_url(base_url, ""), target)
    def origin(url):
        parsed = urlsplit(url)
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    parsed = urlsplit(target)
    if origin(target) != origin(base_url) or parsed.username or parsed.password or parsed.fragment:
        raise WujiApiError("拒绝向不同服务地址转发 API Key")
    return target


async def poll_media(config: dict, task: dict, kind: str, *, interval: float = 3.0) -> dict:
    """Poll once per interval without blocking Comfy's event loop or resubmitting.

    Creative async image tasks use /tasks/{id}; other providers retain their
    generation query paths. Explicit poll_url/Location takes precedence.
    Cancellation stops subsequent polling; an already submitted task is not
    automatically cancelled at the provider.
    """
    if kind not in {"image", "video"}:
        raise ValueError(kind)
    remote = config.get("remote") or {}
    keys = (remote.get("api_key"), config.get("api_key"))
    identity = "；".join(f"{label}={value}" for label, value in (
        ("任务编号", task.get("id") or task.get("task_id")),
        ("请求编号", task.get("request_id")), ("模型", task.get("model")),
    ) if value)
    deadline = time.monotonic() + float(config.get("timeout", 600))
    payload, url = task, None
    while True:
        status = str(payload.get("status") or payload.get("state") or "").strip().lower()
        detail = _payload_error(payload)
        error = payload.get("error")
        has_error = bool(error.strip()) if isinstance(error, str) else bool(
            isinstance(error, dict) and any(value not in (None, "", 0, False) for value in error.values()))
        if has_error or status in {"failed", "error", "cancelled", "canceled", "expired"}:
            ids = "；".join(f"{label}={value}" for label, value in (
                ("任务编号", payload.get("id") or payload.get("task_id") or task.get("id") or task.get("task_id")),
                ("请求编号", payload.get("request_id") or task.get("request_id")),
                ("模型", payload.get("model") or task.get("model")),
            ) if value)
            message = f"创造API{'图像' if kind == 'image' else '视频'}任务失败（{status or 'error'}）"
            message += f"；{ids}" if ids else ""
            # Keep diagnostic IDs even when upstream returns a very long message.
            message = _public_detail(message, *keys)
            message += "；" + _public_detail(detail or "服务端未提供具体原因", *keys)
            raise WujiApiError(message + "；未自动重复提交")
        if status in {"succeeded", "completed", "success", "done"} or (not status and payload.get("data")):
            return payload
        if url is None:
            if not (task.get("id") or task.get("task_id") or task.get("poll_url")):
                raise WujiApiError("创造API未返回媒体结果或任务 ID")
            resource = "tasks" if kind == "image" and config.get("_wuji_async_image_tasks") else f"{kind}s/generations"
            url = _poll_url(remote.get("base_url") or config.get("base_url"), task, resource)
        else:
            await asyncio.sleep(min(max(0.1, interval), max(0, deadline - time.monotonic())))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WujiApiError("创造API媒体任务轮询超时；" + _public_detail(identity, *keys) + "；请保留任务编号查询已有任务，未自动重复提交")
        try:
            payload = await asyncio.to_thread(_request, "GET", url, remote.get("api_key", config.get("api_key")), timeout=min(60.0, remaining))
        except WujiApiError as exc:
            raise WujiApiError(_public_detail(str(exc), *keys) + "；" + _public_detail(identity, *keys) + "；未自动重复提交") from None


def poll_video(config: dict[str, Any], task: dict, *, timeout: float = 900.0, interval: float = 3.0) -> dict:
    task_id = task.get("id") or task.get("task_id")
    if not task_id:
        return task
    remote = config.get("remote") or {}
    base_url = remote.get("base_url") or config.get("base_url")
    key = remote.get("api_key") or config.get("api_key")
    poll_url = task.get("poll_url") or _url(base_url, f"videos/generations/{quote(str(task_id), safe='')}")
    poll_url = urljoin(_base(base_url) + "/", poll_url)
    def origin(url):
        parsed = urlsplit(url)
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    if origin(poll_url) != origin(base_url) or urlsplit(poll_url).username or urlsplit(poll_url).password:
        raise WujiApiError("拒绝向不同服务地址转发 API Key")
    deadline = time.monotonic() + max(10.0, float(timeout))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WujiApiError("创造API视频任务轮询超时")
        payload = _request("GET", poll_url, key, timeout=min(60.0, remaining))
        status = str(payload.get("status") or payload.get("state") or "").lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise WujiApiError(f"视频任务未完成：{status}")
        if status in {"succeeded", "completed", "success"}:
            return payload
        if time.monotonic() >= deadline:
            raise WujiApiError("创造API视频任务轮询超时")
        time.sleep(min(max(0.1, float(interval)), max(0, deadline - time.monotonic())))

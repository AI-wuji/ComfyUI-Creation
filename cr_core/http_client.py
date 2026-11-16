"""HTTP 层：鉴权、重试退避、异步任务轮询。所有对外抛出的信息都经过脱敏。"""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp

from . import config

logger = logging.getLogger("CreativeReasoning")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# 服务端的错误信封（OpenAPI: ApiError）承诺 type / code / message / param 四个字段。
# 其中 type 取值含 upstream_error / overloaded_error / internal_error ——
# 这类是「网关或上游 provider 侧」的问题，属于瞬时故障，即便伴随 4xx 也值得重试。
RETRY_ERROR_TYPES = ("upstream_error", "overloaded_error", "internal_error")
RETRY_ERROR_CODES = (
    "upstream_request_failed",
    "upstream_error",
    "provider_unavailable",
    "no_routes_available",
    "model_service_unavailable",
    "overloaded",
    "internal_error",
)
_RETRY_CODE_FRAGMENTS = ("upstream", "provider_unavailable", "unavailable", "overloaded", "timeout")

TERMINAL_OK = {"completed", "succeeded", "success", "finished"}
TERMINAL_BAD = {"failed", "error", "canceled", "cancelled", "rejected"}
PENDING = {"pending", "processing", "queued", "running", "in_progress", "submitted"}


class CRError(RuntimeError):
    """面向用户的错误，消息已脱敏。"""


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ComfyUI-CreativeReasoning",
    }


_HINTS = {
    400: "服务端认为请求不合法。若下方「出错参数」有值，优先改那一项。",
    401: "API Key 缺失或无效：请在节点 api_key 控件填写，或设置环境变量 CR_API_KEY，或在插件目录放 config.json。",
    403: "当前 API Key 无权调用该模型，请换模型或到控制台开通权限。",
    404: "模型不存在、已下线，或该接口未开放；点节点上的「刷新模型列表」再选一次。",
    413: "请求体过大：减少图片/视频数量，或降低分辨率、缩短视频。节点已按预算压缩参考图，若仍超限请减少素材数量。",
    429: "触发限流，已自动重试仍失败，请稍后再试或降低并发。",
    500: "服务端内部错误，多为瞬时问题（已自动重试）。持续出现请稍后再试。",
    502: "上游模型服务暂时不可用（已自动重试）。多为 provider 瞬时过载，稍后重试通常即可。",
    503: "该模型当前没有可用路由（已自动重试）。可能是该模型临时下线或配额用尽。",
}

_CODE_HINTS = {
    "upstream_request_failed": (
        "上游 provider 这次没接住这次请求（多为瞬时过载、该模型临时不可用，"
        "或素材体积/规格不被上游接受）。节点已按 max_retries 自动重试。"
        "若反复出现：① 减少参考图数量或降低分辨率；② 换一个生成模型；③ 稍后再试。"
    ),
    "provider_unavailable": "该模型的上游 provider 当前不可用，稍后重试或换模型。",
    "no_routes_available": "该模型当前没有可用路由（可能临时下线或配额用尽）。",
    "model_not_found": "模型名不存在：检查「生成模型」下拉与「手动模型 ID」的拼写。",
    "model_access_denied": "当前 API Key 未开通该模型权限。",
    "invalid_api_key": "API Key 无效，请重新获取后填入控件或环境变量 CR_API_KEY。",
    "task_not_found": "任务 ID 不存在或已过期（只有提交任务的同一把 Key 能查询自己的任务）。",
}


def error_info(payload) -> dict:
    """从错误信封里取出全部可用信息。

    规范只承诺 error.type / code / message / param，其中 **param 是「出错参数」**，
    对定位 400 极有价值；旧版把 type 和 param 都丢掉了，用户只能看到一句
    「Upstream provider request failed」，无法行动。这里全部带出。
    """
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            out: dict[str, str] = {}
            for key in ("type", "code", "message", "param"):
                val = err.get(key)
                if isinstance(val, (str, int, float)) and str(val).strip():
                    out[key] = str(val).strip()
            # 规范之外服务端可能附带上游原始信息，一并带出（标量，且限长）
            for key, val in err.items():
                if key in out or isinstance(val, (dict, list)):
                    continue
                if isinstance(val, (str, int, float, bool)) and str(val).strip():
                    out[key] = str(val).strip()[:300]
            if out:
                return out
        elif isinstance(err, str) and err.strip():
            return {"message": err.strip()}
        for key in ("message", "detail"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return {"message": val.strip()[:500]}
    elif payload:
        return {"message": str(payload)[:500]}
    return {}


def _explain(status: int, payload) -> str:
    """把接口错误翻译成可操作的中文提示。"""
    ei = error_info(payload)
    etype = ei.get("type", "")
    code = ei.get("code", "")
    detail = ei.get("message", "")
    param = ei.get("param", "")

    head = [f"HTTP {status}"]
    if etype:
        head.append(etype)
    if code:
        head.append(code)
    msg = " / ".join(head)
    if detail:
        msg += f"：{detail}"
    if param:
        msg += f"\n出错参数：{param}"
    extras = {k: v for k, v in ei.items() if k not in ("type", "code", "message", "param")}
    if extras:
        msg += "\n上游附加信息：" + "；".join(f"{k}={v}" for k, v in extras.items())

    detail_lower = detail.lower()
    if "overdraft limit exceeded" in detail_lower or "insufficient balance" in detail_lower:
        hint = (
            "当前账户余额或授信额度已用尽，服务端因此拒绝调用该模型。"
            "请到 Creative Reasoning 账户后台充值或恢复可用额度后重试；"
            "更换元指令、提示词或文本模型无法解决此错误。"
        )
    else:
        hint = _CODE_HINTS.get(code) or _HINTS.get(status)
    if hint:
        msg += f"\n提示：{hint}"
    return config.scrub(msg)


def is_retryable(status: int, payload) -> bool:
    """是否值得重试。

    除常规瞬时状态码外，**上游侧错误**（type=upstream_error/overloaded_error/internal_error，
    或 code 含 upstream/provider_unavailable…）即便伴随 400 也应重试：
    语义上这是「provider 这次没接住」，而不是我们的请求写错了。
    """
    if status in RETRY_STATUS:
        return True
    if status not in range(400, 500):  # 2xx/3xx 不走这里；5xx 已由 RETRY_STATUS 覆盖
        return False
    ei = error_info(payload)
    etype = (ei.get("type") or "").lower()
    code = (ei.get("code") or "").lower()
    if any(t in etype for t in RETRY_ERROR_TYPES):
        return True
    if code in RETRY_ERROR_CODES:
        return True
    return any(frag in code for frag in _RETRY_CODE_FRAGMENTS)


async def _sleep_backoff(attempt: int, retry_after: str | None, base: float = 1.5) -> None:
    wait = base * (2**attempt)
    if retry_after:
        try:
            wait = max(wait, min(float(retry_after), 60.0))
        except (TypeError, ValueError):
            pass
    await asyncio.sleep(min(wait, 60.0))


def _encode_body(body: dict) -> bytes:
    """把请求体序列化成 UTF-8 JSON 字节。

    刻意不用 aiohttp 的 ``json=``：它内部走 ``json.dumps`` 的默认
    ``ensure_ascii=True``，会把中文转义成 ``\\uXXXX``。服务端有字段（如 ``model``）
    是「借用字符串」，遇到转义串会直接 400：
        invalid type: string "…", expected a borrowed string
    显式下发原始 UTF-8（RFC 8259 规定 JSON 即 UTF-8），兼容面最广。
    """
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


async def request_json(
    method: str,
    path: str,
    api_key: str,
    *,
    body: dict | None = None,
    timeout: float = 300.0,
    max_retries: int = 2,
) -> dict:
    if not api_key:
        raise CRError(
            "未找到 API Key。请在节点 api_key 控件填写，或设置环境变量 CR_API_KEY，"
            "或在插件目录创建 config.json（内容形如 {\"api_key\": \"...\"}）。"
        )

    # 异步任务响应可能返回服务端指定的 poll_url；保留相对路径和完整 URL 两种形态。
    url = path if str(path).lower().startswith(("http://", "https://")) else config.endpoint(path)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    last_err: str = ""

    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                kwargs: dict = {"headers": _headers(api_key)}
                if body is not None:
                    kwargs["data"] = _encode_body(body)
                async with session.request(method.upper(), url, **kwargs) as resp:
                    text = await resp.text()
                    try:
                        payload = json.loads(text) if text else {}
                    except json.JSONDecodeError:
                        payload = {"message": text[:2000]}

                    if resp.status < 300:
                        if not isinstance(payload, dict):
                            raise CRError(config.scrub(f"接口返回了非预期结构：{type(payload).__name__}"))
                        return payload

                    if is_retryable(resp.status, payload) and attempt < max_retries:
                        last_err = _explain(resp.status, payload)
                        logger.warning(
                            "[CreativeReasoning] 请求失败将重试（第 %d/%d 次）: %s",
                            attempt + 1, int(max_retries), last_err.splitlines()[0],
                        )
                        await _sleep_backoff(attempt, resp.headers.get("Retry-After"))
                        continue

                    explained = _explain(resp.status, payload)
                    logger.error(
                        "[CreativeReasoning] HTTP %s 请求失败：%s",
                        resp.status, " | ".join(explained.splitlines()),
                    )
                    raise CRError(explained)

        except asyncio.TimeoutError:
            last_err = f"请求超时（{timeout:.0f}s）。可调大 timeout，或换更快的模型/更短的视频。"
            if attempt < max_retries:
                await _sleep_backoff(attempt, None)
                continue
            raise CRError(last_err) from None
        except (aiohttp.ClientError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            last_err = config.scrub(f"网络错误：{exc}")
            if attempt < max_retries:
                await _sleep_backoff(attempt, None)
                continue
            raise CRError(last_err) from None

    raise CRError(last_err or "请求失败")


def task_status(payload: dict) -> str:
    return str(payload.get("status") or "").strip().lower()


def is_done(payload: dict) -> bool:
    st = task_status(payload)
    if st in TERMINAL_OK:
        return True
    # 同步响应：有 data 而无 status
    return not st and bool(payload.get("data"))


def _looks_not_found(exc: Exception) -> bool:
    """任务 GET 返回 404/410 时，多半是轮询路径与任务实际挂载点不一致。"""
    text = str(exc)
    return any(code in text for code in ("404", "410", "Not Found", "not found", "not_found"))


async def poll_task(
    resource: str,
    task_id: str,
    api_key: str,
    *,
    poll_interval: float = 3.0,
    timeout: float = 600.0,
    max_retries: int = 2,
    on_progress=None,
    poll_url: str | None = None,
) -> dict:
    """轮询异步任务到终态。

    resource 为 GET 路径前缀，例如 'images/edits' / 'images/generations' / 'videos/generations'，
    最终请求为 {resource}/{task_id}。

    图像任务可能挂在 edits 或 generations 路径下（取决于后端把多图编辑判成同步还是异步），
    因此图像类任务在主路径返回 404 时会自动尝试另一个前缀，避免「任务未找到」误报。
    """
    if not task_id:
        raise CRError("接口未返回任务 ID，无法轮询结果。")

    from urllib.parse import quote

    primary = poll_url or f"{resource}/{quote(str(task_id), safe='')}"
    alternates = []
    if resource == "images/edits":
        alternates = ["images/generations"]
    elif resource == "images/generations":
        alternates = ["images/edits"]
    alt_paths = [f"{a}/{quote(str(task_id), safe='')}" for a in alternates]

    deadline = asyncio.get_event_loop().time() + max(10.0, timeout)
    interval = max(1.0, float(poll_interval))
    # 长视频任务通常需要数分钟；逐秒轮询会制造大量 HTTP 连接和日志，
    # 在 Windows 上还可能让 ComfyUI 的 websocket 管道变得不稳定。
    max_interval = max(interval, 15.0)
    waited = 0.0
    used_path = primary
    tried_alternate = False

    while True:
        try:
            payload = await request_json(
                "GET", used_path, api_key, timeout=min(60.0, timeout), max_retries=max_retries
            )
        except CRError as exc:
            if alt_paths and not tried_alternate and _looks_not_found(exc):
                tried_alternate = True
                used_path = alt_paths[0]
                logger.info(
                    "[CreativeReasoning] 任务轮询路径 %s 未找到，自动改用 %s",
                    primary, used_path,
                )
                continue
            raise

        st = task_status(payload)

        if is_done(payload):
            return payload
        if st in TERMINAL_BAD:
            err = payload.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            if any(word in str(msg).lower() for word in ("copyright", "trademark")):
                raise CRError("视频任务被平台内容审核拒绝：提示词或参考素材可能涉及受版权或商标保护的内容。请改用原创人物、原创品牌和自有素材后重试。")
            raise CRError(config.scrub(f"任务失败（{st}）：{msg or '接口未提供原因'}"))
        if st and st not in PENDING:
            logger.debug("[CreativeReasoning] 未知任务状态，继续轮询: %s", st)

        if asyncio.get_event_loop().time() >= deadline:
            raise CRError(
                f"等待任务结果超时（{timeout:.0f}s，任务仍为 {st or '未知'} 状态）。"
                "任务可能仍在服务端运行，调大 timeout 后重试即可，请勿重复提交以免重复计费。"
            )

        if on_progress:
            waited += interval
            on_progress(st or "pending", waited)
        await asyncio.sleep(interval)
        interval = min(max_interval, interval * 1.15)


async def download_bytes(url: str, *, timeout: float = 300.0, max_retries: int = 2) -> bytes:
    """下载结果资源。结果 URL 可能过期，因此拿到即存。"""
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 300:
                        if resp.status in RETRY_STATUS and attempt < max_retries:
                            await _sleep_backoff(attempt, resp.headers.get("Retry-After"))
                            continue
                        raise CRError(config.scrub(f"下载结果失败：HTTP {resp.status}"))
                    return await resp.read()
        except asyncio.TimeoutError:
            if attempt < max_retries:
                await _sleep_backoff(attempt, None)
                continue
            raise CRError("下载结果超时，资源链接可能已过期。") from None
        except (aiohttp.ClientError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            if attempt < max_retries:
                await _sleep_backoff(attempt, None)
                continue
            raise CRError(config.scrub(f"下载结果出错：{exc}")) from None
    raise CRError("下载结果失败")

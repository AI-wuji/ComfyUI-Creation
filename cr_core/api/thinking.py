"""Non-thinking policy, scoped to WujiAPI (never patch llama-cpp globally)."""
import inspect
import logging
import re
from urllib.parse import urlsplit

log = logging.getLogger(__name__)
THINKING_FLAGS = ('enable_thinking', 'preserve_thinking', 'force_reasoning')

FINAL_ONLY = (
    "Return only the final result in the requested language and format. "
    "Do not output thinking, analysis, planning, self-checks, or reasoning tags. "
    "Do not discuss these instructions."
)


def final_messages(messages):
    # Do not mutate caller messages or rebuild image/video content arrays.
    result = [dict(message) for message in messages]
    for message in result:
        if message.get('role') == 'system' and isinstance(message.get('content'), str):
            if FINAL_ONLY not in message['content']:
                message['content'] += '\n\n' + FINAL_ONLY
            break
    else:
        result.insert(0, {'role': 'system', 'content': FINAL_ONLY})
    return result


def _legacy_api_options(base_url, model):
    """Use documented switches; arbitrary OpenAI-compatible servers differ.

    Model names alone do not tell us whether Qwen is served by DashScope,
    vLLM, llama-server, or a proxy. Never send all their flags at once.
    """
    host = (urlsplit(base_url).hostname or '').lower()
    name = str(model).lower()
    if host == 'api.deepseek.com' and name.startswith('deepseek-'):
        return {'thinking': {'type': 'disabled'}}
    if host in {'dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com', 'dashscope-us.aliyuncs.com'}:
        if ('thinking' not in name and 'omni' not in name
                and name.startswith(('qwen3', 'qwen-plus', 'qwen-flash', 'qwen-turbo'))):
            return {'enable_thinking': False}
    if host in {'open.bigmodel.cn', 'api.z.ai'}:
        if re.match(r'glm-(?:4\.[567]|5(?:\.[012])?)(?:-|$)', name):
            return {'thinking': {'type': 'disabled'}}
    if host == 'generativelanguage.googleapis.com':
        if name.startswith('gemini-2.5-flash'):
            return {'reasoning_effort': 'none'}
    # Standard OpenAI field also used by compatible GPT gateways. Pro and
    # Codex variants have different effort support; do not guess their flags.
    if not any(word in name for word in ('pro', 'codex', 'chat')):
        if re.match(r'gpt-5\.(?:1|2|4)(?:-|$)', name):
            return {'reasoning_effort': 'none'}
        if re.match(r'gpt-5(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?$', name):
            log.warning('无极API：此 GPT-5 型号只支持 minimal，不能完全关闭思考；请换用支持 none 的型号')
            return {'reasoning_effort': 'minimal'}  # This generation cannot disable it.
        if re.match(r'(?:o1|o3|o4)(?:-mini)?(?:-\d{4}-\d{2}-\d{2})?$', name) and name != 'o1-mini':
            log.warning('无极API：此推理型号只支持 low，不能完全关闭思考；请换用非思考型号')
            return {'reasoning_effort': 'low'}
    log.warning('无极API：此接口/型号没有已适配的关闭思考参数；最终结果指令和文本过滤不代表服务端已关闭思考')
    return {}



# Public documentation checked 2026-11-16; these are closed sets, not model
# name prefixes. Future snapshots/aliases must be checked before adding them.
# https://developers.openai.com/api/docs/models/gpt-5.1
# https://developers.openai.com/api/docs/models/gpt-5.2
# https://developers.openai.com/api/docs/models/gpt-5.4
_OPENAI_NONE = frozenset({
    'gpt-5.1', 'gpt-5.1-2025-11-13',
    'gpt-5.2', 'gpt-5.2-2025-12-11',
    'gpt-5.4', 'gpt-5.4-2026-03-05',
})
# https://developers.openai.com/api/docs/models/gpt-4o
_OPENAI_NONTHINKING = frozenset({
    'gpt-4o',
    'gpt-4.1',
})
# https://api-docs.deepseek.com/guides/thinking_mode
# https://api-docs.deepseek.com/quick_start/pricing
_DEEPSEEK_TOGGLE = frozenset({
    'deepseek-flash', 'deepseek-v4-pro',
    'deepseek-v4-flash', 'deepseek-v4-flash-vision-exp',
})
# https://help.aliyun.com/zh/model-studio/deep-thinking
# Only explicitly documented hybrid models; thinking-only and new aliases
# are rejected rather than guessed from qwen3/qwen-plus name prefixes.
_QWEN_TOGGLE = frozenset({
    'qwen-plus', 'qwen-plus-latest', 'qwen-plus-2025-04-28',
    'qwen-flash', 'qwen-flash-2025-07-28',
    'qwen3-32b',
    'qwen3.5-plus', 'qwen3.6-plus',
    'qwen3.7-max', 'qwen3.7-max-2026-05-20', 'qwen3.7-max-2026-06-08',
    'qwen3.7-plus', 'qwen3.7-plus-2026-05-26',
    'qwen3.7-flash', 'qwen3.7-flash-2026-07-15',
    'qwen3.8-max', 'qwen3.8-max-0902', 'qwen3.8-flash', 'qwen3.8-27b',
    'qwen4-max', 'qwen4-max-preview',
})
# https://docs.bigmodel.cn/cn/guide/capabilities/thinking
# https://docs.z.ai/guides/capabilities/thinking
# Both document disabled; GLM-5.3 explicitly rejects it. Keep model-specific
# variants out unless their own documentation confirms the same contract.
_GLM_TOGGLE = frozenset({'glm-4.5', 'glm-4.6', 'glm-5', 'glm-5.1', 'glm-5.2'})


def strict_api_options(base_url, model):
    """Build documented non-thinking Chat JSON fields, or raise ValueError.

    No requests, credentials, model substitution, reduced-effort fallback or
    text filtering occurs here. Merge the result into the raw JSON body before
    sending; provider extensions are top-level fields, not an extra_body key.
    The server must honor or reject the requested switch. A successful local
    check does not constitute a live guarantee about a remote model route.
    """
    try:
        url = urlsplit(str(base_url or ''))
        host = (url.hostname or '').lower()
        valid_origin = (url.scheme == 'https' and not url.username and not url.password
                        and not url.query and not url.fragment and url.port in (None, 443))
    except ValueError:
        raise ValueError('严格关闭思考失败：服务地址无效') from None
    name = str(model or '').strip().lower()
    if not valid_origin or not name:
        raise ValueError('严格关闭思考失败：需要有效的 HTTPS 服务地址和模型名')

    if host == 'api.creative-reasoning.com':
        # Official ChatCompletionRequest.reasoning_effort -> ReasoningEffort:
        # none requests disabled reasoning; selected models may support only
        # a subset. This is the only gateway disable field we send.
        # https://creative-reasoning.com/docs-openapi/dataplane.en.json
        # The gateway documents none as a request-level value and returns 400
        # when the selected model does not support it. Never retry without it.
        if name in _OPENAI_NONTHINKING:
            return {}
        if name.startswith(('creative-reasoning-image-', 'creative-reasoning-video-')):
            raise ValueError(f'严格关闭思考失败：型号 {model!r} 不是文本聊天型号')
        return {'reasoning_effort': 'none'}
    elif host == 'api.openai.com':
        if name in _OPENAI_NONE:
            return {'reasoning_effort': 'none'}
        if name in _OPENAI_NONTHINKING:
            return {}
    elif host == 'api.deepseek.com':
        if name in _DEEPSEEK_TOGGLE:
            return {'thinking': {'type': 'disabled'}}
    elif host in {'dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com',
                  'dashscope-us.aliyuncs.com'}:
        if name in _QWEN_TOGGLE:
            return {'enable_thinking': False}
    elif host in {'open.bigmodel.cn', 'api.z.ai'}:
        if name in _GLM_TOGGLE:
            return {'thinking': {'type': 'disabled'}}
    elif host == 'generativelanguage.googleapis.com':
        # https://ai.google.dev/gemini-api/docs/openai#thinking
        # none maps to zero budget for 2.5 Flash/Flash-Lite. 2.5 Pro,
        # Gemini 3 and Gemini 4 cannot disable thinking.
        if name in {'gemini-2.5-flash', 'gemini-2.5-flash-lite'}:
            return {'reasoning_effort': 'none'}
    # Do not echo base_url: a malformed user URL could contain credentials.
    raise ValueError(
        f'严格关闭思考失败：服务 {host or "unknown"} / 型号 {model!r} '
        '未知、仅支持思考或缺少明确关闭依据；已拒绝请求，不会降为 minimal/low'
    )


def api_options(base_url, model):
    """Try a known disable switch without blocking unsupported model routes.

    The caller may remove explicitly rejected optional parameters and retry.
    Never substitute models or mistake final-text filtering for server control.
    """
    try:
        return strict_api_options(base_url, model)
    except ValueError:
        return {}


_THOUGHT_TAGS = (
    ('<think>', '</think>'), ('<thinking>', '</thinking>'),
    ('<analysis>', '</analysis>'), ('<reasoning>', '</reasoning>'),
    ('<thought>', '</thought>'), ('<reasoning_content>', '</reasoning_content>'),
    ('<|think|>', '<|/think|>'), ('<|analysis|>', '<|/analysis|>'),
    ('<|reasoning|>', '<|/reasoning|>'), ('<|thought|>', '<|/thought|>'),
    ('<|begin_of_thought|>', '<|end_of_thought|>'),
    ('<\uFF5Cbegin\u2581of\u2581thinking\uFF5C>', '<\uFF5Cend\u2581of\u2581thinking\uFF5C>'),
)
_THOUGHT_TOKEN = re.compile('|'.join(
    re.escape(token) for pair in _THOUGHT_TAGS for token in pair
), re.I)
_THOUGHT_PREFIX = re.compile(
    r'(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?'
    r'(?:\[\s*(?:analysis|reasoning|thought|thinking)\s*\]|'
    r'(?:analysis|reasoning|thought|thinking|chain[ -]of[ -]thought|'
    r'思考过程|推理过程|分析过程|思考|推理|分析)\s*[:：]|'
    r'(?:analysis|reasoning|thoughts?|thinking|思考过程|推理过程|分析过程)\s*(?:\*\*)?\s*$|'
    r"(?:let me (?:think|analy[sz]e)|let['’]s think|I need to (?:think|analy[sz]e)|"
    r'we need to|the user (?:wants|asks|is asking)|让我(?:先)?(?:思考|分析)|我需要先))',
    re.I | re.M,
)


def strict_final_text(value):
    """Strip balanced explicit thought blocks; reject ambiguous prompt text.

    This cannot semantically identify every possible thought in plain prose.
    Known analysis markers, malformed tags, channel tokens and thought=true
    metadata fail closed. Callers must discard reasoning_content and typed
    thought blocks before passing ordinary content here, and validate planner
    JSON and each image/video prompt separately. Never fall back to raw text
    after ValueError. Error messages intentionally do not repeat the content.
    """
    if isinstance(value, dict):
        if value.get('thought') not in (None, False):
            raise ValueError('严格文本校验失败：content 标记为思考内容')
        if str(value.get('type', '')).lower() in {
            'analysis', 'reasoning', 'thinking', 'thought', 'reasoning_content',
        } or str(value.get('channel', '')).lower() in {'analysis', 'reasoning'}:
            raise ValueError('严格文本校验失败：content 类型或通道为思考内容')
        value = value.get('text', value.get('content'))
    if not isinstance(value, str) or not value.strip():
        raise ValueError('严格文本校验失败：最终文本为空或类型无效')

    # A stack validates nesting before removing entire outer thought blocks.
    # Removing each pair with regex would miss crossed or malformed nesting.
    closing_for = dict(_THOUGHT_TAGS)
    stack, pieces, previous = [], [], 0
    for token in _THOUGHT_TOKEN.finditer(value):
        marker = token.group().lower()
        if marker in closing_for:
            if not stack:
                pieces.append(value[previous:token.start()])
            stack.append(closing_for[marker])
        elif not stack or marker != stack.pop():
            raise ValueError('严格文本校验失败：思考标签缺失或交叉，无法可靠分离')
        if not stack:
            previous = token.end()
    if stack:
        raise ValueError('严格文本校验失败：思考标签未闭合')
    pieces.append(value[previous:])
    cleaned = ''.join(pieces).strip()
    if not cleaned:
        raise ValueError('严格文本校验失败：移除思考内容后没有最终文本')
    if re.search(r'</?\s*(?:think(?:ing)?|analysis|reasoning(?:_content)?|thought)\b', cleaned, re.I):
        raise ValueError('严格文本校验失败：存在无法可靠分离的思考标签')
    if re.search(r'<[|\uFF5C]|[|\uFF5C]>', cleaned):
        raise ValueError('严格文本校验失败：存在无法可靠分离的特殊通道标记')
    if _THOUGHT_PREFIX.search(cleaned):
        raise ValueError('严格文本校验失败：正文包含疑似分析/思考前缀')
    if re.search(r"""(?i)["'](?:thought|is_thinking|thinking)["']\s*[:=]\s*(?:true|1)\b""", cleaned):
        raise ValueError('严格文本校验失败：内容标记 thought=true')
    return cleaned


def generation_text(value, fallback=""):
    """Recover final text, otherwise use clean caller input without raising.

    Never return an ambiguous thinking response just to keep generation alive.
    The fallback comes from the original user request, not reasoning fields.
    """
    if isinstance(value, str):
        candidate = value.strip()
        # Explicit final channels/headings separate a final from prior analysis.
        final_marker = re.compile(
            r'<\|(?:start_header_id|channel|meta_sep)\|>\s*(?:final|answer)(?:<\|end_header_id\|>)?\s*'
            r'|(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?'
            r'(?:final(?:\s+(?:answer|result|prompt))?|最终(?:答案|结果|提示词)|正文)'
            r'\s*(?:\*\*)?\s*[:：]\s*', re.I)
        markers = list(final_marker.finditer(candidate))
        if markers:
            candidate = candidate[markers[-1].end():].strip()
        # Some completion templates already supply the opening think token.
        for opening, closing in _THOUGHT_TAGS:
            if closing.lower() in candidate.lower() and opening.lower() not in candidate.lower():
                candidate = re.split(re.escape(closing), candidate, flags=re.I)[-1].strip()
        try:
            return strict_final_text(candidate)
        except ValueError:
            pass
    try:
        return strict_final_text(fallback)
    except ValueError:
        return ""


def final_text(value):
    """Remove explicitly delimited leading reasoning, not arbitrary prose.

    If no final answer can be recovered, preserve the response. Cleanup must
    not create a new empty-result error or append reports to generated text.
    """
    original = str(value or '').strip()
    cleaned = original
    # A template may already contain the opening token, so the completion
    # starts inside a think block and includes only its closing token.
    for opening, closing in (('<think>', '</think>'), ('<analysis>', '</analysis>'),
                             ('<|think|>', '<|/think|>')):
        while cleaned.lower().startswith(opening):
            end = cleaned.lower().find(closing, len(opening))
            if end < 0:
                break
            cleaned = cleaned[end + len(closing):].strip()
        if closing in cleaned.lower() and opening not in cleaned.lower():
            prefix, suffix = re.split(re.escape(closing), cleaned, maxsplit=1, flags=re.I)
            # JSON/code may contain literal tags. Do not cut such outputs.
            if not prefix.lstrip().startswith(('{', '[', '```')):
                cleaned = suffix.strip()
    return cleaned or original


def make_vision_handler(handler_class, projection_path):
    try:
        parameters = inspect.signature(handler_class).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs = {}
    # llama-cpp has used both names over time. GenericMTMDChatHandler also
    # requires an explicit chat_format; None lets it resolve the GGUF template
    # from the loaded model at call time. Keep clip_model_path for older and
    # model-specific handlers that accept the compatibility alias.
    if 'mmproj_path' in parameters:
        kwargs['mmproj_path'] = projection_path
        if 'chat_format' in parameters:
            kwargs['chat_format'] = None
    else:
        kwargs['clip_model_path'] = projection_path
    if 'verbose' in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs['verbose'] = False
    for flag in THINKING_FLAGS:
        if flag in parameters:
            kwargs[flag] = False
    handler = handler_class(**kwargs)
    disable_handler_thinking(handler)
    return handler


def disable_handler_thinking(handler):
    for flag in THINKING_FLAGS:
        if hasattr(handler, flag):
            setattr(handler, flag, False)
    arguments = getattr(handler, 'extra_template_arguments', None)
    if isinstance(arguments, dict):
        arguments.update(enable_thinking=False, preserve_thinking=False, force_reasoning=False)


def configure_local_thinking(model):
    handler = getattr(model, 'chat_handler', None)
    if handler is not None:
        disable_handler_thinking(handler)
        return
    metadata = getattr(model, 'metadata', None)
    if not isinstance(metadata, dict):
        return
    name = getattr(model, 'chat_format', None)
    template = metadata.get('tokenizer.' + name) if isinstance(name, str) else None
    template = template or metadata.get('tokenizer.chat_template')
    if not isinstance(template, str) or not template.strip():
        log.warning('无极API：模型未提供可配置的 GGUF 对话模板，无法确认思考已关闭')
        return
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        tokens, ids = {}, {}
        for key in ('eos', 'bos', 'eot', 'sep', 'nl', 'pad', 'mask'):
            getter = getattr(model, 'token_' + key, None)
            token_id = getter() if callable(getter) else -1
            if isinstance(token_id, int) and token_id >= 0:
                ids[key] = token_id
                tokens[key + '_token'] = model.detokenize([token_id], special=True).decode('utf-8', errors='replace')
        kwargs = dict(
            template=''.join('{% set ' + flag + ' = false %}' for flag in THINKING_FLAGS) + template,
            eos_token=tokens.get('eos_token', ''), bos_token=tokens.get('bos_token', ''),
        )
        parameters = inspect.signature(Jinja2ChatFormatter).parameters
        if 'stop_token_ids' in parameters:
            kwargs['stop_token_ids'] = list(dict.fromkeys(ids[k] for k in ('eos', 'eot') if k in ids)) or None
        if 'special_tokens_map' in parameters:
            kwargs['special_tokens_map'] = tokens
        formatter = Jinja2ChatFormatter(**kwargs)
        model.chat_handler = formatter.to_chat_handler()
        # This exact formatter also lets prompt callers reserve output space;
        # never estimate tokens using a different template.
        model._wuji_chat_formatter = formatter
    except Exception as exc:
        # Keep generation and downstream quality retries available even when
        # an older runtime cannot configure this model's template.
        log.warning('无极API：非思考模板适配不可用（%s），兼容原处理器继续生成；无法确认思考已关闭', type(exc).__name__)

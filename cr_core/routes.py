"""Model-list refresh route for the standalone Creation API node."""
import asyncio

from aiohttp import web

from .api import client


def register_routes():
    from server import PromptServer

    server = PromptServer.instance
    if getattr(server, "_creation_api_routes_registered", False):
        return

    @server.routes.post("/creation_api/models")
    async def models(request):
        try:
            body = await request.json()
            base_url = str(body.get("base_url") or "").strip()
            api_key = str(body.get("api_key") or "")
            payload = await asyncio.to_thread(
                client._request, "GET", client._url(base_url, "models"), api_key,
                timeout=20,
            )
            values = payload.get("data", payload.get("models", []))
            result = []
            for item in values if isinstance(values, list) else []:
                value = item.get("id") if isinstance(item, dict) else item
                if isinstance(value, str) and value.strip():
                    result.append(value.strip())
            return web.json_response({"models": list(dict.fromkeys(result))})
        except Exception:
            return web.json_response({"error": "模型列表读取失败，请检查服务地址、密钥和 /models 接口。"}, status=400)

    server._creation_api_routes_registered = True

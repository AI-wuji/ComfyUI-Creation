"""Standalone 创造 ComfyUI nodes."""
from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .cr_core.creation_nodes import CreationAPI, CreationImage, CreationText, CreationVideo

WEB_DIRECTORY = "./web"
NODE_DISPLAY_NAME_MAPPINGS = {
    "CreationAPI": "创造API",
    "CreationText": "创造文本扩写",
    "CreationImage": "创造图像生成",
    "CreationVideo": "创造视频生成",
}


class CreationExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [CreationAPI, CreationImage, CreationText, CreationVideo]


async def comfy_entrypoint():
    from .cr_core.routes import register_routes
    register_routes()
    return CreationExtension()


__all__ = ["WEB_DIRECTORY", "NODE_DISPLAY_NAME_MAPPINGS", "CreationExtension", "comfy_entrypoint"]

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const presets = {
    "创意推理": "https://api.creative-reasoning.com/unified/v1",
    "自定义（OpenAI兼容）": "",
    OpenAI: "https://api.openai.com/v1",
    DeepSeek: "https://api.deepseek.com/v1",
    "通义千问（兼容接口）": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "智谱（兼容接口）": "https://open.bigmodel.cn/api/paas/v4",
    "Gemini（兼容接口）": "https://generativelanguage.googleapis.com/v1beta/openai",
};
const modelFields = ["API文本视觉模型", "API图像模型", "API视频模型"];
const get = (node, name) => node.widgets?.find(widget => widget.name === name);
const empty = name => name === "API文本视觉模型" ? "" : "未启用";
const normalizeSeed = value => Math.max(1, Math.min(2147483647, Math.trunc(Number(value) || 1)));
const seedLabels = { fixed: "固定", increment: "递增", decrement: "递减", randomize: "随机" };
const titles = { CreationAPI: "创造API", CreationImage: "创造图像生成", CreationText: "创造文本扩写", CreationVideo: "创造视频生成" };
const seedLabel = value => seedLabels[value] ?? String(value ?? "");
const fieldLabels = {
    meta_instruction: "元指令", prompt: "提示词", image_role_mode: "图像作用",
    aspect_ratio: "画面比例", resolution: "分辨率", duration: "时长（秒）",
    seed: "随机种子", temperature: "温度覆盖", control_after_generate: "种子控制（运行后）",
};
const nodeName = node => node.comfyClass || node.constructor?.comfyClass || node.type;
const inputName = (name, type) => {
    if (typeof name !== "string") return name;
    const [head, ...tail] = name.split(".");
    const label = head === "prompt" && type === "CreationText" ? "文本内容"
        : head === "images" ? (type === "CreationImage" ? "参考图" : "图像输入")
        : { videos: "视频输入", audios: "音频输入", ...fieldLabels }[head] || head;
    // The generated seed control is a frontend widget, not a backend input.
    const mediaGroup = ["images", "videos", "audios", "参考图", "图像输入", "视频输入", "音频输入"].includes(head);
    const slotName = part => part.replace(/^(image|video|audio)_(\d+)$/, (_, kind, index) =>
        `${{ image: type === "CreationImage" ? "参考图" : "图像", video: "视频", audio: "音频" }[kind]}_${index}`);
    return [head === "control_after_generate" ? head : label, ...tail.map(part => mediaGroup ? slotName(part) : part)].join(".");
};
function migrateNode(data, type = data?.type) {
    if (!data || !Object.hasOwn(titles, type)) return;
    for (const input of data.inputs || []) {
        input.name = inputName(input.name, type);
        if (input.widget) input.widget.name = inputName(input.widget.name, type);
    }
    if (data.widgets_values && !Array.isArray(data.widgets_values)) {
        for (const key of Object.keys(data.widgets_values)) {
            const name = inputName(key, type);
            if (name !== key) {
                data.widgets_values[name] = data.widgets_values[key];
                delete data.widgets_values[key];
            }
        }
    }
    if (!data.title || data.title === type || ["创建", "创世"].some(prefix => data.title === titles[type].replace("创造", prefix))) {
        data.title = titles[type];
    }
}

function asCombo(node, name) {
    const current = get(node, name);
    if (!current || current.type === "combo") return current;
    const index = node.widgets.indexOf(current);
    const combo = node.addWidget("combo", name, current.value || empty(name), () => {}, {
        values: [empty(name), current.value || empty(name)],
    });
    node.widgets.pop();
    current.onRemove?.();
    current.inputEl?.remove();
    node.widgets[index] = combo;
    return combo;
}

function seedControl(node) {
    return get(node, "随机种子")?.linkedWidgets?.[0] || get(node, "seed")?.linkedWidgets?.[0]
        || get(node, "control_after_generate");
}

function localizeSeed(node) {
    const seed = get(node, "随机种子") || get(node, "seed");
    if (seed && seed.value !== normalizeSeed(seed.value)) seed.value = normalizeSeed(seed.value);
    const control = seedControl(node);
    if (!control) return;
    control.label = control.localized_name = "种子控制（运行后）";
    control.options ??= {};
    control.options.getOptionLabel = control.options.formatter = seedLabel;
    control.tooltip = "固定：保持种子；递增／递减：按顺序更新；随机：随机更新。";
}

function localize(node) {
    const type = nodeName(node);
    migrateNode(node, type);
    for (const widget of node.widgets || []) {
        const label = widget.name === "control_after_generate" ? fieldLabels[widget.name] : inputName(widget.name, type);
        if (!label || (label === widget.name && !/[\u4e00-\u9fff]/.test(label))) continue;
        widget.label = widget.localized_name = label;
        widget.options ??= {};
        widget.options.label = label;
    }
    localizeSeed(node);
    node.setDirtyCanvas?.(true, true);
}

function updateChoices(widget, values) {
    widget.options ??= {};
    widget.options.values = [...new Set([empty(widget.name), widget.value || empty(widget.name), ...values])];
}

async function refreshModels(node, button) {
    button.label = "正在刷新模型列表…";
    try {
        const response = await api.fetchApi("/creation_api/models", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ base_url: get(node, "API地址")?.value, api_key: get(node, "API密钥")?.value }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "模型列表读取失败");
        for (const name of modelFields) updateChoices(get(node, name), result.models || []);
        button.label = "模型列表已刷新";
    } catch (error) {
        button.label = "刷新失败（可重试）";
        app.ui?.dialog?.show(String(error.message));
    }
}

app.registerExtension({
    name: "Creation.Nodes",
    beforeConfigureGraph(graphData) {
        for (const graph of [graphData, ...(graphData?.definitions?.subgraphs || [])]) {
            for (const node of graph?.nodes || []) migrateNode(node);
        }
    },
    nodeCreated(node) {
        const name = nodeName(node);
        if (!Object.hasOwn(titles, name)) return;
        localize(node);
        if (name !== "CreationAPI" || node.__creationControlsReady) return;
        node.__creationControlsReady = true;
        for (const name of modelFields) asCombo(node, name);
        const refresh = node.addWidget("button", "刷新模型列表", null, () => refreshModels(node, refresh), { serialize: false });
        node.addWidget("button", "手动填写模型 ID", null, () => {
            const dialog = document.createElement("dialog");
            dialog.style.cssText = "padding:20px;max-width:560px;width:85vw;background:var(--comfy-menu-bg,#222);color:var(--input-text,#eee);border:1px solid #666;";
            const inputs = [];
            for (const name of modelFields) {
                const label = document.createElement("label");
                label.textContent = name;
                label.style.display = "block";
                const input = document.createElement("input");
                input.value = get(node, name)?.value === "未启用" ? "" : get(node, name)?.value || "";
                input.style.cssText = "display:block;width:100%;box-sizing:border-box;margin:8px 0 16px;";
                label.append(input);
                dialog.append(label);
                inputs.push([name, input]);
            }
            const save = document.createElement("button");
            save.textContent = "确定";
            save.onclick = () => {
                for (const [name, input] of inputs) {
                    const widget = get(node, name);
                    widget.value = input.value.trim() || empty(name);
                    updateChoices(widget, widget.options?.values || []);
                }
                dialog.close();
            };
            const cancel = document.createElement("button");
            cancel.textContent = "取消";
            cancel.onclick = () => dialog.close();
            dialog.append(save, cancel);
            dialog.addEventListener("close", () => dialog.remove(), { once: true });
            document.body.append(dialog);
            dialog.showModal();
        }, { serialize: false });
        const provider = get(node, "API服务商");
        if (provider) {
            const previous = provider.callback;
            provider.callback = (...values) => {
                previous?.apply(provider, values);
                const address = get(node, "API地址");
                if (address && presets[provider.value] !== undefined) address.value = presets[provider.value];
            };
        }
        const secret = get(node, "API密钥");
        if (secret?.inputEl?.tagName === "INPUT") secret.inputEl.type = "password";
        localize(node);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!Object.hasOwn(titles, nodeData.name)) return;
        // New frontends expose display_name as a getter; do not assign to it.
        if ("backendDisplayName" in nodeData) nodeData.backendDisplayName = titles[nodeData.name];
        else Reflect.set(nodeData, "display_name", titles[nodeData.name]);
        for (const group of [nodeData.input?.required, nodeData.input?.optional]) {
            for (const [name, definition] of Object.entries(group || {})) {
                const label = name === "prompt" && nodeData.name === "CreationText" ? "文本内容" : fieldLabels[name];
                if (label && definition?.[1]) Object.assign(definition[1], { display_name: label, localized_name: label });
            }
        }
        if (Object.hasOwn(nodeType.prototype, "__creationHooksReady")) return;
        Object.defineProperty(nodeType.prototype, "__creationHooksReady", {value: true});
        const configureData = nodeType.prototype.configure;
        if (configureData) nodeType.prototype.configure = function (data, ...args) {
            migrateNode(data, nodeData.name);
            return configureData.call(this, data, ...args);
        };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configure?.apply(this, args);
            localize(this);
            return result;
        };
    },
});

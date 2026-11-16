import { app } from "/scripts/app.js";

const NODE_NAMES = new Set(["CreationImage", "CreationVideo"]);
const editors = new Set(["元指令", "提示词", "meta_instruction", "prompt"]);
const tracked = new Set();
const bindings = new WeakMap();
const cleanups = new WeakMap();
let activeMenu = null;
let styleAdded = false;

function editorFor(widget) {
    const root = widget?.element || widget?.inputEl || widget?.domElement || widget?.options?.element;
    if (!root) return null;
    return /^(TEXTAREA|INPUT)$/.test(root.tagName) ? root : root.querySelector?.("textarea, input");
}

function mediaItems(node) {
    const items = [];
    const groups = (node.comfyClass || node.constructor?.comfyClass || node.type) === "CreationImage"
        ? [["参考图", "图片"]] : [["图像", "图片"], ["视频", "视频"], ["音频", "音频"]];
    for (const [prefix, label] of groups) {
        const slots = (node.inputs || []).filter(input => input.link != null && new RegExp(`(?:^|\\.)${prefix}_\\d+$`).test(input.name));
        slots.sort((a, b) => Number(a.name.match(/\d+$/)[0]) - Number(b.name.match(/\d+$/)[0]));
        slots.forEach((_, index) => items.push({tag: `@${label}${index + 1}`, label: `${label}${index + 1}`}));
    }
    return items;
}

function wire(node, widget, area) {
    if (!area || bindings.has(area)) return;
    if (!styleAdded) {
        const style = document.createElement("style");
        style.textContent = ".creation-mention-menu{position:fixed;z-index:10050;max-height:220px;overflow:auto;background:#252525;border:1px solid #555;border-radius:6px;padding:4px;color:#eee;font:12px sans-serif}.creation-mention-item{padding:7px 10px;cursor:pointer}.creation-mention-active{background:#454545}";
        document.head.append(style);
        styleAdded = true;
    }
    const controller = new AbortController();
    const options = {signal: controller.signal};
    let menu = null, start = -1, active = 0, items = [];
    const close = () => {
        menu?.remove();
        menu = null;
        start = -1;
        if (activeMenu?.close === close) activeMenu = null;
    };
    const insert = tag => {
        const pos = area.selectionStart ?? area.value.length;
        area.setRangeText(`${tag} `, start, pos, "end");
        area.dispatchEvent(new Event("input", {bubbles: true}));
        widget.value = area.value;
        close();
        area.focus();
    };
    const render = query => {
        const all = mediaItems(node);
        items = all.filter(item => item.label.includes(query) || item.tag.includes(query));
        active = Math.min(active, Math.max(0, items.length - 1));
        if (!menu) {
            activeMenu?.close();
            menu = document.createElement("div");
            menu.className = "creation-mention-menu notranslate";
            menu.translate = false;
            document.body.append(menu);
            activeMenu = {menu, area, close};
        }
        menu.replaceChildren();
        if (!items.length) {
            menu.textContent = all.length ? "没有匹配的素材" : "请先连接参考素材";
        } else items.forEach((item, index) => {
            const row = document.createElement("div");
            row.className = `creation-mention-item${index === active ? " creation-mention-active" : ""}`;
            row.textContent = item.tag;
            row.onpointerdown = event => { event.preventDefault(); insert(item.tag); };
            menu.append(row);
        });
        const rect = area.getBoundingClientRect();
        menu.style.left = `${Math.max(8, rect.left)}px`;
        menu.style.top = `${Math.max(8, Math.min(window.innerHeight - 230, rect.bottom + 4))}px`;
        menu.style.minWidth = `${Math.max(190, rect.width)}px`;
    };
    const update = () => {
        const pos = area.selectionStart ?? 0;
        const match = area.value.slice(0, pos).match(/@([\u4e00-\u9fffA-Za-z0-9]*)$/);
        if (!match) return close();
        start = pos - match[0].length;
        active = 0;
        render(match[1]);
    };
    area.addEventListener("input", update, options);
    area.addEventListener("click", update, options);
    area.addEventListener("keydown", event => {
        if (!menu) return;
        if (event.key === "Escape") { event.preventDefault(); close(); return; }
        if (!items.length) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            active = (active + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
            render(area.value.slice(start + 1, area.selectionStart));
        } else if (event.key === "Enter" || event.key === "Tab") {
            event.preventDefault();
            insert(items[active].tag);
        }
    }, options);
    const cleanup = () => { close(); controller.abort(); bindings.delete(area); };
    bindings.set(area, cleanup);
    cleanups.get(node).push(cleanup);
}

// Editors may mount after node construction. Attach on focus rather than scanning
// the whole canvas on every draw or keeping per-node polling timers alive.
document.addEventListener("focusin", event => {
    if (!/^(TEXTAREA|INPUT)$/.test(event.target?.tagName)) return;
    for (const node of tracked) {
        for (const widget of node.widgets || []) {
            if (!editors.has(widget.name)) continue;
            if (editorFor(widget) === event.target) wire(node, widget, event.target);
        }
    }
});
document.addEventListener("pointerdown", event => {
    if (activeMenu && event.target !== activeMenu.area && !activeMenu.menu.contains(event.target)) activeMenu.close();
});

app.registerExtension({
    name: "Creation.Mentions",
    nodeCreated(node) {
        const name = node.comfyClass || node.constructor?.comfyClass || node.type;
        if (!NODE_NAMES.has(name) || tracked.has(node)) return;
        tracked.add(node);
        cleanups.set(node, []);
        for (const widget of node.widgets || []) {
            if (!editors.has(widget.name)) continue;
            wire(node, widget, editorFor(widget));
        }
        const removed = node.onRemoved;
        node.onRemoved = function (...args) {
            tracked.delete(this);
            cleanups.get(this)?.forEach(cleanup => cleanup());
            cleanups.delete(this);
            return removed?.apply(this, args);
        };
    },
});

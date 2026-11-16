// Minimal DOM/event mock: runs offline in Node, never controls a browser.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const root = path.resolve(import.meta.dirname, '..');
const source = fs.readFileSync(path.join(root, 'web/creative_reasoning_mentions.js'), 'utf8');
class MockEvent {
    constructor(type, options = {}) { Object.assign(this, {type, bubbles: false, defaultPrevented: false}, options); }
    preventDefault() { this.defaultPrevented = true; }
}
class Element {
    constructor(tagName, document) {
        this.tagName = tagName.toUpperCase();
        this.document = document;
        this.children = [];
        this.listeners = new Map();
        this.adds = new Map();
        this.style = {};
        this.value = '';
        this.className = '';
        this.selectionStart = this.selectionEnd = 0;
        this.focusCount = 0;
        this._text = '';
    }
    addEventListener(type, callback, options = {}) {
        if (options.signal?.aborted) return;
        const entries = this.listeners.get(type) ?? [];
        if (entries.some(entry => entry.callback === callback)) return;
        this.adds.set(type, (this.adds.get(type) ?? 0) + 1);
        const entry = {callback, options};
        this.listeners.set(type, [...entries, entry]);
        options.signal?.addEventListener('abort', () => this.removeEventListener(type, callback), {once: true});
    }
    removeEventListener(type, callback) { this.listeners.set(type, (this.listeners.get(type) ?? []).filter(entry => entry.callback !== callback)); }
    listenerCount(type) { return (this.listeners.get(type) ?? []).length; }
    dispatchEvent(event) {
        event.target ??= this;
        event.currentTarget = this;
        this['on' + event.type]?.(event);
        for (const entry of [...(this.listeners.get(event.type) ?? [])]) {
            entry.callback.call(this, event);
            if (entry.options.once) this.removeEventListener(event.type, entry.callback);
        }
        if (event.bubbles) this.parentElement?.dispatchEvent(event);
        return !event.defaultPrevented;
    }
    append(...elements) {
        for (const element of elements) { element.remove(); element.parentElement = this; this.children.push(element); }
    }
    replaceChildren(...elements) { for (const child of [...this.children]) child.remove(); this._text = ''; this.append(...elements); }
    remove() {
        if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this);
        this.parentElement = null;
    }
    contains(element) { return this === element || this.children.some(child => child.contains(element)); }
    querySelector(selector) {
        const tags = selector.split(',').map(tag => tag.trim().toUpperCase());
        for (const child of this.children) {
            if (tags.includes(child.tagName)) return child;
            const nested = child.querySelector(selector);
            if (nested) return nested;
        }
        return null;
    }
    set textContent(value) { this.replaceChildren(); this._text = value; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
    getBoundingClientRect() { return {left: 20, bottom: 60, width: 300}; }
    setRangeText(text, start, end, mode) {
        assert.equal(mode, 'end');
        this.value = this.value.slice(0, start) + text + this.value.slice(end);
        this.selectionStart = this.selectionEnd = start + text.length;
    }
    focus() {
        this.focusCount++;
        this.document.activeElement = this;
        this.document.dispatchEvent(new MockEvent('focusin', {target: this}));
    }
}

function harness({withNodes = false} = {}) {
    const document = new Element('document');
    document.document = document;
    document.createElement = tag => new Element(tag, document);
    document.head = document.createElement('head');
    document.body = document.createElement('body');
    document.append(document.head, document.body);
    const extensions = [], timerCalls = [];
    const forbiddenTimer = (...args) => { timerCalls.push(args); throw new Error('Polling/timers forbidden'); };
    const forbiddenNetwork = () => { throw new Error('Network forbidden'); };
    function LGraphNode() {}
    LGraphNode.prototype.draw = function draw() {};
    const prototypeBefore = Object.getOwnPropertyDescriptors(LGraphNode.prototype);
    Object.freeze(LGraphNode.prototype);
    const frozenBefore = Object.getOwnPropertyDescriptors(LGraphNode.prototype);
    const context = vm.createContext({
        app: {registerExtension: extension => extensions.push(extension)},
        document, window: {innerHeight: 800, setTimeout: forbiddenTimer, setInterval: forbiddenTimer, requestAnimationFrame: forbiddenTimer},
        Event: MockEvent, AbortController, LGraphNode, LiteGraph: {LGraphNode},
        setTimeout: forbiddenTimer, setInterval: forbiddenTimer, requestAnimationFrame: forbiddenTimer,
        fetch: forbiddenNetwork, api: {fetchApi: forbiddenNetwork}, console,
    });
    const load = (code, filename) => vm.runInContext('"use strict";\n' + code.replace(/^import .*;\r?\n/gm, ''), context, {filename});
    if (withNodes) load(fs.readFileSync(path.join(root, 'web/creation_nodes.js'), 'utf8'), 'creation_nodes.js');
    load(source, 'creative_reasoning_mentions.js');
    const extension = extensions.find(extension => extension.name === 'Creation.Mentions');
    assert.ok(extension);
    return {
        document, extension, nodes: extensions.find(extension => extension.name === 'Creation.Nodes'),
        area: () => document.createElement('textarea'),
        menu: () => document.body.children.find(child => child.className.includes('creation-mention-menu')),
        globals() {
            assert.equal(document.listenerCount('focusin'), 1);
            assert.equal(document.listenerCount('pointerdown'), 1);
            assert.equal(document.adds.get('focusin'), 1);
            assert.equal(document.adds.get('pointerdown'), 1);
            assert.deepEqual(timerCalls, []);
            assert.deepEqual(Object.getOwnPropertyDescriptors(LGraphNode.prototype), frozenBefore);
            assert.equal(LGraphNode.prototype.draw, prototypeBefore.draw.value);
        },
    };
}
function node(type, widgets = [], inputs = []) {
    return {
        type, widgets, inputs, removed: [],
        onRemoved(...args) { this.removed.push({self: this, args}); return 'removed-result'; },
    };
}
const slot = (name, link) => ({name, link});
function typeAt(area, value, caret = value.length) {
    area.value = value;
    area.selectionStart = area.selectionEnd = caret;
    area.dispatchEvent(new MockEvent('input', {bubbles: true}));
}
function key(area, value) {
    const event = new MockEvent('keydown', {key: value});
    area.dispatchEvent(event);
    return event;
}
function boundOnce(area) {
    for (const type of ['input', 'click', 'keydown']) {
        assert.equal(area.listenerCount(type), 1, type);
        assert.equal(area.adds.get(type), 1, type + ' added once');
    }
}
function unbound(area) {
    for (const type of ['input', 'click', 'keydown']) assert.equal(area.listenerCount(type), 0, type);
}
const rows = h => h.menu()?.children.map(child => child.textContent);

test('no timers or global LGraphNode patches; only image/video nodes tracked; global listeners install once', () => {
    assert.doesNotMatch(source, /\b(?:setTimeout|setInterval|requestAnimationFrame)\s*\(/);
    const h = harness();
    for (const type of ['CreationText', 'CreationAPI', 'ExternalNode']) {
        const area = h.area(), ignored = node(type, [{name: '提示词', inputEl: area}]);
        const removed = ignored.onRemoved;
        h.extension.nodeCreated(ignored);
        area.focus();
        unbound(area);
        assert.equal(ignored.onRemoved, removed);
    }
    for (const type of ['CreationImage', 'CreationVideo']) {
        const area = h.area(), n = node(type, [{name: '提示词', inputEl: area}]);
        h.extension.nodeCreated(n);
        const removed = n.onRemoved;
        for (let i = 0; i < 3; i++) { h.extension.nodeCreated(n); area.focus(); }
        boundOnce(area);
        assert.equal(n.onRemoved, removed);
        n.onRemoved();
        unbound(area);
    }
    assert.equal(h.document.head.children.length, 1, 'single stylesheet');
    h.globals();
});

test('immediate editors bind once across old/new names and DOM widget shapes', () => {
    const h = harness();
    for (const [index, name] of ['元指令', '提示词', 'meta_instruction', 'prompt'].entries()) {
        const area = index % 2 ? h.document.createElement('input') : h.area();
        const wrapper = h.document.createElement('div');
        wrapper.append(area);
        const shapes = [{element: wrapper}, {inputEl: area}, {domElement: wrapper}, {options: {element: wrapper}}];
        const n = node('CreationImage', [{name, ...shapes[index]}]);
        h.extension.nodeCreated(n);
        area.focus();
        area.focus();
        boundOnce(area);
        n.onRemoved();
        unbound(area);
    }
    const area = h.area(), n = node('CreationImage', [{name: 'unrelated', inputEl: area}]);
    h.extension.nodeCreated(n);
    area.focus();
    unbound(area);
    h.globals();
});

test('late editor mount binds on focus exactly once, without polling', () => {
    const h = harness();
    for (const name of ['元指令', '提示词', 'meta_instruction', 'prompt']) {
        const widget = {name}, n = node('CreationVideo', [widget]);
        h.extension.nodeCreated(n);
        const area = h.area(), wrapper = h.document.createElement('div');
        widget.element = wrapper;
        area.focus();
        unbound(area);
        wrapper.append(area);
        area.focus();
        area.focus();
        h.extension.nodeCreated(n);
        boundOnce(area);
        typeAt(area, '@');
        assert.equal(h.menu().textContent, '请先连接参考素材');
        n.onRemoved();
        assert.equal(h.menu(), undefined);
        unbound(area);
    }
    h.globals();
});

test('menu numbers connected media densely and inserts keyboard/pointer choices with value and caret preserved', () => {
    const h = harness(), area = h.area(), widget = {name: '提示词', inputEl: area};
    const inputs = [
        slot('图像输入.图像_10', 8), slot('图像输入.图像_2', 0), slot('图像输入.图像_1', null),
        slot('视频输入.视频_9', 3), slot('音频输入.音频_7', 4), slot('音频输入.音频_1', undefined),
        slot('unrelated_0', 5), slot('图像输入.图像_bad', 6),
    ];
    const n = node('CreationVideo', [widget], inputs), before = structuredClone(inputs);
    h.extension.nodeCreated(n);
    area.focus();
    let inputEvents = 0;
    area.addEventListener('input', () => inputEvents++);
    typeAt(area, '前缀@后缀', 3);
    assert.deepEqual(rows(h), ['@图片1', '@图片2', '@视频1', '@音频1']);
    assert.equal(h.menu().translate, false);
    assert.equal(key(area, 'ArrowDown').defaultPrevented, true);
    assert.equal(h.menu().children[1].className.includes('creation-mention-active'), true);
    assert.equal(key(area, 'Enter').defaultPrevented, true);
    assert.equal(area.value, '前缀@图片2 后缀');
    assert.equal(widget.value, area.value);
    assert.equal(area.selectionStart, '前缀@图片2 '.length);
    assert.equal(inputEvents, 2, 'one typing event plus one insertion event');
    assert.equal(h.menu(), undefined);
    typeAt(area, '@音');
    assert.deepEqual(rows(h), ['@音频1']);
    assert.equal(key(area, 'Tab').defaultPrevented, true);
    assert.equal(widget.value, '@音频1 ');
    typeAt(area, '@');
    key(area, 'ArrowUp');
    assert.equal(h.menu().children[3].className.includes('creation-mention-active'), true);
    key(area, 'ArrowDown');
    assert.equal(h.menu().children[0].className.includes('creation-mention-active'), true);
    const pointer = new MockEvent('pointerdown');
    h.menu().children[2].dispatchEvent(pointer);
    assert.equal(pointer.defaultPrevented, true);
    assert.equal(widget.value, '@视频1 ');
    assert.deepEqual(inputs, before, 'menu sorting must not mutate slots or links');
    h.globals();
});

test('menu filtering, live link changes, Escape, outside clicks and editor switching', () => {
    const h = harness(), area = h.area(), widget = {name: 'prompt', inputEl: area};
    const n = node('CreationImage', [widget], [slot('参考图.参考图_10', 1), slot('参考图_2', 2), slot('参考图.参考图_0', null), slot('视频输入.视频_0', 3)]);
    h.extension.nodeCreated(n);
    typeAt(area, '@');
    assert.deepEqual(rows(h), ['@图片1', '@图片2']);
    h.document.dispatchEvent(new MockEvent('pointerdown', {target: h.menu().children[0]}));
    assert.ok(h.menu(), 'inside menu does not close it');
    h.document.dispatchEvent(new MockEvent('pointerdown', {target: area}));
    assert.ok(h.menu(), 'editor click does not close it');
    typeAt(area, '@不存在');
    assert.equal(h.menu().textContent, '没有匹配的素材');
    assert.equal(key(area, 'Enter').defaultPrevented, false);
    assert.equal(key(area, 'Escape').defaultPrevented, true);
    assert.equal(h.menu(), undefined);
    typeAt(area, '@');
    h.document.dispatchEvent(new MockEvent('pointerdown', {target: h.document.body}));
    assert.equal(h.menu(), undefined);
    typeAt(area, '@');
    n.inputs[0].link = null;
    typeAt(area, '@');
    assert.deepEqual(rows(h), ['@图片1']);
    n.inputs[1].link = null;
    typeAt(area, '@');
    assert.equal(h.menu().textContent, '请先连接参考素材');
    const otherArea = h.area(), other = node('CreationVideo', [{name: '元指令', inputEl: otherArea}], [slot('视频_0', 0)]);
    h.extension.nodeCreated(other);
    typeAt(otherArea, '@');
    assert.deepEqual(rows(h), ['@视频1']);
    assert.equal(h.document.body.children.filter(child => child.className.includes('creation-mention-menu')).length, 1);
    typeAt(otherArea, 'plain text');
    assert.equal(h.menu(), undefined);
    h.globals();
});

test('node removal aborts every editor listener, closes menus and prevents late rebinding', () => {
    const h = harness(), area = h.area(), second = h.area();
    const late = {name: 'meta_instruction'};
    const n = node('CreationImage', [{name: '提示词', inputEl: area}, {name: '元指令', inputEl: second}, late], [slot('参考图_0', 1)]);
    h.extension.nodeCreated(n);
    typeAt(area, '@');
    assert.ok(h.menu());
    assert.equal(n.onRemoved('sentinel', 4), 'removed-result');
    assert.equal(n.removed.length, 1);
    assert.equal(n.removed[0].self, n);
    assert.deepEqual(n.removed[0].args, ['sentinel', 4]);
    assert.equal(h.menu(), undefined);
    unbound(area);
    unbound(second);
    area.focus();
    typeAt(area, '@');
    assert.equal(h.menu(), undefined);
    late.inputEl = h.area();
    late.inputEl.focus();
    unbound(late.inputEl);
    assert.equal(area.adds.get('input'), 1);
    h.globals();
});

test('node migration feeds legacy media slots into mention menus for image and video', () => {
    const h = harness({withNodes: true});
    for (const type of ['CreationImage', 'CreationVideo']) {
        const area = h.area(), widget = {name: 'prompt', inputEl: area};
        const n = node(type, [widget], [slot('images.image_10', 1), slot('images.image_2', 0), slot('videos.video_0', 2), slot('audios.audio_0', 3)]);
        h.nodes.beforeConfigureGraph({nodes: [n]});
        h.nodes.nodeCreated(n);
        h.extension.nodeCreated(n);
        typeAt(area, '@');
        assert.deepEqual(rows(h), type === 'CreationImage' ? ['@图片1', '@图片2'] : ['@图片1', '@图片2', '@视频1', '@音频1']);
        key(area, 'Enter');
        assert.equal(widget.value, '@图片1 ');
        n.onRemoved();
        unbound(area);
    }
    h.globals();
});


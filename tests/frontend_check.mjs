// Offline VM regression contracts; no ComfyUI runtime, browser, or provider.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const root = path.resolve(import.meta.dirname, '..');
const source = fs.readFileSync(path.join(root, 'web/creation_nodes.js'), 'utf8');
const titles = {CreationAPI: '创造API', CreationImage: '创造图像生成', CreationText: '创造文本扩写', CreationVideo: '创造视频生成'};
const labels = {
    meta_instruction: '元指令', prompt: '提示词', image_role_mode: '图像作用',
    aspect_ratio: '画面比例', resolution: '分辨率', duration: '时长（秒）',
    seed: '随机种子', temperature: '温度覆盖', control_after_generate: '种子控制（运行后）',
    images: '图像输入', videos: '视频输入', audios: '音频输入',
};
const models = ['API文本视觉模型', 'API图像模型', 'API视频模型'];
const fieldsFor = type => ({...labels, prompt: type === 'CreationText' ? '文本内容' : '提示词', images: type === 'CreationImage' ? '参考图' : '图像输入'});
const plain = value => JSON.parse(JSON.stringify(value));

function harness() {
    const extensions = [];
    let apiCalls = 0;
    const forbidden = () => { throw new Error('Unexpected browser/network/timer access'); };
    const context = vm.createContext({
        app: {registerExtension: extension => extensions.push(extension)},
        api: {fetchApi() { apiCalls++; return forbidden(); }},
        document: {createElement: forbidden}, fetch: forbidden,
        setTimeout: forbidden, setInterval: forbidden, console,
    });
    vm.runInContext('"use strict";\n' + source.replace(/^import .*;\r?\n/gm, ''), context, {filename: 'creation_nodes.js'});
    assert.equal(extensions.length, 1);
    assert.equal(extensions[0].name, 'Creation.Nodes');
    return {extension: extensions[0], get apiCalls() { return apiCalls; }};
}

function nodeClass(type) {
    class BaseNode {
        constructor() {
            this.type = this.title = type;
            this.configureCalls = this.onConfigureCalls = 0;
            assert.equal(this.widgets, undefined);
            // LiteGraph base construction runs BEFORE ComfyUI builds widgets.
            this.onNodeCreated?.();
        }
        addWidget(type, name, value, callback, options) {
            assert.ok(Array.isArray(this.widgets), 'nodeCreated must run after widget construction');
            const widget = {type, name, value, callback, options};
            this.widgets.push(widget);
            return widget;
        }
        configure(data, ...args) {
            this.configureCalls++;
            this.configuredSnapshot = structuredClone(data);
            this.configureArgs = args;
            this.inputs = data.inputs;
            this.title = data.title;
            this.onConfigure(data);
            return 'configured';
        }
        onConfigure(...args) { this.onConfigureCalls++; this.onConfigureArgs = args; return 'onConfigure-result'; }
        setDirtyCanvas(...args) { this.dirtyArgs = args; }
    }
    return class Node extends BaseNode {};
}

// Model the host: new Node(), build widgets, dispatch extension.nodeCreated(node).
function construct(Node, extension, type, {chinese = false, linked = false, seedValue = 42} = {}) {
    const node = new Node();
    const control = {name: 'control_after_generate', value: 'randomize', options: {values: ['fixed', 'increment', 'decrement', 'randomize']}};
    let seedWrites = 0, storedSeed = seedValue;
    node.widgets = Object.entries(fieldsFor(type))
        .filter(([name]) => !['images', 'videos', 'audios', 'control_after_generate'].includes(name))
        .map(([name, label]) => ({name: chinese ? label : name, value: 'value:' + name, options: {retained: true}}));
    const seed = node.widgets.find(widget => widget.name === (chinese ? '随机种子' : 'seed'));
    Object.defineProperty(seed, 'value', {
        enumerable: true, get: () => storedSeed,
        set(value) { seedWrites++; storedSeed = value; },
    });
    if (linked) seed.linkedWidgets = [control];
    else node.widgets.push(control);
    if (type === 'CreationAPI') {
        node.removedModels = [];
        node.widgets.push(...models.map((name, index) => ({
            name, type: 'text', value: index === 0 ? 'offline-model' : '', options: {},
            onRemove() { node.removedModels.push('widget:' + name); },
            inputEl: {remove() { node.removedModels.push('element:' + name); }},
        })));
        node.widgets.push(
            {name: 'API服务商', value: '自定义（OpenAI兼容）', callback(...args) { node.providerCalls = (node.providerCalls || 0) + 1; node.providerThis = this; node.providerArgs = args; }},
            {name: 'API地址', value: 'offline-placeholder'},
            {name: 'API密钥', value: '', inputEl: {tagName: 'INPUT', type: 'text'}},
        );
    }
    node.inputs = [];
    extension.nodeCreated(node);
    return {node, control, seed, get seedWrites() { return seedWrites; }};
}

test('registration handles legacy display names and read-only getters, without double wrapping', async () => {
    const {extension} = harness();
    for (const [type, title] of Object.entries(titles)) {
        for (const mode of ['legacy', 'backend-getter', 'getter-only']) {
            const Node = nodeClass(type);
            const data = {name: type, input: {required: {}, optional: {}}};
            for (const [index, name] of Object.keys(labels).entries()) {
                data.input[index % 2 ? 'optional' : 'required'][name] = ['STRING', {custom: true}];
            }
            data.input.optional.unspecified = ['STRING'];
            if (mode === 'legacy') data.display_name = 'old';
            else {
                if (mode === 'backend-getter') data.backendDisplayName = 'old';
                Object.defineProperty(data, 'display_name', {get() { return this.backendDisplayName ?? title; }});
            }
            await extension.beforeRegisterNodeDef(Node, data);
            assert.equal(data.display_name, title, type + '/' + mode);
            if (mode === 'backend-getter') assert.equal(data.backendDisplayName, title);
            for (const group of Object.values(data.input)) for (const [name, definition] of Object.entries(group)) {
                if (labels[name] && !['images', 'videos', 'audios'].includes(name)) {
                    assert.equal(definition[1].display_name, fieldsFor(type)[name]);
                    assert.equal(definition[1].localized_name, fieldsFor(type)[name]);
                    assert.equal(definition[1].custom, true);
                }
            }
            const configure = Node.prototype.configure, onConfigure = Node.prototype.onConfigure;
            await extension.beforeRegisterNodeDef(Node, data);
            assert.equal(Node.prototype.configure, configure);
            assert.equal(Node.prototype.onConfigure, onConfigure);
        }
    }
});

test('nodeCreated runs after widget construction, localizes both schemas and never writes valid seeds', async () => {
    const h = harness();
    for (const type of Object.keys(titles)) {
        for (const chinese of [false, true]) for (const linked of [false, true]) {
            const Node = nodeClass(type);
            await h.extension.beforeRegisterNodeDef(Node, {name: type});
            const fixture = construct(Node, h.extension, type, {chinese, linked});
            const {node, control} = fixture;
            assert.equal(node.title, titles[type]);
            for (const [name, label] of Object.entries(fieldsFor(type))) {
                if (['images', 'videos', 'audios', 'control_after_generate'].includes(name)) continue;
                const widget = node.widgets.find(widget => widget.name === (chinese ? label : name));
                assert.equal(widget.label, label);
                assert.equal(widget.localized_name, label);
                assert.equal(widget.options.label, label);
                assert.equal(widget.options.retained, true);
            }
            assert.equal(control.value, 'randomize');
            assert.equal(control.label, '种子控制（运行后）');
            for (const [value, label] of Object.entries({fixed: '固定', increment: '递增', decrement: '递减', randomize: '随机'})) {
                assert.equal(control.options.getOptionLabel(value), label);
                assert.equal(control.options.formatter(value), label);
            }
            assert.equal(control.options.getOptionLabel('future-mode'), 'future-mode');
            assert.deepEqual(control.options.values, ['fixed', 'increment', 'decrement', 'randomize']);
            assert.equal(node.onConfigure('sentinel'), 'onConfigure-result');
            assert.deepEqual(node.onConfigureArgs, ['sentinel']);
            assert.equal(node.onConfigureCalls, 1);
            h.extension.nodeCreated(node);
            assert.equal(fixture.seedWrites, 0, type + ': valid seed setter');
            assert.equal(fixture.seed.value, 42);
            assert.deepEqual(node.dirtyArgs, [true, true]);
        }
    }
    assert.equal(h.apiCalls, 0);
});

test('invalid seed normalization clamps once and remains stable', async () => {
    const {extension} = harness();
    const Node = nodeClass('CreationImage');
    await extension.beforeRegisterNodeDef(Node, {name: 'CreationImage'});
    for (const [value, expected, writes] of [[1, 1, 0], [2147483647, 2147483647, 0], [0, 1, 1], [-4, 1, 1], [2.7, 2, 1], ['42', 42, 1], [NaN, 1, 1], [Infinity, 2147483647, 1]]) {
        const fixture = construct(Node, extension, 'CreationImage', {seedValue: value});
        extension.nodeCreated(fixture.node);
        fixture.node.onConfigure();
        assert.equal(fixture.seed.value, expected);
        assert.equal(fixture.seedWrites, writes);
    }
});

test('API selectors, buttons and provider callbacks install once without network access', async () => {
    const h = harness(), Node = nodeClass('CreationAPI');
    await h.extension.beforeRegisterNodeDef(Node, {name: 'CreationAPI'});
    const {node} = construct(Node, h.extension, 'CreationAPI', {chinese: true});
    const widgets = [...node.widgets];
    for (let i = 0; i < 3; i++) { h.extension.nodeCreated(node); node.onConfigure(); }
    assert.deepEqual(node.widgets, widgets, 'preserve widget objects and order');
    const buttons = node.widgets.filter(widget => widget.type === 'button');
    assert.deepEqual(buttons.map(widget => widget.name), ['刷新模型列表', '手动填写模型 ID']);
    for (const button of buttons) assert.equal(button.options.serialize, false);
    for (const [index, name] of models.entries()) {
        const widget = node.widgets.find(widget => widget.name === name);
        assert.equal(widget.type, 'combo');
        assert.equal(widget.value, index === 0 ? 'offline-model' : '未启用');
        assert.ok(widget.options.values.includes(widget.value));
        assert.equal(node.widgets.filter(widget => widget.name === name).length, 1);
        assert.equal(node.removedModels.filter(item => item === 'widget:' + name).length, 1);
        assert.equal(node.removedModels.filter(item => item === 'element:' + name).length, 1);
    }
    const get = name => node.widgets.find(widget => widget.name === name);
    const provider = get('API服务商');
    provider.callback('synthetic', 7);
    assert.equal(node.providerCalls, 1);
    assert.equal(node.providerThis, provider);
    assert.deepEqual(node.providerArgs, ['synthetic', 7]);
    assert.equal(get('API地址').value, '');
    get('API地址').value = 'offline-placeholder';
    provider.value = 'unknown-provider';
    provider.callback();
    assert.equal(get('API地址').value, 'offline-placeholder');
    assert.equal(get('API密钥').inputEl.type, 'password');
    assert.equal(h.apiCalls, 0);
});

function workflowNode(type, {objectValues = false, chinese = false} = {}) {
    const names = fieldsFor(type);
    const media = type === 'CreationImage' ? '参考图' : '图像';
    const pairs = Object.entries(names).filter(([name]) => name !== 'control_after_generate');
    pairs.push(['images.image_10', names.images + '.' + media + '_10'], ['videos.video_2', '视频输入.视频_2'], ['audios.audio_0', '音频输入.音频_0']);
    return {
        id: 23, type, title: type,
        inputs: pairs.map(([name, localized], index) => ({
            name: chinese ? localized : name, type: 'synthetic', link: index === 0 ? null : index,
            widget: {name: chinese ? localized : name, config: {retained: true}}, slot_index: index,
        })),
        widgets_values: objectValues
            ? Object.fromEntries([...pairs.map(([name, localized], index) => [chinese ? localized : name, {index}]), ['control_after_generate', 'randomize'], ['unknown', 9]])
            : ['instruction', 'prompt', '16:9', 42, 'randomize', {retained: true}],
        outputs: [{name: 'output', links: [7, 8]}], properties: {retained: true},
    };
}

function assertMigration(data, before, type) {
    const names = fieldsFor(type);
    for (const [index, input] of data.inputs.entries()) {
        const oldName = before.inputs[index].name;
        let expected = names[oldName] ?? oldName;
        if (oldName === 'images.image_10') expected = names.images + '.' + (type === 'CreationImage' ? '参考图' : '图像') + '_10';
        if (oldName === 'videos.video_2') expected = '视频输入.视频_2';
        if (oldName === 'audios.audio_0') expected = '音频输入.音频_0';
        assert.equal(input.name, expected, type + ': ' + oldName);
        assert.equal(input.widget.name, expected);
        assert.deepEqual({...input, name: oldName, widget: before.inputs[index].widget}, before.inputs[index]);
        if (!Array.isArray(data.widgets_values)) {
            assert.deepEqual(data.widgets_values[expected], before.widgets_values[oldName]);
            if (expected !== oldName) assert.equal(Object.hasOwn(data.widgets_values, oldName), false);
        }
    }
    assert.deepEqual(data.outputs, before.outputs);
    assert.deepEqual(data.properties, before.properties);
    if (Array.isArray(data.widgets_values)) assert.deepEqual(data.widgets_values, before.widgets_values);
    else {
        assert.equal(data.widgets_values.control_after_generate, 'randomize');
        assert.equal(data.widgets_values.unknown, 9);
    }
}

test('beforeConfigureGraph migrates legacy/new slots and values, including subgraphs, preserving links', () => {
    const {extension} = harness();
    const nodes = Object.keys(titles).flatMap(type => [false, true].flatMap(objectValues => [false, true].map(chinese => workflowNode(type, {objectValues, chinese}))));
    const unknown = workflowNode('ExternalNode'), unknownBefore = structuredClone(unknown);
    const graph = {nodes: [...nodes.slice(0, 8), unknown], definitions: {subgraphs: [{nodes: nodes.slice(8)}]}, links: [[1, 2, 3, 4, 5, 'IMAGE']]};
    const snapshots = nodes.map(node => structuredClone(node));
    const references = nodes.map(node => ({inputs: node.inputs, values: node.widgets_values}));
    const links = graph.links;
    extension.beforeConfigureGraph(graph);
    for (const [index, node] of nodes.entries()) {
        assertMigration(node, snapshots[index], node.type);
        assert.equal(node.title, titles[node.type]);
        assert.equal(node.inputs, references[index].inputs);
        assert.equal(node.widgets_values, references[index].values);
    }
    assert.equal(graph.links, links);
    assert.deepEqual(links, [[1, 2, 3, 4, 5, 'IMAGE']]);
    assert.deepEqual(unknown, unknownBefore);
    const migrated = structuredClone(graph);
    extension.beforeConfigureGraph(graph);
    assert.deepEqual(graph, migrated, 'idempotent migration');
    assert.doesNotThrow(() => extension.beforeConfigureGraph(null));
});

test('configure wrapper migrates before base configure consumes workflow data', async () => {
    const {extension} = harness();
    for (const type of Object.keys(titles)) {
        const Node = nodeClass(type);
        await extension.beforeRegisterNodeDef(Node, {name: type});
        await extension.beforeRegisterNodeDef(Node, {name: type});
        for (const objectValues of [false, true]) {
            const {node} = construct(Node, extension, type);
            const data = workflowNode(type, {objectValues}), before = structuredClone(data), values = data.widgets_values;
            assert.equal(node.configure(data, 'sentinel', 5), 'configured');
            assertMigration(node.configuredSnapshot, before, type);
            assert.equal(node.configureCalls, 1);
            assert.equal(node.onConfigureCalls, 1);
            assert.deepEqual(node.configureArgs, ['sentinel', 5]);
            assert.equal(data.widgets_values, values);
            assert.equal(node.title, titles[type]);
        }
    }
});

test('default/old titles use 创造 and custom titles survive every hook', async () => {
    const {extension} = harness();
    for (const [type, expected] of Object.entries(titles)) {
        const Node = nodeClass(type);
        await extension.beforeRegisterNodeDef(Node, {name: type});
        for (const title of [undefined, '', type, expected.replace('创造', '创建'), expected.replace('创造', '创世'), expected, '自定义标题', '创建：我的画面']) {
            const data = {type, title, inputs: []}, custom = ['自定义标题', '创建：我的画面'].includes(title);
            extension.beforeConfigureGraph({nodes: [data]});
            assert.equal(data.title, custom ? title : expected);
            const {node} = construct(Node, extension, type);
            node.configure(data);
            extension.nodeCreated(node);
            assert.equal(node.title, custom ? title : expected);
        }
    }
});

test('comfyClass lookup works and unrelated nodes remain untouched', async () => {
    const {extension} = harness();
    for (const inherited of [false, true]) {
        const Node = nodeClass('CreationImage');
        if (inherited) Node.comfyClass = 'CreationImage';
        const {node} = construct(Node, {nodeCreated() {}}, 'CreationImage');
        node.type = 'different-type';
        if (!inherited) node.comfyClass = 'CreationImage';
        extension.nodeCreated(node);
        assert.equal(node.widgets.find(widget => widget.name === 'prompt').label, '提示词');
    }
    const Node = nodeClass('ExternalNode'), data = {name: 'ExternalNode', display_name: 'External'};
    await extension.beforeRegisterNodeDef(Node, data);
    const node = new Node();
    node.widgets = [{name: 'seed', value: 0}];
    const before = plain(node);
    extension.nodeCreated(node);
    assert.deepEqual(plain(node), before);
    assert.deepEqual(data, {name: 'ExternalNode', display_name: 'External'});
});

test('zh/zh-CN/en retain 创造 titles and both generations of input IDs', () => {
    for (const locale of ['zh', 'zh-CN', 'en']) {
        const definitions = JSON.parse(fs.readFileSync(path.join(root, 'locales', locale, 'nodeDefs.json'), 'utf8'));
        for (const [type, title] of Object.entries(titles)) {
            assert.equal(definitions[type].display_name, title, locale + '/' + type);
            for (const [name, label] of Object.entries(fieldsFor(type))) {
                assert.equal(definitions[type].inputs[name].name, label, locale + '/' + type + '/' + name);
                assert.equal(definitions[type].inputs[label].name, label, locale + '/' + type + '/' + label);
            }
        }
    }
});


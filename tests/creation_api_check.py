"""Offline contracts; uses the installed ComfyUI runtime and never calls a provider."""
from contextlib import nullcontext
import asyncio
import base64
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'E:\ComfyUI_dapao1\ComfyUI')
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from PIL import Image
import requests

spec = importlib.util.spec_from_file_location('creation_plugin', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
nodes = sys.modules['creation_plugin.cr_core.creation_nodes']
client = nodes.client

def run(coro):
    return asyncio.run(coro)

def reply(content):
    return {'choices': [{'message': {'content': content}}]}

class CreationChecks(unittest.TestCase):
    def setUp(self):
        client._CHAT_UNSUPPORTED.clear()

    def test_registered_nodes_and_chinese_video_fields(self):
        registered = run(plugin.CreationExtension().get_node_list())
        self.assertEqual(registered, [nodes.CreationAPI, nodes.CreationImage, nodes.CreationText, nodes.CreationVideo])
        schema = nodes.CreationVideo.define_schema()
        fields = {item.id: item for item in schema.inputs}
        for name, label in {'画面比例': '画面比例', '分辨率': '分辨率', '时长（秒）': '时长（秒）', '随机种子': '随机种子'}.items():
            self.assertEqual(fields[name].display_name, label)
        self.assertEqual(fields['分辨率'].options, ['自动', '480p', '720p', '1080p'])
        self.assertEqual(nodes.V2_PRESETS['创意推理'], 'https://api.creative-reasoning.com/unified/v1')
        self.assertFalse(any('lulu' in name.lower() for name in nodes.V2_PRESETS))
        self.assertFalse(any('本地' in (item.display_name or item.id) for item in nodes.CreationAPI.define_schema().inputs))
        fields = {item.id: item for item in nodes.CreationImage.define_schema().inputs}
        self.assertEqual(fields['分辨率'].options, ['480p', '720p', '1080p', '1K', '2K', '4K'])
        self.assertFalse(fields['多行列表'].default)
        self.assertEqual(fields['并发数量'].default, 3)
        self.assertEqual((fields['并发数量'].min, fields['并发数量'].max), (1, 8))

    def test_chinese_contract_in_v3_and_v1_without_frontend_translation(self):
        for cls in run(plugin.CreationExtension().get_node_list()):
            schema = cls.GET_SCHEMA()
            name = plugin.NODE_DISPLAY_NAME_MAPPINGS[schema.node_id]
            self.assertEqual(schema.display_name, name)
            v1 = cls.GET_NODE_INFO_V1()
            self.assertEqual(v1['display_name'], name)
            self.assertEqual(v1['input'], cls.INPUT_TYPES())
            fields = {key for group in v1['input'].values() for key in group}
            for field in fields:
                self.assertRegex(field, '[\u4e00-\u9fff]')
            for label in v1['output_name']:
                self.assertRegex(label, '[\u4e00-\u9fff]')
            # JSON/UTF-8 transport is the same contract on Linux and Windows.
            self.assertEqual(json.loads(json.dumps(v1, ensure_ascii=False).encode('utf-8')), json.loads(json.dumps(v1)))

    def test_generation_nodes_require_creation_api_connection_before_execution(self):
        for cls in [nodes.CreationText, nodes.CreationImage, nodes.CreationVideo]:
            self.assertIn('动力源', cls.INPUT_TYPES()['required'])
            with self.assertRaisesRegex(ValueError, '请连接创造API'):
                run(cls.execute())
            with self.assertRaisesRegex(ValueError, '当前动力源不是创造API'):
                run(cls.execute(动力源={'wuji_api_version': 1}))

    def test_chinese_execution_and_legacy_internal_calls_use_same_arguments(self):
        from comfy_api.internal import make_locked_method_func
        config = run(nodes.CreationAPI.execute(API文本视觉模型='chat', API图像模型='image', API视频模型='video'))[0]
        nodes.CreationText.GET_SCHEMA()
        execute = make_locked_method_func(nodes.CreationText, nodes.CreationText.FUNCTION, nodes.CreationText.PREPARE_CLASS_CLONE(None))
        with patch.object(client, 'chat', return_value='完成') as chat:
            run(execute(动力源=config, 元指令='写诗', 文本内容='雨', 温度覆盖=0.5, 随机种子=42))
            chinese_call = chat.call_args
            run(nodes.CreationText.execute(动力源=config, meta_instruction='写诗', prompt='雨', temperature=0.5, seed=42))
            self.assertEqual(chinese_call, chat.call_args)
        with patch.object(client, 'chat', return_value='{"image_prompts":["画面"],"text_output":"诗"}'), \
             patch.object(client, 'generate_image', return_value={'data': []}) as generate, \
             patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
             patch.object(nodes.media, 'images_from_payload', new=AsyncMock(return_value=object())), \
             patch.object(nodes.media, '_pad_and_stack', return_value=object()):
            output = run(nodes.CreationImage.execute(动力源=config, 元指令='画一幅画并写诗', 提示词='雨', 参考图={}, 画面比例='16:9', 分辨率='2K', 多行列表=False, 随机种子=42))
            self.assertEqual(output.result[0], '诗')
            self.assertEqual(generate.call_args.kwargs['size'], '2048x1152')
        with patch.object(client, 'chat', return_value='最终镜头'), \
             patch.object(client, 'generate_video', return_value={'data': []}) as generate, \
             patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
             patch.object(nodes.media, 'video_from_payload', new=AsyncMock(return_value=(object(), ''))):
            run(nodes.CreationVideo.execute(动力源=config, 元指令='扩写', 提示词='雨', 图像输入={}, 视频输入={}, 音频输入={}, 图像作用=nodes.nodes.ROLE_MODE_LABELS['all_reference'], 画面比例='16:9', 分辨率='720p', **{'时长（秒）': 6}, 随机种子=43))
            self.assertEqual(generate.call_args.kwargs['extra'], {'aspect_ratio': '16:9', 'resolution': '720p', 'seed': 43})
            self.assertEqual(generate.call_args.kwargs['duration'], 6)

    def test_chinese_dynamic_media_schema_reconstructs_connected_slots(self):
        from comfy_api.latest import _io as comfy_io
        image = object()
        for cls, group, slot in [(nodes.CreationImage, '参考图', '参考图_0'), (nodes.CreationVideo, '图像输入', '图像_0'), (nodes.CreationVideo, '视频输入', '视频_0'), (nodes.CreationVideo, '音频输入', '音频_0')]:
            # Exercise the runtime's dynamic-input expander, not a local imitation.
            finalized, _, data = comfy_io.get_finalized_class_inputs(cls.INPUT_TYPES(), {f'{group}.{slot}': image})
            self.assertIn(f'{group}.{slot}', finalized['optional'])
            self.assertIn(f'{group}.{slot}', data['dynamic_paths'])
            nested = comfy_io.build_nested_inputs({f'{group}.{slot}': image}, data)
            self.assertIs(nested[group][slot], image)

    def test_seed_directions_mix_adjacent_seeds_and_do_not_repeat_short_cycle(self):
        for kind in ('text', 'image', 'video'):
            directions = [nodes._variation(seed, kind=kind) for seed in range(1, 201)]
            self.assertGreater(len(set(directions)), 180)
            self.assertTrue(all(a != b for a, b in zip(directions, directions[1:])))
            self.assertEqual(nodes._variation(42, kind=kind), nodes._variation(42, kind=kind))
            self.assertNotEqual(nodes._variation(1, kind=kind), nodes._variation(1233, kind=kind))
        self.assertEqual(len({nodes._variation(42, kind=kind) for kind in ('text', 'image', 'video')}), 3)

    def test_text_changes_actual_messages_without_sending_unsupported_seed(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-4o'))[0]
        bodies = []
        def request(*args, **kwargs):
            bodies.append(kwargs['body'])
            return reply('完成')
        with patch.object(client, '_request', side_effect=request):
            for seed in (42, 43, 42):
                output = run(nodes.CreationText.execute(动力源=config, meta_instruction='写一首四行诗', prompt='夏天', seed=seed))
                self.assertEqual(output.result[0], '完成')
            run(nodes.CreationText.execute(动力源=config, prompt='严格任务', temperature=0.2))
        self.assertTrue(all('seed' not in body for body in bodies))
        self.assertNotEqual(bodies[0]['messages'], bodies[1]['messages'])
        self.assertEqual(bodies[0]['messages'], bodies[2]['messages'])
        self.assertIn('写一首四行诗', bodies[0]['messages'][0]['content'])
        self.assertEqual(bodies[0]['temperature'], 1.1)
        self.assertEqual(bodies[3]['temperature'], 0.2)

    def image_run(self, multiline, seed=42, reference=False, image_model='image-model',
                  aspect_ratio='自动', resolution='1K'):
        buffer = io.BytesIO()
        Image.new('RGB', (12, 8), 'red').save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-4o', API图像模型=image_model))[0]
        calls = []
        def request(method, url, key, **kwargs):
            calls.append((method, url, kwargs.get('body')))
            if url.endswith('chat/completions'):
                return reply(json.dumps({'image_prompts': ['第一幅', '第二幅'], 'text_output': '独立诗句'}))
            if method == 'POST':
                return {'id': 'task', 'status': 'queued'}
            return {'status': 'completed', 'data': [{'b64_json': encoded}]}
        refs = {'参考图_0': object()} if reference else None
        with patch.object(client, '_request', side_effect=request), patch.object(nodes.media, 'image_tensor_to_data_uris', return_value=['data:image/png;base64,' + encoded]):
            output = run(nodes.CreationImage.execute(动力源=config, prompt='分镜', 多行列表=multiline,
                seed=seed, images=refs, aspect_ratio=aspect_ratio, resolution=resolution))
        return output, calls

    def test_image_async_batch_extra_text_and_independent_seeds(self):
        output, calls = self.image_run(True)
        self.assertEqual(output.result[0], '独立诗句')
        self.assertEqual(tuple(output.result[1].shape), (2, 8, 12, 3))
        posts = [call for call in calls if call[0] == 'POST' and call[1].endswith('/images/tasks')]
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0][2]['seed'], 42)
        self.assertNotEqual(posts[0][2]['seed'], posts[1][2]['seed'])
        self.assertEqual(posts[0][2]['size'], '1024x1024')
        for value in ('第一幅', '第二幅', '独立诗句'):
            self.assertIn(value, output.result[2])
        _, next_calls = self.image_run(True, seed=43)
        self.assertNotEqual(calls[0][2]['messages'], next_calls[0][2]['messages'])

    def test_image_list_disabled_submits_one_image_and_keeps_extra_text_separate(self):
        output, calls = self.image_run(False)
        posts = [call for call in calls if call[0] == 'POST' and call[1].endswith('/images/tasks')]
        self.assertEqual(len(posts), 1)
        self.assertEqual(tuple(output.result[1].shape), (1, 8, 12, 3))
        self.assertNotIn('独立诗句', posts[0][2]['prompt'])
        self.assertIn('多行列表已关闭', calls[0][2]['messages'][0]['content'])

    def test_image_edit_uses_async_endpoint(self):
        _, calls = self.image_run(True, reference=True)
        posts = [call for call in calls if call[0] == 'POST' and '/images/' in call[1]]
        self.assertEqual(len(posts), 2)
        self.assertTrue(all(call[1].endswith('/images/edit-tasks') and call[2]['image'].startswith('data:') for call in posts))

    def test_image_pixel_sizes(self):
        self.assertEqual(nodes.nodes._image_size('16:9', '1K'), '1024x576')
        self.assertEqual(nodes.nodes._image_size('9:16', '2K'), '1152x2048')
        self.assertEqual(nodes.nodes._image_size('1:1', '4K'), '4096x4096')

    def test_image_p_presets_preserve_selected_ratio_and_reach_request(self):
        ratios = {'自动': (16, 9), '1:1': (1, 1), '16:9': (16, 9), '9:16': (9, 16),
                  '4:3': (4, 3), '3:4': (3, 4), '21:9': (21, 9)}
        with nullcontext():
            for resolution, short in [('480p', 480), ('720p', 720), ('1080p', 1080)]:
                for ratio, (rw, rh) in ratios.items():
                    with self.subTest(resolution=resolution, ratio=ratio):
                        output, calls = self.image_run(False, aspect_ratio=ratio, resolution=resolution)
                        posts = [call for call in calls if call[0] == 'POST' and '/images/' in call[1]]
                        self.assertEqual(len(posts), 1)
                        width, height = map(int, posts[0][2]['size'].split('x'))
                        self.assertEqual(min(width, height), short)
                        self.assertLess(abs(width / height - rw / rh), 0.002)
                        self.assertIn(posts[0][2]['size'], output.result[2])
        self.assertEqual(nodes._image_base_size('16:9', '480p'), '853x480')
        self.assertEqual(nodes._image_base_size('16:9', '720p'), '1280x720')
        self.assertEqual(nodes._image_base_size('16:9', '1080p'), '1920x1080')
        self.assertEqual(nodes._image_base_size('9:16', '1080p'), '1080x1920')

    def test_gpt_image_sizes_satisfy_limits_for_all_ui_options(self):
        for model in ('gpt-image-2.5-flare', 'gpt-image-2.5-sunburst'):
            for ratio in ('', '1:1', '16:9', '9:16', '4:3', '3:4', '21:9'):
                for resolution in ('480p', '720p', '1080p', '1K', '2K', '4K'):
                    with self.subTest(model=model, ratio=ratio, resolution=resolution):
                        size = nodes._image_request_size({'image_model': model}, ratio, resolution)
                        width, height = map(int, size.split('x'))
                        self.assertLessEqual(max(width, height), 3840)
                        self.assertEqual(width % 16, 0)
                        self.assertEqual(height % 16, 0)
                        self.assertGreaterEqual(width * height, 655360)
                        self.assertLessEqual(width * height, 8294400)
                        self.assertLessEqual(max(width, height), 3 * min(width, height))
                        effective_ratio = '16:9' if not ratio and resolution.endswith('p') else ratio
                        expected_ratio = 1 if not effective_ratio else float(effective_ratio.split(':')[0]) / float(effective_ratio.split(':')[1])
                        self.assertLess(abs(width / height / expected_ratio - 1), 0.03)

    def test_gpt_image_real_request_sizes_for_generation_edit_and_batch(self):
        expected_sizes = {
            '1K': {'自动': '1024x1024', '1:1': '1024x1024', '16:9': '1088x608', '9:16': '608x1088'},
            '2K': {'自动': '2048x2048', '1:1': '2048x2048', '16:9': '2048x1152', '9:16': '1152x2048'},
            '4K': {'自动': '2880x2880', '1:1': '2880x2880', '16:9': '3840x2160', '9:16': '2160x3840'},
        }
        ratios = ('自动', '1:1', '16:9', '9:16', '4:3', '3:4', '21:9')
        # Check actual outgoing bodies for all 336 UI/mode combinations.
        with nullcontext():
            for model in ('gpt-image-2.5-flare', 'gpt-image-2.5-sunburst'):
                for reference in (False, True):
                    for multiline in (False, True):
                        for resolution in ('480p', '720p', '1080p', '1K', '2K', '4K'):
                            for ratio in ratios:
                                with self.subTest(model=model, reference=reference, multiline=multiline,
                                                  resolution=resolution, ratio=ratio):
                                    output, calls = self.image_run(multiline, reference=reference, image_model=model,
                                        aspect_ratio=ratio, resolution=resolution)
                                    posts = [call for call in calls if call[0] == 'POST' and '/images/' in call[1]]
                                    count = 2 if multiline else 1
                                    self.assertEqual(len(posts), count)
                                    for _, url, body in posts:
                                        self.assertTrue(url.endswith('/images/edit-tasks' if reference else '/images/tasks'))
                                        width, height = map(int, body['size'].split('x'))
                                        self.assertEqual(width % 16, 0)
                                        self.assertEqual(height % 16, 0)
                                        self.assertLessEqual(max(width, height), 3840)
                                        self.assertGreaterEqual(width * height, 655360)
                                        self.assertLessEqual(width * height, 8294400)
                                        self.assertLessEqual(max(width, height), 3 * min(width, height))
                                        if ratio in expected_sizes.get(resolution, {}):
                                            self.assertEqual(body['size'], expected_sizes[resolution][ratio])
                                        self.assertEqual(body['model'], model)
                                        self.assertIn(body['size'], output.result[2])
                                    self.assertEqual(output.result[0], '独立诗句')
                                    self.assertEqual(tuple(output.result[1].shape), (count, 8, 12, 3))

    def test_image_sizing_follows_effective_remote_model_and_preserves_other_models(self):
        self.assertEqual(nodes._image_request_size({'image_model': 'image'}, '1:1', '4K'), '4096x4096')
        config = {'image_model': 'image', 'remote': {'image': 'gpt-image-2.5-flare'}}
        self.assertEqual(nodes._image_request_size(config, '1:1', '4K'), '2880x2880')
        config = {'image_model': 'gpt-image-2.5-flare', 'remote': {'image': 'image'}}
        self.assertEqual(nodes._image_request_size(config, '1:1', '4K'), '4096x4096')

    def test_image_batch_runs_concurrently_and_preserves_prompt_order(self):
        import threading
        import time
        state = {'active': 0, 'maximum': 0, 'submitted': []}
        lock = threading.Lock()

        def submit(config, prompt, **kwargs):
            with lock:
                state['active'] += 1
                state['maximum'] = max(state['maximum'], state['active'])
                state['submitted'].append(prompt)
            time.sleep(0.04)
            with lock:
                state['active'] -= 1
            return {'prompt': prompt}

        async def decode(payload, **kwargs):
            return torch.full((1, 1, 1, 3), ord(payload['prompt'][0]), dtype=torch.float32)

        import torch
        with patch.object(client, 'generate_image', side_effect=submit), \
             patch.object(client, 'poll_media', new=AsyncMock(side_effect=lambda c, p, k: p)), \
             patch.object(nodes.media, 'images_from_payload', new=AsyncMock(side_effect=decode)), \
             patch.object(nodes.media, '_pad_and_stack', side_effect=lambda values: values):
            outputs, downgraded = run(nodes._image_batch({}, ['A', 'B', 'C', 'D'], [], '1024x1024', 42, 3))
        self.assertGreater(state['maximum'], 1)
        self.assertLessEqual(state['maximum'], 3)
        self.assertEqual(state['submitted'], ['A', 'B', 'C', 'D'])
        self.assertEqual([int(value[0, 0, 0, 0]) for value in outputs], list(map(ord, 'ABCD')))
        self.assertFalse(downgraded)

    def test_image_batch_downgrades_only_explicit_rejection_and_never_resubmits_accepted(self):
        import threading
        import time
        state = {'active': 0, 'maximum': 0, 'calls': {}}
        lock = threading.Lock()

        def submit(config, prompt, **kwargs):
            with lock:
                state['active'] += 1
                state['maximum'] = max(state['maximum'], state['active'])
                state['calls'][prompt] = state['calls'].get(prompt, 0) + 1
            time.sleep(0.02)
            with lock:
                state['active'] -= 1
            if prompt == 'B' and state['calls'][prompt] == 1:
                raise client.WujiApiError('HTTP 429 concurrency limit exceeded', status_code=429,
                    error_code='concurrency_limit_exceeded', concurrency_hint=True)
            return {'prompt': prompt}

        async def decode(payload, **kwargs):
            return torch.zeros((1, 1, 1, 3))

        import torch
        with patch.object(client, 'generate_image', side_effect=submit), \
             patch.object(client, 'poll_media', new=AsyncMock(side_effect=lambda c, p, k: p)), \
             patch.object(nodes.media, 'images_from_payload', new=AsyncMock(side_effect=decode)), \
             patch.object(nodes.media, '_pad_and_stack', side_effect=lambda values: values):
            outputs, downgraded = run(nodes._image_batch({}, ['A', 'B', 'C', 'D'], [], '1024x1024', 42, 3))
        self.assertTrue(downgraded)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(state['calls'], {'A': 1, 'B': 2, 'C': 1, 'D': 1})
        self.assertGreater(state['maximum'], 1)
        self.assertLessEqual(state['maximum'], 3)
        self.assertEqual(client.concurrency_rejected(client.WujiApiError('network timeout')), False)
        self.assertEqual(client.concurrency_rejected(client.WujiApiError('HTTP 429 rate limit', status_code=429)), False)
        self.assertEqual(client.concurrency_rejected(client.WujiApiError('HTTP 429 concurrency limit', status_code=429,
            concurrency_hint=True, task_accepted=True)), False)

    def test_concurrency_rejection_uses_http_evidence_and_edit_does_not_compatibility_retry(self):
        response = requests.Response()
        response.status_code = 429
        response.headers['Retry-After'] = '2'
        response._content = b'{"error":{"code":"concurrency_limit_exceeded","message":"Too many concurrent requests"}}'
        error = client._http_error(response, '')
        self.assertTrue(client.concurrency_rejected(error))
        self.assertEqual(error.retry_after, 2)
        response._content = b'{"task_id":"accepted","error":{"code":"concurrency_limit_exceeded"}}'
        self.assertFalse(client.concurrency_rejected(client._http_error(response, '')))
        config = run(nodes.CreationAPI.execute(API图像模型='image'))[0]
        with patch.object(client, '_request', side_effect=error) as request:
            with self.assertRaises(client.WujiApiError):
                client.generate_image(config, 'prompt', images=['one', 'two'])
            self.assertEqual(request.call_count, 1)

    def test_batch_never_retries_polling_or_network_failures_and_drains_accepted_tasks(self):
        config = {}
        for failure in (client.WujiApiError('network timeout'),
                        client.WujiApiError('429 concurrency', status_code=429, concurrency_hint=True)):
            with self.subTest(failure=str(failure)):
                def submit(config, prompt, **kwargs):
                    return {'id': prompt}

                async def poll(config, payload, kind):
                    if payload['id'] == 'B':
                        raise failure
                    await asyncio.sleep(0.01)
                    return payload

                with patch.object(client, 'generate_image', side_effect=submit) as generate, \
                     patch.object(client, 'poll_media', new=AsyncMock(side_effect=poll)), \
                     patch.object(nodes.media, 'images_from_payload', new=AsyncMock(return_value=object())) as decode:
                    with self.assertRaises(client.WujiApiError):
                        run(nodes._image_batch(config, ['A', 'B', 'C', 'D'], [], '1024x1024', 42, 3))
                    self.assertEqual(generate.call_count, 3)
                    self.assertEqual(decode.await_count, 2)

    def test_reasoning_cannot_be_used_as_image_or_video_prompt(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-4o', API图像模型='image', API视频模型='video'))[0]
        with patch.object(client, 'chat', return_value='<think>先分析用户要求</think>{"image_prompts":["晴天的森林"],"text_output":""}'), \
             patch.object(client, 'generate_image', return_value={'data': []}) as generate, \
             patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
             patch.object(nodes.media, 'images_from_payload', new=AsyncMock(return_value=object())), \
             patch.object(nodes.media, '_pad_and_stack', return_value=object()):
            run(nodes.CreationImage.execute(动力源=config, prompt='森林'))
            self.assertEqual(generate.call_args.args[1], '晴天的森林')
        for contaminated in ('<think>没有闭合的分析', 'Analysis: let me think about the user request',
                             '先分析需求，然后输出 {"image_prompts":["森林"]}'):
            with patch.object(client, 'chat', return_value=contaminated), \
                 patch.object(client, 'generate_image', return_value={'data': []}) as generate, \
                 patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
                 patch.object(nodes.media, 'images_from_payload', new=AsyncMock(return_value=object())), \
                 patch.object(nodes.media, '_pad_and_stack', return_value=object()):
                run(nodes.CreationImage.execute(动力源=config, prompt='森林'))
                self.assertEqual(generate.call_args.args[1], '森林')
        with patch.object(client, 'chat', return_value='<think>先推理镜头</think>镜头沿森林小径缓慢推进'), \
             patch.object(client, 'generate_video', return_value={'data': []}) as generate, \
             patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
             patch.object(nodes.media, 'video_from_payload', new=AsyncMock(return_value=(object(), ''))):
            run(nodes.CreationVideo.execute(动力源=config, prompt='森林'))
            self.assertEqual(generate.call_args.args[1], '镜头沿森林小径缓慢推进')
        with patch.object(client, 'chat', return_value='<think>未闭合镜头分析'), \
             patch.object(client, 'generate_video', return_value={'data': []}) as generate, \
             patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
             patch.object(nodes.media, 'video_from_payload', new=AsyncMock(return_value=(object(), ''))):
            run(nodes.CreationVideo.execute(动力源=config, prompt='森林'))
            self.assertEqual(generate.call_args.args[1], '森林')

    def test_unknown_models_continue_and_rejected_disable_option_retries_then_is_cached(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='unknown-reasoning-model'))[0]
        with patch.object(client, '_request', return_value=reply('可用结果')) as request:
            self.assertEqual(client.chat(config, []), '可用结果')
            self.assertEqual(request.call_count, 1)
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-5.1'))[0]
        with patch.object(client, '_request', side_effect=[
                client.WujiApiError('HTTP 400 unsupported reasoning_effort', status_code=400),
                reply('最终结果'), reply('后续结果')]) as request:
            self.assertEqual(client.chat(config, []), '最终结果')
            self.assertEqual(client.chat(config, []), '后续结果')
            self.assertEqual(request.call_count, 3)
            self.assertEqual(request.call_args_list[0].kwargs['body']['reasoning_effort'], 'none')
            self.assertNotIn('reasoning_effort', request.call_args_list[1].kwargs['body'])
            self.assertNotIn('reasoning_effort', request.call_args_list[2].kwargs['body'])

    def test_server_reasoning_is_discarded_and_final_output_survives(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-4o'))[0]
        responses = [
            {'choices': [{'message': {'content': '最终提示词', 'reasoning_content': '不该出现的思考'}}]},
            {**reply('最终提示词'), 'usage': {'completion_tokens_details': {'reasoning_tokens': 10}}},
            {'choices': [{'message': {'content': [
                {'type': 'text', 'text': '不该出现的分析', 'thought': True},
                {'type': 'text', 'text': '最终提示词'}]}}]},
        ]
        for response in responses:
            with patch.object(client, '_request', return_value=response) as request:
                self.assertEqual(client.chat(config, []), '最终提示词')
                self.assertEqual(request.call_count, 1)

    def test_optional_seed_rejection_recovers_without_changing_prompt_or_model(self):
        config = {'base_url': 'https://unknown.example/v1', 'model': 'custom-chat', 'seed': 42}
        with patch.object(client, '_request', side_effect=[
                client.WujiApiError("HTTP 400 does not support parameters: ['seed']", status_code=400),
                reply('结果')]) as request:
            self.assertEqual(client.chat(config, [{'role': 'user', 'content': '要求'}]), '结果')
        first, second = [call.kwargs['body'] for call in request.call_args_list]
        self.assertEqual(first['seed'], 42)
        self.assertEqual({key: value for key, value in first.items() if key != 'seed'}, second)

    def test_chat_never_retries_accepted_auth_network_or_generic_server_errors(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-5.1'))[0]
        failures = [client.WujiApiError('HTTP 401 unsupported reasoning_effort', status_code=401),
                    client.WujiApiError('HTTP 500 unsupported reasoning_effort', status_code=500),
                    client.WujiApiError('network timeout reasoning_effort'),
                    client.WujiApiError('HTTP 400 unsupported reasoning_effort', status_code=400, task_accepted=True)]
        for error in failures:
            with self.subTest(error=str(error)), patch.object(client, '_request', side_effect=error) as request:
                with self.assertRaises(client.WujiApiError):
                    client.chat(config, [])
                self.assertEqual(request.call_count, 1)

    def test_reasoning_only_or_empty_reply_uses_original_input(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-4o'))[0]
        for payload in [reply(''), reply('<think>unfinished'),
                        {'choices': [{'message': {'reasoning_content': 'hidden thought', 'content': None}}]}]:
            with patch.object(client, '_request', return_value=payload):
                self.assertEqual(client.chat(config, [{'role': 'user', 'content': '原始森林要求'}]), '原始森林要求')

    def test_plan_format_recovery_keeps_extra_text_and_avoids_thought_leakage(self):
        plan = '这是结果：\n```json\n{"image_prompts":["<think>hidden</think>森林"],"text_output":"一首诗"}\n```'
        self.assertEqual(nodes._extract_plan(plan, '原始要求'), (['森林'], '一首诗'))
        self.assertEqual(nodes._extract_plan('{"image_prompts":[', '原始要求'), (['原始要求'], ''))
        self.assertEqual(nodes._extract_plan('森林\n\n海洋', '原始要求', True), (['森林', '海洋'], ''))

    def test_image_expansion_failure_still_generates_from_original_request(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-4o', API图像模型='image'))[0]
        with patch.object(client, 'chat', side_effect=client.WujiApiError('扩写服务不可用')), \
             patch.object(client, 'generate_image', return_value={'data': []}) as generate, \
             patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
             patch.object(nodes.media, 'images_from_payload', new=AsyncMock(return_value=object())), \
             patch.object(nodes.media, '_pad_and_stack', return_value=object()):
            output = run(nodes.CreationImage.execute(动力源=config, prompt='原始森林要求'))
        self.assertEqual(generate.call_args.args[1], '原始森林要求')
        self.assertEqual(output.result[0], '')
        self.assertIn('原始输入', output.result[2])

    def test_video_native_resolution_and_seed_reach_request(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='gpt-4o', API视频模型='video'))[0]
        bodies = []
        plans = []
        events = []
        video = object()
        def request(*args, **kwargs):
            if args[1].endswith('chat/completions'):
                plans.append(kwargs['body'])
                events.append('chat')
                return reply(f'扩写完成的镜头描述 {len(plans)}')
            bodies.append(kwargs['body'])
            events.append('video')
            return {'status': 'completed', 'data': [{'url': 'mock'}]}
        with patch.object(client, '_request', side_effect=request), patch.object(nodes.media, 'video_from_payload', new=AsyncMock(return_value=(video, ''))):
            for resolution, seed in [('480p', 42), ('720p', 43), ('1080p', 44), ('自动', 45)]:
                output = run(nodes.CreationVideo.execute(动力源=config, meta_instruction='请扩写成连贯镜头', prompt='动态画面', resolution=resolution, seed=seed))
                self.assertIs(output.result[1], video)
                self.assertEqual(output.result[0], f'扩写完成的镜头描述 {len(plans)}')
        self.assertEqual([body.get('resolution') for body in bodies], ['480p', '720p', '1080p', None])
        self.assertEqual([body['seed'] for body in bodies], [42, 43, 44, 45])
        self.assertEqual(events, ['chat', 'video'] * 4)
        self.assertTrue(all('seed' not in plan and plan['model'] == 'gpt-4o' for plan in plans))
        self.assertIn('请扩写成连贯镜头', plans[0]['messages'][0]['content'])
        self.assertNotEqual(plans[0]['messages'], plans[1]['messages'])
        self.assertEqual(bodies[0]['content'][0]['text'], '扩写完成的镜头描述 1')
        self.assertEqual(config['seed'], 1)

    def test_video_planner_receives_media_and_keeps_missing_reference_tags(self):
        config = run(nodes.CreationAPI.execute(API文本视觉模型='vision', API视频模型='video'))[0]
        with patch.object(client, 'chat', return_value='最终视频提示词') as chat, \
             patch.object(client, 'generate_video', return_value={'status': 'completed', 'data': []}) as generate, \
             patch.object(nodes.media, 'image_tensor_to_data_uris', return_value=['image-uri']), \
             patch.object(nodes.media, 'video_to_data_uri', return_value='video-uri'), \
             patch.object(nodes.media, 'audio_to_data_uri', return_value='audio-uri'), \
             patch.object(nodes.media, 'video_from_payload', new=AsyncMock(return_value=(object(), ''))):
            output = run(nodes.CreationVideo.execute(动力源=config, prompt='结合所有素材扩写', images={'图像_0': object()}, videos={'视频_0': object()}, audios={'音频_0': object()}, seed=42))
        messages = chat.call_args.args[1]
        types = [item['type'] for item in messages[1]['content']]
        self.assertEqual(types, ['text', 'image_url', 'video_url', 'input_audio'])
        for tag in ('@图片1', '@视频1', '@音频1'):
            self.assertIn(tag, generate.call_args.args[1])
        self.assertTrue(generate.call_args.args[1].startswith('最终视频提示词'))
        self.assertEqual(generate.call_args.kwargs['extra']['seed'], 42)
        self.assertEqual(json.loads(output.result[2])['final_prompt'], output.result[0])

    def test_video_never_bypasses_missing_or_failed_text_model(self):
        config = run(nodes.CreationAPI.execute(API视频模型='video'))[0]
        with patch.object(client, 'generate_video') as generate:
            with self.assertRaisesRegex(ValueError, '文本/视觉模型'):
                run(nodes.CreationVideo.execute(动力源=config, prompt='请求'))
            generate.assert_not_called()
        config = run(nodes.CreationAPI.execute(API文本视觉模型='chat', API视频模型='video'))[0]
        with patch.object(client, 'chat', side_effect=client.WujiApiError('扩写失败')), \
             patch.object(client, 'generate_video', return_value={'data': []}) as generate, \
             patch.object(client, 'poll_media', new=AsyncMock(return_value={'data': []})), \
             patch.object(nodes.media, 'video_from_payload', new=AsyncMock(return_value=(object(), ''))):
            output = run(nodes.CreationVideo.execute(动力源=config, prompt='请求'))
            self.assertEqual(generate.call_args.args[1], '请求')
            self.assertTrue(json.loads(output.result[2])['expansion_fallback'])

    def test_other_api_image_providers_keep_standard_generation_endpoint(self):
        config = run(nodes.CreationAPI.execute(API服务商='OpenAI', API图像模型='image'))[0]
        self.assertFalse(nodes._task_config(config)['_wuji_async_image_tasks'])
        with patch.object(client, '_request', return_value={'data': []}) as request:
            client.generate_image(nodes._task_config(config), 'prompt')
        self.assertTrue(request.call_args.args[1].endswith('/images/generations'))

    def test_video_compatibility_fallback_preserves_generation_parameters(self):
        config = run(nodes.CreationAPI.execute(API视频模型='video'))[0]
        with patch.object(client, '_request', side_effect=[client.WujiApiError('HTTP 400 upstream_request_failed'), {'data': []}]) as request:
            client.generate_video(config, 'prompt', images=['image'], image_roles=['reference_image'], duration=6, extra={'seed': 42, 'resolution': '720p'})
        body = request.call_args.kwargs['body']
        self.assertEqual((body['seed'], body['resolution'], body['duration']), (42, '720p', 6))
        self.assertNotIn('role', body['content'][1])

    def test_large_json_image_response_has_no_previous_small_size_limit(self):
        response = requests.Response()
        response.status_code = 200
        encoded = 'A' * (17 * 1024 * 1024)
        response._content = json.dumps({'data': [{'b64_json': encoded}]}).encode()
        with patch.object(client.requests, 'request', return_value=response):
            payload = client._request('GET', 'https://example.invalid/v1/images/generations/test', '')
        self.assertEqual(len(payload['data'][0]['b64_json']), len(encoded))

    def test_polling_uses_task_url_without_resubmitting(self):
        config = run(nodes.CreationAPI.execute(API图像模型='image'))[0]
        with patch.object(client, '_request', return_value={'status': 'completed', 'data': []}) as request:
            run(client.poll_media(config, {'id': 'task', 'status': 'queued', 'poll_url': '/unified/v1/images/tasks/task'}, 'image'))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0], 'GET')
        self.assertTrue(request.call_args.args[1].endswith('/images/tasks/task'))
        with self.assertRaises(client.WujiApiError):
            run(client.poll_media(config, {'id': 'task', 'poll_url': 'https://other.invalid/task'}, 'image'))

    def test_async_image_default_query_path_matches_unified_task_contract(self):
        config = run(nodes.CreationAPI.execute(API图像模型='image'))[0]
        with patch.object(client, '_request', return_value={'id': 'task', 'status': 'completed', 'data': []}) as request:
            run(client.poll_media(config, {'id': 'task', 'status': 'pending'}, 'image'))
        self.assertEqual(request.call_args.args[:2], ('GET', 'https://api.creative-reasoning.com/unified/v1/tasks/task'))
        config = run(nodes.CreationAPI.execute(API服务商='OpenAI', API图像模型='image'))[0]
        with patch.object(client, '_request', return_value={'status': 'completed', 'data': []}) as request:
            run(client.poll_media(config, {'id': 'task', 'status': 'pending'}, 'image'))
        self.assertTrue(request.call_args.args[1].endswith('/images/generations/task'))

    def test_async_acceptance_respects_location_and_rejects_external_origin(self):
        response = requests.Response()
        response.status_code = 202
        response._content = b'{"id":"task","status":"pending"}'
        response.headers['Location'] = '/unified/v1/tasks/task'
        config = run(nodes.CreationAPI.execute(API图像模型='image'))[0]
        with patch.object(client.requests, 'request', return_value=response):
            task = client._request('POST', 'https://example.invalid/v1/images/tasks', '')
        self.assertEqual(task['poll_url'], response.headers['Location'])
        response.headers['Location'] = 'https://other.invalid/tasks/task'
        with patch.object(client.requests, 'request', return_value=response):
            task = client._request('POST', 'https://example.invalid/v1/images/tasks', '')
        with patch.object(client, '_request') as request:
            with self.assertRaisesRegex(client.WujiApiError, '不同服务地址'):
                run(client.poll_media(config, task, 'image'))
            request.assert_not_called()

    def test_async_failures_keep_structured_reason_and_ids_without_credentials(self):
        config = run(nodes.CreationAPI.execute(API图像模型='image', API密钥='test-secret-12345'))[0]
        failure = {'id': 'task-123', 'request_id': 'request-456', 'model': 'image', 'status': 'failed',
                   'error': {'code': 'unsupported_capability', 'type': 'invalid_request_error', 'param': 'seed',
                             'message': 'unsupported seed; test-secret-12345; Bearer another-secret; https://example.invalid/?api_key=secret; data:image/png;base64,AAAA'}}
        with patch.object(client, '_request', return_value=failure) as request:
            with self.assertRaises(client.WujiApiError) as caught:
                run(client.poll_media(config, {'id': 'task-123', 'status': 'pending'}, 'image'))
        message = str(caught.exception)
        for term in ['unsupported_capability', 'unsupported seed', 'param=seed', 'task-123', 'request-456', '未自动重复提交']:
            self.assertIn(term, message)
        for secret in ['test-secret-12345', 'another-secret', 'https://example', 'AAAA']:
            self.assertNotIn(secret, message)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0], 'GET')

    def test_pending_empty_error_fields_are_not_misreported_as_failed(self):
        config = run(nodes.CreationAPI.execute(API图像模型='image'))[0]
        task = {'id': 'task', 'status': 'pending', 'error': {'code': 0, 'message': '', 'type': None}}
        with patch.object(client, '_request', return_value={'status': 'completed', 'data': []}) as request:
            self.assertEqual(run(client.poll_media(config, task, 'image'))['status'], 'completed')
        self.assertEqual(request.call_count, 1)

    def test_task_failure_without_detail_and_polling_timeout_keep_task_identity(self):
        config = run(nodes.CreationAPI.execute(API图像模型='image'))[0]
        with patch.object(client, '_request') as request:
            with self.assertRaisesRegex(client.WujiApiError, '服务端未提供具体原因') as caught:
                run(client.poll_media(config, {'id': 'task', 'status': 'failed'}, 'image'))
            self.assertIn('任务编号=task', str(caught.exception))
            request.assert_not_called()
        with patch.object(client, 'time') as clock, patch.object(client, '_request') as request:
            clock.monotonic.side_effect = [0, 601]
            with self.assertRaisesRegex(client.WujiApiError, '轮询超时') as caught:
                run(client.poll_media(config, {'id': 'task', 'status': 'pending'}, 'image'))
            self.assertIn('任务编号=task', str(caught.exception))
            request.assert_not_called()

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)

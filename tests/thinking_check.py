"""Offline strict-thinking contracts; no ComfyUI, credentials or API calls.

Run: python -B tests/thinking_check.py
"""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('creation_thinking_check', ROOT / 'cr_core/api/thinking.py')
thinking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thinking)


class StrictOptionsChecks(unittest.TestCase):
    def test_documented_provider_fields(self):
        cases = [
            ('https://api.creative-reasoning.com/unified/v1', 'creative-reasoning-1.5', {'reasoning_effort': 'none'}),
            ('https://api.creative-reasoning.com/unified/v1', 'creative-reasoning-1.5-flash', {'reasoning_effort': 'none'}),
            ('https://api.creative-reasoning.com/unified/v1', 'creative-reasoning-1.0', {'reasoning_effort': 'none'}),
            ('https://api.creative-reasoning.com/unified/v1', 'deepseek-v4-pro', {'reasoning_effort': 'none'}),
            ('https://api.creative-reasoning.com/unified/v1', 'qwen3.7-max-preview', {'reasoning_effort': 'none'}),
            ('https://api.creative-reasoning.com/unified/v1', 'gemini-2.5-pro', {'reasoning_effort': 'none'}),
            ('https://api.creative-reasoning.com/unified/v1', 'custom-model-alias', {'reasoning_effort': 'none'}),
            ('https://api.openai.com/v1', 'gpt-5.1', {'reasoning_effort': 'none'}),
            ('https://api.openai.com/v1', 'gpt-5.2-2025-12-11', {'reasoning_effort': 'none'}),
            ('https://api.openai.com/v1', 'gpt-5.4', {'reasoning_effort': 'none'}),
            ('https://api.deepseek.com/v1', 'deepseek-flash', {'thinking': {'type': 'disabled'}}),
            ('https://api.deepseek.com/v1', 'deepseek-v4-pro', {'thinking': {'type': 'disabled'}}),
            ('https://dashscope.aliyuncs.com/compatible-mode/v1', 'qwen-plus', {'enable_thinking': False}),
            ('https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'qwen3-32b', {'enable_thinking': False}),
            ('https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'qwen3.5-plus', {'enable_thinking': False}),
            ('https://dashscope-us.aliyuncs.com/compatible-mode/v1', 'qwen-flash', {'enable_thinking': False}),
            ('https://open.bigmodel.cn/api/paas/v4', 'glm-5.2', {'thinking': {'type': 'disabled'}}),
            ('https://api.z.ai/api/paas/v4', 'glm-5.2', {'thinking': {'type': 'disabled'}}),
            ('https://generativelanguage.googleapis.com/v1beta/openai', 'gemini-2.5-flash', {'reasoning_effort': 'none'}),
            ('https://generativelanguage.googleapis.com/v1beta/openai', 'gemini-2.5-flash-lite', {'reasoning_effort': 'none'}),
        ]
        with patch('socket.socket.connect', side_effect=AssertionError('offline only')):
            for base, model, expected in cases:
                with self.subTest(model=model, base=base):
                    actual = thinking.strict_api_options(base, model)
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(actual), 1)

    def test_plain_models_need_no_thinking_parameter(self):
        for host in ('api.openai.com', 'api.creative-reasoning.com'):
            for model in ('gpt-4o', 'gpt-4.1'):
                with self.subTest(host=host, model=model):
                    self.assertEqual(thinking.strict_api_options('https://' + host + '/v1', model), {})

    def test_no_fake_model_or_arbitrary_suffix_exemption(self):
        for model in ('', 'creative-reasoning-image-1.5', 'creative-reasoning-video-1.0'):
            with self.subTest(model=model), self.assertRaises(ValueError):
                thinking.strict_api_options('https://api.creative-reasoning.com/unified/v1', model)

    def test_unknown_chat_alias_uses_gateway_none_without_fallback(self):
        self.assertEqual(thinking.strict_api_options(
            'https://api.creative-reasoning.com/unified/v1', 'custom-chat-alias'),
            {'reasoning_effort': 'none'})

    def test_forced_or_unverified_thinking_is_rejected(self):
        cases = {
            'api.openai.com': ('gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'o1', 'o3', 'o4-mini',
                               'gpt-5.2-pro', 'gpt-5.1-codex', 'gpt-5.4-future'),
            'api.deepseek.com': ('deepseek-reasoner', 'deepseek-chat', 'deepseek-unverified'),
            'dashscope.aliyuncs.com': ('qwen3-235b-a22b-thinking-2507', 'qwen3-thinking',
                                       'qwen3.8-2.4t-a95b', 'qwen3.7-max-preview',
                                       'qwen3.7-max-2026-05-17', 'qwen-plus-made-up',
                                       'qwen-turbo', 'qwen-max'),
            'open.bigmodel.cn': ('glm-5.3', 'glm-5.3-flash', 'glm-5.3-flashx', 'glm-5.2-made-up'),
            'generativelanguage.googleapis.com': ('gemini-2.5-pro', 'gemini-3-pro-preview',
                                                  'gemini-3-flash-preview', 'gemini-4-pro',
                                                  'gemini-2.5-flash-thinking'),
        }
        with patch('socket.socket.connect', side_effect=AssertionError('offline only')):
            for host, models in cases.items():
                for model in models:
                    with self.subTest(host=host, model=model), self.assertRaises(ValueError):
                        thinking.strict_api_options('https://' + host + '/v1', model)

    def test_wrong_provider_and_spoofed_or_invalid_urls_are_rejected(self):
        for base in ('https://unknown.example/v1', 'https://api.openai.com.evil.example/v1',
                     'https://api.deepseek.com/v1', 'https://api.openai.com:8443/v1',
                     'http://api.openai.com/v1', '', 'https://[broken'):
            with self.subTest(base=base), self.assertRaises(ValueError):
                thinking.strict_api_options(base, 'gpt-5.2')

    def test_errors_do_not_repeat_url_credentials(self):
        secret = 'do-not-print-this-secret'
        for base in ('https://user:' + secret + '@api.openai.com/v1',
                     'https://api.openai.com/v1?key=' + secret):
            with self.subTest(base=base), self.assertRaises(ValueError) as error:
                thinking.strict_api_options(base, 'gpt-5.2')
            self.assertNotIn(secret, str(error.exception))

    def test_options_are_fresh_and_legacy_entry_point_still_exists(self):
        first = thinking.strict_api_options('https://api.deepseek.com/v1', 'deepseek-flash')
        first['thinking']['type'] = 'enabled'
        self.assertEqual(thinking.strict_api_options('https://api.deepseek.com/v1', 'deepseek-flash'),
                         {'thinking': {'type': 'disabled'}})
        self.assertTrue(callable(thinking.api_options))


class StrictFinalTextChecks(unittest.TestCase):
    def test_plain_prompt_and_strict_json_survive(self):
        for text in ('A small red fox in a forest.', '慢镜头推进，金色夕阳下的城市。',
                     '{"image_prompts":["A forest"],"text_output":""}',
                     '```json\n{"image_prompts":["A forest"]}\n```'):
            with self.subTest(text=text):
                self.assertEqual(thinking.strict_final_text('  ' + text + '  '), text)

    def test_all_complete_pairs_are_removed(self):
        for opening, closing in thinking._THOUGHT_TAGS:
            with self.subTest(opening=opening):
                self.assertEqual(thinking.strict_final_text(opening + 'hidden thought' + closing + 'A forest'), 'A forest')
                self.assertEqual(thinking.strict_final_text('A ' + opening + 'hidden' + closing + 'forest'), 'A forest')

    def test_nested_multiple_and_case_insensitive_blocks(self):
        text = '<THINK>one<analysis>nested</analysis></THINK><reasoning>two</reasoning>A forest'
        self.assertEqual(thinking.strict_final_text(text), 'A forest')
        self.assertEqual(thinking.strict_final_text('<think>a</think>A forest<think>b</think>'), 'A forest')

    def test_unbalanced_or_crossed_tags_never_fall_back_to_original(self):
        for text in ('<think>unfinished', 'thought</think>A forest',
                     '<think>hidden</think><analysis>unfinished',
                     '<think><analysis>hidden</think></analysis>A forest',
                     '<think>only reasoning</think>', '<think>A<think>B</think>',
                     '<thinking class="analysis">hidden</thinking>A forest',
                     '<analysis >hidden</analysis>A forest'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                thinking.strict_final_text(text)

    def test_analysis_prefix_and_mixed_reasoning_are_rejected(self):
        for text in ('Analysis: let me think about the user request',
                     '[analysis] We need to choose a camera angle',
                     '## Analysis\nWe need to choose a camera angle',
                     '**Reasoning:** Plan first', '思考过程：先分析用户需求',
                     'Let me think about this request.', 'I need to analyze the request.',
                     'A forest\nAnalysis: choose lighting.',
                     '<think>removed</think>Analysis: more thoughts',
                     '<|im_start|>assistant<|meta_sep|>analysis hidden',
                     '<|channel|>analysis\nI need to create an image'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                thinking.strict_final_text(text)

    def test_thought_metadata_and_non_text_content_are_rejected(self):
        for value in ({'thought': True, 'text': 'A forest'}, {'thought': 'true', 'text': 'A forest'},
                      {'type': 'analysis', 'text': 'A forest'}, {'type': 'reasoning', 'text': 'A forest'},
                      {'channel': 'analysis', 'content': 'A forest'},
                      '{"thought":true,"text":"A forest"}',
                      None, [], 123, '', '  ', {'text': ''}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                thinking.strict_final_text(value)

    def test_ordinary_structured_text_is_supported(self):
        self.assertEqual(thinking.strict_final_text({'thought': False, 'type': 'text', 'text': 'A forest'}), 'A forest')
        self.assertEqual(thinking.strict_final_text({'content': 'A forest'}), 'A forest')

    def test_failures_do_not_repeat_reasoning_in_error(self):
        with self.assertRaises(ValueError) as error:
            thinking.strict_final_text('<think>private hidden content')
        self.assertNotIn('private hidden content', str(error.exception))


class GenerationCompatibilityChecks(unittest.TestCase):
    def test_unverified_models_do_not_block_requests(self):
        for base, model in [('https://unknown.example/v1', 'custom-chat'),
                            ('https://api.openai.com/v1', 'o3'),
                            ('https://open.bigmodel.cn/v4', 'glm-5.3')]:
            self.assertEqual(thinking.api_options(base, model), {})

    def test_disable_switch_remains_first_attempt(self):
        self.assertEqual(thinking.api_options('https://api.creative-reasoning.com/v1', 'custom-chat'),
                         {'reasoning_effort': 'none'})

    def test_final_extraction_and_original_input_recovery(self):
        cases = [('<think>hidden</think>A forest', 'A forest'),
                 ('hidden</think>A forest', 'A forest'),
                 ('Analysis: choose lighting\nFinal prompt: A forest', 'A forest'),
                 ('思考过程：先推理\n最终提示词：森林', '森林'),
                 ('<think>unfinished hidden', 'original user request'),
                 ('Analysis: hidden', 'original user request'),
                 ('<think>hidden</think>', 'original user request'),
                 ('', 'original user request')]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(thinking.generation_text(value, 'original user request'), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)

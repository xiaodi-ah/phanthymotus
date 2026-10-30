"""Opt-in, per-instance MOSS Chinese and Thai / Supertonic six-language routing."""
import logging
from .moss_tts_runtime import MossAdapter, language_for
from .supertonic_runtime import SupertonicAdapter, LANGUAGES as SUPER_LANGUAGES

LANGUAGES = ('zh', 'th', *SUPER_LANGUAGES)


def route_language(text, language='auto', default='auto'):
    requested = default if language in (None, '', 'auto') else language
    code = language_for(text, requested)
    if code not in LANGUAGES:
        raise ValueError('Language not enabled in MOSS/Supertonic routing: ' + code)
    return code


class RoutedAdapter:
    fast_delivery = True

    def __init__(self, model_dir, supertonic_model_dir, cuda_graph=True, language='auto', seed=42):
        route_language('test', language)
        self.language = language
        self.moss = MossAdapter(model_dir, cuda_graph=cuda_graph, language='zh', seed=seed)
        self.supertonic = SupertonicAdapter(supertonic_model_dir, seed=seed)

    def warmup(self):
        # Both backends must work before the MCP instance declares readiness.
        yield from self.moss.synthesize_stream_language('你好。', 'zh')
        yield from self.supertonic.synthesize_stream_language('Hello.', 'en')

    def synthesize_stream_language(self, text, language=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('text must be a nonempty string')
        code = route_language(text, language, self.language)
        backend = self.moss if code in ('zh', 'th') else self.supertonic
        logging.getLogger(__name__).info('TTS route: language=%s backend=%s', code, 'moss' if code in ('zh', 'th') else 'supertonic')
        yield from backend.synthesize_stream_language(text, code)

    def synthesize_stream(self, text):
        yield from self.synthesize_stream_language(text)

    def synthesize(self, text, language=None):
        return b''.join(self.synthesize_stream_language(text, language))

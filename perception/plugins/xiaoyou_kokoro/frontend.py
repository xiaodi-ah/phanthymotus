"""One version-pinned frontend shared by training and evaluation."""
import json
import re
import unicodedata

from pathlib import Path
import importlib.util
from importlib.metadata import version
from .text import prepare_text


LATIN = r'A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024f\u1e00-\u1eff'


class Frontend:
    def __init__(self, config_path):
        if importlib.util.find_spec("en_core_web_sm") is None:
            raise RuntimeError("The fixed en_core_web_sm model must be installed at build time")
        if version('en-core-web-sm') != '3.8.0' or version('misaki') != '0.9.4':
            raise RuntimeError('The training frontend requires Misaki 0.9.4 and en_core_web_sm 3.8.0')
        from misaki import en, espeak, zh
        self.en = en.G2P(trf=False, british=False,
                         fallback=espeak.EspeakFallback(british=False), unk='❓')
        self.zh = zh.ZHG2P(version=None, unk='❓')
        self.vocab = json.loads(Path(config_path).read_text())['vocab']

    @staticmethod
    def prepare(text, language=None):
        return prepare_text(text, language)

    def __call__(self, text, language):
        return self.phonemize(self.prepare(text, language), language)

    def _english(self, text):
        if re.fullmatch(r'[A-Za-z]', text.strip()):
            return self.en.lexicon.get_NNP(text.strip())[0]
        # The formula's A is a letter; Misaki otherwise tags it as an article.
        text = re.sub(r'\ba(?=\s+(?:squared|cubed|equals|plus|minus|times|divided)\b)',
                      '[a](/ˈA/)', text, flags=re.IGNORECASE)
        return self.en(text)[0]

    def phonemize(self, text, language):
        normalized = unicodedata.normalize('NFKC', text).strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        # Explicit punctuation equivalences; never remove phonetic content.
        normalized = normalized.replace('·', ' ').replace('–', '—')
        if language == 'en':
            phonemes = self._english(normalized)
        else:
            # Misaki v1.0's legacy Chinese frontend leaves Latin words raw,
            # even when en_callable is supplied. Explicitly route those spans.
            spans = re.split(rf"([{LATIN}][{LATIN}0-9_'’\-]*(?:[ \t]+[{LATIN}][{LATIN}0-9_'’\-]*)*)", normalized)
            result = []
            for span in spans:
                if not span.strip():
                    continue
                if re.match(f'[{LATIN}]', span):
                    result.append(self._english(span))
                else:
                    result.append(self.zh(span)[0])
            phonemes = ' '.join(result)
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        unknown = sorted(set(phonemes) - set(self.vocab))
        if unknown:
            raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        ids = [self.vocab[p] for p in phonemes]
        if not ids:
            raise ValueError('Empty phoneme sequence')
        return phonemes, ids

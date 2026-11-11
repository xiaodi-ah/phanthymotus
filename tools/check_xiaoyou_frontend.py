"""Check deployment phonemes against cached training phonemes, without audio."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
from plugins.xiaoyou_kokoro.frontend import Frontend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    frontend = Frontend(args.config)
    report = {'checked': 0, 'mismatches': 0, 'examples': [], 'versions': {
        name: importlib.metadata.version(name) for name in ['misaki', 'spacy', 'en-core-web-sm']}}
    with Path(args.fixtures).open() as stream:
        for line in stream:
            row = json.loads(line)
            try:
                actual, _ = frontend(row['text'], row['language'])
            except Exception as exc:
                actual = type(exc).__name__ + ': ' + str(exc)
            report['checked'] += 1
            if actual != row['phonemes']:
                report['mismatches'] += 1
                if len(report['examples']) < 20:
                    report['examples'].append({'id': row['id'], 'text': row['text'],
                                               'expected': row['phonemes'], 'actual': actual})
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report['checked'] and not report['mismatches'], 'Frontend differs from training'


if __name__ == '__main__':
    main()

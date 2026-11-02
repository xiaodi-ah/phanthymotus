"""Oversized single sentences, expanded tokenization and retry/cleanup regression."""
import sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.supertonic_runtime.text_chunks import text_chunks
from plugins.supertonic_runtime import SupertonicAdapter,DurationLimitError

def processor(texts,languages):
    return np.zeros((1,len(texts[0])*4+20)),None

class Tests(unittest.TestCase):
    def test_no_loss_and_hard_token_bound(self):
        for language,text in [('de','Das System verwaltet alle Ressourcen '*40),('ru','Система управляет ресурсами '*50),('ja','これは長い文章です。'*100),('en','x'*2000),('fr','cafe\u0301 '*300),('es','el sistema '*150)]:
            chunks=list(text_chunks(text,language,processor))
            self.assertEqual(''.join(''.join(chunks).split()),''.join(text.split()))
            self.assertTrue(all(0<processor([c],[language])[0].shape[-1]<=384 for c in chunks))
            self.assertLessEqual(len(chunks[0]),40 if language=='ja' else 120)
            self.assertTrue(all(not __import__('unicodedata').combining(c[0]) for c in chunks))
    def test_short_sentence_preserved(self):
        self.assertEqual(list(text_chunks('Hello, world.','en',processor)),['Hello, world.'])
    def test_early_comma_does_not_hide_word_boundaries(self):
        text='Bonjour, '+ 'extraordinaire '*30
        chunks=list(text_chunks(text,'fr',processor))
        self.assertEqual([word for chunk in chunks for word in chunk.split()],text.split())
    def test_first_clause_keeps_short_sentence_intact(self):
        plain=lambda texts,languages:(np.zeros((1,len(texts[0])+20)),None)
        for language,text in [
            ('fr','Sur le plan international, il existe près de 200 organisations de visite en course à pied. La plupart d’entre elles fonctionnent indépendamment.'),
            ('es','A nivel mundial, existen cerca de 200 organizaciones de turismo para corredores. La mayoría opera de manera independiente.')]:
            self.assertTrue(next(text_chunks(text,language,plain)).endswith('.'))
    def adapter(self,fail_duration=False):
        a=SupertonicAdapter.__new__(SupertonicAdapter);a.lock=threading.Lock();a.seed=42;a.style=None
        seen=[];released=[]
        def infer(texts,languages,style,steps,speed):
            self.assertEqual((steps,speed),(8,1.05))
            if fail_duration and len(texts[0])>24:raise DurationLimitError()
            seen.append(texts[0]);return np.ones((1,200),np.float32),None
        a.model=SimpleNamespace(text_processor=processor,_infer=infer);a.release_temporary=lambda:released.append(True)
        return a,seen,released
    def test_duration_retry_preserves_content_and_cleans_up(self):
        a,seen,released=self.adapter(True);text='The system manages graphics processors and provides models to users.'
        blocks=list(a.native_blocks(text,'en'));self.assertGreater(len(blocks),1)
        self.assertEqual(''.join(''.join(seen).split()),''.join(text.split()));self.assertEqual(released,[True])
    def test_cancellation_and_failure_release_temporary(self):
        a,_,released=self.adapter();g=a.native_blocks('Hello. '*100,'en');next(g);g.close();self.assertEqual(released,[True]);self.assertFalse(a.lock.locked())
        def fail(*args):raise RuntimeError('inference failure')
        a.model._infer=fail
        with self.assertRaises(RuntimeError):list(a.native_blocks('Hello.','en'))
        self.assertEqual(released,[True,True]);self.assertFalse(a.lock.locked())
if __name__=='__main__':unittest.main()

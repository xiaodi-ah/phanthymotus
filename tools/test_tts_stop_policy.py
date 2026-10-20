"""Changing EOS selection must preserve audio uniforms and the slot embedding."""
import ast,os,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'perception'))
from plugins.moss_tts_runtime import MossAdapter
source=root/'perception/plugins/moss_tts_runtime/__init__.py'
adapter=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MossAdapter')
init=next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
method=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='fused_frame')

class StopPolicyTests(unittest.TestCase):
    def exercise(self,mode,probability=.75):
        namespace={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        captured={}
        def execute(requested,feeds):
            captured.update(random=feeds['random_u'].copy(),requested=requested)
            slot=feeds['random_u'][0,0]<=probability
            outputs={'should_continue':np.array([[int(slot)]]),'text_scores':np.array([[np.log(probability),np.log(1-probability)]],np.float32),'frame_token_ids':np.full((1,16),7 if slot else 9,np.int32)}
            return [outputs[n] for n in requested]
        runtime=SimpleNamespace(greedy_stop=mode=='greedy',rng=np.random.default_rng(42),sessions={'local_fixed_sampled_frame':SimpleNamespace(run=execute)})
        result=namespace['fused_frame'](runtime,np.zeros((1,768),np.float32),previous_token_sets_by_channel=[])
        return result,captured,runtime.rng
    def test_sample_preserves_v1_uniforms_and_stop(self):
        result,captured,rng=self.exercise('sample')
        reference=np.random.default_rng(42);expected=np.clip(reference.random((1,17)),0,.99999994).astype(np.float32)
        self.assertFalse(result[0]);np.testing.assert_array_equal(captured['random'],expected)
        self.assertEqual(captured['requested'],['should_continue','frame_token_ids']);self.assertEqual(rng.random(),reference.random())
    def test_greedy_continuation_uses_slot_audio_and_same_audio_uniforms(self):
        result,captured,rng=self.exercise('greedy')
        reference=np.random.default_rng(42);expected=np.clip(reference.random((1,17)),0,.99999994).astype(np.float32)
        self.assertTrue(result[0]);self.assertEqual(result[1],[7]*16);self.assertEqual(captured['random'][0,0],0)
        np.testing.assert_array_equal(captured['random'][:,1:],expected[:,1:]);self.assertEqual(rng.random(),reference.random())
    def test_greedy_stops_when_eos_is_more_probable(self):
        result,captured,_=self.exercise('greedy',probability=.1)
        self.assertFalse(result[0]);self.assertEqual(captured['requested'],['text_scores','frame_token_ids'])
    def test_tie_continues_as_first_candidate_argmax(self):
        self.assertTrue(self.exercise('greedy',probability=.5)[0][0])
    def test_invalid_scores_fail_instead_of_silent_truncation(self):
        with self.assertRaisesRegex(RuntimeError,'continuation/EOS'):self.exercise('greedy',probability=float('nan'))
    def test_submission_defaults_are_enabled(self):
        defaults={}
        for node in ast.walk(init):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='get' and len(node.args)==2 and all(isinstance(x,ast.Constant) for x in node.args):
                defaults[node.args[0].value]=node.args[1].value
        self.assertEqual(defaults['MOSS_TTS_STOP_MODE'],'greedy')
        self.assertEqual(defaults['MOSS_TTS_TEXT_FRONTEND'],'numbers')
    def test_invalid_policy_fails_before_loading_models(self):
        with patch.dict(os.environ,{'MOSS_TTS_DELIVERY':'optimized','MOSS_TTS_STOP_MODE':'invalid'}):
            with self.assertRaisesRegex(ValueError,'MOSS_TTS_STOP_MODE'):MossAdapter('/missing')

if __name__=='__main__':unittest.main()

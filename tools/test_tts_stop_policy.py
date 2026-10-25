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
    def exercise(self,mode,probability=.75,histories=None,seed=42):
        namespace={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        captured={'calls':[]}
        def execute(requested,feeds):
            captured.update(random=feeds['random_u'].copy(),requested=requested,mask=feeds.get('repetition_seen_mask'))
            captured['calls'].append(dict(random=feeds['random_u'].copy(),mask=feeds.get('repetition_seen_mask')))
            slot=feeds['random_u'][0,0]<=probability
            outputs={'should_continue':np.array([[int(slot)]]),'text_scores':np.array([[np.log(probability),np.log(1-probability)]],np.float32),'frame_token_ids':np.full((1,16),7 if slot else 9,np.int32)}
            return [outputs[n] for n in requested]
        runtime=SimpleNamespace(greedy_stop=mode=='greedy',guard_stop=mode=='guarded',rng=np.random.default_rng(seed),sessions={'local_fixed_sampled_frame':SimpleNamespace(run=execute,inputs=['global_hidden','random_u']+(['repetition_seen_mask'] if histories is not None else []))})
        result=namespace['fused_frame'](runtime,np.zeros((1,768),np.float32),previous_token_sets_by_channel=[] if histories is None else histories)
        captured['last_stop']=runtime.last_stop
        return result,captured,runtime.rng
    def test_sample_preserves_v1_uniforms_and_stop(self):
        result,captured,rng=self.exercise('sample')
        reference=np.random.default_rng(42);expected=np.clip(reference.random((1,17)),0,.99999994).astype(np.float32)
        self.assertFalse(result[0]);np.testing.assert_array_equal(captured['random'],expected)
        self.assertEqual(captured['requested'],['should_continue','frame_token_ids','text_scores']);self.assertEqual(rng.random(),reference.random())
    def test_greedy_continuation_uses_slot_audio_and_same_audio_uniforms(self):
        result,captured,rng=self.exercise('greedy')
        reference=np.random.default_rng(42);expected=np.clip(reference.random((1,17)),0,.99999994).astype(np.float32)
        self.assertTrue(result[0]);self.assertEqual(result[1],[7]*16);self.assertEqual(captured['random'][0,0],0)
        np.testing.assert_array_equal(captured['random'][:,1:],expected[:,1:]);self.assertEqual(rng.random(),reference.random())
    def test_greedy_stops_when_eos_is_more_probable(self):
        result,captured,_=self.exercise('greedy',probability=.1)
        self.assertFalse(result[0]);self.assertEqual(captured['requested'],['should_continue','frame_token_ids','text_scores'])
    def test_tie_continues_as_first_candidate_argmax(self):
        self.assertTrue(self.exercise('greedy',probability=.5)[0][0])
    def test_guard_recomputes_slot_audio_without_extra_random_draws(self):
        histories=[{2}]+[set() for _ in range(15)]
        result,captured,rng=self.exercise('guarded',probability=.995,histories=histories,seed=1074)
        reference=np.random.default_rng(1074);expected=np.clip(reference.random((1,17)),0,.99999994).astype(np.float32)
        self.assertGreater(expected[0,0],.995);self.assertEqual(result,(True,[7]*16))
        self.assertEqual(len(captured['calls']),2);first,second=captured['calls']
        np.testing.assert_array_equal(first['random'],expected)
        self.assertEqual(second['random'][0,0],0)
        np.testing.assert_array_equal(second['random'][:,1:],expected[:,1:])
        self.assertIs(first['mask'],second['mask']);self.assertEqual(int(second['mask'].sum()),1)
        self.assertEqual(rng.random(),reference.random());self.assertTrue(captured['last_stop']['rare_eos_overridden'])
        self.assertEqual(captured['last_stop']['random_u'],float(expected[0,0]))
    def test_guard_preserves_likely_eos_and_normal_continuation(self):
        for probability in (.2,.98,.99999):
            guarded,captured,_=self.exercise('guarded',probability=probability,seed=1074)
            sampled,_,_=self.exercise('sample',probability=probability,seed=1074)
            self.assertEqual(guarded,sampled);self.assertEqual(len(captured['calls']),1)
            self.assertFalse(captured['last_stop']['rare_eos_overridden'])
    def test_invalid_scores_fail_instead_of_silent_truncation(self):
        with self.assertRaisesRegex(RuntimeError,'continuation/EOS'):self.exercise('greedy',probability=float('nan'))
    def test_nano_history_uses_channel_specific_mask(self):
        histories=[set() for _ in range(16)];histories[0]={0,1023};histories[15]={9}
        _,captured,_=self.exercise('sample',histories=histories)
        mask=captured['mask'];self.assertEqual(mask.shape,(1,16,1024));self.assertEqual(mask.dtype,np.int32)
        self.assertEqual(int(mask.sum()),3);self.assertEqual(mask[0,0,1023],1);self.assertEqual(mask[0,15,9],1)
        _,next_call,_=self.exercise('sample',histories=[set() for _ in range(16)])
        self.assertEqual(int(next_call['mask'].sum()),0)
    def test_invalid_history_fails(self):
        with self.assertRaisesRegex(ValueError,'16 repetition'):self.exercise('sample',histories=[])
        histories=[set() for _ in range(16)];histories[0]={1024}
        with self.assertRaisesRegex(ValueError,'Invalid audio token'):self.exercise('sample',histories=histories)
    def test_eos_trace_keeps_original_random_uniform(self):
        result,captured,_=self.exercise('greedy')
        self.assertAlmostEqual(captured['last_stop']['continue_probability'],.75,places=6)
        self.assertAlmostEqual(captured['last_stop']['random_u'],.773956,places=5)
        self.assertTrue(captured['last_stop']['continuing'])
    def test_submission_defaults_are_enabled(self):
        defaults={}
        for node in ast.walk(init):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='get' and len(node.args)==2 and all(isinstance(x,ast.Constant) for x in node.args):
                defaults[node.args[0].value]=node.args[1].value
        self.assertEqual(defaults['MOSS_TTS_STOP_MODE'],'guarded')
        self.assertEqual(defaults['MOSS_TTS_TEXT_FRONTEND'],'numbers')
    def test_invalid_policy_fails_before_loading_models(self):
        with patch.dict(os.environ,{'MOSS_TTS_DELIVERY':'optimized','MOSS_TTS_STOP_MODE':'invalid'}):
            with self.assertRaisesRegex(ValueError,'MOSS_TTS_STOP_MODE'):MossAdapter('/missing')

if __name__=='__main__':unittest.main()

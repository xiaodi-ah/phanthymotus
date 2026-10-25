"""Single-model quality smoke test with existing Orin engines; not a GSB test."""
import argparse, hashlib, json, os, sys, time, wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.moss_tts_runtime import MossAdapter,VOICES

CASES={
    'zh':'你好，请前往会议室。',
    'ja':'室温は23℃です。バッテリーの残量は85%です。',
    'de':'Die Raumtemperatur beträgt 23°C, und der Akku ist zu 85% geladen.',
    'en':'The room temperature is 23°C, and the battery is at 85%.',
    'ru':'Пожалуйста, пройдите в конференц-зал.',
    'es':'Por favor, dirígete a la sala de reuniones.',
    'fr':'Veuillez vous rendre dans la salle de réunion.'}
LONG='Before leaving the lobby, check the battery, cameras, and distance sensors, and confirm that every device is working correctly.'

def reserve():
    fields=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(fields['MemAvailable'].split()[0])/1024
    if available<1024:raise RuntimeError('Host available memory below reserved 1024MiB')
    return available

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',default='/models/moss-tts-trt-jp61-v2');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not os.environ.get('MOSS_TTS_STOP_MODE') and not os.environ.get('MOSS_TTS_TEXT_FRONTEND'),'Test submission defaults without environment overrides'
    args.output.mkdir(parents=True,exist_ok=False)
    report={'state':'running','scope':'one independent model, native adapter; excludes MCP/ROS and three containers','results':[],'minimum_host_available_mb':reserve()}
    def save():
        (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save()
    try:
        model=MossAdapter(args.model_dir,cuda_graph=True)
        assert model.stop_mode=='guarded' and model.text_frontend=='numbers'
        assert VOICES['ja']=='Saki' and VOICES['de']=='Adam'
        assert all(name in model.voices for name in VOICES.values())
        jobs=[(language,text,seed) for seed in (42,2026) for language,text in CASES.items()]+[('en',LONG,seed) for seed in (42,2026)]
        for index,(language,text,seed) in enumerate(jobs):
            report['minimum_host_available_mb']=min(report['minimum_host_available_mb'],reserve())
            model.seed=seed;first=None;parts=[];started=time.perf_counter()
            for pcm in model.synthesize_stream_language(text,language):
                if first is None:first=time.perf_counter()-started
                report['minimum_host_available_mb']=min(report['minimum_host_available_mb'],reserve());parts.append(pcm)
            elapsed=time.perf_counter()-started;audio=b''.join(parts)
            assert audio and len(audio)%2==0
            assert model.runtime.greedy_stop==(language=='en')
            assert model.runtime.guard_stop==(language!='en')
            path=args.output/f'{index:02d}-{language}-{seed}.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(audio)
            row={'language':language,'seed':seed,'text':text,'voice':VOICES[language],'greedy_stop':model.runtime.greedy_stop,'wav':path.name,'pcm_sha256':hashlib.sha256(audio).hexdigest(),'audio_seconds':len(audio)/32000,'first_pcm_seconds':first,'generation_seconds':elapsed,'native_rtf':elapsed/(len(audio)/32000)}
            report['results'].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
        report['state']='completed';report['gsb_or_listening_verified']=False;save()
    except Exception as error:
        report.update(state='failed',error=repr(error));save();raise

if __name__=='__main__':main()

"""MOSS adapter behind the official tts MCP/ROS contract."""
import copy,hashlib,threading
from .tts import TOOLS as BASE_TOOLS,_TTSNode
from .moss_tts_runtime import MossAdapter,LANGUAGES,language_for

TOOLS=copy.deepcopy(BASE_TOOLS)
TOOLS[0]['description']='Local MOSS-TTS-Nano with frame-fused TensorRT, JP6.1 trial'
TOOLS[0]['inputSchema']['properties']['language']={'type':'string','enum':['auto',*LANGUAGES],'default':'auto'}
TOOLS[0]['inputSchema']['properties']['instance_id']={'type':'string'}
TOOLS[0]['configSchema']['properties']={'language':{'type':'string','enum':['auto',*LANGUAGES],'default':'auto','scope':'shared'}}

class TTSPlugin:
    PREFIX='tts'
    def __init__(self,plugin_cfg,executor):
        self.cfg=dict(plugin_cfg);self.executor=executor;self.nodes={};self.load_lock=threading.RLock()
    def get_tools(self):return TOOLS
    def _node(self,key,input_topic=''):
        with self.load_lock:
            if key not in self.nodes:
                # Each MCP instance owns separate engines, KV and codec state.
                adapter=MossAdapter(self.cfg.get('model_dir','/models/moss-tts-trt-jp61-v2'),cuda_graph=bool(self.cfg.get('cuda_graph',True)),language=self.cfg.get('language','auto'),seed=self.cfg.get('seed',42))
                node=_TTSNode(input_topic or None,adapter,node_suffix=hashlib.sha256(key.encode()).hexdigest()[:12])
                self.executor.add_node(node);self.nodes[key]=node
            elif input_topic and self.nodes[key]._input_topic!=input_topic:
                previous=self.nodes[key]
                if previous.state!='idle' or (previous._worker_thread and previous._worker_thread.is_alive()):
                    raise ValueError('Stop the instance before changing its input topic')
                node=_TTSNode(input_topic,previous._adapter,node_suffix=hashlib.sha256(key.encode()).hexdigest()[:12])
                self.executor.remove_node(previous);previous.destroy_node()
                self.executor.add_node(node);self.nodes[key]=node
            return self.nodes[key]
    def dispatch(self,name,args):
        action=args.get('action') if name=='tts' else name
        key=args.get('instance_id') or args.get('input_topic') or '_default'
        if action=='info':
            node=self.nodes.get(key)
            if node:result=node._status_dict()
            elif self.nodes and not args.get('instance_id'):
                states=[n._status_dict() for n in self.nodes.values()]
                result={'state':'running' if any(n['state']=='running' for n in states) else 'idle','topic_in':[t for n in states for t in n['topic_in']],'topic_out':[t for n in states for t in n['topic_out']]}
            else:result={'state':'idle','topic_in':[],'topic_out':[]}
            return {**result,'name':'TTS','manufacture':'Embodied','model':'MOSS-TTS-Nano-100M/TRT-frame-v2','independent_models':len(self.nodes),'languages':list(LANGUAGES),'desc':'JP6.1 experimental submission; quality and three-instance stability pending'}
        if action=='start':return self._node(key,args.get('input_topic','')).start()
        if action=='stop':
            keys=[key] if args.get('instance_id') else list(self.nodes)
            for item in keys:
                if item in self.nodes:
                    node=self.nodes[item];node.stop()
                    if node._worker_thread and node._worker_thread.is_alive():return {'state':'stopping'}
                    # Platform stops after every case; retain this instance's model to avoid reload peaks.
            return {'state':'idle'}
        if action=='config':
            language=args.get('language',args.get('tts_language',self.cfg.get('language','auto')))
            language_for('test',language)
            if float(args.get('speed',1.0))!=1.0 or int(args.get('speaker_id',0))!=0:raise ValueError('Trial supports speed=1 and default speaker only')
            self.cfg['language']=language
            for node in self.nodes.values():node._adapter.language=language
            return {'status':'configured','language':language}
        if action=='speak':
            text=args.get('text','');language=args.get('language',args.get('lang','auto'))
            if not isinstance(text,str) or not text.strip():raise ValueError('text must be a nonempty string')
            language_for(text,language)
            if not args.get('instance_id') and not args.get('input_topic') and key not in self.nodes and self.nodes:
                if len(self.nodes)!=1:raise ValueError('Specify instance_id or input_topic when multiple TTS instances are running')
                key=next(iter(self.nodes))
            node=self._node(key,args.get('input_topic',''))
            if node.state!='running':
                status=node.start()
                if status.get('state')!='running':return status
            node.enqueue(text,trace_id=args.get('_trace_id',''),language=language)
            return {'status':'queued','text':text}
        return None
    def synthesize_raw(self,text,language='auto'):
        node=self.nodes.get('_default') or next(iter(self.nodes.values()),None) or self._node('_default')
        return node._adapter.synthesize(text,language=language)

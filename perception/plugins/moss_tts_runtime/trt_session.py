"""ORT-shaped adapter with GPU-resident recurrent state and bounded buffers."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import tensorrt as trt

class TrtSession:
 def __init__(self,path):
  self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
  self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
  if self.engine is None:raise RuntimeError('Cannot load TensorRT engine: '+str(path))
  self.context=self.engine.create_execution_context();self.stream=torch.cuda.Stream()
  self.names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
  self.inputs=[n for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
  self.outputs=[n for n in self.names if n not in self.inputs];self.buffers={};self.parity=0;self.input_shapes={};self.addresses={}
  self.info={}
  for name in self.names:
   npdtype=trt.nptype(self.engine.get_tensor_dtype(name));dtype=torch.from_numpy(np.empty((),dtype=npdtype)).dtype
   device='cpu' if self.engine.get_tensor_location(name)==trt.TensorLocation.HOST else 'cuda'
   self.info[name]=(npdtype,dtype,device)
  self.output_info=[SimpleNamespace(name=n) for n in self.outputs]
 def get_outputs(self):return self.output_info
 def buffer(self,key,count,dtype,device):
  old=self.buffers.get(key)
  if old is None or old.numel()<count:
   old=torch.empty(max(1,count if old is None else max(count,old.numel()*2)),dtype=dtype,device=device);self.buffers[key]=old
  return old[:count]
 def bind(self,name,tensor):
  address=tensor.data_ptr()
  if self.addresses.get(name)!=address:
   if not self.context.set_tensor_address(name,address):raise RuntimeError('Tensor binding failed: '+name)
   self.addresses[name]=address
 def run(self,requested,feeds):
  if set(feeds)!=set(self.inputs):raise ValueError('Engine input names mismatch')
  self.parity=1-self.parity;alive=[];out={}
  with torch.cuda.stream(self.stream):
   for name in self.inputs:
    value=feeds[name];npdtype,dtype,device=self.info[name]
    if isinstance(value,torch.Tensor):
     tensor=value.to(device=device,dtype=dtype).contiguous()
    else:
     array=np.ascontiguousarray(value,dtype=npdtype);tensor=self.buffer(('in',name),array.size,dtype,device).reshape(array.shape)
     if array.size:tensor.copy_(torch.from_numpy(array))
    shape=tuple(tensor.shape)
    if self.input_shapes.get(name)!=shape:
     if not self.context.set_input_shape(name,shape):raise RuntimeError('Invalid input shape: '+name)
     self.input_shapes[name]=shape
    alive.append(tensor)
    if tensor.numel()==0:tensor=self.buffer(('empty',name),1,dtype,device);alive.append(tensor)
    self.bind(name,tensor)
   missing=self.context.infer_shapes()
   if missing:raise RuntimeError('Unresolved shape inputs: '+str(missing))
   for name in self.outputs:
    shape=tuple(self.context.get_tensor_shape(name))
    if any(d<0 for d in shape):raise RuntimeError('Data-dependent output needs allocator: '+name)
    npdtype,dtype,device=self.info[name]
    count=1
    for dim in shape:count*=dim
    buffer=self.buffer(('out',name,self.parity),max(1,count),dtype,device)
    out[name]=buffer[:count].reshape(shape)
    self.bind(name,buffer)
   if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('TensorRT enqueue failed')
  self.stream.synchronize()
  # Recurrent outputs keep their own device storage; logits/audio are consumed on CPU.
  selected=self.outputs if requested is None else requested
  for name in selected:
   if name in ('text_logits','audio_logits','audio','audio_lengths'):
    value=out[name]
    if name=='audio_logits' and 'step_type' in feeds:
     step=int(feeds['step_type'][0]);channel=0 if step==1 else int(feeds['channel_index'][0])+1
     value=value[:,channel,:]
    out[name]=value.cpu().numpy().copy()
  return [out[n] for n in (self.outputs if requested is None else requested)]

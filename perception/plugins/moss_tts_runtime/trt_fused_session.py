"""Static fused-frame execution with fixed addresses and optional CUDA Graph."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import tensorrt as trt

class FusedSession:
 def __init__(self,path,cuda_graph=False):
  self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
  self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
  if self.engine is None:raise RuntimeError('Fused engine deserialization failed')
  self.context=self.engine.create_execution_context();self.stream=torch.cuda.Stream();self.graph=None
  self.inputs=[];self.outputs=[];self.buffers={};self.host={}
  for i in range(self.engine.num_io_tensors):
   name=self.engine.get_tensor_name(i);shape=tuple(self.engine.get_tensor_shape(name));dtype=trt.nptype(self.engine.get_tensor_dtype(name))
   if any(d<0 for d in shape) or self.engine.get_tensor_location(name)!=trt.TensorLocation.DEVICE:raise RuntimeError('Fused engine must have static device IO')
   tensor_dtype=torch.from_numpy(np.empty((),dtype=dtype)).dtype
   self.buffers[name]=torch.empty(shape,device='cuda',dtype=tensor_dtype)
   if not self.context.set_tensor_address(name,self.buffers[name].data_ptr()):raise RuntimeError('Binding failed: '+name)
   if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
    self.inputs.append(name);self.host[name]=torch.empty(shape,dtype=tensor_dtype,pin_memory=True)
   else:self.outputs.append(name)
  assert set(self.inputs)=={'global_hidden','random_u'}
  assert self.buffers['audio_topk_scores'].dtype==torch.float32 and self.buffers['text_scores'].dtype==torch.float32
  with torch.cuda.stream(self.stream):
   self.buffers['global_hidden'].zero_();self.buffers['random_u'].fill_(.5)
   for _ in range(3):self.enqueue()
  self.stream.synchronize()
  if cuda_graph:
   self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph,stream=self.stream):self.enqueue()
   self.stream.synchronize()
 def enqueue(self):
  if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('Fused TensorRT enqueue failed')
 def get_outputs(self):return [SimpleNamespace(name=n) for n in self.outputs]
 def run(self,requested,feeds):
  if set(feeds)!=set(self.inputs):raise ValueError('Fused frame inputs mismatch')
  with torch.cuda.stream(self.stream):
   for name in self.inputs:
    target=self.buffers[name];value=feeds[name]
    if tuple(value.shape)!=tuple(target.shape):raise ValueError('Fused input shape mismatch: '+name)
    if isinstance(value,torch.Tensor):target.copy_(value,non_blocking=True)
    else:
     host=self.host[name];host.copy_(torch.from_numpy(np.ascontiguousarray(value,dtype=host.numpy().dtype)))
     target.copy_(host,non_blocking=True)
   if self.graph:self.graph.replay()
   else:self.enqueue()
  self.stream.synchronize()
  return [self.buffers[n].cpu().numpy().copy() for n in (self.outputs if requested is None else requested)]

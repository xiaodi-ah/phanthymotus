"""Fixed-shape streaming codec with GPU recurrence and one graph/sync per frame."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import tensorrt as trt

class CodecGraphSession:
    def __init__(self,path,meta):
        self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('Codec engine deserialization failed')
        self.context=self.engine.create_execution_context();self.stream=torch.cuda.Stream()
        self.inputs=[];self.outputs=[];self.buffers={};self.host={};self.np_dtype={}
        self.state_outputs={}
        for spec in meta['streaming_decode']['transformer_offsets']:
            self.state_outputs[spec['output_name']]=spec['input_name']
        for spec in meta['streaming_decode']['attention_caches']:
            for source,target in [('offset_output_name','offset_input_name'),('cached_keys_output_name','cached_keys_input_name'),('cached_values_output_name','cached_values_input_name'),('cached_positions_output_name','cached_positions_input_name')]:
                self.state_outputs[spec[source]]=spec[target]
        names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for name in names:
            if self.engine.get_tensor_location(name)!=trt.TensorLocation.DEVICE or self.engine.is_shape_inference_io(name):
                raise RuntimeError('Codec graph requires device IO without shape inputs')
            self.np_dtype[name]=trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
                self.inputs.append(name);shape=tuple(self.engine.get_tensor_shape(name))
                if any(dim<0 for dim in shape):
                    low,shape,high=self.engine.get_tensor_profile_shape(name,0);shape=tuple(shape)
                    if tuple(low)!=shape or tuple(high)!=shape:raise RuntimeError('Codec graph requires a fixed profile')
                if not self.context.set_input_shape(name,shape):raise RuntimeError('Codec shape binding failed')
                dtype=torch.from_numpy(np.empty((),dtype=self.np_dtype[name])).dtype
                self.buffers[name]=torch.empty(shape,dtype=dtype,device='cuda')
                if not self.context.set_tensor_address(name,self.buffers[name].data_ptr()):raise RuntimeError('Codec input binding failed')
            else:self.outputs.append(name)
        if set(self.inputs)!=set(self.state_outputs.values())|{'audio_codes','audio_code_lengths'}:
            raise RuntimeError('Codec state metadata does not match engine inputs')
        if self.context.infer_shapes():raise RuntimeError('Unresolved codec shapes')
        for name in self.outputs:
            shape=tuple(self.context.get_tensor_shape(name))
            if any(dim<0 for dim in shape):raise RuntimeError('Codec output must have a fixed shape')
            dtype=torch.from_numpy(np.empty((),dtype=self.np_dtype[name])).dtype
            self.buffers[name]=torch.empty(shape,dtype=dtype,device='cuda')
            if not self.context.set_tensor_address(name,self.buffers[name].data_ptr()):raise RuntimeError('Codec output binding failed')
        self.output_info=[SimpleNamespace(name=name) for name in self.outputs]
        for name in ('audio_codes','audio_code_lengths','audio','audio_lengths'):
            value=self.buffers[name];self.host[name]=torch.empty(tuple(value.shape),dtype=value.dtype,pin_memory=True)
        positions={spec['cached_positions_input_name'] for spec in meta['streaming_decode']['attention_caches']}
        with torch.cuda.stream(self.stream):
            for name in self.inputs:self.buffers[name].fill_(-1 if name in positions else (1 if name=='audio_code_lengths' else 0))
            for _ in range(3):self.enqueue()
        self.stream.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.stream):self.enqueue()
        self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            for name in self.state_outputs.values():self.buffers[name].fill_(-1 if name in positions else 0)
        self.stream.synchronize()
    def enqueue(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('Codec TensorRT enqueue failed')
        # Returned recurrent values alias this input bank after the captured copy.
        for output,input_name in self.state_outputs.items():self.buffers[input_name].copy_(self.buffers[output])
    def get_outputs(self):return self.output_info
    def run(self,requested,feeds):
        if set(feeds)!=set(self.inputs):raise ValueError('Codec inputs mismatch')
        with torch.cuda.stream(self.stream):
            for name in self.inputs:
                value=feeds[name];target=self.buffers[name]
                if value is target:continue
                if tuple(value.shape)!=tuple(target.shape):raise ValueError('Codec input shape mismatch: '+name)
                if isinstance(value,torch.Tensor):target.copy_(value,non_blocking=True)
                else:
                    array=np.ascontiguousarray(value,dtype=self.np_dtype[name])
                    if name=='audio_code_lengths' and (array.size!=1 or int(array.flat[0])!=1):raise ValueError('Codec graph supports one frame')
                    if name in self.host:
                        self.host[name].copy_(torch.from_numpy(array));target.copy_(self.host[name],non_blocking=True)
                    else:target.copy_(torch.from_numpy(array))
            self.graph.replay()
            for name in ('audio','audio_lengths'):self.host[name].copy_(self.buffers[name],non_blocking=True)
        self.stream.synchronize()
        return [self.host[name].numpy().copy() if name in ('audio','audio_lengths') else self.buffers[self.state_outputs.get(name,name)] for name in (self.outputs if requested is None else requested)]

"""Small NumPy/CUDA TensorRT bindings for the existing serial Supertonic stages."""
import ctypes
import ctypes.util
import mmap
from pathlib import Path

import numpy as np

_cuda = None
allocated_bytes = 0
peak_bytes = 0


class Cuda:
    def __init__(self):
        self.lib = ctypes.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so.12')
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.check(self.lib.cudaSetDevice(0))

    def check(self, result):
        if result:
            raise RuntimeError(self.lib.cudaGetErrorString(result).decode())

    def malloc(self, count):
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), count))
        return pointer.value

    def free(self, pointer):
        self.check(self.lib.cudaFree(ctypes.c_void_p(pointer)))

    def stream(self):
        value = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreateWithFlags(ctypes.byref(value), 1))
        return value.value

    def sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)))

    def copy(self, destination, source, count, kind, stream):
        self.check(self.lib.cudaMemcpyAsync(destination, source, count, kind, stream))


def cuda():
    global _cuda
    if _cuda is None:
        _cuda = Cuda()
    return _cuda


class Storage:
    def __init__(self, count):
        global allocated_bytes, peak_bytes
        self.size = max(1, int(count))
        self.owner = cuda()
        self.pointer = self.owner.malloc(self.size)
        allocated_bytes += self.size
        peak_bytes = max(peak_bytes, allocated_bytes)

    def __del__(self):
        global allocated_bytes
        if getattr(self, 'pointer', None):
            self.owner.free(self.pointer)
            allocated_bytes -= self.size
            self.pointer = None


class CpuValue:
    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


class DeviceValue:
    def __init__(self, storage, shape, dtype, stream):
        self.storage = storage
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.stream = stream

    def cpu(self):
        result = np.empty(self.shape, dtype=self.dtype)
        if result.size:
            cuda().copy(result.ctypes.data, self.storage.pointer, result.nbytes, 2, self.stream)
            cuda().sync(self.stream)
        return CpuValue(result)


class TrtSession:
    def __init__(self, path, workspace=None):
        import tensorrt as trt
        self.trt = trt
        if trt.__version__ != '10.4.0':
            raise RuntimeError('Expected the existing TensorRT10.4 Orin engines')
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with Path(path).open('rb') as source:
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
                self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError('Cannot deserialize engine: '+str(path))
        self.context = self.engine.create_execution_context_without_device_memory()
        if self.context is None:
            raise RuntimeError('Cannot create TensorRT context')
        self.workspace = {} if workspace is None else workspace
        self.stream = cuda().stream()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.buffers = {}
        self.parity = 0

    def buffer(self, key, size):
        old = self.buffers.get(key)
        if old is None or old.size < max(1, size):
            old = Storage(size)
            self.buffers[key] = old
        return old

    def release_temporary(self):
        cuda().sync(self.stream)
        self.buffers.clear()

    def run(self, requested, feeds):
        if set(feeds) != set(self.inputs):
            raise ValueError('TensorRT input names mismatch')
        self.parity = 1-self.parity
        alive = []
        for name in self.inputs:
            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            value = feeds[name]
            if isinstance(value, DeviceValue) and value.dtype == dtype:
                shape, address = value.shape, value.storage.pointer
                alive.append(value)
            else:
                if isinstance(value, DeviceValue):
                    value = value.cpu().numpy()
                array = np.ascontiguousarray(value, dtype=dtype)
                shape = array.shape
                alive.append(array)
                if self.engine.get_tensor_location(name) == self.trt.TensorLocation.HOST:
                    address = array.ctypes.data
                else:
                    storage = self.buffer(('in', name), array.nbytes)
                    address = storage.pointer
                    if array.size:
                        cuda().copy(address, array.ctypes.data, array.nbytes, 1, self.stream)
            if not self.context.set_input_shape(name, shape):
                raise ValueError('Invalid TensorRT shape: '+name)
            if not self.context.set_tensor_address(name, address):
                raise RuntimeError('Cannot bind TensorRT input: '+name)
        missing = self.context.infer_shapes()
        if missing:
            raise RuntimeError('Unresolved shapes: '+str(missing))
        required = max(256, self.context.update_device_memory_size_for_shapes())
        scratch = self.workspace.get('buffer')
        if scratch is None or scratch.size < required:
            scratch = Storage(required)
            self.workspace['buffer'] = scratch
        self.context.device_memory = scratch.pointer
        out = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError('Unresolved TensorRT output shape: '+name)
            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            storage = self.buffer(('out', name, self.parity), int(np.prod(shape))*dtype.itemsize)
            out[name] = DeviceValue(storage, shape, dtype, self.stream)
            if not self.context.set_tensor_address(name, storage.pointer):
                raise RuntimeError('Cannot bind TensorRT output: '+name)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError('TensorRT enqueue failed')
        cuda().sync(self.stream)
        return [out[name] for name in (self.outputs if requested is None else requested)]

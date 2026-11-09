"""CPU mock verifies byte copies, recurrent buffers, resizing and cleanup."""
import ctypes
import gc
from types import SimpleNamespace
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from plugins.supertonic_runtime import direct_trt_session as runtime


class FakeCuda:
    def __init__(self):
        self.memory = {}

    def malloc(self, count):
        value = ctypes.create_string_buffer(count)
        pointer = ctypes.addressof(value)
        self.memory[pointer] = value
        return pointer

    def free(self, pointer):
        del self.memory[pointer]

    def copy(self, destination, source, count, kind, stream):
        ctypes.memmove(destination, source, count)

    def sync(self, stream):
        pass


class Context:
    def __init__(self):
        self.addresses = {}

    def set_input_shape(self, name, shape):
        self.shape = shape
        return True

    def set_tensor_address(self, name, address):
        self.addresses[name] = address
        return True

    def infer_shapes(self):
        return []

    def update_device_memory_size_for_shapes(self):
        return 256

    def get_tensor_shape(self, name):
        return self.shape

    def execute_async_v3(self, stream):
        assert self.addresses['x'] != self.addresses['y']
        size = int(np.prod(self.shape))
        values = np.ctypeslib.as_array((ctypes.c_float*size).from_address(self.addresses['x']))
        output = (values+1).astype('float32')
        ctypes.memmove(self.addresses['y'], output.ctypes.data, output.nbytes)
        return True


def main():
    runtime._cuda = FakeCuda()
    session = runtime.TrtSession.__new__(runtime.TrtSession)
    session.trt = SimpleNamespace(nptype=lambda v:v, TensorLocation=SimpleNamespace(HOST='host'))
    session.engine = SimpleNamespace(get_tensor_dtype=lambda n:np.float32,
                                     get_tensor_location=lambda n:'device')
    session.context = Context()
    session.inputs, session.outputs = ['x'], ['y']
    session.stream, session.workspace, session.buffers, session.parity = 1, {}, {}, 0
    first = session.run(None, {'x':np.array([1, 2], dtype='float32')})[0]
    np.testing.assert_array_equal(first.cpu().numpy(), [2, 3])
    second = session.run(None, {'x':first})[0]
    np.testing.assert_array_equal(second.cpu().numpy(), [3, 4])
    third = session.run(None, {'x':second})[0]
    np.testing.assert_array_equal(third.cpu().numpy(), [4, 5])
    resized = session.run(['y'], {'x':np.arange(10, dtype='float32')})[0]
    np.testing.assert_array_equal(resized.cpu().numpy(), np.arange(1, 11))
    try:
        session.run(None, {'wrong':np.ones(2)})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid feed names accepted')
    del first, second, third, resized
    session.release_temporary()
    session.workspace.clear()
    gc.collect()
    assert runtime.allocated_bytes == 0 and not runtime._cuda.memory
    print('Roundtrip, recurrent double buffers, resize, validation and GPU buffer cleanup passed')


if __name__ == '__main__':
    main()

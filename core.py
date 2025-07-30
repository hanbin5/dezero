import contextlib
import weakref

import numpy as np

import dezero
from dezero.cuda import get_array_module, is_cupy_available, to_cpu, to_gpu

if is_cupy_available:
    import cupy


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def test_mode():
    return using_config('train', False)

def no_grad():
    return using_config('enable_backprop', False)



class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            # Allow cupy.ndarray
            if not isinstance(data, (np.ndarray, cupy.ndarray) if is_cupy_available else np.ndarray):
                raise TypeError('{}: invalid type'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property 
    def shape(self):
        return self.data.shape

    @property 
    def ndim(self):
        return self.data.ndim

    @property 
    def size(self):
        return self.data.size

    @property 
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        
        # Support cupy
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = to_cpu(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = to_gpu(self.data)


    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # Support cupy
            xp = get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def __add__(self, other):
        return dezero.functions.add(self, other)

    def __radd__(self, other):
        return dezero.functions.add(other, self)

    def __mul__(self, other):
        return dezero.functions.mul(self, other)

    def __rmul__(self, other):
        return dezero.functions.mul(other, self)

    def __neg__(self):
        return dezero.functions.neg(self)

    def __sub__(self, other):
        return dezero.functions.sub(self, other)

    def __rsub__(self, other):
        return dezero.functions.sub(other, self)

    def __truediv__(self, other):
        return dezero.functions.div(self, other)

    def __rtruediv__(self, other):
        return dezero.functions.div(other, self)

    def __pow__(self, other):
        return dezero.functions.pow(self, other)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)

    @property 
    def T(self):
        return self.transpose()

# Support cupy
def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        # Support cupy
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        xp = get_array_module(*xs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y, xp)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

class Parameter(Variable):
    pass

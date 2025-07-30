try:
    import cupy
    is_cupy_available = True
except ImportError:
    is_cupy_available = False
    cupy = None


def get_array_module(*args):
    """Returns the array module for the given arguments.

    If any of the arguments is a CuPy array, this function returns the `cupy` module.
    Otherwise, this function returns the `numpy` module.

    Args:
        *args: Values to inspect.

    Returns:
        module: `cupy` or `numpy` is returned.
    """
    if not is_cupy_available:
        import numpy
        return numpy

    for x in args:
        if isinstance(x, cupy.ndarray):
            return cupy
    import numpy
    return numpy


def to_cpu(x):
    """Converts a CuPy array to a NumPy array.

    If the input is a NumPy array, it is returned as is.

    Args:
        x (numpy.ndarray or cupy.ndarray): Array to be converted.

    Returns:
        numpy.ndarray: Converted array.
    """
    if is_cupy_available and isinstance(x, cupy.ndarray):
        return cupy.asnumpy(x)
    return x


def to_gpu(x):
    """Converts a NumPy array to a CuPy array.

    If the input is a CuPy array, it is returned as is.

    Args:
        x (numpy.ndarray or cupy.ndarray): Array to be converted.

    Returns:
        cupy.ndarray: Converted array.
    """
    if is_cupy_available:
        return cupy.asarray(x)
    else:
        raise ImportError("CuPy is not available. Cannot move data to GPU.")

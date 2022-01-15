from pathlib import Path
from typing import Union, Generator
from mmap_ninja import base

# See: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
import numpy as np


def save_mmap_kwargs(out_dir: Path,
                     dtype,
                     shape,
                     order):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    base.str_to_file(np.dtype(dtype).name, out_dir / f'dtype.ninja')
    base.shape_to_file(shape, out_dir / f'shape.ninja')
    base.str_to_file(order, out_dir / f'order.ninja')


def read_mmap_kwargs(out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    return {
        'dtype': base.file_to_str(out_dir / 'dtype.ninja'),
        'shape': base.file_to_shape(out_dir / 'shape.ninja'),
        'order': base.file_to_str(out_dir / 'order.ninja')
    }


def empty(out_dir: Union[str, Path], dtype, shape, order) -> np.memmap:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    save_mmap_kwargs(out_dir, dtype, shape, order)
    memmap = np.memmap(str(out_dir / 'data.ninja'),
                       mode='w+',
                       dtype=dtype,
                       shape=shape,
                       order=order)
    return memmap


def from_ndarray(arr: np.ndarray, out_dir: Union[str, Path]) -> np.memmap:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    dtype = arr.dtype
    shape = arr.shape
    order = 'F' if np.isfortran(arr) else 'C'
    memmap = np.memmap(str(out_dir / 'data.ninja'),
                       mode='w+',
                       dtype=dtype,
                       shape=shape,
                       order=order)
    memmap[:] = arr
    save_mmap_kwargs(out_dir, dtype, shape, order)
    return memmap


def write_samples(memmap, out_dir, samples, start, total):
    arr = np.stack(samples)
    if memmap is None:
        dtype = arr.dtype
        shape = (total,) + arr.shape[1:]
        order = 'F' if np.isfortran(arr) else 'C'
        memmap = np.memmap(str(out_dir / 'data.ninja'),
                           mode='w+',
                           dtype=dtype,
                           shape=shape,
                           order=order)
    end = start + len(samples)
    memmap[start:end] = arr
    return memmap, end


def from_generator(sample_generator: Generator[str, None, None],
                   out_dir: Union[str, Path],
                   batch_size: int, n: int) -> np.memmap:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    samples = []
    memmap = None
    start = 0
    for sample in sample_generator:
        samples.append(sample)
        if len(samples) % batch_size != 0:
            continue
        memmap, start = write_samples(memmap, out_dir, samples, start, n)
        samples = []
    if len(samples) > 0:
        memmap, start = write_samples(memmap, out_dir, samples, start, n)
    return memmap


def open_existing(out_dir: Union[str, Path], mode='r'):
    out_dir = Path(out_dir)
    kwargs = read_mmap_kwargs(out_dir)
    memmap = np.memmap(str(out_dir / 'data.ninja'),
                       mode=mode,
                       **kwargs)
    return memmap


def extend_dir(out_dir: Union[str, Path], arr: np.ndarray):
    out_dir = Path(out_dir)
    kwargs = read_mmap_kwargs(out_dir)
    shape = kwargs['shape']
    order = kwargs['order']
    dtype = np.dtype(kwargs['dtype'])
    assert shape[1:] == arr.shape[1:], f'Trying to append samples with incorrect shape: {arr.shape[1:]}, ' \
                                       f'expected: {shape[1:]}'
    with open(out_dir / 'data.ninja', 'ab') as data_file:
        data_file.write(arr.astype(dtype).tobytes(order=order))
        data_file.flush()
    new_shape = (shape[0] + arr.shape[0], *shape[1:])
    base.shape_to_file(new_shape, out_dir / f'shape.ninja')


def extend(np_mmap: np.memmap, arr: np.ndarray):
    extend_dir(Path(np_mmap.filename).parent, arr)

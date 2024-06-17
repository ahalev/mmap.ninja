from enum import Enum
from functools import partial
from math import inf
from tqdm.auto import tqdm

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel, delayed = None, None
    HAS_JOBLIB = False
else:
    HAS_JOBLIB = True


class _Exhausted(Enum):
    exhausted = 'EXHAUSTED'


EXHAUSTED = _Exhausted.exhausted


class ParallelBatchCollector:
    _parallel: Parallel = None

    def __init__(self, indexable, batch_size, n_jobs=None, verbose=False, **kwargs):
        self.indexable, self._obj_length, self._num_batches = self.verify(indexable, batch_size, n_jobs)
        self.batch_size = batch_size

        self._pbar = self._init_pbar(verbose)
        self._parallel = self.begin(n_jobs, **kwargs)
        self._batch_num = 0
        self._exhausted = False

    @staticmethod
    def verify(indexable, batch_size, n_jobs):
        try:
            _ = indexable.__getitem__
        except AttributeError:
            if callable(indexable):
                indexable = _IndexableWrap(indexable)
            else:
                msg = 'indexable must implement __getitem__ or be callable and take one integer argument.'
                raise TypeError(msg)

        try:
            length = len(indexable)
        except TypeError as e:
            if n_jobs not in (1, None):
                msg = 'Passed object has no len() and cannot be utilized in parallel. ' \
                      'Pass n_jobs=None or define __len__.'
                raise TypeError(msg) from e
            length = inf
            num_batches = None
        else:
            num_batches = length // batch_size + (length % batch_size != 0)

        return indexable, length, num_batches

    @staticmethod
    def begin(n_jobs: int, **kwargs):
        if n_jobs in (None, 1):
            return
        elif not HAS_JOBLIB:
            msg = 'joblib is not installed. Install joblib or run with n_jobs=None to ignore parallelization.'
            raise ImportError(msg)

        _parallel = Parallel(n_jobs=n_jobs, return_as='generator', **kwargs)
        _parallel.__enter__()
        return _parallel

    def batches(self):
        if self._parallel is None:
            yield from self._collect_no_parallel_batches()
        else:
            yield from self._collect_parallel_batches()

    def _init_pbar(self, verbose):
        if not verbose:
            return None
        return tqdm(total=self._obj_length)

    def _update_pbar(self, batch):
        if self._pbar is not None:
            self._pbar.update(len(batch))

        return batch

    def _collect_no_parallel_batches(self):
        while not self.exhausted():
            yield self._update_pbar(self._collect_no_parallel_batch())

    def _collect_no_parallel_batch(self):
        results = _collect_batch(self.indexable, self._rng())

        if self.exhausted(results):
            results = [r for r in results if r is not EXHAUSTED]

        return results

    def _collect_parallel_batches(self):
        func = delayed(partial(_collect_batch, self.indexable))

        yield from filter(self._parallel_filter, self._parallel(func(rng) for rng in self._all_ranges()))

        self._parallel.__exit__(None, None, None)

    def _parallel_filter(self, results):
        self._batch_num += 1
        self.exhausted(results)
        self._update_pbar(results)

        return results

    def exhausted(self, results=()):
        self._exhausted = (
                self._exhausted or
                any(r is EXHAUSTED for r in results) or
                self.completed_batches()
        )

        return self._exhausted

    def completed_batches(self):
        return self._num_batches is not None and self._batch_num == self._num_batches

    def _all_ranges(self):
        batch_range = partial(_batch_range, batch_size=self.batch_size, total_length=self._obj_length)
        return map(batch_range, range(self._num_batches))

    def _rng(self):
        rng = _batch_range(self._batch_num, self.batch_size, self._obj_length)
        self._batch_num += 1

        return rng


class _IndexableWrap:
    def __init__(self, func):
        self._func = func

    def __getitem__(self, item):
        return self._func(item)

    @property
    def wrapped(self):
        return self._func


class _IndexableLengthWrap(_IndexableWrap):
    def __init__(self, func, length):
        super().__init__(func)
        self.length = length

    def __len__(self):
        return self.length


def make_indexable(func, length=None):
    if length is not None:
        return _IndexableLengthWrap(func, length)
    return _IndexableWrap(func)


def _collect_batch(indexable, rng):
    results = [_get_from_indexable(indexable, j) for j in rng]

    return results


def _batch_range(batch_num, batch_size, total_length=inf):
    start = batch_size * batch_num
    stop = batch_size * (1 + batch_num)

    if stop >= total_length:
        stop = total_length

    return range(start, stop)


def _get_from_indexable(indexable, item,):
    try:
        return indexable[item]
    except (IndexError, KeyError):
        return EXHAUSTED

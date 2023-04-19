import operator

import numpy as np
import builtins
from typing import Sequence, Any, Optional, Union, List, Tuple, NamedTuple, Dict

from numpy import expand_dims

_sum = builtins.sum
_any = builtins.any
_int = builtins.int
_all = builtins.all
DimSize = Union[int, Any]  # extensible

Array = Union[
    np.ndarray,  # NumPy array type
    np.bool_,
    np.number,  # NumPy scalar types
    bool,
    int,
    float,
    complex,  # Python scalar types
]

python_scalar_dtypes: Dict[type, np.dtype] = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    complex: np.dtype("complex128"),
}


def is_python_scalar(x: Any) -> bool:
    try:
        return x.aval.weak_type and np.ndim(x) == 0
    except AttributeError:
        return type(x) in python_scalar_dtypes


def dtype(x: Any, *, canonicalize: bool = False) -> np.dtype:
    if x is None:
        raise ValueError(f"Invalid argument to dtype: {x}.")
    elif isinstance(x, type) and x in python_scalar_dtypes:
        dt = python_scalar_dtypes[x]
    elif type(x) in python_scalar_dtypes:
        dt = python_scalar_dtypes[type(x)]
    else:
        try:
            dt = np.result_type(x)
        except TypeError as err:
            raise TypeError(f"Cannot determine dtype of {x}") from err
    return np.dtype(dt) if canonicalize else dt


def _dtype(x: Any):
    return dtype(x, canonicalize=True)


def scalar_type_of(x: Any) -> type:
    """Return the scalar type associated with a JAX value."""
    typ = dtype(x)
    if np.issubdtype(typ, np.bool_):
        return bool
    elif np.issubdtype(typ, np.integer):
        return int
    elif np.issubdtype(typ, np.floating):
        return float
    elif np.issubdtype(typ, np.complexfloating):
        return complex
    else:
        raise TypeError(f"Invalid scalar value {x}")


def _const(example, val):
    dtype = _dtype(example)
    if is_python_scalar(example):
        val = scalar_type_of(example)(val)
        return val if dtype == _dtype(val) else np.array(val, dtype)
    return np.array(val, dtype)


class GatherDimensionNumbers(NamedTuple):
    """
    Describes the dimension number arguments to an `XLA's Gather operator
    <https://www.tensorflow.org/xla/operation_semantics#gather>`_. See the XLA
    documentation for more details of what the dimension numbers mean.

    Args:
      offset_dims: the set of dimensions in the `gather` output that offset into
        an array sliced from `operand`. Must be a tuple of integers in ascending
        order, each representing a dimension number of the output.
      collapsed_slice_dims: the set of dimensions `i` in `operand` that have
        `slice_sizes[i] == 1` and that should not have a corresponding dimension
        in the output of the gather. Must be a tuple of integers in ascending
        order.
      start_index_map: for each dimension in `start_indices`, gives the
        corresponding dimension in `operand` that is to be sliced. Must be a
        tuple of integers with size equal to `start_indices.shape[-1]`.

    Unlike XLA's `GatherDimensionNumbers` structure, `index_vector_dim` is
    implicit; there is always an index vector dimension and it must always be the
    last dimension. To gather scalar indices, add a trailing dimension of size 1.
    """

    offset_dims: Tuple[int, ...]
    collapsed_slice_dims: Tuple[int, ...]
    start_index_map: Tuple[int, ...]


def _is_int_arraylike(x):
    """Returns True if x is array-like with integer dtype, False otherwise."""
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        or np.issubdtype(getattr(x, "dtype", None), np.integer)
        or isinstance(x, (list, tuple))
        and _all(_is_int_arraylike(e) for e in x)
    )


def _is_scalar(x):
    """Checks if a Python or NumPy scalar."""
    return np.isscalar(x) or (isinstance(x, (np.ndarray, Array)) and np.ndim(x) == 0)


# TODO(mattjj): clean up this logic
def _is_advanced_int_indexer(idx):
    """Returns True if idx should trigger int array indexing, False otherwise."""
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    assert isinstance(idx, tuple)
    if _all(
        e is None
        or e is Ellipsis
        or isinstance(e, slice)
        or _is_scalar(e)
        and np.issubdtype(_dtype(e), np.integer)
        for e in idx
    ):
        return False
    return _all(
        e is None or e is Ellipsis or isinstance(e, slice) or _is_int_arraylike(e)
        for e in idx
    )


class _Indexer(NamedTuple):
    # The expected shape of the slice output.
    slice_shape: Sequence[int]
    # The slice shape to pass to lax.gather().
    gather_slice_shape: Sequence[int]
    # The gather indices to use.
    gather_indices: Array
    # A GatherDimensionNumbers object describing the gather to perform.
    dnums: GatherDimensionNumbers

    # Are the gather_indices known to be non-overlapping and/or sorted?
    # (In practice, these translate to "there no advanced indices", because
    # only advanced indices could lead to index repetition.)
    unique_indices: bool
    indices_are_sorted: bool

    # Slice dimensions that have negative strides, and so must be reversed after
    # the gather.
    reversed_y_dims: Sequence[int]

    # Keep track of any axes created by `newaxis`. These must be inserted for
    # gathers and eliminated for scatters.
    newaxis_dims: Sequence[int]


def convert_element_type(operand, dtype):
    return np.asarray(operand, dtype=dtype)


def is_constant_dim(d: DimSize) -> bool:
    # Whether the dimension is a static integer constant.
    try:
        operator.index(d)
        return True
    except:
        return False


def broadcast_in_dim(operand, shape, broadcast_dimensions):
    in_reshape = np.ones(len(shape), dtype=np.int32)
    for i, bd in enumerate(broadcast_dimensions):
        in_reshape[bd] = operand.shape[i]
    return np.broadcast_to(np.reshape(operand, in_reshape), shape)


def _normalize_index(index, axis_size):
    """Normalizes an index value in the range [-N, N) to the range [0, N)."""
    if np.issubdtype(_dtype(index), np.unsignedinteger):
        return index
    if is_constant_dim(axis_size):
        axis_size_val = _const(index, axis_size)
    else:
        axis_size_val = convert_element_type(axis_size, _dtype(index))
    if isinstance(index, (int, np.integer)):
        return np.add(index, axis_size_val) if index < 0 else index
    else:
        return np.select(index < 0, np.add(index, axis_size_val), index)


def _canonicalize_tuple_index(arr_ndim, idx, array_name="array"):
    """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
    len_without_none = _sum(1 for e in idx if e is not None and e is not Ellipsis)
    if len_without_none > arr_ndim:
        raise IndexError(
            f"Too many indices for {array_name}: {len_without_none} "
            f"non-None/Ellipsis indices for dim {arr_ndim}."
        )
    ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
    ellipsis_index = next(ellipses, None)
    if ellipsis_index is not None:
        if next(ellipses, None) is not None:
            raise IndexError(
                f"Multiple ellipses (...) not supported: {list(map(type, idx))}."
            )
        colons = (slice(None),) * (arr_ndim - len_without_none)
        idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1 :]
    elif len_without_none < arr_ndim:
        colons = (slice(None),) * (arr_ndim - len_without_none)
        idx = tuple(idx) + colons
    return idx


def _static_idx(idx: slice, size):
    """Helper function to compute the static slice start/limit/stride values."""
    if isinstance(size, int):
        start, stop, step = idx.indices(size)
    else:
        raise TypeError(size)

    if (step < 0 and stop >= start) or (step > 0 and start >= stop):
        return 0, 0, 1, False  # sliced to size zero

    if step > 0:
        return start, stop, step, False
    else:
        k = (start - stop - 1) % (-step)
        return stop + k + 1, start + 1, -step, True


def _index_to_gather(
    x_shape: Sequence[int], idx: Sequence[Any], normalize_indices: bool = True
) -> _Indexer:
    # Remove ellipses and add trailing slice(None)s.
    idx = _canonicalize_tuple_index(len(x_shape), idx)

    # Check for advanced indexing:
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

    # Do the advanced indexing axes appear contiguously? If not, NumPy semantics
    # move the advanced axes to the front.
    advanced_axes_are_contiguous = False

    advanced_indexes: Optional[Sequence[Union[Array, np.ndarray]]] = None

    # The positions of the advanced indexing axes in `idx`.
    idx_advanced_axes: Sequence[int] = []

    # The positions of the advanced indexes in x's shape.
    # collapsed, after None axes have been removed. See below.
    x_advanced_axes: Optional[Sequence[int]] = None

    if _is_advanced_int_indexer(idx):
        idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
        advanced_pairs = (
            (np.asarray(e), i, j)
            for j, (i, e) in enumerate(idx_no_nones)
            if np.isscalar(e) or isinstance(e, (Sequence, Array, np.ndarray))
        )
        if normalize_indices:
            advanced_pairs = (
                (_normalize_index(e, x_shape[j]), i, j) for e, i, j in advanced_pairs
            )
        advanced_indexes, idx_advanced_axes, x_advanced_axes = zip(*advanced_pairs)
        advanced_axes_are_contiguous = bool(np.all(np.diff(idx_advanced_axes) == 1))

    x_axis = 0  # Current axis in x.
    y_axis = 0  # Current axis in y, before collapsing. See below.
    collapsed_y_axis = 0  # Current axis in y, after collapsing.

    # Scatter dimension numbers.
    offset_dims: Sequence[int] = []
    collapsed_slice_dims: Sequence[int] = []
    start_index_map: Sequence[int] = []

    use_64bit_index = _any([not is_constant_dim(d) or d >= (1 << 31) for d in x_shape])
    index_dtype = np.int64 if use_64bit_index else np.int32

    # Gather indices.
    # Pairs of (array, start_dim) values. These will be broadcast into
    # gather_indices_shape, with the array dimensions aligned to start_dim, and
    # then concatenated.
    gather_indices: List[Tuple[Array, int]] = []
    gather_indices_shape: List[int] = []

    # We perform three transformations to y before the scatter op, in order:
    # First, y is broadcast to slice_shape. In general `y` only need broadcast to
    # the right shape.
    slice_shape: Sequence[int] = []

    # Next, y is squeezed to remove newaxis_dims. This removes np.newaxis/`None`
    # indices, which the scatter cannot remove itself.
    newaxis_dims: Sequence[int] = []

    # Finally, we reverse reversed_y_dims to handle slices with negative strides.
    reversed_y_dims: Sequence[int] = []

    gather_slice_shape: Sequence[int] = []

    for idx_pos, i in enumerate(idx):
        # Handle the advanced indices here if:
        # * the advanced indices were not contiguous and we are the start.
        # * we are at the position of the first advanced index.
        if advanced_indexes is not None and (
            advanced_axes_are_contiguous
            and idx_pos == idx_advanced_axes[0]
            or not advanced_axes_are_contiguous
            and idx_pos == 0
        ):
            advanced_indexes = np.broadcast_arrays(advanced_indexes)
            shape = advanced_indexes[0].shape
            ndim = len(shape)

            start_dim = len(gather_indices_shape)
            gather_indices += (
                (convert_element_type(a, index_dtype), start_dim)
                for a in advanced_indexes
            )
            gather_indices_shape += shape

            start_index_map.extend(x_advanced_axes)
            collapsed_slice_dims.extend(x_advanced_axes)
            slice_shape.extend(shape)
            y_axis += ndim
            collapsed_y_axis += ndim

        # Per-index bookkeeping for advanced indexes.
        if idx_pos in idx_advanced_axes:
            x_axis += 1
            gather_slice_shape.append(1)
            continue

        if i is None:
            slice_shape.append(1)
            newaxis_dims.append(y_axis)
            y_axis += 1

        elif isinstance(i, slice):
            # Normalize the slice to use None when possible
            start, stop, step = i.start, i.stop, i.step
            try:
                if step is None or step == 1:
                    step = None
                if step is None:
                    if start is None or start == 0:
                        start = None
                    if stop is None or (stop >= x_shape[x_axis]):
                        stop = None
                elif step == -1:
                    step = -1
            except TypeError:
                pass

            # Handle slice(None) and slice(None, None, -1)
            if (
                start is None
                and stop is None
                and (step is None or isinstance(step, int) and step == -1)
            ):
                if step == -1:
                    reversed_y_dims.append(collapsed_y_axis)
                slice_shape.append(x_shape[x_axis])
                gather_slice_shape.append(x_shape[x_axis])
                offset_dims.append(collapsed_y_axis)
                collapsed_y_axis += 1
                y_axis += 1
                x_axis += 1
            # Handle slice index (only static, otherwise an error is raised)
            else:
                start, limit, stride, needs_rev = _static_idx(
                    slice(start, stop, step), x_shape[x_axis]
                )
                if needs_rev:
                    reversed_y_dims.append(collapsed_y_axis)
                if stride == 1:
                    i = convert_element_type(start, index_dtype)
                    gather_indices.append((i, len(gather_indices_shape)))
                    slice_shape.append(limit - start)
                    gather_slice_shape.append(limit - start)
                    offset_dims.append(collapsed_y_axis)
                    start_index_map.append(x_axis)
                else:
                    i = np.arange(start, limit, stride, dtype=index_dtype)
                    size = i.shape[0]
                    slice_shape.append(size)
                    gather_slice_shape.append(1)
                    gather_indices.append((i, len(gather_indices_shape)))
                    gather_indices_shape.append(size)

                    start_index_map.append(x_axis)
                    collapsed_slice_dims.append(x_axis)

                collapsed_y_axis += 1
                y_axis += 1
                x_axis += 1
        else:
            msg = "Indexing mode not yet supported. Open a feature request!\n{}"
            raise IndexError(msg.format(idx))

    if len(gather_indices) == 0:
        gather_indices_array = np.zeros((0,), dtype=index_dtype)
    elif len(gather_indices) == 1:
        g, _ = gather_indices[0]
        gather_indices_array = expand_dims(g, (g.ndim,))
    else:
        last_dim = len(gather_indices_shape)
        gather_indices_shape.append(1)
        gather_indices_array = np.concatenate(
            [
                broadcast_in_dim(g, gather_indices_shape, tuple(range(i, i + g.ndim)))
                for g, i in gather_indices
            ],
            last_dim,
        )

    dnums = GatherDimensionNumbers(
        offset_dims=tuple(offset_dims),
        collapsed_slice_dims=tuple(sorted(collapsed_slice_dims)),
        start_index_map=tuple(start_index_map),
    )
    return _Indexer(
        slice_shape=slice_shape,
        newaxis_dims=tuple(newaxis_dims),
        gather_slice_shape=gather_slice_shape,
        reversed_y_dims=reversed_y_dims,
        dnums=dnums,
        gather_indices=gather_indices_array,
        unique_indices=advanced_indexes is None,
        indices_are_sorted=advanced_indexes is None,
    )

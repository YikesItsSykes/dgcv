import numbers

from ..._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols
from ..._aux._utilities._config import dgcv_warning
from ...core.arrays import matrix_dgcv
from ..linear_algebra import _is_canonical_structure_array, _structure_array

_skew_violation = "`assume_skew=True` was passed to the algebra contructor, but the accompanying structure data was not skew symmetric or skewable. If specifying [a,b] and [b,a] as nonzeros, the [a,b]=-[b,a] is required."

_key_format = "Structure data supplied as a dictionary must have keys that are tuples of non-negative integers, either (i,j) paired with a list-like of coefficients, or (i,j,k) paired with a single coefficient."


def _column_source(value):
    data = getattr(value, "_data", None)
    if data is None:
        return None
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        return None
    if len(shape) == 1:
        return data
    if len(shape) == 2 and 1 in shape:
        return data
    return None


def _read_column(value, key):
    if isinstance(value, (list, tuple)):
        entries = {
            i: c
            for i, c in enumerate(value)
            if c is not None and not _scalar_is_zero(c)
        }
        return entries, len(value)
    source = _column_source(value)
    if source is None:
        raise ValueError(
            f"The structure data entry at key {key} could not be read as a vector of coefficients. Supported entry types are list, tuple, 1-dimensional array_dgcv, and matrix_dgcv of shape (n,1) or (1,n). Received {type(value)}."
        )
    entries = {
        i: c for i, c in source.items() if c is not None and not _scalar_is_zero(c)
    }
    shape = value.shape
    length = shape[0] if len(shape) == 1 else shape[0] * shape[1]
    return entries, length


def _complete_skew(cells, dim):
    seen = set()
    for key in list(cells):
        if key in seen:
            continue
        idx1, idx2 = key
        if idx1 == idx2:
            raise ValueError(_skew_violation)
        mirror = (idx2, idx1)
        seen.add(key)
        seen.add(mirror)
        other = cells.get(mirror)
        if other is None:
            cells[mirror] = matrix_dgcv(
                {k: -v for k, v in cells[key]._data.items()}, shape=(dim, 1)
            )
        else:
            left, right = cells[key]._data, other._data
            for k in set(left) | set(right):
                if not _scalar_is_zero(left.get(k, 0) + right.get(k, 0)):
                    raise ValueError(_skew_violation)


def _normalize_sparse_structure(items, *, dim=None, assume_skew=False):
    raw = {}
    params = set()
    arity = None
    max_index = -1
    max_index_key = None
    common_length = None
    length_key = None

    for key, value in items:
        if not isinstance(key, tuple) or len(key) not in (2, 3):
            raise ValueError(_key_format)
        if arity is None:
            arity = len(key)
        elif len(key) != arity:
            raise ValueError(
                f"Structure data keys must all have the same length, but both {arity}-tuple and {len(key)}-tuple keys were supplied (e.g. {key})."
            )
        for idx in key:
            if not isinstance(idx, numbers.Integral) or idx < 0:
                raise ValueError(_key_format)
            if idx > max_index:
                max_index, max_index_key = idx, key

        if arity == 3:
            if value is None or _scalar_is_zero(value):
                continue
            params |= get_free_symbols(value)
            raw.setdefault(key[:2], {})[key[2]] = value
        else:
            entries, length = _read_column(value, key)
            if common_length is None:
                common_length, length_key = length, key
            elif length != common_length and dim is None:
                raise ValueError(
                    f"When the algebra dimension is inferred from a structure data dictionary, all coefficient vectors must have the same length. The entry at key {key} has length {length}, but the entry at key {length_key} has length {common_length}."
                )
            if not entries:
                continue
            for c in entries.values():
                params |= get_free_symbols(c)
            raw[key] = entries

    if dim is None:
        if arity == 3:
            dim = max_index + 1
            dgcv_warning(
                f"The algebra dimension {dim} was inferred from the largest index in the structure data; pass `dimension=` to set it explicitly."
            )
        else:
            dim = common_length if common_length is not None else 0
    if max_index >= dim:
        raise ValueError(
            f"The structure data key {max_index_key} contains the index {max_index}, which requires an algebra dimension of at least {max_index + 1}, but the dimension was determined to be {dim}."
        )

    cells = {}
    for key, entries in raw.items():
        overrun = max(entries)
        if overrun >= dim:
            raise ValueError(
                f"The structure data entry at key {key} has a nonzero coefficient at index {overrun}, which exceeds the algebra dimension {dim}."
            )
        cells[key] = matrix_dgcv(entries, shape=(dim, 1))

    if assume_skew:
        _complete_skew(cells, dim)

    return _structure_array(cells, dim), params


def _structure_data_from_array(data, *, assume_skew=False):
    shape = data.shape
    if _is_canonical_structure_array(data):
        if assume_skew:
            cells = dict(data._data_unspooled)
            _complete_skew(cells, shape[0])
            return _structure_array(cells, shape[0]), data.free_symbols
        return data, data.free_symbols
    if len(shape) in (2, 3) and len(set(shape)) == 1:
        return _normalize_sparse_structure(
            data._data_unspooled.items(), dim=shape[0], assume_skew=assume_skew
        )
    raise ValueError(
        f"An array received as structure data must have shape (dim, dim), with entries that are coefficient vectors, or shape (dim, dim, dim), with entries that are scalar coefficients. The supplied array has shape {shape}."
    )

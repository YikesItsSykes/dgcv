from fractions import Fraction

from ..._aux._backends._engine import engine_kind
from ..._aux._backends._exact_arith import exact_reciprocal, ratio
from ..._aux._backends._symbolic_router import (
    _fast_simplify,
    _scalar_is_one,
    _scalar_is_zero,
    as_numer_denom,
    exact_nonzero,
    get_free_symbols,
    subs,
)
from ..._aux._backends._types_and_constants import (
    _disposable_symbols,
    constant_scalar_types,
    fast_scalar_types,
    rational,
)
from ..._aux._vmf._safeguards import create_key
from ._indexing import _spool


def _invert(*, record_divisors=False, allow_formal=True):
    if record_divisors:
        divisors = []

        def _inv(denominator):
            divisors.append(denominator)
            try:
                return exact_reciprocal(denominator)
            except Exception:  ###!!! optimize for the formal divisor path
                if allow_formal:
                    return 1 / denominator
                raise

        return _inv, divisors

    def _inv(denominator):
        try:
            return exact_reciprocal(denominator)
        except Exception:
            if allow_formal:  ###!!! optimize for the formal divisor path
                return 1 / denominator
            raise

    return _inv, None


def _expands_to_zero(value):
    expand = getattr(value, "expand", None)
    if not callable(expand):
        return False
    try:
        return _scalar_is_zero(expand())
    except Exception:
        return False


_PROBE_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67)


def _probe_point(atoms, round_idx):
    count = len(_PROBE_PRIMES)
    point = {}
    for pos, atom in enumerate(atoms):
        numer = _PROBE_PRIMES[(pos + 7 * round_idx) % count]
        denom = _PROBE_PRIMES[(pos + 3 * round_idx + 5) % count]
        point[atom] = rational(numer, denom)
    return point


def _certainly_nonzero(value):
    try:
        atoms = sorted(get_free_symbols(value) or (), key=str)
    except Exception:
        return False
    if not atoms:
        return False
    for round_idx in range(2):
        try:
            probe = subs(value, _probe_point(atoms, round_idx))
        except Exception:
            return False
        verdict = exact_nonzero(probe)
        if verdict is None:
            return False
        if verdict:
            return True
    return False


_plain_number_types = (int, Fraction, float)


def _pivot_is_zero(value):
    if type(value) in _plain_number_types:
        return value == 0
    if _scalar_is_zero(value):
        return True
    if engine_kind() == "builtin" and getattr(value, "_nf_cache", None) is not None:
        return False
    if _certainly_nonzero(value):
        return False
    if _expands_to_zero(value):
        return True
    try:
        numerator, _denominator = as_numer_denom(value)
    except Exception:
        return False
    return _expands_to_zero(numerator)


def _eliminate(
    work_rows,
    *,
    rhs_cols=0,
    record_divisors=False,
    allow_formal_inverse=True,
    simplify_steps=False,
    fast_only=False,
    want_pivmap=False,
):
    row_count = len(work_rows)
    total_cols = len(work_rows[0]) if row_count else 0
    col_count = total_cols - rhs_cols

    if row_count == 0 or col_count == 0:
        if want_pivmap:
            return 0, {}, [], [] if record_divisors else None
        return 0, [] if record_divisors else None
    _inv, divisors = _invert(
        record_divisors=record_divisors, allow_formal=allow_formal_inverse
    )
    pivot_index = 0
    pivcol_to_row = {}
    pivot_cols = []

    for col_idx in range(col_count):
        pivot_row = None
        for scan_row in range(pivot_index, row_count):
            if not _pivot_is_zero(work_rows[scan_row][col_idx]):
                pivot_row = scan_row
                break
        if pivot_row is None:
            continue

        if pivot_row != pivot_index:
            work_rows[pivot_index], work_rows[pivot_row] = (
                work_rows[pivot_row],
                work_rows[pivot_index],
            )

        pivot_value = work_rows[pivot_index][col_idx]
        inv_pivot = _inv(pivot_value)
        do_simplify = simplify_steps and not fast_only

        if not _scalar_is_one(inv_pivot):
            for entry_idx in range(col_idx, total_cols):
                entry = work_rows[pivot_index][entry_idx]
                if _scalar_is_zero(entry):
                    continue
                entry = entry * inv_pivot
                if do_simplify and not isinstance(entry, constant_scalar_types()):
                    entry = _fast_simplify(entry)
                work_rows[pivot_index][entry_idx] = entry

        for row_idx in range(row_count):
            if row_idx == pivot_index:
                continue
            factor = work_rows[row_idx][col_idx]
            if _pivot_is_zero(factor):
                continue
            for entry_idx in range(col_idx, total_cols):
                pivot_entry = work_rows[pivot_index][entry_idx]
                if _scalar_is_zero(pivot_entry):
                    continue
                value = work_rows[row_idx][entry_idx] - factor * pivot_entry
                if do_simplify and not isinstance(value, constant_scalar_types()):
                    value = _fast_simplify(value)
                work_rows[row_idx][entry_idx] = value

        pivcol_to_row[col_idx] = pivot_index
        pivot_cols.append(col_idx)
        pivot_index += 1
        if pivot_index == row_count:
            break

    if want_pivmap:
        return pivot_index, pivcol_to_row, pivot_cols, divisors
    return pivot_index, divisors


class _matrix_linalg:
    def det(self):
        if self.nrows != self.ncols:
            raise ValueError("det only defined for square matrices")

        size = self.nrows
        if size == 0:
            return 1
        if size == 1:
            return self[0, 0]

        work_rows = [
            [self[idx, col_idx] for col_idx in range(size)] for idx in range(size)
        ]
        prev_pivot = 1
        sign = 1

        for step in range(size - 1):
            pivot_row = None
            for scan_row in range(step, size):
                if not _scalar_is_zero(work_rows[scan_row][step]):
                    pivot_row = scan_row
                    break
            if pivot_row is None:
                return 0
            if pivot_row != step:
                work_rows[step], work_rows[pivot_row] = (
                    work_rows[pivot_row],
                    work_rows[step],
                )
                sign = -sign

            pivot = work_rows[step][step]
            for idx in range(step + 1, size):
                leading_entry = work_rows[idx][step]
                for col_idx in range(step + 1, size):
                    work_rows[idx][col_idx] = ratio(
                        (
                            work_rows[idx][col_idx] * pivot
                            - leading_entry * work_rows[step][col_idx]
                        ),
                        prev_pivot,
                    )
                work_rows[idx][step] = 0

            prev_pivot = pivot
            if _scalar_is_zero(prev_pivot):
                return 0

        return sign * work_rows[size - 1][size - 1]

    def charpoly(self, var):
        """
        Characteristic polynomial of the matrix

        Parameters
        ----------
        var : symbolic variable
            used as the polynomial's variable
        """
        if self.nrows != self.ncols:
            raise ValueError("charpoly only defined for square matrices")
        return (var * self.identity(self.nrows) - self).det()

    def inverse(self):
        if self.nrows != self.ncols:
            raise ValueError("inverse only defined for square matrices")

        size = self.nrows
        work_rows = [
            [self[idx, col_idx] for col_idx in range(size)] for idx in range(size)
        ]
        inverse_rows = [[0 for _ in range(size)] for _ in range(size)]
        for idx in range(size):
            inverse_rows[idx][idx] = 1

        for pivot_col in range(size):
            pivot_row = None
            for scan_row in range(pivot_col, size):
                if not _scalar_is_zero(work_rows[scan_row][pivot_col]):
                    pivot_row = scan_row
                    break
            if pivot_row is None:
                raise ZeroDivisionError("matrix is singular")

            if pivot_row != pivot_col:
                work_rows[pivot_col], work_rows[pivot_row] = (
                    work_rows[pivot_row],
                    work_rows[pivot_col],
                )
                inverse_rows[pivot_col], inverse_rows[pivot_row] = (
                    inverse_rows[pivot_row],
                    inverse_rows[pivot_col],
                )

            pivot = work_rows[pivot_col][pivot_col]
            inv_pivot = exact_reciprocal(pivot)

            for col_idx in range(size):
                work_rows[pivot_col][col_idx] = (
                    work_rows[pivot_col][col_idx] * inv_pivot
                )
                inverse_rows[pivot_col][col_idx] = (
                    inverse_rows[pivot_col][col_idx] * inv_pivot
                )

            for idx in range(size):
                if idx == pivot_col:
                    continue
                factor = work_rows[idx][pivot_col]
                if _scalar_is_zero(factor):
                    continue
                for col_idx in range(size):
                    work_rows[idx][col_idx] = (
                        work_rows[idx][col_idx] - factor * work_rows[pivot_col][col_idx]
                    )
                    inverse_rows[idx][col_idx] = (
                        inverse_rows[idx][col_idx]
                        - factor * inverse_rows[pivot_col][col_idx]
                    )

        return self.__class__(inverse_rows)

    def rank(
        self,
        assume_fast_data_types=False,
        allow_formal_inverse=False,
        simplify_steps=False,
        record_divisors=False,
    ):
        if self.nrows == 0 or self.ncols == 0:
            if record_divisors:
                return 0, []
            return 0

        fast_types = fast_scalar_types()
        fast_case = assume_fast_data_types or all(
            isinstance(entry, fast_types) for entry in self._data.values()
        )

        work_rows = self._dense_copy()
        rank_value, divisors = _eliminate(
            work_rows,
            rhs_cols=0,
            record_divisors=record_divisors,
            allow_formal_inverse=allow_formal_inverse,
            simplify_steps=simplify_steps,
            fast_only=fast_case,
            want_pivmap=False,
        )
        if record_divisors:
            return rank_value, divisors
        return rank_value

    def solve_right_batch(
        self, rhs, *, allow_formal_inverse=True, simplify_steps=False
    ):
        n_unknowns = self.ncols
        rhs_matrix = rhs if isinstance(rhs, matrix_dgcv) else matrix_dgcv(rhs)
        if rhs_matrix.nrows != self.nrows:
            raise ValueError("solve_right_batch requires matching row counts")
        n_rhs = rhs_matrix.ncols
        if self.nrows == 0 or n_unknowns == 0:
            return (
                [[0] * n_unknowns for _ in range(n_rhs)],
                set(),
                list(range(n_unknowns)),
            )

        work_rows = []
        for row_idx in range(self.nrows):
            row = [self[row_idx, col] for col in range(n_unknowns)]
            row += [rhs_matrix[row_idx, col] for col in range(n_rhs)]
            work_rows.append(row)

        _, pivcol_to_row, _pivot_cols, _ = _eliminate(
            work_rows,
            rhs_cols=n_rhs,
            record_divisors=False,
            allow_formal_inverse=allow_formal_inverse,
            simplify_steps=simplify_steps,
            fast_only=False,
            want_pivmap=True,
        )

        inconsistent = set()
        for row in work_rows:
            if any(not _pivot_is_zero(row[col]) for col in range(n_unknowns)):
                continue
            for idx in range(n_rhs):
                if not _pivot_is_zero(row[n_unknowns + idx]):
                    inconsistent.add(idx)

        solutions = [[0] * n_unknowns for _ in range(n_rhs)]
        for pivot_col, pivot_row in pivcol_to_row.items():
            for idx in range(n_rhs):
                solutions[idx][pivot_col] = work_rows[pivot_row][n_unknowns + idx]

        free_cols = [col for col in range(n_unknowns) if col not in pivcol_to_row]
        return solutions, inconsistent, free_cols

    def pivot_columns(self, record_divisors=False, simplify_steps=False):
        if self.nrows == 0 or self.ncols == 0:
            return ([], []) if record_divisors else []

        fast_types = fast_scalar_types()
        fast_case = all(isinstance(entry, fast_types) for entry in self._data.values())

        work_rows = self._dense_copy()
        _, _pivcol_to_row, pivot_cols, divisors = _eliminate(
            work_rows,
            rhs_cols=0,
            record_divisors=record_divisors,
            allow_formal_inverse=True,
            simplify_steps=simplify_steps,
            fast_only=fast_case,
            want_pivmap=True,
        )
        if record_divisors:
            return pivot_cols, (divisors or [])
        return pivot_cols

    def nullspace(self):
        row_count = self.nrows
        col_count = self.ncols

        if col_count == 0:
            return []
        if row_count == 0:
            out = []
            for idx in range(col_count):
                basis_vector = self.__class__.zeros(col_count, 1)
                basis_vector._data[_spool((idx, 0), basis_vector.shape)] = 1
                out.append(basis_vector)
            return out

        work_rows = self._dense_copy()
        _, pivcol_to_row, pivot_cols, _ = _eliminate(
            work_rows,
            rhs_cols=0,
            record_divisors=False,
            allow_formal_inverse=False,
            simplify_steps=False,
            fast_only=False,
            want_pivmap=True,
        )

        free_cols = [idx for idx in range(col_count) if idx not in pivcol_to_row]
        if not free_cols:
            return []

        out = []
        for free_col in free_cols:
            components = [0] * col_count
            components[free_col] = 1
            for pivot_col in pivot_cols:
                pivot_row = pivcol_to_row[pivot_col]
                components[pivot_col] = -work_rows[pivot_row][free_col]

            basis_vector = self.__class__.zeros(col_count, 1)
            for idx, value in enumerate(components):
                if not _scalar_is_zero(value):
                    basis_vector._data[_spool((idx, 0), basis_vector.shape)] = value
            out.append(basis_vector)

        return out

    def solve_right(
        self,
        b,
        *,
        return_divisors=False,
        simplify_steps=False,
        stamp_divisors=False,
        allow_formal_inverse=True,
        parametric_vars=None,
    ):
        if self.nrows == 0:
            sol = [] if (getattr(b, "nrows", 0) == 0) else None
            return (sol, []) if return_divisors else sol

        rhs_matrix = b if isinstance(b, matrix_dgcv) else matrix_dgcv(b)
        if rhs_matrix.ncols != 1 or rhs_matrix.nrows != self.nrows:
            raise ValueError("b must be a column vector with matching nrows")

        fast_types = fast_scalar_types()
        fast_case = all(
            isinstance(entry, fast_types) for entry in self._data.values()
        ) and all(isinstance(entry, fast_types) for entry in rhs_matrix._data.values())

        row_count = self.nrows
        col_count = self.ncols
        work_rows = [
            [self[idx, col_idx] for col_idx in range(col_count)] + [rhs_matrix[idx, 0]]
            for idx in range(row_count)
        ]

        _rank, pivcol_to_row, _, divisors = _eliminate(
            work_rows,
            rhs_cols=1,
            record_divisors=return_divisors,
            allow_formal_inverse=allow_formal_inverse,
            simplify_steps=simplify_steps,
            fast_only=fast_case,
            want_pivmap=True,
        )

        for row_idx in range(_rank, row_count):
            if not _pivot_is_zero(work_rows[row_idx][col_count]):
                out = None
                if return_divisors:
                    divisor_list = divisors or []
                    if stamp_divisors and divisor_list:
                        seen = set()
                        deduped = []
                        for divisor in divisor_list:
                            if divisor in seen:
                                continue
                            seen.add(divisor)
                            deduped.append(divisor)
                        return out, deduped
                    return out, divisor_list
                return out

        if parametric_vars is None:
            prefix = create_key("_x", True, 4)
            params = _disposable_symbols(f"{prefix}_", col_count)
        else:
            params = list(parametric_vars)
            if len(params) != col_count:
                raise ValueError("parametric_vars must have length equal to ncols")

        free_cols = [idx for idx in range(col_count) if idx not in pivcol_to_row]
        solution = [0] * col_count
        for free_col in free_cols:
            solution[free_col] = params[free_col]

        for pivot_col in sorted(pivcol_to_row):
            pivot_row = pivcol_to_row[pivot_col]
            value = work_rows[pivot_row][col_count]
            for free_col in free_cols:
                value = value - work_rows[pivot_row][free_col] * solution[free_col]
            if simplify_steps and not fast_case:
                value = _fast_simplify(value)
            solution[pivot_col] = value

        out = solution

        if return_divisors:
            divisor_list = divisors or []
            if stamp_divisors and divisor_list:
                seen = set()
                deduped = []
                for divisor in divisor_list:
                    if divisor in seen:
                        continue
                    seen.add(divisor)
                    deduped.append(divisor)
                return out, deduped
            return out, divisor_list
        return out

    def try_solve_right(self, b):
        return self.solve_right(b, return_divisors=False)

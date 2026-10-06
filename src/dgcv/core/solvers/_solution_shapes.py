import operator

from ..._aux._backends._symbolic_router import subs
from ..._aux._backends._types_and_constants import is_atomic
from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry


def _is_relation(obj):
    return (
        getattr(obj, "lhs", None) is not None and getattr(obj, "rhs", None) is not None
    )


def _rel_lhs_rhs(rel):
    op = getattr(rel, "operator", None)
    if callable(op):
        try:
            if op() is not operator.eq:
                return None, None
        except Exception:
            return None, None
    f = getattr(rel, "lhs", None)
    g = getattr(rel, "rhs", None)
    if callable(f) and callable(g):
        try:
            return rel.lhs(), rel.rhs()
        except Exception:
            return None, None
    if f is not None and g is not None and not callable(f) and not callable(g):
        return f, g
    return None, None


def _sage_branch_to_dict(branch, wanted):
    if _is_relation(branch):
        rels = [branch]
    else:
        try:
            rels = list(branch)
        except TypeError:
            return None
    if not rels:
        return None
    d = {}
    for rel in rels:
        L, R = _rel_lhs_rhs(rel)
        if L is None:
            return None
        var = wanted.get(str(L))
        if var is None:
            return None
        d[var] = R
    return d


def _warn_dropped_branches(dropped):
    if get_dgcv_settings_registry().get("forgo_warnings", False):
        return
    shown = str(dropped[0]).replace("\n", " ")
    if len(shown) > 120:
        shown = shown[:117] + "..."
    dgcv_warning(
        f"solve_dgcv: the sage solver returned {len(dropped)} solution branch(es) it could not "
        f"make explicit (e.g. `{shown}`). They were omitted, so the returned solutions may be "
        "incomplete.",
        stacklevel=6,
    )


def _sage_solve_to_dicts(
    sols, vars_, input_symbols=frozenset(), fold_parameters=True, warn=True
):
    if sols is None:
        return []

    if isinstance(sols, dict) or _is_relation(sols):
        sols = [sols]
    else:
        try:
            sols = list(sols)
        except TypeError:
            sols = [sols]

    wanted = {str(v): v for v in vars_}
    input_names = {str(s) for s in input_symbols}
    out = []
    dropped = []

    for s in sols:
        if isinstance(s, dict):
            out.append({wanted[str(k)]: v for k, v in s.items() if str(k) in wanted})
            continue

        d = _sage_branch_to_dict(s, wanted)
        if d is None:
            dropped.append(s)
            continue

        if fold_parameters:
            replacements = {}
            for var, val in d.items():
                if (
                    is_atomic(val)
                    and str(val) not in wanted
                    and str(val) not in input_names
                ):
                    replacements[val] = var
            if replacements:
                d = {k: subs(v, replacements) for k, v in d.items()}

        out.append(d)

    if dropped and warn:
        _warn_dropped_branches(dropped)

    return out


def _linsolve_to_dicts(solset, vars_):
    if not solset:
        return []
    out = []
    for tup in solset:
        if isinstance(tup, dict):
            out.append(tup)
            continue
        try:
            tup = tuple(tup) if hasattr(tup, "__iter__") else ()
        except Exception:
            tup = ()
        if len(tup) == len(vars_):
            out.append(dict(zip(vars_, tup)))
    return out


def _engine_solve_to_dicts(sols, vars_):
    if sols is None:
        return []
    if isinstance(sols, dict):
        return [sols]
    if not isinstance(sols, (list, tuple)):
        sols = [sols]

    out = []
    vars_set = set(vars_)

    for s in sols:
        if isinstance(s, dict):
            out.append(s)
            continue

        rels = list(s) if isinstance(s, (list, tuple)) else [s]

        d = {}
        ok = True
        for rel in rels:
            lhs, rhs = _rel_lhs_rhs(rel)
            if lhs is None:
                ok = False
                break
            if lhs in vars_set:
                d[lhs] = rhs

        if ok:
            out.append(d)

    return out

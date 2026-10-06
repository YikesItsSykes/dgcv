import random
import string

from ..._aux._backends._symbolic_router import subs
from ..._aux._backends._types_and_constants import (
    as_engine_scalar,
    expr_numeric_types,
    is_atomic,
    symbol,
    to_active_engine,
)
from ..._aux._utilities._config import working_namespace
from ..._aux._vmf._safeguards import get_dgcv_category


def _generate_str_id(base_str, *dicts):
    candidate = base_str
    while any(candidate in d for d in dicts):
        random_suffix = "".join(
            random.choices(string.ascii_letters + string.digits, k=6)
        )
        candidate = f"{base_str}_{random_suffix}"
    return candidate


def _identifier(hint):
    hint = str(hint)
    if hint.isidentifier():
        return hint
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in hint)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


class SolveBridge:
    def __init__(self):
        self.standins = {}
        self.reverse = {}
        self._by_obj = {}
        self._lifters = {}

    @property
    def symbols(self):
        return set(self.reverse)

    def lookup(self, obj):
        try:
            return self._by_obj.get(obj, obj)
        except TypeError:
            return obj

    def adopt(self, identifier, obj, sym):
        self.standins[identifier] = (obj, sym)
        self.reverse[sym] = obj
        try:
            self._by_obj[obj] = sym
        except TypeError:
            pass

    def standin(self, obj, hint=None):
        sym = self.lookup(obj)
        if sym is not obj:
            return sym
        if is_atomic(obj):
            return obj
        hint = str(obj) if hint is None else str(hint)
        identifier = _generate_str_id(hint, self.standins, working_namespace())
        try:
            sym = symbol(identifier)
        except Exception:
            identifier = _generate_str_id(
                _identifier(hint), self.standins, working_namespace()
            )
            sym = symbol(identifier)
        self.adopt(identifier, obj, sym)
        return sym

    def lower(self, x):
        if x is None or (isinstance(x, (list, tuple)) and len(x) == 0):
            return []
        hook = getattr(x, "__dgcv_solve_bridge__", None)
        if hook is not None:
            return list(hook(self))
        obstruction = getattr(x, "__dgcv_zero_obstr__", None)
        if obstruction is not None:
            return list(obstruction[0])
        category = get_dgcv_category(x)
        if category == "expression":
            return [as_engine_scalar(x.polyExpr)]
        if category in ("algebra_element", "subalgebra_element"):
            return [as_engine_scalar(term) for term in x.coeffs]
        if category in ("tensorProduct", "fastTensorProduct"):
            return [as_engine_scalar(term) for term in x.coeff_dict.values()]
        if isinstance(x, expr_numeric_types()):
            return [as_engine_scalar(x)]
        return [to_active_engine(x)]

    def register_lifter(self, key, lifter):
        self._lifters.setdefault(key, lifter)

    def lift_expr(self, expr):
        if not hasattr(expr, "subs"):
            return expr
        for lifter in self._lifters.values():
            lifted = lifter(self, expr)
            if lifted is not None:
                return lifted
        if not self.reverse:
            return expr
        try:
            return subs(expr, self.reverse)
        except Exception:
            return expr

    def lift_var(self, var):
        try:
            if var in self.reverse:
                return self.reverse[var]
        except TypeError:
            return var
        entry = self.standins.get(str(var))
        return entry[0] if entry is not None else var

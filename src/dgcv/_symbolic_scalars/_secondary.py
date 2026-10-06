# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import numbers
import signal
import threading
import time
from fractions import Fraction

from .._aux._backends._engine import (
    _get_sage_module,
    _get_sympy_module,
    is_sage_available,
    is_sympy_available,
)
from .._aux._backends._types_and_constants import to_active_engine
from .._aux._utilities._config import get_dgcv_settings_registry
from ..core.solvers._solution_shapes import _sage_solve_to_dicts
from ..eds._atoms import zero_form_atom
from ..eds._constants import PI, E, NamedConstant
from ..eds._zero_forms import _nf_of, zero_form_class
from . import _poly
from ._foreign import from_sage_rf, from_sympy_rf
from ._functions import ElementaryFunction, FunctionAtom, Radical
from ._normal_form import _not_finite, nf_from_base


class SecondaryUnavailable(NotImplementedError):
    pass


class SecondaryBudgetExpired(Exception):
    pass


_QUALITY_FEATURES = frozenset({"simplify", "factor"})
_SHORTEST_TIMER = 1e-6


def quality_budget(kind):
    if kind != "sympy":
        return None
    budget = get_dgcv_settings_registry().get("secondary_time_budget")
    if not budget:
        return None
    if threading.current_thread() is not threading.main_thread():
        return None
    if not hasattr(signal, "setitimer"):
        return None
    return budget


def _run_with_budget(call, budget, feature):
    host_remaining, host_interval = signal.getitimer(signal.ITIMER_REAL)
    if 0 < host_remaining <= budget:
        return call()

    def expire(_signum, _frame):
        raise SecondaryBudgetExpired(
            f"the secondary engine exceeded the `secondary_time_budget` of {budget} s on `{feature}`; the builtin form was kept"
        )

    previous = signal.signal(signal.SIGALRM, expire)
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

        if host_remaining:
            left = host_remaining - (time.monotonic() - started)
            signal.setitimer(
                signal.ITIMER_REAL, max(left, _SHORTEST_TIMER), host_interval
            )


_UNSET = object()
_cache = {"setting": _UNSET, "kind": None}
_bridges = {}


def invalidate_secondary_cache():
    _cache["setting"] = _UNSET
    _cache["kind"] = None
    for bridge in _bridges.values():
        bridge.forget()
    _bridges.clear()


def secondary_kind():
    setting = get_dgcv_settings_registry().get("secondary_symbolic_engine", "auto")
    if _cache["setting"] == setting:
        return _cache["kind"]
    if setting is None:
        kind = None
    elif setting == "sympy":
        kind = "sympy" if is_sympy_available() else None
    elif setting == "sage":
        kind = "sage" if is_sage_available() else None
    elif is_sympy_available():
        kind = "sympy"
    elif is_sage_available():
        kind = "sage"
    else:
        kind = None

    _cache["setting"] = setting
    _cache["kind"] = kind
    return kind


def secondary_module(kind=None):
    if kind is None:
        kind = secondary_kind()
    if kind == "sympy":
        return _get_sympy_module()
    if kind == "sage":
        return _get_sage_module()
    return None


def simplify_secondary_kind():
    if secondary_kind() is None:
        return None
    return "sympy" if is_sympy_available() else None


def _require(feature, kind=None):
    if kind is None:
        kind = secondary_kind()
    if kind is None:
        setting = get_dgcv_settings_registry().get("secondary_symbolic_engine", "auto")
        if setting is None:
            cause = "the secondary engine is switched off; select one with `set_dgcv_settings(secondary_engine=...)`"
        elif setting in ("sympy", "sage"):
            cause = f"the selected secondary engine ({setting}) is not installed"
        else:
            cause = "no secondary engine is installed; install sympy or sage"

        raise SecondaryUnavailable(
            f"`{feature}` is not available in dgcv's builtin symbolic engine and {cause}"
        )
    return kind


def _bridge_for(kind, mod):
    bridge = _bridges.get(kind)
    if bridge is None or bridge.mod is not mod:
        bridge = SecondaryBridge(kind, mod)
        _bridges[kind] = bridge
    return bridge


class SecondaryBridge:
    def __init__(self, kind, mod):
        self.kind = kind
        self.mod = mod
        self.standins = {}
        self.reverse = {}
        self.heads = {}
        self.gens = {}
        self.names = {}

    def forget(self):
        if self.kind == "sage":
            for _atom, sym, _sig in list(self.standins.values()):
                self._forget_sym(sym)
        self.standins.clear()
        self.reverse.clear()
        self.names.clear()
        self.heads.clear()
        self.gens.clear()

    def _forget_sym(self, sym):
        try:
            for a in self.mod.assumptions(sym):
                self.mod.forget(a)
        except Exception:
            pass

    def _assumptions(self, atom):
        markers = atom._markers
        out = {}
        if atom.role in ("real", "imaginary") or "real" in markers:
            out["real"] = True
        for key in ("positive", "negative", "nonnegative", "nonpositive", "integer"):
            if key in markers:
                out[key] = True
        return out

    def _signature(self, atom, base=None):
        return (atom._markers, atom.role, None if base is None else id(base))

    def _evict(self, atom):
        entry = self.standins.pop(atom, None)
        if entry is None:
            return
        _held, sym, _sig = entry
        if self.reverse.get(sym) is _held:
            del self.reverse[sym]
            self.names.pop(str(sym), None)
        if self.kind == "sage":
            self._forget_sym(sym)
        self.gens.clear()

    def symbol_for(self, atom):
        entry = self.standins.get(atom)
        if entry is not None:
            held, sym, sig = entry
            if held is atom:
                return sym
            base = None
            if atom.role == "antiholomorphic":
                partner = atom.partner("holomorphic")
                if partner is not None:
                    pentry = self.standins.get(partner)
                    base = None if pentry is None else pentry[1]

            if sig == self._signature(atom, base):
                self.standins[atom] = (atom, sym, sig)
                if self.reverse.get(sym) is held:
                    self.reverse[sym] = atom
                    self.names[str(sym)] = atom

                return sym

            self._evict(held)

        if atom.role == "antiholomorphic":
            partner = atom.partner("holomorphic")
            if partner is not None:
                base = self.symbol_for(partner)
                sym = self.mod.conjugate(base)
                self.standins[atom] = (atom, sym, self._signature(atom, base))
                return sym

        name = str(atom)
        assumptions = self._assumptions(atom)
        if self.kind == "sympy":
            sym = self.mod.Symbol(name, **assumptions)
        else:
            if assumptions.get("real"):
                sym = self.mod.var(name, domain="real")
            else:
                sym = self.mod.var(name, domain="complex")

            if assumptions.get("positive"):
                self.mod.assume(sym > 0)

        other = self.reverse.get(sym)
        if other is not None and other is not atom:
            self._evict(other)

        self.standins[atom] = (atom, sym, self._signature(atom))
        self.reverse[sym] = atom
        self.names[str(sym)] = atom
        return sym

    def head_for(self, label):
        head = self.heads.get(label)
        if head is None:
            if self.kind == "sympy":
                head = self.mod.Function(label)
            else:
                head = self.mod.function(label)
            self.heads[label] = head
        return head

    def number(self, c):
        if isinstance(c, Fraction):
            if self.kind == "sympy":
                return self.mod.Rational(c.numerator, c.denominator)
            return self.mod.QQ(c.numerator) / self.mod.QQ(c.denominator)
        if isinstance(c, bool):
            return int(c)
        if isinstance(c, int):
            return self.mod.Integer(c) if self.kind == "sympy" else self.mod.ZZ(c)
        return c

    def gen_cas(self, g):
        leaf = _poly.gen_leaf(g)
        if g == _poly.I_GEN:
            return self.mod.I
        if isinstance(leaf, zero_form_atom):
            return self.symbol_for(leaf)
        cached = self.gens.get(g)
        if cached is not None:
            return cached
        out = self.leaf_cas(leaf, lambda a: self.lower_rf(nf_from_base(a)))
        self.gens[g] = out
        return out

    def leaf_cas(self, leaf, lower):
        if isinstance(leaf, NamedConstant):
            if leaf == E:
                return self.mod.E if self.kind == "sympy" else self.mod.e
            if leaf == PI:
                return self.mod.pi
            return self.symbol_for_label(leaf.label, leaf)

        if isinstance(leaf, ElementaryFunction):
            return getattr(self.mod, leaf.name)(lower(leaf.args[0]))
        if isinstance(leaf, Radical):
            arg = lower(leaf.args[0])
            if self.kind == "sympy":
                return self.mod.Pow(arg, self.mod.Rational(1, leaf.q))
            return arg ** (self.mod.QQ(1) / leaf.q)

        if isinstance(leaf, FunctionAtom):
            head = self.head_for(leaf.label_base)
            args = [lower(a) for a in leaf.args]
            applied = head(*args)
            if not any(leaf.derivs):
                return applied
            if self.kind == "sympy":
                if not all(isinstance(a, self.mod.Symbol) for a in args):
                    raise SecondaryUnavailable(
                        "derivatives of functions with non-symbol arguments cannot be lowered to the secondary engine"
                    )
                return self.mod.Derivative(
                    applied, *[(a, n) for a, n in zip(args, leaf.derivs) if n]
                )

            for a, n in zip(args, leaf.derivs):
                if n:
                    applied = applied.diff(a, n)

            return applied

        raise SecondaryUnavailable(
            f"`{leaf}` cannot be lowered to the secondary engine"
        )

    def symbol_for_label(self, label, leaf=None):
        sym = self.heads.get(("symbol", label))
        if sym is None:
            name = f"{label}_const"
            sym = self.mod.Symbol(name) if self.kind == "sympy" else self.mod.var(name)
            self.heads[("symbol", label)] = sym
            if leaf is not None:
                self.reverse[sym] = leaf
                self.names[str(sym)] = leaf
        return sym

    def lower_poly(self, poly):
        if not poly:
            return self.number(0)
        gen_cas = self.gen_cas
        number = self.number
        cas_gens = {}
        terms = []
        if self.kind == "sympy":
            Mul = self.mod.Mul
            Pow = self.mod.Pow
            for m, c in poly.items():
                factors = [] if c == 1 else [number(c)]
                for g, e in m:
                    x = cas_gens.get(g)
                    if x is None:
                        x = cas_gens[g] = gen_cas(g)

                    factors.append(x if e == 1 else Pow(x, e))

                if not factors:
                    terms.append(number(1))
                elif len(factors) == 1:
                    terms.append(factors[0])
                else:
                    terms.append(Mul(*factors))

            if len(terms) == 1:
                return terms[0]
            return self.mod.Add(*terms)

        for m, c in poly.items():
            term = number(c)
            factors = []
            for g, e in m:
                x = cas_gens.get(g)
                if x is None:
                    x = cas_gens[g] = gen_cas(g)

                factors.append(x if e == 1 else x**e)

            if factors:
                term = self.mod.SR(term).mul(*factors)

            terms.append(term)

        if len(terms) == 1:
            return terms[0]
        return self.mod.SR(0).add(*terms)

    def lower_rf(self, rf):
        num = self.lower_poly(rf.num)
        if rf.den.get(()) == 1 and len(rf.den) == 1:
            out = num
        else:
            out = num / self.lower_poly(rf.den)
        if self.kind == "sage":
            return self.mod.SR(out)
        return out

    def lower(self, x):
        if isinstance(x, (list, tuple)):
            return type(x)(self.lower(a) for a in x)
        if isinstance(x, bool):
            return x
        if isinstance(x, numbers.Number):
            return self.number(x)
        x = to_active_engine(x)
        return self.lower_rf(_nf_of(x))

    def lift(self, x):
        if isinstance(x, dict):
            return {self.lift(k): self.lift(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return type(x)(self.lift(a) for a in x)
        if isinstance(x, bool) or x is None:
            return x
        if self.kind == "sympy":
            if not isinstance(x, self.mod.Basic):
                return x
            if isinstance(x, self.mod.Symbol):
                atom = self.reverse.get(x)
                if atom is not None:
                    return atom
            if x.is_Number:
                return _lift_number(x)
            rf = from_sympy_rf(x, symbol_map=self.reverse)
        else:
            mod = type(x).__module__ or ""
            if not (mod.startswith("sage") or hasattr(x, "operator")):
                return x
            if hasattr(x, "is_symbol") and x.is_symbol():
                atom = self.names.get(str(x))
                if atom is not None:
                    return atom
            rf = from_sage_rf(x, names=self.names)
        if rf.is_constant:
            return rf.constant_value()
        return zero_form_class._from_nf(rf)


def _lift_number(x):
    if x.is_Integer:
        return int(x)
    if x.is_Rational:
        return Fraction(int(x.p), int(x.q))
    if not x.is_finite:
        raise _not_finite(x)
    return float(x)


def with_secondary(exprs, fn, feature, lift=True, kind=None):
    kind = _require(feature, kind)
    mod = secondary_module(kind)
    bridge = _bridge_for(kind, mod)
    lowered = [bridge.lower(e) for e in exprs]
    budget = quality_budget(kind) if feature in _QUALITY_FEATURES else None
    if budget is None:
        out = fn(mod, lowered, bridge)
    else:
        out = _run_with_budget(lambda: fn(mod, lowered, bridge), budget, feature)

    if not lift:
        return out
    try:
        return bridge.lift(out)
    except NotImplementedError as exc:
        raise SecondaryUnavailable(
            f"the secondary engine ({kind}) returned `{out}` for `{feature}`, which dgcv's builtin engine cannot represent: {exc}"
        ) from None


def solve_with_secondary(eqns, vars_):
    n_eqns = len(eqns)

    def fn(mod, lowered, bridge):
        eqs = list(lowered[:n_eqns])
        vs = list(lowered[n_eqns:])
        if bridge.kind == "sympy":
            return mod.solve(eqs, vs, dict=True)
        if len(eqs) == 1 and len(vs) > 1:
            eqs = eqs * 2
        return _sage_solution_dicts(mod.solve(eqs, vs), vs)

    return with_secondary(list(eqns) + list(vars_), fn, feature="solve")


def _sage_solution_dicts(sols, vs):
    return _sage_solve_to_dicts(sols, vs, fold_parameters=False)


def integrate(expr, *args, **kwargs):
    def fn(mod, lowered, bridge):
        e, *vs = lowered
        if bridge.kind == "sympy":
            return mod.integrate(e, *vs, **kwargs)
        return e.integral(*vs)

    return with_secondary([expr, *args], fn, feature="integrate")

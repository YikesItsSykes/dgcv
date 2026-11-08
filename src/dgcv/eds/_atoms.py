"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._atoms

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from collections import Counter
from fractions import Fraction
from functools import total_ordering

from ..core.base import dgcv_class
from .._aux._utilities._config import dgcv_warning, get_variable_registry
from .._aux._vmf._safeguards import retrieve_passkey
from .._aux.printing.printing._eds import conjugation_prefix, ordinal_latex
from .._aux.printing.printing._string_processing import _process_var_label
from .._aux._backends._calculus import diff as _routed_diff
from .._aux._backends._cls_coercion import register_legacy_sympy_class
from .._aux._backends import _engine as _engine_mod
from .._aux._backends._engine import engine_kind
from .._aux._backends._symbolic_router import register_zero_test
from .._aux._backends._symbolic_router import conjugate as _routed_conjugate
from .._aux._backends._types_and_constants import expr_numeric_types

_nf_hooks = {}


def _builtin():
    kind = _engine_mod._engine_kind
    if kind is None:
        kind = engine_kind()
    return kind == "builtin"
from ._roles import (
    ASSUMPTION_NAMES,
    CONJUGATE_ROLE,
    FAMILY_ROLES,
    NON_INHERITED_MARKERS,
    REAL_IMPLYING_ASSUMPTIONS,
    ROLE_MARKERS,
    family_of_label,
    parse_assumptions,
    role_of_label,
)


@total_ordering
class SortableObj:
    def __init__(self, pair: tuple):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Input must be a tuple of (value, sortable_key)")
        self.pair = pair
        self.value = pair[0]
        self.place = pair[1]

    def __lt__(self, other):
        if not isinstance(other, SortableObj):
            return NotImplemented
        return self.place < other.place  # Comparison is based on the sortable key

    def __eq__(self, other):
        if not isinstance(other, SortableObj):
            return NotImplemented
        return self.place == other.place  # Equality is based on the sortable key

    def __repr__(self):
        return f"SortableObj(value={self.value}, place={self.place})"


@total_ordering
class barSortedStr:
    def __init__(self, label=None):
        if label:
            if not isinstance(label, str):
                raise ValueError(f"Input must be a str. Recieved {label}")
            self.str = label
        else:
            self.str = None

    def __lt__(self, other):
        pref = conjugation_prefix()
        prefL = len(pref)
        if self.str is None:
            return True
        if other is None:
            return False
        if isinstance(other, barSortedStr):
            other = other.str
            if other is None:
                return False
        if not isinstance(other, str):
            return NotImplemented
        if self.str[0:prefL] == pref:
            if other[0:prefL] == pref:
                return self.str < other
            else:
                return False
        else:
            if other[0:prefL] == pref:
                return True
            else:
                return self.str < other

    def __eq__(self, other):
        if self.str is None:
            if other is None or (isinstance(other, barSortedStr) and other.str is None):
                return True
            else:
                return False
        if isinstance(other, barSortedStr):
            other = other.str
            if other is None:
                return False
        if not isinstance(other, str):
            return NotImplemented
        return self.str == other  # Equality is based on str label

    def __repr__(self):
        return f"barSortedStr(value={self.str})"


def _custom_conj(expr):
    if isinstance(
        expr,
        (
            zero_form_class,
            zero_form_atom,
            abstract_differential_form_atom,
            abstract_differential_form_monomial,
            abstract_differential_form,
        ),
    ):
        return expr._eval_conjugate()
    else:
        return _routed_conjugate(expr)


_EMPTY_COFRAME = None


def _derivative_key(coframe_derivatives):
    return tuple((elem[0].hash_key,) + tuple(elem[1:]) for elem in coframe_derivatives)


def _empty_coframe():
    global _EMPTY_COFRAME
    if _EMPTY_COFRAME is None:
        _EMPTY_COFRAME = abstract_coframe(tuple(), {})
    return _EMPTY_COFRAME


def _imag_unit():
    from .._aux._backends._types_and_constants import imag_unit

    return imag_unit()


def _half():
    from .._aux._backends._types_and_constants import half

    return half()


def _generic_re(atom):
    return zero_form_class(("mul", _half(), ("add", atom, atom._eval_conjugate())))


def _generic_im(atom):
    return zero_form_class(
        ("mul", -1, _imag_unit(), _half(), ("sub", atom, atom._eval_conjugate()))
    )


def _new_bridge():
    from ..core.solvers._bridge import SolveBridge

    return SolveBridge()


def _zero_obstruction(obj):
    bridge = _new_bridge()
    return obj.__dgcv_solve_bridge__(bridge), bridge.symbols


def _normal_order(cf, indices):
    if cf._all_directions_commute():
        return tuple(sorted(indices))
    out = []
    for a in indices:
        start = len(out)
        while start > 0 and cf._directions_commute(out[start - 1], a):
            start -= 1
        pos = start
        while pos < len(out) and out[pos] <= a:
            pos += 1
        out.insert(pos, a)
    return tuple(out)


def _in_normal_order(atom):
    for cd in atom.coframe_derivatives:
        idx = cd[1:]
        if len(idx) > 1 and not (
            cd[0]._all_directions_commute() and list(idx) == sorted(idx)
        ):
            return False
    return True


def _derivative_remainder(cf, mine, theirs):
    if cf._all_directions_commute():
        left = Counter(theirs)
        left.subtract(Counter(mine))
        if any(v < 0 for v in left.values()):
            return None
        return tuple(sorted(left.elements()))
    n = len(mine)
    if theirs[:n] != mine:
        return None
    return theirs[n:]


_ATOM_KINDS = ("zero_form", "zero_form_conjugate")
_LABEL_ASSUMPTIONS = frozenset({"real", "constant"})


def _family_of(label):
    registry = get_variable_registry()
    info = registry["paths"].get(label)
    if info is None or info.get("kind") not in _ATOM_KINDS:
        return None, None
    system_label = info["path"][1]
    return system_label, registry["eds"]["atoms"].get(system_label)


def _family_entry(label):
    return _family_of(label)[1]


def _assumption_record(atom, create=False):
    family = _family_entry(atom.label)
    if family is None:
        return None
    record = family.get("assumptions")
    if record is None and create:
        record = family["assumptions"] = {}
    return record


def _independence_rules(independent_of, coframe_independants):
    if independent_of is None:
        return coframe_independants
    rules = {cf: set(idx) for cf, idx in (coframe_independants or {}).items()}
    if isinstance(independent_of, dict):
        for cf, idx in independent_of.items():
            rules.setdefault(cf, set()).update(idx)
        return rules
    forms = (
        list(independent_of)
        if isinstance(independent_of, (list, tuple, set, frozenset))
        else [independent_of]
    )
    for form in forms:
        if not isinstance(form, abstract_differential_form_atom):
            raise TypeError(
                "`independent_of` expects 1-forms of a coframe or a dictionary {coframe: indices}"
            )
        cf = form.primary_coframe
        if cf is None:
            continue
        rules.setdefault(cf, set()).add(cf.forms.index(form))
    return rules


def _mapped_independence(rules, conjugate=False):
    if not conjugate:
        return {cf: set(idx) for cf, idx in rules.items()}
    return {cf: {cf.conj_rules.get(i, i) for i in idx} for cf, idx in rules.items()}


def _independent_of_form(rules, form):
    return any(
        form in other.forms and other.forms.index(form) in indices
        for other, indices in rules.items()
    )


def _declare_real(system_label, family, pair):
    primary, conj = pair
    marks = family.setdefault("declared", {})
    if "real" in marks.get(primary.label, ()) or "real" in primary._markers:
        return
    marks[primary.label] = marks.get(primary.label, frozenset()) | {"real"}
    family["family_relatives"][primary.label] = (primary, primary)
    if conj.label != primary.label:
        registry = get_variable_registry()
        family["family_relatives"].pop(conj.label, None)
        family["conjugates"].pop(conj.label, None)
        registry["paths"].pop(conj.label, None)
        owner = registry["_labels"].get(system_label)
        if owner is not None:
            owner.get("children", set()).discard(conj.label)
        family.setdefault("retired", []).append(conj.label)
        dgcv_warning(
            f"`{primary.label}` is now real: `{conj.label}` has left the variable management framework but "
            "stays bound in the namespace and may occur in expressions built earlier; replace it with "
            f"`subs({{{conj.label}: {primary.label}}})`"
        )
    if all("real" in marks.get(name, ()) for name in family.get("family_names", ())):
        family["real"] = True


class zero_form_atom(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "zeroFormAtom"
    _dgcv_categories = {"zeroFormAtom", "eds_scalar"}

    def __init__(
        self,
        label,
        coframe_derivatives=tuple(),
        coframe=None,
        _markers=frozenset(),
        coframe_independants=None,
        assumptions=None,
    ):
        """
        Create a new zeroFormAtom instance.

        Parameters:
        - label (str): The base function label.
        - coframe_derivatives (tuple, optional): tuple of tuples whose first entry is a coframe and whose subsequent entries are indices of coframe derivatives.
        - coframe (abstract_coframe, optional): marks the primary abstract_coframe w.r.t. the zero forms printing/display behavior may be adjusted
        """
        if not isinstance(label, str):
            raise TypeError(
                f"label must be type `str`. Instead received `{type(label)}`\n given label: {label}"
            )

        if coframe_independants is None:
            coframe_independants = {}
        if assumptions is not None:
            _markers = frozenset(_markers) | {
                name for name, value in parse_assumptions(assumptions).items() if value
            }
        if not isinstance(coframe_derivatives, (tuple, list)):
            raise ValueError(
                f"coframe_derivatives must be a tuple of tuples whose first entry is a coframe and whose subsequent entries are indices of coframe derivatives.\n given `coframe_derivatives` of type: {type(coframe_derivatives)}"
            )
        for elem in coframe_derivatives:
            if not len(elem) == 0 and not isinstance(elem[0], abstract_coframe):
                raise ValueError(
                    f"first elements in the `coframe_derivatives` tuples must be coframe type. Given object {elem} instead with type {type(elem)}"
                )
            if not all(
                isinstance(order, numbers.Integral)
                and order in range(elem[0].dimension)
                for order in elem[1:]
            ):
                raise ValueError(
                    f"tuples in `coframe_derivatives` must begin with an abstCoframe instance followed by non-negative integers in the range of the corresponding coframe dimension. Instead of an integer tuple, recieved: {elem[1:]} \n with associated coframe basis: {elem[0]}"
                )
        coframe_derivatives = tuple(
            (elem[0],) + _normal_order(elem[0], tuple(elem[1:]))
            for elem in coframe_derivatives
            if len(elem) > 1
        )
        if coframe_derivatives and not NON_INHERITED_MARKERS.isdisjoint(_markers):
            _markers = frozenset(_markers) - NON_INHERITED_MARKERS
        if coframe is None or (coframe_derivatives and coframe is _empty_coframe()):
            if len(coframe_derivatives) > 0:
                coframe = coframe_derivatives[0][0]
            else:
                coframe = _empty_coframe()
        elif len(coframe_derivatives) > 0 and coframe != coframe_derivatives[0][0]:
            coframe_derivatives = ((coframe,),) + coframe_derivatives
        if not isinstance(coframe, abstract_coframe):
            raise TypeError(
                "Expected given `coframe` to be None or have type `abstract_coframe`"
            )

        # Using SymPy's Basic constructor
        self.label = label
        self.coframe_derivatives = coframe_derivatives
        self._hash = hash((label, _derivative_key(coframe_derivatives)))
        self.coframe = coframe

        self._markers = _markers
        self._coframe_independants = coframe_independants
        constant = "constant" in _markers
        self.is_one = self.label == "_1" and constant
        self._is_zero = self.label == "_0" and constant
        self._rf_cache = None
        self._self_conj = None
        self.secondary_coframes = [
            elem[0] for elem in self.coframe_derivatives if elem[0] != self.coframe
        ]
        self.related_coframes = (
            [self.coframe] + self.secondary_coframes
            if self.coframe is not _empty_coframe()
            else self.secondary_coframes
        )

    is_minus_one = False

    def __dgcv_solve_bridge__(self, bridge):
        if engine_kind() == "builtin":
            return [self]
        bridge.register_lifter("eds", _eds_lift)
        return [bridge.standin(self, str(self))]

    def diff(self, *args, **kwargs):
        if engine_kind() != "builtin":
            return self.__dgcv_apply__(lambda e: _routed_diff(e, *args, **kwargs))
        pairs = []
        i = 0
        while i < len(args):
            order = 1
            if i + 1 < len(args) and isinstance(args[i + 1], numbers.Integral):
                order = int(args[i + 1])
                i += 1
            var = args[i]
            if isinstance(var, zero_form_class):
                var = var.base
            pairs.append((var, order))
            i += 1
        total = sum(order for _, order in pairs)
        if total == 0:
            return self
        if total == 1:
            var = next(v for v, order in pairs if order == 1)
            return 1 if var == self else 0
        return 0

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    @property
    def is_zero(self):
        """Property to safely expose the zero check."""
        return self._is_zero

    @property
    def _nf(self):
        rf = getattr(self, "_rf_cache", None)
        if rf is None:
            rf = _nf_hooks["rf_gen"](self)
            self._rf_cache = rf
        return rf

    def _nf_operand(self, other):
        if isinstance(other, (int, Fraction)):
            return not isinstance(other, bool)
        if isinstance(other, zero_form_atom):
            return True
        if isinstance(other, zero_form_class):
            return not _nf_hooks["has_pow_base"](other)
        return _nf_hooks["is_leaf"](other)

    @property
    def primary_coframe(self):
        if self.coframe is not _empty_coframe():
            return self.coframe
        family = _family_entry(self.label)
        return None if family is None else family.get("primary_coframe")

    @property
    def coframe_independants(self):
        family = _family_entry(self.label)
        if family is None:
            return self._coframe_independants
        return family.get("independent_of", {}).get(self.label, {})

    def _declared(self):
        family = _family_entry(self.label)
        if family is None:
            return frozenset()
        return family.get("declared", {}).get(self.label, frozenset())

    def _declared_real(self):
        return "real" in self._markers or "real" in self._declared()

    @property
    def is_constant(self):
        return "constant" in self._markers or "constant" in self._declared()

    @property
    def is_parameter(self):
        return self.is_constant

    @property
    def differential_order(self):
        if not hasattr(self, "_differential_order"):
            self._differential_order = sum(
                [len(elem[1:]) for elem in self.coframe_derivatives]
            )
        return self._differential_order

    @property
    def role(self):
        for marker in self._markers:
            if marker in ROLE_MARKERS:
                return marker
        return role_of_label(self.label)

    @property
    def family(self):
        return family_of_label(self.label, zero_form_atom)

    def _has_indices(self):
        return any(len(elem) > 1 for elem in self.coframe_derivatives)

    def _real_valued(self):
        if not (self.role in ("real", "imaginary") or self._declared_real()):
            return False
        if not self._has_indices():
            return True
        fixed = self._self_conj
        if fixed is None:
            fixed = self._self_conj = self._eval_conjugate() == self
        return fixed

    @property
    def assumptions(self):
        out = {m for m in self._markers if m in ASSUMPTION_NAMES and m != "real"}
        record = _assumption_record(self)
        if record:
            out |= record.get(self, frozenset())
        if self._real_valued():
            out.add("real")
        return frozenset(out)

    @property
    def source_zero_form(self):
        if not self._has_indices():
            return self
        return zero_form_atom(
            self.label,
            coframe=self.coframe,
            _markers=self._markers,
            coframe_independants=self._coframe_independants,
        )

    def assume(self, *assumptions, **flags):
        independent_of = flags.pop("independent_of", None)
        differential = flags.pop("differential", None)
        declared = {}
        for item in assumptions:
            if isinstance(item, dict):
                declared.update(item)
            elif isinstance(item, str):
                declared[item] = True
            else:
                declared.update(dict.fromkeys(item, True))
        declared.update(flags)
        for name, value in declared.items():
            if name not in ASSUMPTION_NAMES or not isinstance(value, bool):
                raise ValueError(
                    f"unknown assumption {name!r}; the accepted names are "
                    f"{', '.join(sorted(ASSUMPTION_NAMES))}, each mapped to True or False"
                )
            if name in _LABEL_ASSUMPTIONS and value is False:
                raise ValueError(f"`{name}` cannot be withdrawn once declared")
        label_level = {name for name in _LABEL_ASSUMPTIONS if declared.pop(name, False)}
        system_label, family = _family_of(self.label)
        if family is None:
            raise ValueError(
                f"`{self.label}` is not registered in the variable management framework; "
                "create it with `createZeroForm` before declaring assumptions"
            )
        if (label_level or independent_of is not None or differential is not None) and self._has_indices():
            names = sorted(label_level) + [
                name
                for name, value in (("independent_of", independent_of), ("differential", differential))
                if value is not None
            ]
            raise ValueError(
                f"declare {', '.join(f'`{n}`' for n in names)} on the zero form `{self.label}` itself, not on a derivative"
            )
        writes = self._check_records(differential, label_level, declared, independent_of)
        if writes:
            owner = differential.primary_coframe
            owner._write_potentials(writes)
            if self.primary_coframe is None:
                family["primary_coframe"] = owner
        if declared:
            record = family.setdefault("assumptions", {})
            current = set(record.get(self, ()))
            for name, value in declared.items():
                if value:
                    current.add(name)
                else:
                    current.discard(name)
            if current:
                record[self] = frozenset(current)
            else:
                record.pop(self, None)
        if label_level or independent_of is not None:
            pair = family["family_relatives"].get(self.label, (self, self))
            labels = {atom.label for atom in pair}
            if "constant" in label_level:
                marks = family.setdefault("declared", {})
                for name in labels:
                    marks[name] = marks.get(name, frozenset()) | {"constant"}
            if independent_of is not None:
                rules = family.setdefault("independent_of", {})
                new = _independence_rules(independent_of, rules.get(self.label))
                for name in labels:
                    rules[name] = _mapped_independence(new, name != self.label)
            if "real" in label_level:
                _declare_real(system_label, family, pair)
        changed = _nf_hooks.get("assumptions_changed")
        if changed is not None:
            changed(self)
        return self

    def _check_records(self, differential, label_level, declared, independent_of):
        real = "real" in label_level or any(declared.get(name) for name in REAL_IMPLYING_ASSUMPTIONS)
        constant = "constant" in label_level
        rules = {} if independent_of is None else _independence_rules(independent_of, None)
        home = self.primary_coframe
        if differential is None:
            row = None if home is None else home._potential_index.get(self)
            for index in row or ():
                home._check_differential(
                    home.forms[index],
                    self.label,
                    real=real,
                    constant=constant,
                    independent=_independent_of_form(rules, home.forms[index]),
                )
            return None
        owner = getattr(differential, "primary_coframe", None)
        if owner is None:
            raise ValueError("`differential` must be a basis 1-form of a registered coframe")
        if home is not None and home is not owner:
            raise ValueError(f"{differential} belongs to another coframe than {self.label}")
        owner._check_differential(
            differential,
            self.label,
            real=real or self._declared_real() or self.role == "real",
            constant=constant or self.is_constant,
            independent=_independent_of_form(rules, differential)
            or _independent_of_form(self.coframe_independants, differential),
        )
        return owner._potential_writes(differential, self)

    @property
    def is_real(self):
        a = self.assumptions
        if "real" in a:
            return True
        if self.role in ("holomorphic", "antiholomorphic"):
            return False
        return None

    @property
    def is_positive(self):
        a = self.assumptions
        if "positive" in a:
            return True
        if a & {"negative", "nonpositive"}:
            return False
        return None

    @property
    def is_negative(self):
        a = self.assumptions
        if "negative" in a:
            return True
        if a & {"positive", "nonnegative"}:
            return False
        return None

    @property
    def is_nonzero(self):
        a = self.assumptions
        if a & {"positive", "negative", "nonzero"}:
            return True
        return None

    def _with_label(self, label, markers=None):
        return zero_form_atom(
            label,
            coframe_derivatives=self.coframe_derivatives,
            coframe=self.coframe,
            _markers=self._markers if markers is None else markers,
            coframe_independants=self._coframe_independants,
        )

    def partner(self, role):
        if role not in FAMILY_ROLES:
            raise ValueError(f"unknown complex role {role!r}")
        fam = self.family
        if fam is None:
            return None
        member = fam.partner(role)
        if member.label == self.label:
            return self
        return self._with_label(member.label, member._markers)

    def conjugate_atom(self):
        role = self.role
        target = CONJUGATE_ROLE[role]
        if role in FAMILY_ROLES and target != role:
            partner = self.partner(target)
            if partner is not None:
                return partner
        return None

    def to_real(self):
        role = self.role
        if role not in ("holomorphic", "antiholomorphic"):
            return self
        x = self.partner("real")
        y = self.partner("imaginary")
        if x is None or y is None:
            return self
        sign = 1 if role == "holomorphic" else -1
        return zero_form_class(("add", x, ("mul", sign, _imag_unit(), y)))

    def to_hol(self):
        role = self.role
        if role not in ("real", "imaginary"):
            return self
        z = self.partner("holomorphic")
        zbar = self.partner("antiholomorphic")
        if z is None or zbar is None:
            return self
        if role == "real":
            return zero_form_class(("mul", _half(), ("add", z, zbar)))
        return zero_form_class(("mul", -1, _imag_unit(), _half(), ("sub", z, zbar)))

    def __dgcv_re__(self):
        role = self.role
        if role in ("holomorphic", "antiholomorphic"):
            x = self.partner("real")
            return x if x is not None else _generic_re(self)
        if self._real_valued():
            return self
        return _generic_re(self)

    def __dgcv_im__(self):
        role = self.role
        if role in ("holomorphic", "antiholomorphic"):
            y = self.partner("imaginary")
            if y is None:
                return _generic_im(self)
            return y if role == "holomorphic" else zero_form_class(("mul", -1, y))
        if self._real_valued():
            return 0
        return _generic_im(self)

    def _sage_(self):
        raise AttributeError

    def __eq__(self, other):
        """
        Check equality of two zeroFormAtom instances.
        """
        if not isinstance(other, zero_form_atom):
            return NotImplemented
        return (
            self.label == other.label
            and self.coframe_derivatives == other.coframe_derivatives
        )

    def __hash__(self):
        """
        Hash the zeroFormAtom instance based on its label and coframe_derivatives.
        """
        return self._hash

    def __reduce__(self):
        state = {key: value for key, value in self.__dict__.items() if key != "_hash"}
        return (
            zero_form_atom._from_pickle,
            (self.label, _derivative_key(self.coframe_derivatives)),
            state,
        )

    @classmethod
    def _from_pickle(cls, label, key):
        obj = object.__new__(cls)
        obj._hash = hash((label, key))
        return obj

    def __lt__(self, other):
        if not isinstance(other, zero_form_atom):
            return NotImplemented

        self_key = (
            self.label,
            len(self.coframe_derivatives),
            tuple(elem[1:] for elem in self.coframe_derivatives),
        )
        other_key = (
            other.label,
            len(other.coframe_derivatives),
            tuple(elem[1:] for elem in other.coframe_derivatives),
        )
        return self_key < other_key

    def sort_key(self, order=None):
        return (
            3,
            self.label,
            len(self.coframe_derivatives),
            tuple(elem[1:] for elem in self.coframe_derivatives),
        )

    def _eval_conjugate(self):
        """
        Define how `sympy.conjugate()` should behave for zeroFormAtom instances.
        """
        pref = conjugation_prefix()
        prefL = len(pref)
        conjugated_markers = self._markers
        role = self.role
        partner = self.conjugate_atom() if role in FAMILY_ROLES else None
        if partner is not None:
            conjugated_label = partner.label
            conjugated_markers = partner._markers
        elif role in ("real", "imaginary") or self._declared_real():
            conjugated_label = self.label
        elif self.label.startswith(pref):
            conjugated_label = self.label[prefL:]  # Remove conjugate prefix
        else:
            conjugated_label = f"{pref}{self.label}"  # Add conjugate prefix

        newCD = []
        for elem in self.coframe_derivatives:
            k = elem[0]
            v = elem[1:]
            newCD += [tuple([k] + [k.conj_rules[index] for index in v])]

        # Return a new zeroFormAtom with the conjugated label
        return zero_form_atom(
            conjugated_label,
            coframe_derivatives=newCD,
            coframe=self.coframe,
            _markers=conjugated_markers,
            coframe_independants=self._coframe_independants,
        )

    def _eval_simplify(self, **kws):
        constraints = kws.pop("constraints", None)
        with_conjugates = kws.pop("with_conjugates", True)
        if constraints is not None:
            return _simplify_under_constraints(self, constraints, with_conjugates, **kws)
        return self

    def __mul__(self, other):
        """
        Multiplication of zeroFormAtom:
        - With another zeroFormAtom --> Becomes a structured `abstract_ZF` multiplication.
        - With a scalar (int/float/sympy.Expr) --> Wraps in `abstract_ZF`.
        """
        if self.is_one:
            return other
        if self.is_zero:
            return zero_form_class(0)
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("mul", self, other), lambda: self._nf * _nf_hooks["nf_of"](other)
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("mul", self, other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        if _builtin():
            return _nf_hooks["with_nf"](("mul", -1, self), lambda: -self._nf)
        return zero_form_class(("mul", -1, self))

    def __truediv__(self, other):
        """
        Division with zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("div", self, other), lambda: self._nf / _nf_hooks["nf_of"](other)
            )
        if isinstance(other, zero_form_atom) or isinstance(other, expr_numeric_types()):
            return zero_form_class(("div", self, other))
        return NotImplemented

    def __rtruediv__(self, other):
        """
        Division with zeroFormAtom instances
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("div", other, self), lambda: _nf_hooks["nf_of"](other) / self._nf
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("div", other, self))

        return NotImplemented

    def __pow__(self, exp):
        """
        Exponentiation of zeroFormAtom:
        - Returns a structured `abstract_ZF` exponentiation.
        """
        if (
            _builtin()
            and isinstance(exp, numbers.Integral)
            and not isinstance(exp, bool)
        ):
            return _nf_hooks["with_nf"](
                ("pow", self, exp), lambda: self._nf ** int(exp)
            )
        if isinstance(exp, (zero_form_atom, zero_form_class)) or isinstance(
            exp, expr_numeric_types()
        ):
            return zero_form_class(("pow", self, exp))
        return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("pow", other, self))

        return NotImplemented

    def __add__(self, other):
        """
        Addition with zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("add", self, other), lambda: self._nf + _nf_hooks["nf_of"](other)
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("add", self, other))
        return NotImplemented

    def __radd__(self, other):
        """
        Addition with zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("add", other, self), lambda: _nf_hooks["nf_of"](other) + self._nf
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("add", self, other))
        return NotImplemented

    def __sub__(self, other):
        """
        Subtraction of zeroFormAtom:
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("sub", self, other), lambda: self._nf - _nf_hooks["nf_of"](other)
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("sub", self, other))
        return NotImplemented

    def __rsub__(self, other):
        """
        Subtraction of zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("sub", other, self), lambda: _nf_hooks["nf_of"](other) - self._nf
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("sub", other, self))
        return NotImplemented

    def _canonicalize_step(self):
        for count, elem in enumerate(self.coframe_derivatives):
            zf = self
            partialsCount = len(elem)
            if partialsCount > 2:
                for idx1 in range(1, partialsCount - 1):
                    idx2 = idx1 + 1
                    if elem[idx2] < elem[idx1]:
                        return _swap_CFD_order(
                            zf, count, idx1
                        ), False  # False for stabilized status
        return self, True  # True for stabilized status

    def _eval_canonicalize(self, depth=1000):
        if _in_normal_order(self):
            return self
        zf = self
        stabilized = False
        count = 0
        while stabilized is False and count < depth:
            if hasattr(zf, "_canonicalize_step"):
                zf, stabilized = zf._canonicalize_step()
            count += 1
        return zf

    @property
    def free_symbols(self):
        return {self}

    def is_primitive(self, other, returnCD=False):
        "compute if other element is a coframe derivative (of some order) of self"
        if isinstance(other, zero_form_class) and isinstance(other.base, zero_form_atom):
            other = other.base
        trip = False
        trailingCD = tuple()
        if isinstance(other, zero_form_atom) and self.label == other.label:
            own = self.coframe_derivatives
            theirs = other.coframe_derivatives
            n = len(own)
            if n == 0:
                trip, trailingCD = True, theirs
            elif n <= len(theirs) and own[:-1] == theirs[: n - 1]:
                tail, their_tail = own[-1], theirs[n - 1]
                if tail[0] == their_tail[0]:
                    rest = _derivative_remainder(tail[0], tail[1:], their_tail[1:])
                    if rest is not None:
                        trip = True
                        trailingCD = theirs[n:]
                        if rest:
                            trailingCD = ((tail[0],) + tuple(rest),) + trailingCD
        if returnCD:
            return trip, trailingCD
        return trip

    def is_diff_corollary(self, other, returnCD=False):
        if isinstance(other, zero_form_class) and isinstance(other.base, zero_form_atom):
            return other.base.is_primitive(self, returnCD=returnCD)
        if isinstance(other, zero_form_atom):
            return other.is_primitive(self, returnCD=returnCD)
        return False

    def as_coeff_Mul(self, **kwds):
        return 1, self

    def as_ordered_factors(self):
        return (self,)

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def subs(self, data, with_diff_corollaries=False, with_conjugates=None):
        """
        Symbolic substitution in zeroFormAtom.
        """
        if isinstance(data, (list, tuple)) and all(
            isinstance(j, tuple) and len(j) == 2 for j in data
        ):
            l1 = len(data)
            data = dict(data)
            if len(data) < l1:
                dgcv_warning(
                    "Provided substitution rules had repeat keys, and only one was used."
                )

        if isinstance(data, dict):
            close = with_diff_corollaries if with_conjugates is None else with_conjugates
            if close:
                data = _conjugate_closure(data, "subs")
            if with_diff_corollaries:
                for key in data.keys():
                    truthVal, coefDer = self.is_diff_corollary(key, returnCD=True)
                    if truthVal:
                        new_value = data[key]
                        if isinstance(
                            new_value, (zero_form_atom, zero_form_class)
                        ) or isinstance(new_value, expr_numeric_types()):
                            for cd in coefDer:
                                new_value = coframe_derivative(new_value, *cd)
                            return new_value
                        else:
                            raise TypeError(
                                f"subs() cannot replace a `zeroFormAtom` with object type {type(new_value)}."
                            )
            if self in data:
                new_value = data[self]
                if isinstance(new_value, (zero_form_atom, zero_form_class)) or isinstance(
                    new_value, expr_numeric_types()
                ):
                    return new_value
                else:
                    raise TypeError(
                        f"subs() cannot replace a `zeroFormAtom` with object type {type(new_value)}."
                    )
            else:
                return self
        else:
            raise TypeError("`zeroFormAtom.subs()` received unsupported subs data.")

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __dgcv_apply__(self, func, **kwargs):
        bridge = _new_bridge()
        result = bridge.lift_expr(func(bridge.lower(self)[0], **kwargs))
        if isinstance(result, zero_form_class) and result.base is self:
            return self
        return result

    def simplify(self, **kw):
        return self._eval_simplify(**kw)

    def factor(self, **kw):
        return self

    def expand(self, **kw):
        return self

    def trigsimp(self, **kw):
        return self

    def cancel(self, **kw):
        return self

    def together(self, **kw):
        return self

    def apart(self, **kw):
        return self

    def ratsimp(self, **kw):
        return self

    def powsimp(self, **kw):
        return self

    def logcombine(self, **kw):
        return self

    def expand_log(self, **kw):
        return self

    def expand_trig(self, **kw):
        return self

    def expand_power_exp(self, **kw):
        return self

    def expand_power_base(self, **kw):
        return self

    def numer(self, **kw):
        return self

    def denom(self, **kw):
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Fallback string representation.
        """
        if self.is_one:
            return "1"
        if self.is_zero:
            return "0"

        if len(self.coframe_derivatives) > 0 and (
            len(self.coframe_derivatives[0]) > 1 or len(self.coframe_derivatives) > 1
        ):
            return_str = self.label
            count = 0
            partials_str = "_".join(
                map(str, [j + 1 for j in self.coframe_derivatives[0][1:]])
            )
            return_str = f"D_{partials_str}({return_str})"
            if len(self.coframe_derivatives[0]) > 1:
                count = 1
            for elem in self.coframe_derivatives[1:]:
                v = elem[1:]
                count_str = "" if count == 0 else f"_{count}"
                partials_str = "_".join(map(str, [j + 1 for j in v]))
                return_str = f"D_{partials_str}({return_str}){count_str}"
                count += 1
            return return_str

        return self.label

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        LaTeX representation for zeroFormAtom.
        """
        if self.is_one:
            return "1"
        if self.is_zero:
            return "0"

        base_label = self.label
        conjugated = False
        pref = conjugation_prefix()
        prefL = len(pref)
        if base_label.startswith(pref):
            base_label = base_label[prefL:]
            conjugated = True
        if (
            "engine_symbol" in self._markers
            and not self.coframe_derivatives
            and _builtin()
        ):
            return _process_var_label(self.label)

        index_start = None
        if "_low_" in base_label:
            index_start = base_label.index("_low_")
        elif "_hi_" in base_label:
            index_start = base_label.index("_hi_")

        if index_start is not None:
            first_part = base_label[:index_start]
            index_part = base_label[index_start:]
        else:
            first_part = base_label
            index_part = ""

        # Process the base part
        formatted_label = _process_var_label(first_part)

        if len(self.coframe_derivatives) > 0 and (
            len(self.coframe_derivatives[0]) > 1 or len(self.coframe_derivatives) > 1
        ):
            partials = True
        else:
            partials = False
        if "_" in formatted_label and index_part is not None and partials:
            formatted_label = f"\\left({formatted_label}\\right)"

        # Extract lower and upper indices
        lower_list, upper_list = [], []
        if "_low_" in index_part:
            lower_part = index_part.split("_low_")[1]
            if "_hi_" in lower_part:
                lower_part, upper_part = lower_part.split("_hi_")
                upper_list = upper_part.split("_")
            lower_list = lower_part.split("_")
        elif "_hi_" in index_part:
            upper_part = index_part.split("_hi_")[1]
            upper_list = upper_part.split("_")

        # Conjugate index formatter
        def cIdx(idx, cf):
            idx = int(idx)
            if isinstance(cf, abstract_coframe) and idx - 1 in cf.inverted_conj_rules:
                return f"\\overline{{{1 + cf.inverted_conj_rules[idx - 1]}}}"
            else:
                return f"{idx}"

        # Convert string indices to LaTeX-compatible integers
        lower_list = [cIdx(idx, self.coframe) for idx in lower_list if idx]
        upper_list = [cIdx(idx, self.coframe) for idx in upper_list if idx]

        # Extract partial derivative indices
        partials_strs = []
        if partials and len(self.coframe_derivatives[0]) > 1:
            new_indices = [j + 1 for j in self.coframe_derivatives[0][1:]]
            new_indices_str = ",".join([cIdx(j, self.coframe) for j in new_indices])
            partials_strs.extend([new_indices_str])
        elif self.coframe is not None:
            partials_strs = [""]
        for elem in self.coframe_derivatives[1:]:
            new_indices = [j + 1 for j in elem[1:]]
            new_indices_str = ",".join([cIdx(j, elem[0]) for j in new_indices])
            partials_strs.extend([new_indices_str])

        # Combine indices into the LaTeX string
        lower_str = ",".join(lower_list)
        # partials_strs = [",".join(map(cIdx, j)) for j in partials_indices]
        first_partials_str = partials_strs[0] if len(partials_strs) > 0 else ""
        upper_str = ",".join(upper_list)

        indices_str = ""
        indices_str_partials = ""  # only update if conjugated==True
        if upper_str:
            indices_str += f"^{{{upper_str}}}"
            if first_partials_str and conjugated:
                indices_str_partials += f"^{{\\vphantom{{{upper_str}}}}}"
        if lower_str or "verbose" in self._markers:
            if conjugated:
                indices_str += (
                    f"_{{{lower_str}\\vphantom{{;{first_partials_str}}}}}".replace(
                        r"\vphantom{;}}", "}"
                    )
                )
                if first_partials_str:
                    indices_str_partials += (
                        f"_{{\\vphantom{{{lower_str}}};{first_partials_str}}}"
                    )
            else:
                indices_str += f"_{{{lower_str};{first_partials_str}}}".replace(
                    ";}", "}"
                ).replace(r"\vphantom{;}}", "}")
        elif first_partials_str:
            if conjugated:
                if upper_str:
                    indices_str_partials += f"_{{;{first_partials_str}}}"
                else:
                    indices_str_partials += f"_{{{first_partials_str}}}"
            else:
                indices_str += f"_{{{first_partials_str}}}"
        pre_final_str = f"{formatted_label}{indices_str}"
        if indices_str_partials != "":  # implies conjugated
            pre_final_str = f"\\smash{{\\overline{{{pre_final_str}}}}}\\vphantom{{{formatted_label}}}{indices_str_partials}"
        elif conjugated:
            pre_final_str = f"\\overline{{{pre_final_str}}}"

        final_str = pre_final_str

        count = 2
        for new_partials_str in partials_strs[1:]:
            final_str = f"\\left.\\smash{{{final_str}}}\\vphantom{{{pre_final_str}}}\\right|_{{{new_partials_str}}}^{{\\boxed{{\\tiny{ordinal_latex(count)}}}}}"
            count += 1
        return final_str

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"


register_legacy_sympy_class(zero_form_atom)

register_zero_test(zero_form_atom, zero_form_atom.is_zero.fget)

zeroFormAtom = zero_form_atom

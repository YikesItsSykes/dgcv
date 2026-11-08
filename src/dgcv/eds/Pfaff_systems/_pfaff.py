"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._pfaff

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
import random

from ..._aux._backends._display import latex as _routed_latex
from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._backends._symbolic_router import subs as _routed_subs
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import (
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from ..._aux.printing.printing._eds import conjugation_prefix
from ...core.arrays import matrix_dgcv
from ...core.base import dgcv_class
from ...core.conversions.conversions import holToReal, realToSym, symToHol, symToReal
from ...core.solvers import solve_dgcv
from .. import _bridges
from .._calculus import _conjugate_closure
from .._coframes import abstract_coframe
from .._forms import (
    abstract_differential_form,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
)
from ..eds_operations import transform_coframe
from ..pde._systems import _coordinate_kind, _equation_list, is_pde_input
from ._chart import _is_complex_formula, _restrict_forms, _warn_on_divisors
from ._coordinates import (
    _abstract_closed,
    _appended_rows,
    _closed_equations,
    _conjugated_rows,
    _conjugated_scalar,
    _coordinate_reality,
    _forms_from_rows,
    _local_coordinates,
    _registered_coordinates,
)
from ._flag import (
    _chain_data,
    _combination,
    _exact_chain_work_limit,
    _exact_polar_level,
    _ExactChainTooLarge,
    _flag_data,
    _flag_weights,
    _general_solution,
    _polar,
    _polar_basis,
    _projects_onto,
    _sample_runs,
    _sampled_run,
)
from ._sampling import (
    _completed_point,
    _conjugate_consistent,
    _designated_coordinates,
    _equation_list_or_none,
    _loci_from_divisors,
    _locus_key,
    _polynomial_in,
    _pruned_loci,
    _random_values,
    _rank_drop_loci,
    _sampling_tail,
    _singular_factors,
    _unavailable_message,
)
from ._summary import (
    integral_flag_summary as _integral_flag_summary,
)
from ._summary import (
    system_summary as _system_summary,
)
from ._util import (
    _as_list,
    _dot,
    _form_is_zero,
    _free,
    _is_abstract,
    _is_vector,
    _rank,
    _subs,
    _symbols,
    _tidy,
    _translated,
    _unit,
    _zero_test,
)

_passkey = retrieve_passkey()


def _closed_labels(labels, given_count, sources, rows, subset, width):
    labels = [None if name is None else str(name) for name in labels]
    rest = width - len(subset)
    if len(labels) in (len(subset), len(subset) + rest):
        return labels
    pre = _independent_rows(rows[:given_count])
    if len(labels) != len(pre):
        raise ValueError(
            f"`labels` lists the system's forms ({len(subset)}, the conjugates added by the closure "
            f"included) or every form of the new coframe ({len(subset) + rest}); received {len(labels)}"
        )
    table = dict(zip(pre, labels))
    prefix = conjugation_prefix()
    for k, src in enumerate(sources):
        if table.get(src) is not None:
            table[given_count + k] = f"{prefix}{table[src]}"
    return [table.get(idx) for idx in subset]


class _SingularPointError(ValueError):
    pass


class _Chart:
    def __init__(
        self, coframe, Pfaffian_forms, complementary_forms, matrix=None, images=None
    ):
        self.coframe = coframe
        self.Pfaffian_forms = tuple(Pfaffian_forms)
        self.complementary_forms = tuple(complementary_forms)
        self.matrix = matrix
        self.images = tuple(coframe.forms if images is None else images)
        self.complementary_index = [coframe.forms.index(f) for f in self.complementary_forms]
        self.complementary_position = {f: k for k, f in enumerate(self.complementary_forms)}
        self.Pfaffian_index = [coframe.forms.index(f) for f in self.Pfaffian_forms]
        self._inverse = None

    @property
    def frame(self):
        return self.coframe.dual()

    @property
    def inverse(self):
        if self._inverse is None:
            self._inverse = matrix_dgcv(self.matrix).inverse()
        return self._inverse

    def element(self, components):
        return self.frame.element(dict(zip(self.complementary_index, components)))

    def structure_pairs(self, form, at=None):
        for (fi, fj, _), value in self.coframe.coefficients(of=form).items():
            if fi in self.complementary_position and fj in self.complementary_position:
                yield self.complementary_position[fi], self.complementary_position[fj], _subs(value, at)


class PfaffSystem(dgcv_class):
    _dgcv_class_check = _passkey
    _dgcv_category = "pfaff_system"

    def __init__(
        self,
        Pfaffian_forms,
        *,
        coframe=None,
        labels="omega",
        parameters=None,
        coordinates=None,
        horizontal_forms=None,
        filter_horizontals_for_indep_mod_Pfaff=True,
        dependent=None,
        restrict=None,
        _objects=None,
    ):
        self.defining_equations = ()
        self._pde = None
        self._pde_inverse = {}
        self._coordinate_tex = {}
        self._singularities = {}
        attach = None
        if get_dgcv_category(Pfaffian_forms) == "distribution":
            Pfaffian_forms = list(Pfaffian_forms.df_basis)
        elif is_pde_input(Pfaffian_forms):
            if coordinates is not None or horizontal_forms is not None:
                raise TypeError(
                    "a Pfaff system built from differential equations fixes its own coordinates "
                    "and horizontal forms"
                )
            space = jetSpace.from_equations(
                Pfaffian_forms, dependent=dependent, parameters=parameters, coframe=coframe
            )
            Pfaffian_forms, coordinates, horizontal_forms, defining, divisors, _ = space._pfaff_inputs(
                None, restrict
            )
            coframe, parameters, _objects = None, space.parameter_symbols, space._objects
            attach = (defining, divisors, space)
        elif dependent is not None or restrict is not None:
            raise TypeError(
                "`dependent` and `restrict` belong to a Pfaff system built from differential equations"
            )
        if _objects is None:
            _objects = _registered_coordinates()
        given = [_unit_atom(form) for form in _as_list(Pfaffian_forms)]
        if not given:
            raise TypeError("a Pfaff system needs at least one 1-form")
        if not isinstance(labels, (str, list, tuple)):
            raise TypeError("`labels` is a stem, or a list of labels")
        self._source = _source_coframe(given, coframe)
        self._labels = labels
        self._objects = _objects
        self._parameters = _bridges._as_parameters(parameters)
        if self._source is None:
            self._columns = _coordinate_columns(given, self._parameters, coordinates)
            self._reality = _coordinate_reality(
                self._columns,
                getattr(self._objects, "conjugates", None),
                getattr(self._objects, "minted", ()),
            )
        else:
            self._columns = self._source.forms
            self._reality = _coordinate_reality(
                _source_coordinates(self._source, self._parameters)
            )
        rows = self._source_rows(given)
        extra, sources = [], []
        if self._closed:
            extra, sources = _appended_rows(rows, self._conjugated(rows))
            given += self._forms_of(extra)
            rows += extra
        self._given = tuple(given)
        self._rows = rows
        subset = _independent_rows(self._rows)
        if extra and not isinstance(labels, str):
            labels = _closed_labels(
                labels, len(rows) - len(extra), sources, rows, subset, len(self._columns)
            )
            self._labels = labels
        self._choice = (subset, _pivot_columns([self._rows[idx] for idx in subset]))
        self._default = self._local_coframe(*self._choice, labels)
        if self._default.coframe is self._source and not isinstance(labels, str):
            raise ValueError(
                "the forms are members of their coframe, so there are no new forms to label"
            )
        self._charts = {}
        self._flags = {}
        self._matrices = {}
        self._summary_cache = {}
        self.coordinates = (
            tuple(self._columns)
            if self._source is None
            else _source_coordinates(self._source, self._parameters)
        )
        self._filter_horizontals = bool(filter_horizontals_for_indep_mod_Pfaff)
        self.horizontal_forms = ()
        self._dropped_horizontals = ()
        self._set_horizontal_forms(
            self._checked_forms(horizontal_forms, strict=not self._filter_horizontals)
        )
        if attach is not None:
            self._attach_equations(*attach)

    def _attach_equations(self, equations, divisors, space):
        self.defining_equations = tuple(equations)
        self._coordinate_tex = dict(getattr(space, "coordinate_tex", None) or {})
        if space.pde is not None:
            self._pde = space.pde
            self._pde_inverse = {
                coordinate: atom for atom, coordinate in space.jet_coordinate_rules.items()
            }
        self._record_singularities("chart", _loci_from_divisors(divisors, ()))
        if equations:
            self._record_singularities(
                "distribution_rank",
                _rank_drop_loci(self._rows, len(space.contact_forms), len(equations)),
            )

    def _record_singularities(self, source, loci):
        if not loci:
            return
        known = self._singularities.setdefault(source, [])
        seen = {_locus_key(locus) for locus in known}
        fresh = [locus for locus in loci if _locus_key(locus) not in seen]
        known.extend(fresh)
        if self._pde is not None and fresh:
            translated = [
                tuple(_translated(condition, self._pde_inverse) for condition in locus)
                for locus in fresh
            ]
            cache = self._pde._cache.setdefault("singularities", {})
            held = cache.setdefault(source, [])
            held_keys = {_locus_key(locus) for locus in held}
            held.extend(locus for locus in translated if _locus_key(locus) not in held_keys)

    def singularities(self):
        return {
            source: [list(locus) for locus in _pruned_loci(loci)]
            for source, loci in self._singularities.items()
            if loci
        }

    def random_point(self, seed=None, bound=3, tries=40):
        point, verdict, detail = self._sample_point(seed, bound, tries)
        if verdict == "found":
            return point
        if verdict == "unavailable":
            raise NotImplementedError(detail)
        raise ValueError(
            f"no point of the defining equations was found in {tries} random samples"
            + _sampling_tail(verdict, detail)
        )

    def _sample_point(self, seed=None, bound=3, tries=40):
        if not self.coordinates:
            raise TypeError("a point needs coordinates, and this system has none")
        rng = random.Random(seed)
        equations = list(self.defining_equations)
        reality = self._reality
        pairs = reality.pairs if reality.closed else {}
        designated = _designated_coordinates(equations, self.coordinates, reality)
        parameters = set(self._parameters)
        others = [x for x in self.coordinates if x not in designated]
        if not any(x in _free(F) for F in equations for x in others):
            tries = min(tries, 1)
        roots_seen, non_real = False, []
        is_zero = _zero_test()
        for _ in range(tries):
            values = _random_values(rng, bound, self.coordinates, pairs)
            if not equations:
                return values, "found", None
            fixed = {x: value for x, value in values.items() if x not in designated}
            reduced = _equation_list_or_none([_subs(F, fixed) for F in equations], is_zero)
            if reduced is None:
                continue
            if not reduced:
                return fixed | {y: values[y] for y in designated}, "found", None
            try:
                solutions = solve_dgcv(reduced, list(designated))
            except NotImplementedError as exc:
                return None, "unavailable", _unavailable_message(exc)
            except ValueError:
                solutions = []
            accepted = []
            for solution in solutions:
                roots_seen = True
                point = _completed_point(rng, bound, fixed, solution, designated, parameters, pairs)
                if point is None:
                    continue
                bad = [y for y in designated if reality.is_real(y) and _is_complex_formula(point[y], is_zero)]
                if bad:
                    non_real.extend(y for y in bad if y not in non_real)
                    continue
                if pairs and not _conjugate_consistent(point, reality):
                    continue
                accepted.append(point)
            if accepted:
                return rng.choice(accepted), "found", None
        if not roots_seen:
            if all(_polynomial_in(F, designated) for F in equations):
                return None, "inconsistent", None
            return None, "inconclusive", None
        if non_real:
            return None, "non_real", [y for y in self.coordinates if y in non_real]
        return None, "inconclusive", None

    def _local_coframe(self, subset, pivots, labels):
        source, given, columns = self._source, self._given, self._columns
        rest = [idx for idx in range(len(columns)) if idx not in pivots]
        if source is None:
            kept = [None] * len(subset)
            names = [_complementary_name(columns[idx], self._reality) for idx in rest]
            images = [self._objects.form(columns[idx]) for idx in rest]
            taken = names
        else:
            kept = [_member_label(given[idx], source) for idx in subset]
            if None not in kept:
                forms = tuple(
                    source.forms[source.forms.index(given[idx])] for idx in subset
                )
                return _Chart(
                    source, forms, tuple(f for f in source.forms if f not in forms)
                )
            names = [source.forms[idx].label for idx in rest]
            images = [source.forms[idx] for idx in rest]
            taken = [form.label for form in source.forms]
        names = _form_labels(labels, kept, names, taken)
        images = [given[idx] for idx in subset] + images
        if source is None:
            coframe = _coordinate_coframe(
                names, images, columns, self._parameters, self._objects
            )
        else:
            coframe = _rebased_coframe(source, names, images)
        count = len(subset)
        matrix = [self._rows[idx] for idx in subset] + [
            _unit(len(columns), idx) for idx in rest
        ]
        return _Chart(
            coframe, coframe.forms[:count], coframe.forms[count:], matrix, images
        )

    def _rows_at(self, at):
        return [[_subs(entry, at) for entry in row] for row in self._rows]

    def _chosen_at(self, at):
        rows = self._rows_at(at)
        subset, pivots = self._choice
        minor = [[rows[i][j] for j in pivots] for i in subset]
        if not minor or _rank(matrix_dgcv(minor)) == len(subset):
            return self._default
        subset = _independent_rows(rows)
        pivots = _pivot_columns([rows[idx] for idx in subset])
        return self._local_coframe(
            subset, pivots, self._labels if isinstance(self._labels, str) else "omega"
        )

    def _local(self, at=None):
        if at is None:
            return self._default
        key = frozenset(at.items())
        if key not in self._charts:
            self._charts[key] = self._chosen_at(at)
        return self._charts[key]

    def _chart(self, at=None):
        chart = self._local(at)
        if len(chart.Pfaffian_forms) < self.Pfaff_distribution_codim():
            raise _SingularPointError(
                f"the point is singular for this system: the forms span {len(chart.Pfaffian_forms)} dimensions "
                f"there, below their rank {self.Pfaff_distribution_codim()} over functions, and these numbers "
                "are defined at regular points"
            )
        return chart

    def _coordinate_rows(self, forms):
        return _rows_on_fields(forms, [self._objects.field(x) for x in self._columns])

    @property
    def _closed(self):
        if self._source is None:
            return self._reality.closed
        return _abstract_closed(self._source)

    def _source_rows(self, forms):
        if self._source is None:
            return self._coordinate_rows(forms)
        return _abstract_rows(forms, self._source)

    def _conjugated(self, rows):
        reality = self._reality
        if self._source is None:
            columns = self._columns
            index = {x: idx for idx, x in enumerate(columns)}
            return _conjugated_rows(
                rows, reality, lambda idx: index[reality.partner(columns[idx])]
            )
        return _conjugated_rows(rows, reality, self._source.conj_rules.__getitem__)

    def _forms_of(self, rows):
        if self._source is None:
            return _forms_from_rows(rows, self._columns, self._objects)
        return [_unit_atom(form) for form in _on_forms(rows, self._source.forms)]

    def _form_rows(self, forms, chart):
        if _on_coframe(forms, chart.coframe):
            return _abstract_rows(forms, chart.coframe), True
        if self._source is None:
            _check_coordinate_forms(forms)
            return self._coordinate_rows(forms), False
        return _abstract_rows(forms, self._source), False

    def _checked_forms(self, forms, strict=True):
        forms = tuple(_unit_atom(form) for form in _as_list(forms))
        if (
            strict
            and forms
            and _rank(matrix_dgcv(self._form_rows(forms, self._default)[0]))
            < len(forms)
        ):
            raise ValueError("the forms of the projection must be independent")
        return forms

    def _closed_horizontals(self, forms):
        forms = list(forms)
        chart = self._default
        if not forms or not chart.complementary_index or not self._closed:
            return forms
        candidates = self._forms_of(self._conjugated(self._source_rows(forms)))
        rank = _rank(matrix_dgcv(self._on_complementary(forms, chart)))
        for candidate in candidates:
            if _form_is_zero(candidate):
                continue
            raised = _rank(matrix_dgcv(self._on_complementary(forms + [candidate], chart)))
            if raised > rank:
                forms.append(candidate)
                rank = raised
        return forms

    def _set_horizontal_forms(self, forms):
        forms = tuple(
            self._closed_horizontals([_unit_atom(form) for form in _as_list(forms)])
        )
        dropped = ()
        if forms and self._filter_horizontals:
            rows = self._on_complementary(forms, self._default)
            if not self._default.complementary_index or _rank(matrix_dgcv(rows)) < len(forms):
                keep = self._independent_mod_Pfaff(rows)
                dropped = tuple(f for idx, f in enumerate(forms) if idx not in keep)
                forms = tuple(forms[idx] for idx in keep)
                dgcv_warning(
                    f"{len(dropped)} of the {len(dropped) + len(forms)} horizontal forms dropped as dependent modulo "
                    "the forms of the Pfaff system; Use `filter_horizontals_for_indep_mod_Pfaff=False` to keep them all"
                )
        self.horizontal_forms = forms
        self._dropped_horizontals = dropped

    @staticmethod
    def _independent_mod_Pfaff(rows):
        if not rows or not rows[0]:
            return []
        columns = [[row[idx] for row in rows] for idx in range(len(rows[0]))]
        return sorted(
            matrix_dgcv(columns).pivot_columns(
                simplify_steps=True, zero_test=_zero_test()
            )
        )

    def _on_complementary(self, forms, chart):
        rows, on_chart = self._form_rows(forms, chart)
        if on_chart:
            return [[row[idx] for idx in chart.complementary_index] for row in rows]
        if self._source is None:
            frame = chart.coframe.dual_frame_in_coordinates
            return _rows_on_fields(forms, [frame[idx] for idx in chart.complementary_index])
        inverse = chart.inverse
        return [
            _tidy([_dot(row, inverse.col(idx)) for idx in chart.complementary_index])
            for row in rows
        ]

    @property
    def coframe(self):
        return self._default.coframe

    @property
    def Pfaffian_forms(self):
        return self._default.Pfaffian_forms

    @property
    def complementary_forms(self):
        return self._default.complementary_forms

    @property
    def dimension(self):
        return self._default.coframe.dimension

    @property
    def frame(self):
        return self.coframe.dual()

    def _point(self, at):
        if at is None:
            return None
        if not self.coordinates:
            raise TypeError(
                "a point needs coordinates, and this system has none; "
                "subs(system, rules) substitutes into its definition instead"
            )
        at = dict(at)
        ignored = [key for key in at if key not in self.coordinates]
        if ignored:
            dgcv_warning(
                f"ignored {len(ignored)} key(s) of `at` that are not coordinates of this system: "
                + ", ".join(str(key) for key in ignored)
                + "; subs(system, rules) substitutes such symbols",
                wc_label="dgcvImportantWarning",
            )
            for key in ignored:
                del at[key]
        if self._reality.closed:
            is_zero = _zero_test()
            for key in list(at):
                if key not in self._reality.pairs:
                    continue
                partner = self._reality.pairs[key]
                conj = _conjugated_scalar(at[key], self._reality)
                if partner not in at:
                    at[partner] = conj
                elif not is_zero(at[partner] - conj):
                    raise ValueError(
                        f"`at` gives {partner} = {at[partner]} where the conjugate of {key} = {at[key]} is {conj}"
                    )
        return at

    def Pfaff_distribution_codim(self, at=None):
        at = self._point(at)
        if at is None:
            return len(self.Pfaffian_forms)
        return _rank(matrix_dgcv(self._rows_at(at)))

    def Pfaff_distribution_rank(self, at=None):
        return self.dimension - self.Pfaff_distribution_codim(at)

    def is_regular_point(self, at):
        return self.Pfaff_distribution_codim(at) == self.Pfaff_distribution_codim()

    @property
    def free_symbols(self):
        out = set()
        for form in (*self._given, *self.horizontal_forms):
            out |= _free(form)
        if self._source is not None:
            out |= set(self._source.free_symbols)
        return out

    def subs(self, rules, with_diff_corollaries=False, with_conjugates=None):
        rules = _rule_dict(rules)
        if not rules:
            return self
        fixed = [key for key in rules if key in self.coordinates]
        if fixed:
            raise ValueError(
                "a rule on a coordinate ("
                + ", ".join(str(key) for key in fixed)
                + ") gives no new system; evaluate at a point with `at`"
            )
        close = (
            with_conjugates
            if with_conjugates is not None
            else self._reality.closed
            or (self._source is not None and _abstract_closed(self._source))
        )
        if close:
            rules = _conjugate_closure(rules, escape="subs")
        kwargs = {
            "with_diff_corollaries": with_diff_corollaries,
            "with_conjugates": with_conjugates,
        }
        parameters = _substituted_parameters(self._parameters, rules, self.coordinates)
        horizontal = self._in_source(self.horizontal_forms)
        if self._source is None:
            coframe = None
            given = _subs_forms(self._given, rules)
            horizontal = _subs_forms(horizontal, rules)
        else:
            coframe = _subs_source(self._source, rules, self.coordinates, kwargs)
            given = _subs_forms(self._given, rules, kwargs)
            horizontal = _subs_forms(horizontal, rules, kwargs)
            if coframe is not self._source:
                source = coframe
                coframe = _rebased_coframe(
                    source, [form.label for form in source.forms], source.forms
                )
                given = _on_forms(_abstract_rows(given, source), coframe.forms)
                horizontal = _on_forms(_abstract_rows(horizontal, source), coframe.forms)
        system = PfaffSystem(
            given,
            coframe=coframe,
            labels=self._fitting_labels(given, coframe),
            parameters=parameters,
            coordinates=self.coordinates if coframe is None else None,
            filter_horizontals_for_indep_mod_Pfaff=self._filter_horizontals,
            _objects=self._objects,
        )
        system._set_horizontal_forms(horizontal)
        return system

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def __dgcv_converter__(
        self,
        conv,
        skipVar=None,
        variables_scope=None,
        convert_everything=False,
        conversion_dict=None,
    ):
        entry = _CONVERTERS.get(conv)
        if entry is None:
            return None
        if self._source is not None or isinstance(self._objects, _local_coordinates):
            return self
        convert, basis = entry
        if basis is None:
            return self
        kwargs = {
            "skipVar": skipVar,
            "convert_everything": convert_everything,
            "variables_scope": variables_scope,
        }
        coordinates = _converted_symbols(self._columns, basis, kwargs)
        parameters = frozenset(_converted_symbols(self._parameters, basis, kwargs))
        if [str(x) for x in coordinates] == [str(x) for x in self._columns]:
            return self
        system = PfaffSystem(
            [convert(form, **kwargs) for form in self._given],
            labels=self._labels if isinstance(self._labels, str) else "omega",
            parameters=parameters,
            coordinates=coordinates,
            filter_horizontals_for_indep_mod_Pfaff=self._filter_horizontals,
        )
        system._set_horizontal_forms(
            [convert(form, **kwargs) for form in self.horizontal_forms]
        )
        return system

    def _fitting_labels(self, given, coframe):
        labels = self._labels
        if isinstance(labels, str):
            return labels
        given = [_unit_atom(form) for form in given]
        rows = (
            self._coordinate_rows(given) if coframe is None else _abstract_rows(given, coframe)
        )
        subset = _independent_rows(rows)
        if coframe is not None and all(
            _member_label(given[idx], coframe) is not None for idx in subset
        ):
            return "omega"
        if len(labels) in (len(subset), self.dimension):
            return list(labels)
        return "omega"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        head = ", ".join(str(f) for f in self.Pfaffian_forms) or "0"
        return f"PfaffSystem({head} = 0)"

    def _latex(self, printer=None, raw=True, **kwargs):
        body = (
            ", ".join(_routed_latex(f) + " = 0" for f in self.Pfaffian_forms) or "0 = 0"
        )
        return r"\left\{" + body + r"\right\}"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def _in_source(self, forms):
        rows, on_chart = self._form_rows(forms, self._default)
        if not on_chart or self._default.coframe is self._source:
            return list(forms)
        return _on_forms(rows, self._default.images)

    def restrict(self, equations):
        if self._source is not None:
            raise TypeError("restriction needs coordinates, and this system has none")
        _, divisors, chart, (given, horizontal) = _restrict_forms(
            _closed_equations(_equation_list(equations, _zero_test()), self._reality),
            self._columns,
            self._objects,
            self._reality,
            self._given,
            self._in_source(self.horizontal_forms),
        )
        _warn_on_divisors(divisors)
        system = PfaffSystem(
            given,
            labels=self._labels if isinstance(self._labels, str) else "omega",
            parameters=self._parameters,
            coordinates=chart,
            filter_horizontals_for_indep_mod_Pfaff=self._filter_horizontals,
            _objects=self._objects,
        )
        system._set_horizontal_forms(horizontal)
        return system

    def is_completely_integrable(self):
        chart = self._default
        is_zero = _zero_test()
        return all(
            is_zero(value)
            for form in chart.Pfaffian_forms
            for _, _, value in chart.structure_pairs(form)
        )

    def _structure_matrices(self, chart, at):
        key = None if at is None else frozenset(at.items())
        if key not in self._matrices:
            f = len(chart.complementary_forms)
            is_zero = _zero_test()
            out = []
            for form in chart.Pfaffian_forms:
                matrix = matrix_dgcv.zeros(f, f)
                for i, j, value in chart.structure_pairs(form, at):
                    if is_zero(value):
                        continue
                    matrix[i, j] = value
                    matrix[j, i] = -value
                out.append(matrix)
            self._matrices[key] = out
        return self._matrices[key]

    def structure_coefficients_on_annihilator(self, at=None):
        at = self._point(at)
        return list(self._structure_matrices(self._chart(at), at))

    def _omega_check(self, values, chart, at):
        is_zero = _zero_test()
        for idx, form in zip(chart.Pfaffian_index, chart.Pfaffian_forms):
            value = _subs(values(idx), at)
            if not is_zero(value):
                raise ValueError(
                    f"the vector is not annihilated by {form}: {form}(X) = {value}"
                )

    def _to_components(self, data, chart, at):
        forms = chart.complementary_forms
        if _is_vector(data):
            if data.frame is not chart.frame:
                raise TypeError(
                    "the vector field belongs to another frame than the one at this point; "
                    "give a coordinate vector field or a vector of that frame"
                )
            values = data.coeff_dict
            self._omega_check(lambda idx: values.get(idx, 0), chart, at)
            return [_subs(values.get(idx, 0), at) for idx in chart.complementary_index]
        if isinstance(data, matrix_dgcv):
            if data.nrows > 1 and data.ncols > 1:
                raise ValueError(
                    "a vector is one row or one column of a matrix, not several"
                )
            data = data.col(0) if data.ncols == 1 else data.row(0)
        if isinstance(data, dict):
            components = [0] * len(forms)
            for form, value in data.items():
                if form not in chart.complementary_position:
                    raise ValueError(
                        f"{form} is not a complementary form of the system: {forms}"
                    )
                components[chart.complementary_position[form]] = _subs(value, at)
            return components
        if query_dgcv_categories(data, {"vector_field"}):
            images = chart.coframe.forms_in_coordinates
            if images is None:
                raise TypeError(
                    "a coordinate vector field needs a coframe with coordinate forms"
                )
            self._omega_check(
                lambda idx: _bridges._scalar(images[idx](data)), chart, at
            )
            return [
                _subs(_bridges._scalar(images[idx](data)), at) for idx in chart.complementary_index
            ]
        data = list(data)
        if len(data) != self.dimension:
            raise ValueError(
                f"a vector has {self.dimension} components in coframe order, received {len(data)}"
            )
        self._omega_check(lambda idx: data[idx], chart, at)
        return [_subs(data[idx], at) for idx in chart.complementary_index]

    def vector(self, data, at=None):
        at = self._point(at)
        chart = self._chart(at)
        components = self._to_components(data, chart, at)
        if _is_vector(data) and at is None:
            return data
        return chart.element(components)

    def _subspace(self, N, chart, at):
        if (
            isinstance(N, (dict, matrix_dgcv))
            or _is_vector(N)
            or query_dgcv_categories(N, {"vector_field"})
        ):
            N = [N]
        vectors = [self._to_components(v, chart, at) for v in N]
        basis = _independent_rows(vectors) if chart.complementary_index else []
        if not basis:
            raise ValueError("a subspace needs a nonzero vector")
        return [vectors[idx] for idx in basis]

    def polar_matrix(self, N, at=None):
        at = self._point(at)
        chart = self._chart(at)
        vectors = self._subspace(N, chart, at)
        return _polar(
            self._structure_matrices(chart, at), vectors, len(chart.complementary_index)
        )

    def polar_rank(self, N, at=None):
        return _rank(self.polar_matrix(N, at))

    def polar_system_solutions(self, N, at=None):
        at = self._point(at)
        chart = self._chart(at)
        return [
            chart.element(v)
            for v in _polar_basis(self.polar_matrix(N, at), len(chart.complementary_index))
        ]

    @staticmethod
    def _integral(vectors, matrices):
        f = len(vectors[0])
        is_zero = _zero_test()
        for matrix in matrices:
            for a in range(len(vectors)):
                for b in range(a + 1, len(vectors)):
                    value = _dot(
                        vectors[a], [_dot(matrix.row(i), vectors[b]) for i in range(f)]
                    )
                    if not is_zero(value):
                        return False
        return True

    def is_integral(self, N, at=None):
        at = self._point(at)
        chart = self._chart(at)
        vectors = self._subspace(N, chart, at)
        return self._integral(vectors, self._structure_matrices(chart, at))

    def _chain(self, chart, at):
        f = len(chart.complementary_index)
        flag, ranks, divisors = [], [], []
        if f == 0:
            return _chain_data(flag, ranks, divisors, ())
        matrices = self._structure_matrices(chart, at)
        generic = self._flags.get(None) if at is not None else None
        exact_levels = None if generic is None else generic.exact_levels
        vector = _combination([_unit(f, k) for k in range(f)], "t_1_")
        while True:
            if exact_levels is not None and len(flag) >= exact_levels:
                return self._sampled_chain(matrices, f, flag, ranks, divisors)
            polar = _polar(matrices, flag + [vector], f)
            try:
                rank, divs, rows, pivot_rows = _exact_polar_level(
                    polar, _exact_chain_work_limit
                )
            except _ExactChainTooLarge:
                return self._sampled_chain(matrices, f, flag, ranks, divisors)
            flag.append(vector)
            p = len(flag)
            ranks.append(rank)
            divisors.extend(divs)
            if rank >= f - p or p >= f:
                return _chain_data(flag, ranks, divisors, ())
            vector = _general_solution(rows, pivot_rows, _symbols(f"t_{p + 1}_", f))

    @staticmethod
    def _sampled_chain(matrices, f, exact_flag, exact_ranks, exact_divisors):
        runs = []
        for seed in range(_sample_runs):
            run = _sampled_run(matrices, f, exact_flag, exact_ranks, random.Random(seed))
            twin = next((other for other in runs if other.ranks == run.ranks), None)
            runs.append(run)
            if twin is not None:
                break
        else:
            dgcv_warning(
                f"the polar system ranks of {_sample_runs} randomly sampled integral flags disagree; "
                "the largest ranks are reported"
            )
            run = max(runs, key=lambda other: other.ranks)
        return _chain_data(
            run.vectors, run.ranks, exact_divisors, run.samples, len(exact_ranks)
        )

    def _warm_caches(self, at=None):
        at = self._point(at)
        key = None if at is None else frozenset(at.items())
        if key not in self._flags:
            chart = self._chart(at)
            data = self._chain(chart, at)
            vectors, ranks, divisors = data.vectors, data.ranks, data.divisors
            f = len(chart.complementary_index)
            flag_symbols = [
                w for p in range(1, len(vectors) + 2) for w in _symbols(f"t_{p}_", f)
            ]
            if at is None:
                self._record_singularities(
                    "integral_flag", _loci_from_divisors(divisors, flag_symbols)
                )
            self._flags[key] = _flag_data(
                chart=chart,
                vectors=tuple(vectors),
                polar_ranks=tuple(ranks),
                characters=tuple(ranks[:1])
                + tuple(ranks[q] - ranks[q - 1] for q in range(1, len(ranks))),
                genre=len(ranks) or None,
                singularities=tuple(_singular_factors(divisors)),
                weights=tuple(_flag_weights(vectors, f)),
                samples=tuple(data.samples),
                exact_levels=data.exact_levels,
            )
        return self._flags[key]

    def characters(self, at=None):
        return self._warm_caches(at).characters

    def polar_ranks(self, at=None):
        return self._warm_caches(at).polar_ranks

    def genre(self, at=None):
        return self._warm_caches(at).genre

    def integral_flag(self, at=None):
        return [list(subspace) for subspace in self._warm_caches(at).flag]

    def integral_flag_singularities(self, at=None):
        return list(self._warm_caches(at).singularities)

    def summary(
        self,
        theme=None,
        use_latex=None,
        *,
        plain_text=None,
        return_displayable=False,
        at=None,
        interrupt_to_partial_report=True,
        _reporting_threshold_s=7.0,
        **kwargs,
    ):
        at = self._point(at)
        return _system_summary(
            self,
            at=at,
            theme=theme,
            use_latex=use_latex,
            plain_text=plain_text,
            return_displayable=return_displayable,
            interrupt_to_partial_report=interrupt_to_partial_report,
            reporting_threshold_s=_reporting_threshold_s,
            **kwargs,
        )

    def integral_flag_summary(
        self,
        theme=None,
        use_latex=None,
        *,
        plain_text=None,
        return_displayable=False,
        at=None,
        **kwargs,
    ):
        at = self._point(at)
        return _integral_flag_summary(
            self,
            self._warm_caches(at),
            at=at,
            theme=theme,
            use_latex=use_latex,
            plain_text=plain_text,
            return_displayable=return_displayable,
            **kwargs,
        )

    @staticmethod
    def _ordinary(vectors, polar_ranks, matrices):
        count = len(vectors) - 1
        if count > len(polar_ranks):
            return False
        f = len(vectors[0])
        flag = []
        for q in range(count):
            flag.append(_combination(vectors, f"u_{q + 1}_"))
            if _rank(_polar(matrices, flag, f)) != polar_ranks[q]:
                return False
        return True

    def is_ordinary(self, N, at=None):
        at = self._point(at)
        chart = self._chart(at)
        vectors = self._subspace(N, chart, at)
        matrices = self._structure_matrices(chart, at)
        if not self._integral(vectors, matrices):
            return False
        return self._ordinary(vectors, self.polar_ranks(at), matrices)

    def is_regular(self, N, at=None):
        at = self._point(at)
        chart = self._chart(at)
        vectors = self._subspace(N, chart, at)
        matrices = self._structure_matrices(chart, at)
        if not self._integral(vectors, matrices):
            return False
        polar_ranks = self.polar_ranks(at)
        p = len(vectors)
        if p > len(polar_ranks) or not self._ordinary(vectors, polar_ranks, matrices):
            return False
        return _rank(_polar(matrices, vectors, len(vectors[0]))) == polar_ranks[p - 1]

    def characteristic_distribution(self, at=None):
        at = self._point(at)
        chart = self._chart(at)
        f = len(chart.complementary_index)
        if f == 0:
            return []
        rows = [
            row
            for matrix in self._structure_matrices(chart, at)
            for row in matrix.tolist()
        ]
        return [
            chart.element(v) for v in _polar_basis(matrix_dgcv(rows or [[0] * f]), f)
        ]

    def is_Cartan_involutive(self, horizontal_forms=None, at=None):
        try:
            return self._Cartan_obstruction(horizontal_forms, at) is None
        except _SingularPointError:
            return None

    def _Cartan_obstruction(self, horizontal_forms=None, at=None):
        at = self._point(at)
        own = horizontal_forms is None
        forms = (
            self.horizontal_forms
            if own
            else tuple(self._closed_horizontals(self._checked_forms(horizontal_forms)))
        )
        if not forms:
            if own and self._dropped_horizontals:
                return "every horizontal form given is dependent on the distribution"
            raise TypeError(
                "no horizontal forms: pass them as `horizontal_forms`, or build the system with them"
            )
        data = self._warm_caches(at)
        chart = data.chart
        n = len(forms)
        rows = [[_subs(entry, at) for entry in row] for row in self._on_complementary(forms, chart)]
        if not chart.complementary_index or _rank(matrix_dgcv(rows)) < n:
            return "the horizontal forms are dependent on the distribution"
        if data.genre < n:
            return f"the genre {data.genre} is smaller than {n}"
        projection = [[_dot(h, v) for v in data.vectors[:n]] for h in rows]
        if not _projects_onto(projection, n, data.weights):
            return f"no {n}-dimensional horizontal ordinary subspace"
        return None


def _hol_to_real(expr, **kwargs):
    return symToReal(holToReal(expr, **kwargs), **kwargs)


_CONVERTERS = {
    "holToReal": (_hol_to_real, _hol_to_real),
    "symToReal": (symToReal, symToReal),
    "realToSym": (realToSym, realToSym),
    "realToHol": (realToSym, realToSym),
    "symToHol": (symToHol, None),
}


def _sub_type(x):
    return _coordinate_kind(x)[0]


def _converted_symbols(symbols, basis, kwargs):
    out = []
    for x in symbols:
        atoms = sorted(
            _free(basis(x, **kwargs)),
            key=lambda atom: (_sub_type(atom) in ("anti", "imag"), str(atom)),
        ) or [x]
        for atom in atoms:
            if not any(str(atom) == str(seen) for seen in out):
                out.append(atom)
    return out


def _unit_atom(form):
    if isinstance(
        form, (abstract_differential_form_monomial, abstract_differential_form)
    ):
        items = list(form.coefficients().items())
        if len(items) == 1:
            key, value = items[0]
            if len(key) == 1 and _scalar_is_zero(value - 1):
                return key[0]
    return form


def _source_coordinates(source, parameters):
    images = source.forms_in_coordinates
    if images is None:
        return ()
    return tuple(
        _bridges._coordinate_space(images, set(parameters) | set(source.parameters or ()))
    )


def _on_forms(rows, forms):
    return [
        sum(
            (
                value * form
                for value, form in zip(row, forms)
                if not _scalar_is_zero(value)
            ),
            0,
        )
        for row in rows
    ]


def _rule_dict(rules):
    if isinstance(rules, dict):
        return dict(rules)
    if isinstance(rules, (list, tuple)) and all(
        isinstance(rule, tuple) and len(rule) == 2 for rule in rules
    ):
        return dict(rules)
    raise TypeError("`subs` takes a dict of rules or a list of (key, value) pairs")


def _substituted_parameters(parameters, rules, coordinates):
    kept = {p for p in parameters if p not in rules}
    for key, value in rules.items():
        if key in parameters:
            kept |= {s for s in _free(value) if s not in coordinates}
    return frozenset(kept)


def _coefficients(form):
    if isinstance(form, numbers.Number):
        return [form]
    if _is_abstract(form):
        return list(form.coefficients().values())
    return list(getattr(form, "coeff_dict", {}).values())


def _is_infinite(value):
    return getattr(value, "is_finite", None) is False


def _subs_forms(forms, rules, kwargs=None):
    out = []
    for form in forms:
        try:
            new = _routed_subs(form, rules, **(kwargs or {}))
        except (ZeroDivisionError, ValueError):
            raise ValueError(
                f"the rules give a pole in a coefficient of {form}"
            ) from None
        if any(_is_infinite(value) for value in _coefficients(new)):
            raise ValueError(f"the rules give a pole in a coefficient of {form}")
        out.append(new)
    return out


def _slot_symbols(slots):
    out = set()
    for name in ("frame", "coordinate_forms"):
        for obj in slots[name] or ():
            out |= _free(obj)
    for value in slots["potentials"].values():
        out |= _free(value)
    return out


def _subs_source(source, rules, coordinates, kwargs):
    slots = source._slots()
    if not set(rules) & (set(source.free_symbols) | _slot_symbols(slots)):
        return source
    equations = source.subs(rules, **kwargs).structure_equations
    for name in ("frame", "coordinate_forms"):
        if slots[name] is not None:
            slots[name] = tuple(_subs_forms(slots[name], rules))
    slots["potentials"] = {
        form: _routed_subs(value, rules) for form, value in slots["potentials"].items()
    }
    slots["parameters"] = _substituted_parameters(
        set(slots["parameters"] or ()), rules, coordinates
    )
    return abstract_coframe(source.forms, equations, source.min_conj_rules, **slots)


def _source_coframe(forms, coframe):
    nonzero = [form for form in forms if not _form_is_zero(form)]
    abstract = all(_is_abstract(form) for form in nonzero)
    if coframe is not None:
        if not isinstance(coframe, abstract_coframe):
            raise TypeError("`coframe` must be an abstract coframe")
        if not abstract:
            raise TypeError("`coframe` takes abstract 1-forms")
        return coframe
    if nonzero and abstract:
        owners = {
            atom.primary_coframe
            for form in nonzero
            for key in form.coefficients()
            for atom in key
        }
        if len(owners) != 1 or None in owners:
            raise TypeError(
                "the abstract 1-forms given belong to no single coframe; pass `coframe=`"
            )
        return owners.pop()
    _check_coordinate_forms(nonzero)
    return None


def _check_coordinate_forms(forms):
    for form in forms:
        if _form_is_zero(form):
            continue
        if (
            not query_dgcv_categories(form, {"differential_form"})
            or getattr(form, "degree", None) != 1
        ):
            raise TypeError(
                f"expected coordinate 1-forms or abstract 1-forms; received {form!r}"
            )


def _abstract_rows(forms, coframe):
    rows = []
    for form in forms:
        row = [0] * coframe.dimension
        if not _form_is_zero(form):
            for key, value in form.coefficients().items():
                if len(key) != 1:
                    raise TypeError(f"expected 1-forms; received {form!r}")
                if key[0] not in coframe.forms:
                    raise ValueError(
                        f"{form} is not a combination of the forms of {coframe}"
                    )
                row[coframe.forms.index(key[0])] = value
        rows.append(row)
    return rows


def _on_coframe(forms, coframe):
    return all(
        _form_is_zero(form)
        or (
            _is_abstract(form)
            and all(
                atom in coframe.forms for key in form.coefficients() for atom in key
            )
        )
        for form in forms
    )


def _coordinate_columns(forms, parameters, coordinates):
    if coordinates is None:
        nonzero = [form for form in forms if not _form_is_zero(form)]
        coordinates = _bridges._coordinate_space(nonzero, parameters)
    if not coordinates:
        raise TypeError(
            "zero forms name no manifold; pass `coframe=` or `coordinates=`"
        )
    return list(coordinates)


def _rows_on_fields(forms, fields):
    return [
        [0] * len(fields)
        if _form_is_zero(form)
        else [_bridges._scalar(form(field)) for field in fields]
        for form in forms
    ]


def _vanishes_nowhere(row):
    return any(
        _bridges._is_number(entry) and not _scalar_is_zero(entry) for entry in row
    )


def _columns_numbers_first(rows):
    return sorted(
        range(len(rows[0])),
        key=lambda idx: sum(not _bridges._is_number(row[idx]) for row in rows),
    )


def _independent_rows(rows):
    if not rows:
        return []
    order = sorted(range(len(rows)), key=lambda idx: not _vanishes_nowhere(rows[idx]))
    transposed = [[rows[i][j] for i in order] for j in _columns_numbers_first(rows)]
    pivots = matrix_dgcv(transposed).pivot_columns(
        simplify_steps=True, zero_test=_zero_test()
    )
    return sorted(order[k] for k in pivots)


def _pivot_columns(rows):
    if not rows:
        return []
    order = _columns_numbers_first(rows)
    pivots = matrix_dgcv([[row[idx] for idx in order] for row in rows]).pivot_columns(
        simplify_steps=True, zero_test=_zero_test()
    )
    return sorted(order[k] for k in pivots)


def _member_label(form, coframe):
    if isinstance(form, abstract_differential_form_atom) and form in coframe.forms:
        return form.label
    return None


def _form_labels(labels, kept, rest, taken):
    if isinstance(labels, str):
        taken = set(taken)
        names = []
        index = 0
        for label in kept:
            if label is None:
                index += 1
                while f"{labels}{index}" in taken:
                    index += 1
                label = f"{labels}{index}"
            names.append(label)
        return names + rest
    if any(name is None for name in labels):
        labels = _form_labels(
            "omega",
            [None if name is None else str(name) for name in labels],
            [],
            set(taken) | {str(name) for name in labels if name is not None},
        )
    labels = [str(name) for name in labels]
    if len(labels) == len(kept):
        return labels + rest
    if len(labels) == len(kept) + len(rest):
        return labels
    raise ValueError(
        f"`labels` lists the system's forms ({len(kept)}) or every form of the new coframe "
        f"({len(kept) + len(rest)}); received {len(labels)}"
    )


def _complementary_name(x, reality):
    partner = reality.pairs.get(x)
    prefix = conjugation_prefix()
    if partner is not None and str(x).startswith(prefix):
        return f"{prefix}d{partner}"
    return f"d{x}"


def _fresh_atoms(names):
    return [abstract_differential_form_atom(1, 1, label=name) for name in names]


def _owned_coframe(atoms, coefficients, **slots):
    coframe = abstract_coframe(
        atoms, {atom: abstract_differential_form([]) for atom in atoms}, **slots
    )
    coframe._own()
    terms = {}
    for (i, j, k), value in coefficients.items():
        terms.setdefault(atoms[k], []).append(value * atoms[i] * atoms[j])
    coframe.update_structure_equations(
        replace_eqns={
            form: abstract_differential_form(items) for form, items in terms.items()
        },
        simplify=False,
        with_conjugates=False,
    )
    return coframe


def _coordinate_coframe(names, images, coordinates, parameters, objects):
    images = tuple(images)
    frame = tuple(objects.frame(images, coordinates))
    atoms = _fresh_atoms(names)
    return _owned_coframe(
        atoms,
        objects.structure(images, frame, parameters),
        frame=frame,
        coordinate_forms=images,
        parameters=parameters,
        potentials=_bridges.coordinate_potentials(atoms, images, parameters),
    )


def _rebased_coframe(source, names, images):
    rebased = transform_coframe(
        source, images, new_coframe_basis=_fresh_atoms(names), label=None
    )
    position = {form: idx for idx, form in enumerate(rebased.forms)}
    return _owned_coframe(
        _fresh_atoms(names),
        {
            (position[a], position[b], position[c]): value
            for (a, b, c), value in rebased.coefficients().items()
        },
        frame=rebased.dual_frame_in_coordinates,
        coordinate_forms=rebased.forms_in_coordinates,
        parameters=rebased.parameters,
        potentials=rebased.potentials,
        potential_index=rebased._potential_index,
    )

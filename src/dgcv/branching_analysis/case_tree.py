from __future__ import annotations

from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
    subs,
)
from .._aux._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
    latex_in_html,
)
from .._aux.printing.printing._string_processing import _strip_displaystyles
from ._conditions import backsub_store, merge_conditions, normalize_conditions
from ._seeding import add_cases_from as _add_cases_from
from ._seeding import seeded_from as _seeded_from
from ._tasks import (
    _UNSET,
    clear_profile_cache,
    compute_status,
    constraint_rules,
    has_equation_system,
    inherited,
    inherited_constraints,
    invoke,
    invoke_with,
    is_builder,
    result_message,
    sample_point,
    sampled_conditions,
    sampled_relations,
    substitutions,
    symbol_pool,
)
from ._traversal import walk
from .printing._conditions_latex import conditions_str
from .printing._leaves_html import tree_leaves_html
from .printing._summaries import case_summary as _case_summary
from .printing._tree_html import full_tree_html, latex_verbose_tree, verbose_tree

__all__ = ["case_tree"]


class case_tree:
    """
    Directory structure for organizing branching case analysis of an equation system.

    Parameters:
    -----------
    label: str (optional, default='root')
        string label of node on the case tree. On the root this is also the title shown in
        `tree_summary`'s root node. `add_case`, `sampled_subcase`, `add_corollary` and
        `add_cases_from` register what they build and return nothing, so a notebook cell
        ending in one of them stays quiet; the subcase is reachable as an attribute of its
        parent or through `parent._subcases`, and `return_created_object=True` hands it
        back directly
    equation_system: iterable (optional, default = [])
        An iterable containing objects that dgcv can compare with zero
    parameters: list|tuple|set (optional, default = [])
        list of variables in the equation system that can be constrained in the case tree.
        `add_case` also accepts `parameters`/`variables` to replace a subcase's declared
        atoms, and `new_parameters`/`new_variables` to extend them. Extending is what a
        reparametrizing subcase needs: substituting new symbols in through its closed
        conditions leaves them undeclared, so `free_parameters` drops them and the reported
        count drifts to zero. Declaring them on the whole tree instead would count them as
        free on every other branch
    completion_condition: callable (optional, default compares elements in case-rules-reduced eqn system to 0),
        This should be a callable function that can be applied to the self.reduced_equation_system returning a bool indicating if the eqn system is satisfied
    evaluate_with_simplifies: bool (optional, default = True)
        Determines if internal simplify calls are usesd or omitted. Can affect compute times and completion_condition accuracy
    completion_message: str|callable (optional)
        This messages is displayed in tree summaries at complete state nodes. If str, then the str literal is displayed. if callable, then the callable(self) is displayed, so callable(self) should return a str
    summary: callable (optional)
        If callable the default self.case_summary will be replaced by `summary(self)`
    subject: object|callable (optional)
        A parameterized object that each branch specializes by substituting its closed
        case rules, or a function building it from them, called as subject(constraints)
        or subject(constraints, node). Substitution reaches objects supporting `subs`
        and the entries of lists, tuples, sets and dicts; pass a function to specialize
        anything else. Defaults to the equation system. Available per branch as
        self.specialized. The `constraints` passed in are self.substitutions: the branch's
        own closed conditions, any inherited from a tree it was seeded from, and any
        sample point in effect, with the sample point resolved through the conditions'
        values. self.constraints is the same less the sample point, since a sample point is
        a representative value rather than a constraint
    profile: callable|dict|str (optional)
        The per-branch computation, reported in tree and leaf summaries. Called as
        profile(specialized) or profile(specialized, node) according to how many positional
        parameters it accepts. Returning a dict names the reported fields; a "status"
        key sets the branch status and a "message" key its summary line. A non-callable is
        reported as-is without computing anything, so the branch's subject is never built:
        a dict is used as the result verbatim, and any other value becomes its message
    status: callable (optional)
        Called like `profile` and returning the branch status. Bools are reported as
        complete/incomplete. Overrides any "status" key in the profile result
    report: list (optional)
        Names of profile result fields to report, in order. Defaults to every field
        returned across the leaves
    other_data: object (optional)
        Free-form data carried by the tree and ignored by every standard method, provided
        for `subject`, `profile` and `status` callables to reference as node.other_data. It
        is stored exactly as passed and resolves through the parent chain, so setting it
        once at the root reaches every branch and any subcase can override it
    raise_on_profile_error: bool (optional, default = False)
        By default an exception raised by `profile` or `status` is recorded on self.error
        and reported as the status "error". Set True to let it propagate
    """

    def __init__(
        self,
        label: str = None,
        equation_system=None,
        variables=None,
        parameters=None,
        completion_condition=None,
        evaluate_with_simplifies=True,
        subject=None,
        profile=None,
        status=None,
        report=None,
        raise_on_profile_error=None,
        sample_point=None,
        conditions_to_sample=None,
        comparison=None,
        other_data=None,
        _inherited_constraints=None,
        _parent=None,
        _explicit_completion=None,
        **kwargs,
    ):
        legacy_cr = kwargs.pop("case_rules", None)

        self._parent = _parent
        self._subject = subject
        self._profile = profile
        self._status_rule = status
        self._report = report
        self._sample_point = sample_point
        self._sampled_conditions = conditions_to_sample
        self._comparison = comparison
        self._other_data = other_data
        self._inherited_constraints = _inherited_constraints
        self._raise_on_profile_error = raise_on_profile_error
        self._explicit_completion = (
            callable(completion_condition)
            if _explicit_completion is None
            else _explicit_completion
        )

        self.general_equation_system = (
            [] if equation_system is None else equation_system
        )
        if "_simplify_rule" in kwargs:
            s = kwargs["_simplify_rule"]
        elif evaluate_with_simplifies is True:

            def s(x):
                return simplify(x)
        else:

            def s(x):
                return x

        self._internal_simplify = s

        pool = (
            symbol_pool(self, subject)
            if variables is None or parameters is None
            else ()
        )
        if variables is None:
            try:
                vari = set()
                for obj in pool:
                    vari |= get_free_symbols(obj)
                if parameters:
                    vari = {v for v in vari if v not in parameters}
                self.system_variables = vari
            except Exception:
                self.system_variables = set()
        else:
            self.system_variables = set(variables)
        if parameters is None:
            try:
                all_symbols = set()
                for obj in pool:
                    all_symbols |= get_free_symbols(obj)
                self.system_parameters = {
                    x for x in all_symbols if x not in self.system_variables
                }
            except Exception:
                self.system_parameters = set()
        else:
            self.system_parameters = set(parameters)
        if not callable(completion_condition):

            def _cd(obj):
                try:

                    def ztest(x):
                        return _scalar_is_zero(self._internal_simplify(x))

                    return all(ztest(x) for x in obj)
                except Exception:
                    return False

            self.completion_condition = _cd
        else:
            self.completion_condition = completion_condition
        self.label = label if label else "root"
        self._subcases = {}
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not isinstance(getattr(self, "_initial_rules", None), dict):
            self._initial_rules = {}
        if legacy_cr:
            merge_conditions(self._initial_rules, normalize_conditions(legacy_cr), "d")
        self._complete = None
        self._ev_eqn_system = None
        self._free_variables = None
        self._free_parameters = None
        self._processed_cr = None
        self._closed_cr = None
        self._open_cr = None
        self._profile_result = _UNSET
        self._profile_error = None
        self._status = _UNSET
        self._specialized = _UNSET
        self._sampled_relations = _UNSET

    @property
    def complete(self):
        if self._complete is None:
            self._complete = self.completion_condition(self.reduced_equation_system)
        return self._complete

    @property
    def case_rules(self):
        if self._processed_cr is None:
            s = self._internal_simplify
            out = {}
            for e, slot in self._initial_rules.items():
                ns = {}
                if "closed" in slot:
                    v, src = slot["closed"]
                    ns["closed"] = (s(v), src)
                if "open" in slot:
                    ns["open"] = tuple((s(v), src) for v, src in slot["open"])
                out[e] = ns
            self._processed_cr = out
        return self._processed_cr

    @property
    def closed_case_rules(self):
        if self._closed_cr is None:
            self._closed_cr = {
                e: slot["closed"][0]
                for e, slot in self.case_rules.items()
                if "closed" in slot
            }
        return self._closed_cr

    @property
    def open_case_rules(self):
        if self._open_cr is None:
            self._open_cr = {
                e: {v for v, _ in slot["open"]}
                for e, slot in self.case_rules.items()
                if slot.get("open")
            }
        return self._open_cr

    @property
    def constraints(self):
        return constraint_rules(self)

    @property
    def open_conditions(self):
        return self.open_case_rules

    @property
    def other_data(self):
        return inherited(self, "_other_data")

    @property
    def inherited_constraints(self):
        return inherited_constraints(self)

    @property
    def constraint_rules(self):
        return constraint_rules(self)

    @property
    def sample_point(self):
        return sample_point(self)

    @property
    def substitutions(self):
        return substitutions(self)

    @property
    def conditions_to_sample(self):
        return sampled_conditions(self)

    @property
    def sampled_relations(self):
        return sampled_relations(self)

    @property
    def is_sampled(self):
        return bool(sampled_conditions(self))

    @property
    def subject(self):
        declared = inherited(self, "_subject")
        if declared is not None:
            return declared
        return self.general_equation_system if has_equation_system(self) else None

    @property
    def profile(self):
        return inherited(self, "_profile")

    @property
    def status_rule(self):
        return inherited(self, "_status_rule")

    @property
    def report_fields(self):
        return inherited(self, "_report")

    def setting(self, name, default=None):
        value = inherited(self, name)
        return default if value is None else value

    def set_subject(self, obj):
        self._subject = obj
        clear_profile_cache(self)
        return obj

    def set_profile(self, f):
        self._profile = f
        clear_profile_cache(self)
        return f

    def set_status(self, f):
        self._status_rule = f
        clear_profile_cache(self)
        return f

    def apply(self, obj, simplify=True, iterate=None, rules=None):
        if rules is None:
            rules = self.substitutions
        s = self._internal_simplify if simplify else (lambda x: x)
        if hasattr(obj, "subs"):
            return s(subs(obj, rules))
        if isinstance(obj, dict):
            return {k: s(subs(v, rules)) for k, v in obj.items()}
        if iterate is None:
            iterate = isinstance(obj, (list, tuple, set, frozenset))
        if iterate:
            try:
                items = list(obj)
            except TypeError:
                return s(subs(obj, rules))
            return [s(subs(x, rules)) for x in items]
        return s(subs(obj, rules))

    @property
    def specialized(self):
        if self._specialized is _UNSET:
            subj = self.subject
            rules = self.substitutions
            if subj is None:
                self._specialized = dict(rules)
            elif is_builder(subj):
                self._specialized = invoke_with(self, subj, dict(rules))
            elif subj is self.general_equation_system and not self.sample_point:
                self._specialized = self.reduced_equation_system
            else:
                self._specialized = self.apply(subj)
        return self._specialized

    @property
    def result(self):
        if self._profile_result is _UNSET:
            f = self.profile
            if f is None:
                self._profile_result = None
            elif not callable(f):
                self._profile_result = f if isinstance(f, dict) else {"message": str(f)}
            else:
                try:
                    self._profile_result = invoke(self, f)
                except Exception as e:
                    if inherited(self, "_raise_on_profile_error") is True:
                        raise
                    self._profile_error = e
                    self._profile_result = None
        return self._profile_result

    @property
    def error(self):
        _ = self.result
        return self._profile_error

    @property
    def result_fields(self):
        r = self.result
        if r is None:
            return {}
        if isinstance(r, dict):
            return {k: v for k, v in r.items() if k not in ("status", "message")}
        return {"result": r}

    @property
    def status(self):
        if self._status is _UNSET:
            self._status = compute_status(self)
        return self._status

    @property
    def message(self):
        return result_message(self)

    @property
    def leaves(self):
        return dict(walk(self))

    def results(self, include_interior=False):
        return {p: n.result for p, n in walk(self, include_interior)}

    def statuses(self, include_interior=False):
        return {p: n.status for p, n in walk(self, include_interior)}

    def errors(self, include_interior=False):
        return {
            p: n.error for p, n in walk(self, include_interior) if n.error is not None
        }

    def run(self, include_interior=False):
        for _, node in walk(self, include_interior):
            _ = node.status
        return self

    def add_case(
        self,
        label: str = None,
        defining_conditions: dict = None,
        corollary_conditions: dict = None,
        return_created_object: bool = False,
        **kwargs,
    ):
        legacy_cr = kwargs.pop("case_rules", None)
        warn_open = kwargs.pop("_warn_open", True)
        override_params = kwargs.pop("parameters", None)
        override_vars = kwargs.pop("variables", None)
        added_params = kwargs.pop("new_parameters", None)
        added_vars = kwargs.pop("new_variables", None)

        child_params = (
            self.system_parameters if override_params is None else set(override_params)
        )
        if added_params is not None:
            child_params = child_params | set(added_params)
        child_vars = (
            self.system_variables if override_vars is None else set(override_vars)
        )
        if added_vars is not None:
            child_vars = child_vars | set(added_vars)

        if label is None:
            for idx in range(1, len(self._subcases) + 2):
                pref = f"_{idx}"
                if any(str(x) == f"_{idx}" for x in self._subcases):
                    continue
                label = pref
                break

        verbose = kwargs.get("verbose", None)
        if not isinstance(verbose, bool):
            verbose = getattr(self, "verbose", False) is True
        if not isinstance(label, str):
            raise TypeError("The `label` parameter must be a string.")
        if label.isnumeric():
            raise ValueError(
                "Pure numeric labels for subcases are not supported. Recommendation: prepend the numeric label with an underscore."
            )
        if label in self._subcases:
            if verbose:
                dgcv_warning(
                    "Overwriting an existing subcase branch.",
                    wc_label="dgcvOperationsNote",
                )
        elif hasattr(self, label):
            raise ValueError(
                "subcases cannot be assigned names coinciding with the `case_tree` class' base attributes."
            )

        defining = normalize_conditions(defining_conditions)
        if legacy_cr:
            defining["closed"] = {
                **defining["closed"],
                **normalize_conditions(legacy_cr)["closed"],
            }
        corollary = normalize_conditions(corollary_conditions)

        new_closed = {**defining["closed"], **corollary["closed"]}
        child_rules = backsub_store(self.case_rules, new_closed)
        merge_conditions(child_rules, defining, "d", warn_open=warn_open)
        merge_conditions(child_rules, corollary, "c", warn_open=warn_open)

        kwargs["_initial_rules"] = child_rules
        new_tree = case_tree(
            label=label,
            equation_system=self.general_equation_system,
            variables=child_vars,
            parameters=child_params,
            completion_condition=self.completion_condition,
            **{
                "_simplify_rule": self._internal_simplify,
                "_parent": self,
                "_explicit_completion": self._explicit_completion,
                **kwargs,
            },
        )
        setattr(self, label, new_tree)
        self._subcases[label] = new_tree
        return new_tree if return_created_object else None

    def sampled_subcase(
        self,
        label: str = None,
        conditions_to_sample=None,
        sample_point: dict = None,
        comparison: str = None,
        return_created_object: bool = False,
        **kwargs,
    ):
        return self.add_case(
            label=label,
            conditions_to_sample=list(conditions_to_sample or []),
            sample_point=dict(sample_point or {}),
            comparison=comparison,
            return_created_object=return_created_object,
            **kwargs,
        )

    def add_corollary(
        self,
        label: str,
        corollary_conditions: dict = None,
        return_created_object: bool = False,
        **kwargs,
    ):
        legacy_cr = kwargs.pop("case_rules", None)
        if corollary_conditions is None and legacy_cr is not None:
            corollary_conditions = legacy_cr
        return self.add_case(
            label=label,
            corollary_conditions=corollary_conditions,
            return_created_object=return_created_object,
            **kwargs,
        )

    def add_cases_from(
        self,
        source,
        select=None,
        prefix="c",
        label_rule=None,
        transfer_open=True,
        include_interior=False,
        return_created_object: bool = False,
        **kwargs,
    ):
        return _add_cases_from(
            self,
            source,
            select=select,
            prefix=prefix,
            label_rule=label_rule,
            transfer_open=transfer_open,
            include_interior=include_interior,
            return_created_object=return_created_object,
            **kwargs,
        )

    @classmethod
    def seeded_from(
        cls,
        source,
        select=None,
        prefix="",
        label_rule=None,
        transfer_open=True,
        include_interior=False,
        **kwargs,
    ):
        return _seeded_from(
            cls,
            source,
            select=select,
            prefix=prefix,
            label_rule=label_rule,
            transfer_open=transfer_open,
            include_interior=include_interior,
            **kwargs,
        )

    def remove_case(self, label):
        if label in self._subcases:
            _ = self._subcases.pop(label, None)
            if hasattr(self, label):
                delattr(self, label)

    def _repr_latex_(self, raw=False, **kwargs):
        out = conditions_str(
            {"closed": self.closed_case_rules, "open": self.open_case_rules},
            _punct="",
            **kwargs,
        )
        if raw is True:
            out = _strip_displaystyles(out)
        return out

    def __str__(self):
        return getattr(self, "inheritance_path", "") + self.label

    def show_case_rules(self, plain_text=False, **kwargs):
        if plain_text is True:
            print({"closed": self.closed_case_rules, "open": self.open_case_rules})
        else:
            from .._aux.printing.printing._dgcv_display import show

            show(self._repr_latex_(**kwargs))

    @property
    def reduced_equation_system(self):
        if self._ev_eqn_system is None:
            es = self.general_equation_system
            self._ev_eqn_system = self.apply(
                es,
                iterate=not hasattr(es, "subs") and not isinstance(es, dict),
                rules=constraint_rules(self),
            )
        return self._ev_eqn_system

    @property
    def variable_constraints(self):
        vari = self.system_variables
        return {
            "closed": {e: v for e, v in self.closed_case_rules.items() if e in vari},
            "open": {e: vals for e, vals in self.open_case_rules.items() if e in vari},
        }

    @property
    def parameter_conditions(self):
        params = self.system_parameters
        return {
            "closed": {e: v for e, v in self.closed_case_rules.items() if e in params},
            "open": {
                e: vals for e, vals in self.open_case_rules.items() if e in params
            },
        }

    @property
    def free_variables(self):
        if self._free_variables is None:
            scr = self.closed_case_rules
            fvd = {v: subs(v, scr) for v in self.system_variables}
            fv = set()
            for _, v in fvd.items():
                fv |= get_free_symbols(v)
            self._free_variables = [v for v in fv if v in self.system_variables]
        return self._free_variables

    @property
    def free_parameters(self):
        if self._free_parameters is None:
            scr = self.closed_case_rules
            fvd = {v: subs(v, scr) for v in self.system_parameters}
            fv = set()
            for v in fvd.values():
                fv |= get_free_symbols(v)
            self._free_parameters = [v for v in fv if v in self.system_parameters]
        return self._free_parameters

    def tree_summary(
        self,
        theme: str = None,
        filter_conditions=None,
        use_latex=None,
        foldable=True,
        collapse_depth=None,
        collapse_threshold: int = 6,
        generate_full_report: bool = True,
    ):
        use_latex = (
            get_dgcv_settings_registry().get("use_latex", False)
            if use_latex is None
            else use_latex
        )
        tree = (
            latex_verbose_tree(
                self,
                filter_conditions=filter_conditions,
                evaluate=generate_full_report,
                sep="\n",
            )
            if use_latex
            else verbose_tree(
                self,
                filter_conditions=filter_conditions,
                evaluate=generate_full_report,
                sep="\n",
            )
        )
        return latex_in_html(
            full_tree_html(
                tree,
                theme=theme,
                root_label=self.label,
                foldable=foldable,
                collapse_depth=collapse_depth,
                collapse_threshold=collapse_threshold,
            )
        )

    def leaf_summary(self, theme=None, sort_by=None, reverse=False, **kwargs):
        return tree_leaves_html(
            self, theme=theme, sort_by=sort_by, reverse=reverse, **kwargs
        )

    def case_summary(self, plain_text=False):
        return _case_summary(self, plain_text=plain_text)

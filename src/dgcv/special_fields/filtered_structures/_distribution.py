from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any, Literal

from ..._aux._backends._numeric_router import rational_sample
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    conjugate,
    get_free_symbols,
    simplify,
    subs,
)
from ..._aux._backends._types_and_constants import exact_fraction, rational, symbol
from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from ..._aux._vmf.vmf import order_coordinates
from ...algebras import algebra_class, createAlgebra
from ...algebras.threads.util import _span_solver
from ...core.base import dgcv_class
from ...core.conversions.conversions import allToReal, allToSym, symToHol
from ...core.dgcv_core import tensor_field_class, wedge
from ...core.solvers import solve_dgcv
from ...core.vector_fields_and_differential_forms import (
    LieDerivative,
    _extract_basis_by_wedge_vectorized,
    annihilator,
    decompose,
)


class _flag_span:
    def __init__(self, basis, coordinates, use_numeric, point=None):
        self.coordinates = tuple(coordinates)
        self.use_numeric = use_numeric
        self.basis = []
        self.divisors = []
        self._rank_cap = len(self.coordinates)
        self._fixed = point is not None
        self.point = point
        self._reset_point()
        for vf in basis:
            self.independent(vf)

    def _reset_point(self):
        for _ in range(1 if self._fixed else 8):
            if not self._fixed:
                self.point = {var: rational_sample() for var in self.coordinates}
            self.num = None
            self._num_ok = True
            self._num_images = []
            self._images = []
            self._subsets = {}
            images = [self._image(vf) for vf in self.basis]
            if any(image is None for image in images):
                self._pole()
                continue
            if all(self._numeric_insert(image) for image in images):
                return

    def _pole(self):
        if self._fixed:
            raise ValueError(
                "the distribution's coefficients have a pole at the given point."
            )

    def _image(self, vf):
        try:
            image = {k: subs(v, self.point) for k, v in vf.coeff_dict.items()}
        except ZeroDivisionError:
            return None
        except ValueError as err:
            if "division by zero" not in str(err):
                raise
            return None
        for v in image.values():
            if exact_fraction(v) is None and (
                getattr(v, "is_finite", None) is False or str(v) == "nan"
            ):
                return None
        return image

    @staticmethod
    def _symbolic(vf):
        return {k: v for k, v in vf.coeff_dict.items() if not _scalar_is_zero(v)}

    def _numeric_insert(self, image):
        if not self._num_ok:
            return True
        if all(_scalar_is_zero(v) for v in image.values()):
            return False
        symbolic = any(exact_fraction(v) is None for v in image.values())
        if self.num is None:
            solver = _span_solver.build_from_vectors([image], force_symbolic=symbolic)
            if solver is None:
                self._num_ok = False
                return True
            self.num = solver
            self._num_images = [image]
            return True
        res = self.num.insert_vector(image)
        if res is None and self.num.numeric:
            solver = _span_solver.build_from_vectors(
                self._num_images, force_symbolic=True
            )
            if solver is not None:
                self.num = solver
                res = self.num.insert_vector(image)
        if res is None:
            self._num_ok = False
            return True
        if res:
            self._num_images.append(image)
        return res

    def _numeric_verdict(self, vf):
        image = self._image(vf)
        for _ in range(8):
            if image is not None:
                break
            self._pole()
            self._reset_point()
            image = self._image(vf)
        else:
            raise ValueError(
                "the distribution's coefficients have a pole at every sampled point."
            )
        if self._num_ok:
            res = self._numeric_insert(image)
            if self._num_ok:
                if res:
                    return True, None
                coords = None if self.num is None else self.num.reduce_vector(image)
                if coords is None:
                    return False, None
                return False, tuple(
                    i for i, c in enumerate(coords) if not _scalar_is_zero(c)
                )
        while len(self._images) < len(self.basis):
            self._images.append(subs(self.basis[len(self._images)], self.point))
        probe = subs(vf, self.point)
        if not self._images:
            return (not probe.is_zero), ()
        coeffs = decompose(probe, self._images, assume_basis=True)[0]
        if len(coeffs) == 0:
            return True, None
        return False, tuple(i for i, c in enumerate(coeffs) if not _scalar_is_zero(c))

    def _hints(self, rows):
        if self.num is None:
            return [None] * len(rows)
        keys = self.num.pivot_keys()
        return [keys[i] if i < len(keys) else None for i in rows]

    def _symbolic_solver(self, rows):
        vecs = [self._symbolic(self.basis[i]) for i in rows]
        solver = _span_solver.build_from_vectors(
            vecs[:1], force_symbolic=True, membership_only=True
        )
        if solver is None:
            return None
        hints = self._hints(rows)
        for vec, hint in zip(vecs[1:], hints[1:]):
            solver.insert_vector(vec, pivot_hint=hint)
        self.divisors.extend(solver.divisors)
        return solver

    def _dependent_on(self, rows, vf):
        rows = tuple(rows)
        if not rows:
            return vf.is_zero
        cache = self._subsets
        solver = cache.get(rows, False)
        if solver is False:
            solver = self._symbolic_solver(rows)
            cache[rows] = solver
        if solver is not None:
            res = solver.in_span(self._symbolic(vf))
            if res is not None:
                return res
        return bool(
            decompose(
                vf,
                [self.basis[i] for i in rows],
                assume_basis=True,
                only_check_decomposability=True,
            )
        )

    def independent(self, vf):
        if len(self.basis) >= self._rank_cap:
            return False
        indep, support = self._numeric_verdict(vf)
        if indep:
            self.basis.append(vf)
            return True
        if self.use_numeric or self._fixed:
            return False
        if support is not None and len(support) < len(self.basis):
            if self._dependent_on(support, vf):
                return False
        if self._dependent_on(range(len(self.basis)), vf):
            return False
        self.basis.append(vf)
        self._reset_point()
        return True


class distribution(dgcv_class):
    def __init__(
        self,
        spanning_vf_set=None,
        spanning_df_set=None,
        assume_compatibility: bool = False,
        check_compatibility_aggressively: bool = False,
        _assume_minimal_Data=None,
        *,
        coordinate_space: None | Sequence[Any] = None,
        find_basis: bool = False,
        find_polynomial_spanners=False,
        assume_starting_objs_polynomial=False,
        assume_spanning_sections_linearly_indep=False,
        formatting: None | Literal["complex", "real"] = None,
        dimension_hint=None,
    ):
        """
        The optional `dimension hint` keyword is used to minimize expensive linear independence
        checks when extracting a basis. Only set this to a known upper bound on possible
        distribution rank or else computed basis may be wrong.
        """
        if assume_spanning_sections_linearly_indep:
            _assume_minimal_Data = retrieve_passkey()
        if spanning_vf_set is not None:
            if not isinstance(spanning_vf_set, (list, tuple)):
                raise TypeError(
                    "`distribution` spanning_vf_set must be a list/tuple of vector fields."
                )
            spanning_vf_set = tuple(spanning_vf_set)
            if not all(
                query_dgcv_categories(vf, {"vector_field"}) for vf in spanning_vf_set
            ):
                raise TypeError(
                    "`distribution` spanning_vf_set must contain only vector fields."
                )

        if spanning_df_set is not None:
            if not isinstance(spanning_df_set, (list, tuple)):
                raise TypeError(
                    "`distribution` spanning_df_set must be a list/tuple of degree-1 differential forms."
                )
            spanning_df_set = tuple(spanning_df_set)
            if not all(
                query_dgcv_categories(df, {"differential_form"})
                and getattr(df, "degree", None) == 1
                for df in spanning_df_set
            ):
                raise TypeError(
                    "`distribution` spanning_df_set must contain only degree-1 differential forms."
                )

        if spanning_vf_set is None and spanning_df_set is None:
            self._prefered_data_type = 1
            self._spanning_vf_set = tuple()
            self._spanning_df_set = tuple()
            self.coordinates = tuple()
            self.formatting = None
            self._vf_basis = (
                tuple() if _assume_minimal_Data == retrieve_passkey() else None
            )
            self._df_basis = (
                tuple() if _assume_minimal_Data == retrieve_passkey() else None
            )
            self._derived_flag = None
            self._wderived_flag = None
            self._derived_flag_divisors = None
            self._wderived_flag_divisors = None
            return

        self._simplifying_preference = find_polynomial_spanners
        if formatting not in (None, "complex", "real"):
            formatting = None

        self._prefered_data_type = 1 if spanning_df_set is None else 0

        vfs = (
            None
            if spanning_vf_set is None
            else self._normalize_spanning_set(
                spanning_vf_set,
                formatting=formatting,
                scale_to_poly=find_polynomial_spanners
                and not assume_starting_objs_polynomial,
            )
        )
        dfs = (
            None
            if spanning_df_set is None
            else self._normalize_spanning_set(
                spanning_df_set,
                formatting=formatting,
                scale_to_poly=find_polynomial_spanners
                and not assume_starting_objs_polynomial,
            )
        )

        if vfs is not None and dfs is not None and assume_compatibility is False:
            for df in dfs:
                for vf in vfs:
                    val = df(vf)
                    if check_compatibility_aggressively:
                        val = simplify(val)
                    if not _scalar_is_zero(val):
                        raise TypeError(
                            "Unable to verify that the provided vector fields and differential forms annihilate each other. "
                            "Use assume_compatibility=True to bypass this check, or set check_compatibility_aggressively=True."
                        )

        def _flatten_unique(seq_of_seqs):
            out = []
            seen = set()
            for seq in seq_of_seqs:
                for v in seq:
                    if v in seen:
                        continue
                    seen.add(v)
                    out.append(v)
            return tuple(out)

        obj_spanners = (
            spanning_vf_set
            if spanning_vf_set
            else spanning_df_set
            if spanning_df_set
            else []
        )
        inferred_min_vs = _flatten_unique(
            obj.infer_minimal_varSpace() for obj in obj_spanners
        )

        if coordinate_space is not None:
            if not isinstance(coordinate_space, (list, tuple, set)):
                raise TypeError(
                    "coordinate_space must be a list/tuple/set if provided."
                )
            vs = tuple(coordinate_space)
            missing = [v for v in inferred_min_vs if v not in vs]
            if missing:
                raise ValueError(
                    "Provided coordinate_space does not contain all inferred minimal coordinates: "
                    + ", ".join(str(x) for x in missing)
                )
            self.coordinates = order_coordinates(vs)
        else:
            self.coordinates = order_coordinates(inferred_min_vs)
        self._dim_hint = dimension_hint
        self._spanning_vf_set = vfs
        self._spanning_df_set = dfs
        self._characteristic = None
        self._ext_power_class_cache = None
        self.formatting = formatting
        self._anticanonical_bundle = None
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "distribution"

        if _assume_minimal_Data == retrieve_passkey():
            self._vf_basis = self._spanning_vf_set
            self._df_basis = self._spanning_df_set
        else:
            self._vf_basis = None
            self._df_basis = None

        self._derived_flag = None
        self._wderived_flag = None
        self._derived_flag_divisors = None
        self._wderived_flag_divisors = None

        if find_basis is True:
            if vfs:
                self._spanning_vf_set = self.vf_basis
            else:
                self._spanning_df_set = self.df_basis

        self.varSpace = self.coordinates  # deprecated

    @staticmethod
    def _infer_minimal_vs(obj) -> tuple[Any, ...]:
        f = getattr(obj, "infer_minimal_varSpace", None)
        if callable(f):
            vs = f()
            if isinstance(vs, tuple):
                return vs
            if isinstance(vs, (list, set)):
                return tuple(vs)
        vs2 = getattr(obj, "varSpace", None)
        if isinstance(vs2, tuple):
            return vs2
        if isinstance(vs2, (list, set)):
            return tuple(vs2)
        return tuple()

    @property
    def anticanonical_bundle(self):
        if self._anticanonical_bundle is None:
            vf_basis = self.vf_basis
            if len(vf_basis) == 0:
                self._anticanonical_bundle = tensor_field_class(coeff_dict={tuple(): 1})
            else:
                self._anticanonical_bundle = wedge(*vf_basis)
        return self._anticanonical_bundle

    @property
    def rank(self):
        return len(self.vf_basis)

    @staticmethod
    def _validated_format_of(obj) -> str | None:
        fmt = getattr(obj, "_validated_format", None)
        if fmt in ("standard", "complex", "real"):
            return fmt
        return None

    @staticmethod
    def _needs_mixed_conversion(formats: set[str]) -> bool:
        return ("real" in formats) and ("complex" in formats)

    @staticmethod
    def _preferred_formatting_from_settings() -> str:
        pref = get_dgcv_settings_registry().get("preferred_variable_format", None)
        return "real" if pref == "real" else "complex"

    @staticmethod
    def _convert_obj(obj, target: str):
        if target == "real":
            return allToReal(obj)
        return allToSym(obj)

    def _normalize_spanning_set(
        self, spanning_set, *, formatting: None | str, scale_to_poly: bool = False
    ):
        elems = tuple(spanning_set)
        if not elems:
            return tuple()

        if formatting is not None:
            target = formatting
            out = []
            for e in elems:
                ef = self._validated_format_of(e)
                if ef != target and ef in ("real", "complex"):
                    new_e = self._convert_obj(e, target)
                    if scale_to_poly:
                        new_e = new_e.scale_to_polynomial_attempt()
                    out.append(new_e)
                else:
                    if scale_to_poly:
                        new_e = e.scale_to_polynomial_attempt()
                        out.append(new_e)
                    else:
                        out.append(e)
            return tuple(out)

        fmts = set()
        new_elems = []
        for e in elems:
            if scale_to_poly:
                e = e.scale_to_polynomial_attempt()
            new_elems.append(e)
            ef = self._validated_format_of(e)
            if ef is not None:
                fmts.add(ef)
        elems = new_elems

        if not self._needs_mixed_conversion(fmts):
            return elems

        target = self._preferred_formatting_from_settings()
        out = []
        for e in elems:
            ef = self._validated_format_of(e)
            if ef in ("real", "complex") and ef != target:
                out.append(self._convert_obj(e, target))
            else:
                out.append(e)
        return tuple(out)

    @property
    def spanning_vf_set(self):
        if self._spanning_vf_set is None:
            vfs = annihilator(
                self.df_basis,
                coordinate_space=self.coordinates,
                coherent_coordinates_checked=False,
                polynomial_bases=self._simplifying_preference,
            )
            self._spanning_vf_set = vfs
            self._vf_basis = vfs
        return self._spanning_vf_set

    @property
    def vf_basis(self):
        if self._vf_basis is None:
            if self._spanning_vf_set is None:
                return self.spanning_vf_set
            self._vf_basis = _extract_basis_by_wedge_vectorized(
                self._spanning_vf_set, dimension_hint=self._dim_hint
            )
        return self._vf_basis

    @property
    def df_basis(self):
        if self._df_basis is None:
            if self._spanning_df_set is None:
                return self.spanning_df_set
            self._df_basis = _extract_basis_by_wedge_vectorized(
                self._spanning_df_set, dimension_hint=self._dim_hint
            )
        return self._df_basis

    @property
    def spanning_df_set(self):
        if self._spanning_df_set is None:
            sdf = annihilator(
                self.vf_basis,
                coordinate_space=self.coordinates,
                coherent_coordinates_checked=False,
                polynomial_bases=self._simplifying_preference,
            )
            self._spanning_df_set = sdf
            self._df_basis = sdf
        return self._spanning_df_set

    def intersection(self, other, formatting=None):
        if not isinstance(other, distribution):
            raise TypeError(
                "dgcv.distribution.intersection can only operate on other dgcv.distribution instances."
            )
        svf = self._vf_basis if self._vf_basis is not None else self.spanning_vf_set
        sdf = self._df_basis if self._df_basis is not None else self.spanning_df_set
        ovf = other._vf_basis if other._vf_basis is not None else other.spanning_vf_set
        odf = other._df_basis if other._df_basis is not None else other.spanning_df_set
        if len(svf) * len(odf) < len(ovf) * len(sdf):
            interVF = annihilator(odf, control_distribution=svf)
        else:
            interVF = annihilator(sdf, control_distribution=ovf)
        formatting = (
            self.formatting
            if formatting is None and self.formatting == other.formatting
            else formatting
        )
        return distribution(
            interVF,
            formatting=formatting,
        )

    def union(self, other, extract_basis=False):
        if not isinstance(other, distribution):
            raise TypeError(
                "dgcv.distribution.intersection can only operate on other dgcv.distribution instances."
            )
        return distribution(
            self.spanning_vf_set + other.spanning_vf_set, extract_basis=extract_basis
        )

    def _derive_flag(
        self, weak, find_polynomial_spanners, max_iterations, use_numeric, point=None
    ):
        tiered_list = [list(self.vf_basis)]
        span = _flag_span(tiered_list[0], self.coordinates, use_numeric, point)
        for _ in range(max_iterations):
            left = tiered_list[0] if weak else sum(tiered_list, [])
            top = tiered_list[-1]
            new_tier = []
            for vf1 in left:
                for vf2 in top:
                    nb = LieDerivative(vf1, vf2)
                    if nb.is_zero:
                        continue
                    if find_polynomial_spanners is True:
                        nb = nb.scale_to_polynomial_attempt(factor=True)
                    if span.independent(nb):
                        new_tier.append(nb)
            if not new_tier:
                break
            tiered_list.append(new_tier)
        divisors = list(span.divisors)
        if point is not None and span.num is not None:
            divisors.extend(span.num.divisors)
        return tiered_list, divisors

    def _generic_flag_sizes(self, max_iterations):
        if self._wderived_flag is not None:
            return [len(level) for level in self._wderived_flag]
        best = None
        draws = 0
        for _ in range(6):
            point = {var: rational_sample() for var in self.coordinates}
            try:
                flag = self._derive_flag(True, False, max_iterations, True, point)[0]
            except ValueError:
                continue
            sizes = [len(level) for level in flag]
            if best is None or self._cumulative(sizes) > self._cumulative(best):
                best = sizes
            draws += 1
            if draws == 2:
                break
        return best

    @staticmethod
    def _cumulative(sizes):
        out = []
        total = 0
        for size in sizes:
            total += size
            out.append(total)
        return tuple(out) + (total,) * (16 - len(out))

    @staticmethod
    def _flag_point(at):
        if not isinstance(at, dict):
            raise TypeError(
                f"`at` must be a dict mapping coordinates to values; got {type(at).__name__}."
            )
        return at

    def derived_flag(
        self,
        find_polynomial_spanners=True,
        max_iterations=10,
        use_numeric_methods=False,
        at=None,
    ):
        if at is not None:
            return self._derive_flag(
                False,
                find_polynomial_spanners,
                max_iterations,
                True,
                self._flag_point(at),
            )[0]
        use_numeric = use_numeric_methods or bool(
            get_dgcv_settings_registry().get("use_numeric_methods", False)
        )
        if self._derived_flag is None:
            self._derived_flag, self._derived_flag_divisors = self._derive_flag(
                False, find_polynomial_spanners, max_iterations, use_numeric
            )
        return self._derived_flag

    def weak_derived_flag(
        self,
        find_polynomial_spanners=False,
        max_iterations=10,
        use_numeric_methods=False,
        at=None,
    ):
        if at is not None:
            return self._derive_flag(
                True,
                find_polynomial_spanners,
                max_iterations,
                True,
                self._flag_point(at),
            )[0]
        use_numeric = use_numeric_methods or bool(
            get_dgcv_settings_registry().get("use_numeric_methods", False)
        )
        if self._wderived_flag is None:
            self._wderived_flag, self._wderived_flag_divisors = self._derive_flag(
                True, find_polynomial_spanners, max_iterations, use_numeric
            )
        return self._wderived_flag

    def nilpotent_approximation(
        self,
        approximation_point=None,
        label=None,
        basis_labels=None,
        exclude_from_VMF=False,
        return_created_object=True,
        randomize_approximation_point=False,
        use_numeric_methods=False,
        **kwargs,
    ):
        if randomize_approximation_point:
            from random import randint

            approximation_point = dict()
            for var in self.coordinates:
                in1 = randint(1, 20)
                in2 = randint(in1 + 1, in1 + 20)
                ins = [in1, in2]
                idx = randint(0, 1)
                approximation_point[var] = rational(ins[idx], ins[1 - idx])
            # Add plain text printing
            from ..._aux.printing.printing._dgcv_display import (
                LaTeX_eqn_system,
                show,
            )

            print("Evaluating nilpotent approximation at the randomly chosen point:")
            show(LaTeX_eqn_system(approximation_point, one_line=True))
        approximation_point = kwargs.get(
            "expansion_point", approximation_point
        )  # old syntax support
        if approximation_point is None:
            approximation_point = {var: 0 for var in self.coordinates}

        dimension = len(self.coordinates)
        derFlag = self.weak_derived_flag(at=approximation_point)
        local_sizes = [len(level) for level in derFlag]
        reference_sizes = self._generic_flag_sizes(10)
        if reference_sizes is not None and self._cumulative(
            local_sizes
        ) != self._cumulative(reference_sizes):
            dgcv_warning(
                f"The expansion point is a growth-vector singularity: the derived flag there has tier sizes {local_sizes} against {reference_sizes} at a generic point. The nilpotent approximation returned is the graded algebra of the flag at the point."
            )
        evaluated_flag = [
            list([subs(vf, approximation_point) for vf in level]) for level in derFlag
        ]
        evaluated_basis = _extract_basis_by_wedge_vectorized(
            sum(evaluated_flag, []), use_numeric_methods=use_numeric_methods
        )
        depth = len(derFlag)
        basisVF = sum(derFlag, [])

        discrep = len(self.coordinates) - len(evaluated_basis)
        if discrep > 0:
            dgcv_warning(
                f"The distribution is not bracket generating or the expansion point is a growth-vector singularity singularity (note: currently `dgcv.distribution` methods are not intended for analysis at such singularities). A complement to its bracket-generated envelope has been assigned weight {-depth} and added to the nilpotent approximation as a component commuting with everything."
            )
        elif discrep < 0:  # old logic, never happens; refactor reminder
            raise TypeError(
                f"The distribution is singular at the point {approximation_point}. Nilpotent approximations are not yet supported for singular distributions."
            )
        vlabel = create_key("var")
        vars = [symbol(f"{vlabel}{j}") for j in range(len(evaluated_basis))]
        gen_elem = sum(coef * elem for coef, elem in zip(vars, evaluated_basis))

        def _decomp(elem, ge=gen_elem, variables=vars):
            eqns = list((elem - ge).coeff_dict.values())
            sol = solve_dgcv(
                eqns, variables, method="linear_parametric", simplify_result=False
            )
            return sol

        level_dimensions = [len(level) for level in derFlag]

        def i_to_w_rule(idx):
            cap = 0
            for level, ld in enumerate(level_dimensions):
                cap += ld
                if idx < cap:
                    return -1 - level
            return -depth

        idx_to_weight_assignment = {j: i_to_w_rule(j) for j in range(dimension)}
        grading_vec = [idx_to_weight_assignment[idx] for idx in range(dimension)]
        VFC_enum = list(enumerate(basisVF))
        algebra_data = dict()
        for count1, elem1 in VFC_enum:
            for count2, elem2 in VFC_enum[count1 + 1 :]:
                newLevelWeight = (
                    idx_to_weight_assignment[count1] + idx_to_weight_assignment[count2]
                )
                if newLevelWeight < -depth:
                    coeffs = [0] * len(self.coordinates)
                else:
                    eqns = subs(LieDerivative(elem1, elem2), approximation_point)
                    coeff_sol = _decomp(eqns)
                    if len(coeff_sol) == 0:
                        raise RuntimeError(
                            "failed to extract algebra structure during nilpotent approximation."
                        )
                    coeffs = [
                        coeff_sol[0].get(var, var)
                        if idx_to_weight_assignment[idx] == newLevelWeight
                        else 0
                        for idx, var in enumerate(vars)
                    ] + ([0] * discrep)
                algebra_data[(count1, count2)] = coeffs
                algebra_data[(count2, count1)] = [-j for j in coeffs]
        if label is None:
            if basis_labels is not None:
                dgcv_warning(
                    "`basis_labels` was provided but no `label` was provided; `basis_labels` is ignored."
                )
            printWarning = (
                "This algebra was initialized via `distribution.nilpotent_approximation` with no label; "
                "automatic labels were assigned. Provide `label=...` (and optionally `basis_labels=...`) to control labeling, "
                "or use exclude_from_VMF=True to suppress warnings."
            )
            childPrintWarning = (
                "This algebraElement's parent algebra was initialized via `distribution.nilpotent_approximation` with no label; "
                "automatic labels were assigned."
            )
            exclusionPolicy = retrieve_passkey() if exclude_from_VMF is True else None
            return algebra_class(
                algebra_data,
                grading=[grading_vec],
                assume_skew=True,
                _callLock=retrieve_passkey(),
                _print_warning=printWarning,
                _child_print_warning=childPrintWarning,
                _exclude_from_VMF=exclusionPolicy,
            )

        return createAlgebra(
            algebra_data,
            label,
            basis_labels=basis_labels,
            grading=[grading_vec],
            assume_skew=True,
            return_created_object=return_created_object,
            forgo_vmf_registry=exclude_from_VMF,
        )

    def contains_section(
        self,
        section,
        section_parameters: list = None,
        linear_section_parameters: bool = False,
    ):
        if query_dgcv_categories(section, {"vector_field"}):
            bas = self.vf_basis
        elif query_dgcv_categories(section, {"differential_form"}):
            bas = self.df_basis
        if len(bas) == 0 and section_parameters is None:
            return section.is_zero
        indepcheck = decompose(
            section,
            bas,
            assume_basis=True,
            variables_to_constrain=section_parameters,
            assume_VTC_linear=linear_section_parameters,
        )
        if section_parameters is None:
            return len(indepcheck[0]) != 0
        out = len(indepcheck[0]) != 0 and indepcheck[2]
        return out

    @property
    def _ext_power_class(self):
        if self._ext_power_class_cache is None:
            self._ext_power_class_cache = simplify(wedge(*self.vf_basis))
        return self._ext_power_class_cache

    @property
    def characteristic(self):
        if self._characteristic is None:
            vfs = self.vf_basis
            genVF, section_par = linear_combination(vfs)
            sp_contraints = []
            for vf in vfs:
                bracket = LieDerivative(genVF, vf)
                deco = self.contains_section(
                    bracket,
                    section_parameters=section_par,
                    linear_section_parameters=True,
                )
                sp_contraints += [k - v for k, v in deco.items()]
            sol = solve_dgcv(
                sp_contraints,
                section_par,
                method="linear_parametric",
                simplify_result=False,
            )[0]
            solution = subs(genVF, sol)
            free_vars = set()
            for val in sol.values():
                free_vars |= get_free_symbols(val)
            free_vars = set(section_par) & free_vars
            zeroing = {vari: 0 for vari in free_vars}
            char_dist = [
                subs(solution, zeroing | {vari: 1}).scale_to_polynomial_attempt()
                for vari in free_vars
            ]
            self._characteristic = char_dist
        return self._characteristic

    def __add__(self, other):
        if other == 0:
            return self
        if get_dgcv_category(other) == "distribution":
            return distribution(self.spanning_vf_set + other.spanning_vf_set)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if get_dgcv_category(other) == "distribution":
            lbs = [
                LieDerivative(vf1, vf2)
                for vf1 in self.vf_basis
                for vf2 in other.vf_basis
            ]
            return distribution(
                list(self.spanning_vf_set) + list(other.spanning_vf_set) + lbs,
                find_polynomial_spanners=True,
                find_basis=True,
            )
        return NotImplemented

    def __pow__(self, other):
        if not isinstance(other, numbers.Integral) or other >= 0:
            return NotImplemented
        out = distribution([])
        for _ in range(other):
            out *= self
        return out

    def __str__(self):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        max_dim = 20
        vs = getattr(self, "varSpace", None) or tuple()
        fmt = getattr(self, "formatting", None)

        if getattr(self, "_prefered_data_type", 1) == 1:
            span = getattr(self, "_spanning_vf_set", None)
            if span is None:
                span = self.spanning_vf_set
        else:
            span = getattr(self, "_spanning_df_set", None)
            if span is None:
                span = self.spanning_df_set

        span = tuple(span) if span is not None else tuple()

        def _trunc(seq):
            if len(seq) <= max_dim:
                return seq
            k = max_dim // 2
            return seq[:k] + ("...",) + seq[-k:]

        core = "<" + ", ".join(str(e) for e in _trunc(span)) + ">"

        if not vlp:
            return core

        vs_core = "<" + ", ".join(str(v) for v in _trunc(vs)) + ">"
        tag = "distribution"
        if fmt in ("complex", "real"):
            tag += f"[{fmt}]"
        return f"{tag} on {vs_core}: {core}"

    def _repr_latex_(self, raw: bool = False, abbrev: bool = False, **kwargs):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        max_dim = 20
        vs = getattr(self, "varSpace", None) or tuple()
        fmt = getattr(self, "formatting", None)

        if getattr(self, "_prefered_data_type", 1) == 1:
            span = getattr(self, "_spanning_vf_set", None)
            if span is None:
                span = self.spanning_vf_set
        else:
            span = getattr(self, "_spanning_df_set", None)
            if span is None:
                span = self.spanning_df_set

        span = tuple(span) if span is not None else tuple()

        def _trunc(seq):
            if len(seq) <= max_dim:
                return seq
            k = max_dim // 2
            return seq[:k] + (r"\dots",) + seq[-k:]

        def _tex(obj):
            if get_dgcv_settings_registry().get("compile_latex_conjugation", True):
                f = getattr(
                    symToHol(obj, convert_everything=False), "_repr_latex_", None
                )
            else:
                f = getattr(obj, "_repr_latex_", None)
            if callable(f):
                s = f(raw=True)
                return str(s).replace("$", "").replace(r"\displaystyle", "")
            return str(obj)

        if abbrev:
            out = r"\mathcal{D}"
            return out if raw else rf"$\displaystyle {out}$"

        inner = ", ".join(_tex(e) for e in _trunc(span))
        core = rf"\left\langle {inner}\right\rangle"

        if not vlp:
            out = core
            return out if raw else rf"$\displaystyle {out}$"

        vs_inner = ", ".join(_tex(v) for v in _trunc(vs))
        vs_core = rf"\left\langle {vs_inner}\right\rangle"

        tag = r"\mathcal{D}"
        if fmt == "real":
            tag = r"\mathcal{D}_{\mathbb{R}}"
        elif fmt == "complex":
            tag = r"\mathcal{D}_{\mathbb{C}}"

        out = rf"{tag}\ \text{{on}}\ {vs_core}:\ {core}"
        return out if raw else rf"$\displaystyle {out}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw, **kwargs)

    def apply(self, operator, *args, **kwds):
        return distribution(
            [operator(vf, *args, **kwds) for vf in self.vf_basis],
            formatting=self.formatting,
        )

    def __dgcv_apply__(self, operator):
        return self.apply(operator)

    def __dgcv_conjugate__(self, symbolic=False):
        return self.apply(conjugate, symbolic=symbolic)

from __future__ import annotations

from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    get_free_symbols,
)
from ..._aux._utilities._misc import linear_combination
from ...core.vector_fields_and_differential_forms.decomposition import decompose
from ._tensor_products import _fast_tensor_products


def _scaled_sum(lst):
    hook = getattr(type(lst[0][1]), "_dgcv_multiadd_scaled", None)
    if hook is not None:
        return hook(lst, 0)
    total = 0
    for c, v in lst:
        total = total + c * v
    return total


class _symbol_generators:
    def _generator_data(self):
        gens = self._GLA_generators
        neg = self.negativePart
        node_elem = gens["node_elem"]
        S = []
        s_index = dict()
        s_weight = []
        s_parents = []
        for w in sorted(gens["map"], reverse=True):
            for node, elem, dep in gens["map"][w]:
                s_index[id(elem)] = len(S)
                S.append(elem)
                s_weight.append(w)
                if dep == 1:
                    s_parents.append(None)
                else:
                    ea = node_elem[id(node[0])]
                    eb = node_elem[id(node[1])]
                    s_parents.append((s_index[id(ea)], s_index[id(eb)]))
        s_atoms = [
            _fast_tensor_products({(s,): 1}, neg, _validated=1) for s in range(len(S))
        ]
        s_dual = [dict() for _ in S]
        scalars = []
        for w, rows in gens["coords"].items():
            if not rows:
                continue
            positions = gens["level_positions"][w]
            base = s_index[id(gens["spanning"][w][0])]
            for i, row in enumerate(rows):
                for sl, c in enumerate(row):
                    if not _scalar_is_zero(c):
                        s_dual[base + sl][positions[i]] = c
        parent_of = {p: s for s, p in enumerate(s_parents) if p is not None}
        pairs = []
        for j, sj in enumerate(S):
            if s_parents[j] is not None:
                continue
            for t, st in enumerate(S):
                if t == j or (s_parents[t] is None and t < j):
                    continue
                if (j, t) in parent_of or (t, j) in parent_of:
                    continue
                prod = sj * st
                rel = dict()
                if not _fast_tensor_products(prod).is_zero:
                    Sw = gens["spanning"][s_weight[j] + s_weight[t]]
                    base = s_index[id(Sw[0])]
                    for sl, c in enumerate(decompose(prod, Sw, assume_basis=True)[0]):
                        if not _scalar_is_zero(c):
                            rel[base + sl] = c
                            scalars.append(c)
                pairs.append((j, t, rel))
        if self._parameters:
            divisors = []
            for c in scalars:
                _, d = as_numer_denom(c)
                if get_free_symbols(d):
                    divisors.append(d)
            if divisors:
                self._singularities["prolongation"] = expr_union_primitives(
                    list(self._singularities.get("prolongation", [])) + divisors,
                    self._parameters,
                    process_rationals=True,
                    fail_quietly=True,
                )
        gens["S"] = S
        gens["s_index"] = s_index
        gens["s_weight"] = s_weight
        gens["s_parents"] = s_parents
        gens["s_atoms"] = s_atoms
        gens["s_dual"] = s_dual
        by_position = [dict() for _ in range(neg.dimension)]
        for s, row in enumerate(s_dual):
            for p, c in row.items():
                by_position[p][s] = c
        gens["s_by_position"] = by_position
        gens["s_pairs"] = pairs
        gens["s_generators"] = {
            w: [s for s in range(len(S)) if s_parents[s] is None and s_weight[s] == w]
            for w in gens["generators"]
        }
        gens["ds_coords"] = dict()

    def _s_split(self, op):
        cache = getattr(op, "_s_split_cache", None)
        if cache is None:
            cache = dict()
            for key, v in op.coeff_dict.items():
                if len(key) == 2:
                    cache.setdefault(key[1], []).append((key[0], v))
            op._s_split_cache = cache
        return cache

    def _s_contract(self, op, s):
        terms = self._s_split(op).get(s)
        if not terms:
            return None
        return _fast_tensor_products(
            {(i,): v for i, v in terms}, self.negativePart, _validated=1
        )

    def _s_combine(self, pairs):
        pairs = [(c, x) for c, x in pairs if x is not None and not x.is_zero]
        if not pairs:
            return None
        out = _fast_tensor_products._dgcv_multiadd_scaled(pairs)
        if out.is_zero:
            return None
        if out.algebra is None:
            out = _fast_tensor_products(out.coeff_dict, self.negativePart, _validated=1)
        return out

    def _operator_S(self, idx):
        alias = self._aliasing[idx]
        op = alias.get("operator_S")
        if op is None:
            base = alias.get("operator")
            if base is None:
                raise RuntimeError(
                    "`Tanaka_symbol.prolong` with precomputed generators met a nonnegative element without a one-level operator form."
                )
            terms = dict()
            for s, elem in enumerate(self._GLA_generators["S"]):
                img = base * elem
                if not isinstance(img, _fast_tensor_products):
                    img = _fast_tensor_products(img)
                if img.is_zero:
                    continue
                for key, v in img.coeff_dict.items():
                    terms[(key[0], s)] = v
            op = _fast_tensor_products(terms, self.negativePart, _validated=2)
            alias["operator_S"] = op
        return op

    def _s_bracket(self, X, b):
        if X is None or X.is_zero:
            return None
        neg = self.negativePart
        if next(iter(X.coeff_dict))[0] < neg.dimension:
            alg = X._to_algebra(neg)
            if alg is False:
                return None
            out = _fast_tensor_products(alg * self._GLA_generators["S"][b])
            return None if out.is_zero else out
        pairs = []
        for key, v in X.coeff_dict.items():
            img = self._s_contract(self._operator_S(key[0]), b)
            if img is not None:
                pairs.append((v, img))
        return self._s_combine(pairs)

    def _s_images(self, g):
        gens = self._GLA_generators
        parents = gens["s_parents"]
        images = []
        for s in range(len(gens["S"])):
            p = parents[s]
            if p is None:
                images.append(self._s_contract(g, s))
            else:
                a, b = p
                images.append(
                    self._s_combine(
                        [
                            (1, self._s_bracket(images[a], b)),
                            (-1, self._s_bracket(images[b], a)),
                        ]
                    )
                )
        return images

    def _ds_coords(self, elem, fresh):
        gens = self._GLA_generators
        cached = None if fresh else gens["ds_coords"].get(id(elem))
        if cached is None:
            by_position = gens["s_by_position"]
            pending = dict()
            for p, c in enumerate(elem.coeffs):
                if _scalar_is_zero(c):
                    continue
                for s, d in by_position[p].items():
                    pending.setdefault(s, []).append((c, d))
            cached = dict()
            for s, lst in pending.items():
                total = _scaled_sum(lst)
                if not _scalar_is_zero(total):
                    cached[s] = total
            if not fresh:
                gens["ds_coords"][id(elem)] = cached
        return cached

    def _generator_equations(self, general_elem, height, DS_records, esVars, accumulate):
        gens = self._GLA_generators
        images = self._s_images(general_elem)
        for record in DS_records:
            source_bound = record.cap - height - 1
            for (low, high), source in record.components.items():
                if high >= 0 or low > source_bound:
                    continue
                if low == high:
                    target = record.component(low + height + 1)
                    dsSpanners = target.spanners if target else []
                    sources = source.spanners
                    fresh = False
                else:
                    dsSpanners = record.target_spanners(
                        low + height + 1, min(high, source_bound) + height + 1
                    )
                    fresh = high > source_bound
                    sources = (
                        source.truncated_spanners(source_bound)
                        if fresh
                        else source.spanners
                    )
                for elem in sources:
                    coords = self._ds_coords(elem, fresh)
                    image = self._s_combine([(c, images[s]) for s, c in coords.items()])
                    if dsSpanners:
                        newGE, newVars = linear_combination(dsSpanners)
                        esVars += newVars
                        accumulate(newGE if image is None else image + newGE)
                    elif image is not None:
                        accumulate(image)
        for j, t, rel in gens["s_pairs"]:
            terms = [
                (-1, self._s_bracket(images[j], t)),
                (1, self._s_bracket(images[t], j)),
            ]
            for s, c in rel.items():
                terms.append((c, images[s]))
            expr = self._s_combine(terms)
            if expr is not None:
                accumulate(expr)

    def _complete_S(self, el):
        g = self._aliased_expansion(el, partial=True)
        terms = dict()
        if isinstance(g, _fast_tensor_products):
            for s, img in enumerate(self._s_images(g)):
                if img is None:
                    continue
                for key, v in img.coeff_dict.items():
                    terms[(key[0], s)] = v
        return _fast_tensor_products(terms, self.negativePart, _validated=2)

    def _materialize_S_atom(self, alias):
        op = alias["operator_S"]
        dual = self._GLA_generators["s_dual"]
        neg = self.negativePart
        pending = dict()
        for (i, s), v in op.coeff_dict.items():
            for p, c in dual[s].items():
                pending.setdefault((i, p), []).append((c, v))
        terms = dict()
        for key, lst in pending.items():
            total = _scaled_sum(lst)
            if not _scalar_is_zero(total):
                terms[key] = total
        alias["operator"] = _fast_tensor_products(terms, neg, _validated=2)
        alias.pop("_pending_S", None)

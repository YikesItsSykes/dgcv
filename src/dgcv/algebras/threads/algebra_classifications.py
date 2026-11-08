"""
Reference data for simple Lie algebras
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

_min_rank = {"A": 1, "B": 2, "C": 3, "D": 4}

_exceptional_dim_rank = {
    "G2": (14, 2),
    "F4": (52, 4),
    "E6": (78, 6),
    "E7": (133, 7),
    "E8": (248, 8),
}

_exceptional_forms = {
    "G2": ((-14, ()), (2, ())),
    "F4": ((-52, ()), (4, ("FI",)), (-20, ("FII",))),
    "E6": (
        (-78, ()),
        (6, ("EI",)),
        (2, ("EII",)),
        (-14, ("EIII",)),
        (-26, ("EIV",)),
    ),
    "E7": ((-133, ()), (7, ("EV",)), (-5, ("EVI",)), (-25, ("EVII",))),
    "E8": ((-248, ()), (8, ("EVIII",)), (-24, ("EIX",))),
}

_exceptional_root_data = {
    "G2": (6, 6, 3, 4),
    "F4": (24, 24, 2, 9),
    "E6": (72, 0, 1, 12),
    "E7": (126, 0, 1, 18),
    "E8": (240, 0, 1, 30),
}

# low rank coincidences, and the sp(n) vs sp(2n) naming clash
_extra_names = {
    "su(2)": ("sp(1)", "so(3)"),
    "sl(2,R)": ("sp(2,R)", "so(2,1)"),
    "so(5)": ("sp(2)",),
    "so(4,1)": ("sp(1,1)",),
    "so(3,2)": ("sp(4,R)",),
    "su(4)": ("so(6)",),
    "su(3,1)": ("so*(6)",),
    "su(2,2)": ("so(4,2)",),
    "sl(4,R)": ("so(3,3)",),
    "su*(4)": ("so(5,1)",),
    "sl(2,C)_R": ("so(3,1)",),
}

_same_algebra = {
    "su(1,1)": "sl(2,R)",
    "so*(8)": "so(6,2)",
}

_forms_cache = {}


def _parse_type(tag):
    if tag in _exceptional_dim_rank:
        return tag, None
    fam = tag[:1]
    if fam not in _min_rank:
        return None, None
    rk = int(tag[1:])
    return (None, None) if rk is None else (fam, rk)


def _family_dim(fam, rk):
    if rk is None or fam not in _min_rank or rk < _min_rank[fam]:
        return None
    if fam == "A":
        return rk * (rk + 2)
    if fam in {"B", "C"}:
        return rk * (2 * rk + 1)
    if fam == "D":
        return rk * (2 * rk - 1)
    return None


def _type_dimension_rank(tag):
    fam, rk = _parse_type(tag)
    if fam is None:
        return None
    if rk is None:
        return _exceptional_dim_rank[fam]
    dim = _family_dim(fam, rk)
    return None if dim is None else (dim, rk)


def _types_at_rank(dim, rk):
    out = [f"{fam}{rk}" for fam in "ABCD" if _family_dim(fam, rk) == dim]
    out += [tag for tag, size in _exceptional_dim_rank.items() if size == (dim, rk)]
    return out


def _admissible_ranks(dim):
    """Ranks at which some type could have dimension dim, by inverting the formulas."""
    ranks = set()
    root = isqrt(dim + 1)
    if root * root == dim + 1:
        ranks.add(root - 1)
    root = isqrt(8 * dim + 1)
    if root * root == 8 * dim + 1:
        if (root - 1) % 4 == 0:
            ranks.add((root - 1) // 4)
        if (root + 1) % 4 == 0:
            ranks.add((root + 1) // 4)
    for tag_dim, tag_rank in _exceptional_dim_rank.values():
        if tag_dim == dim:
            ranks.add(tag_rank)
    return sorted(ranks)


def complex_type_candidates(dimension, rank=None):
    """
    Dynkin types of a complex simple Lie algebra of the given size.
    """
    dim = int(dimension)
    if dim is None or dim < 3:
        return ()
    if rank is not None:
        rk = int(rank)
        return () if rk is None or rk < 1 else tuple(_types_at_rank(dim, rk))
    out = []
    for rk in _admissible_ranks(dim):
        for tag in _types_at_rank(dim, rk):
            if tag not in out:
                out.append(tag)
    return tuple(out)


def _root_data(tag):
    fam, rk = _parse_type(tag)
    if fam is None:
        return None
    if rk is None:
        return _exceptional_root_data[fam]
    if _family_dim(fam, rk) is None:
        return None
    if fam == "A":
        return (rk * (rk + 1), 0, 1, rk + 1)
    if fam == "B":
        return (2 * rk * (rk - 1), 2 * rk, 2, 2 * rk - 1)
    if fam == "C":
        return (2 * rk, 2 * rk * (rk - 1), 2, rk + 1)
    return (2 * rk * (rk - 1), 0, 1, 2 * rk - 2)


def killing_root_lengths(tag):
    """
    Squared root lengths of a complex simple Lie algebra, Killing normalized.
    """
    data = _root_data(tag)
    if data is None:
        return ()
    long_count, short_count, ratio, coxeter = data
    out = []
    if short_count:
        out.append((Fraction(1, ratio * coxeter), short_count))
    if long_count:
        out.append((Fraction(1, coxeter), long_count))
    return tuple(out)


def complex_type_from_root_lengths(dimension, rank, lengths):
    """
    Dynkin type matching a measured multiset of squared root lengths.
    """
    try:
        target = tuple(
            sorted((Fraction(value), int(count)) for value, count in lengths)
        )
    except (TypeError, ValueError):
        target = None
    dim = int(dimension)
    rk = int(rank)
    if not target or dim is None or rk is None:
        return (), False
    out = [
        tag
        for tag in complex_type_candidates(dim, rk)
        if killing_root_lengths(tag) == target
    ]
    if not out and dim % 2 == 0 and rk % 2 == 0:
        if all(count % 2 == 0 for _, count in target):
            halved = tuple((value, count // 2) for value, count in target)
            for tag in complex_type_candidates(dim // 2, rk // 2):
                if killing_root_lengths(tag) == halved:
                    out.append(f"{tag}+{tag}")
    return tuple(out), len(out) == 1


@dataclass(frozen=True, slots=True)
class RealFormRecord:
    """
    One real form of a simple Lie algebra.
    """

    label: str
    aliases: tuple
    complex_type: str
    dimension: int
    rank: int
    signature: int
    maximal_compact_dimension: int
    is_realification: bool


def _classical_entries(fam, rk):
    if fam == "A":
        n = rk + 1
        out = [(f"su({n})", n * n - 1), (f"sl({n},R)", n * (n - 1) // 2)]
        if n >= 4 and n % 2 == 0:
            out.append((f"su*({n})", (n // 2) * (n + 1)))
        for q in range(1, n // 2 + 1):
            p = n - q
            out.append((f"su({p},{q})", p * p + q * q - 1))
        return out
    if fam in {"B", "D"}:
        m = 2 * rk + 1 if fam == "B" else 2 * rk
        out = [(f"so({m})", m * (m - 1) // 2)]
        for q in range(1, m // 2 + 1):
            p = m - q
            out.append((f"so({p},{q})", p * (p - 1) // 2 + q * (q - 1) // 2))
        if fam == "D":
            out.append((f"so*({m})", rk * rk))
        return out
    out = [(f"sp({rk})", rk * (2 * rk + 1)), (f"sp({2 * rk},R)", rk * rk)]
    for q in range(1, rk // 2 + 1):
        p = rk - q
        out.append((f"sp({p},{q})", p * (2 * p + 1) + q * (2 * q + 1)))
    return out


def _all_names(label, extra):
    for name in [label] + list(extra):
        for alt in _extra_names.get(name, ()):
            if alt != label and alt not in extra:
                extra.append(alt)
    return tuple(extra)


def _classical_records(tag, dim, rk, entries):
    order = []
    slots = {}
    for label, compact_dim in entries:
        merge_into = _same_algebra.get(label, None)
        if merge_into in slots:
            slots[merge_into][1].append(label)
            continue
        slots[label] = [compact_dim, []]
        order.append(label)
    records = []
    for label in order:
        compact_dim, extra = slots[label]
        records.append(
            RealFormRecord(
                label=label,
                aliases=_all_names(label, extra),
                complex_type=tag,
                dimension=dim,
                rank=rk,
                signature=dim - 2 * compact_dim,
                maximal_compact_dimension=compact_dim,
                is_realification=False,
            )
        )
    return tuple(records)


def _exceptional_records(tag, dim, rk):
    return tuple(
        RealFormRecord(
            label=f"{tag}({signature})",
            aliases=aliases,
            complex_type=tag,
            dimension=dim,
            rank=rk,
            signature=signature,
            maximal_compact_dimension=(dim - signature) // 2,
            is_realification=False,
        )
        for signature, aliases in _exceptional_forms[tag]
    )


def real_forms_of_complex_type(tag):
    """
    Every real form whose complexification has the given Dynkin type.
    """
    if tag in _forms_cache:
        return _forms_cache[tag]
    fam, rk = _parse_type(tag)
    size = _type_dimension_rank(tag)
    if fam is None or size is None:
        return ()
    dim, type_rank = size
    if rk is None:
        records = _exceptional_records(tag, dim, type_rank)
    else:
        records = _classical_records(tag, dim, type_rank, _classical_entries(fam, rk))
    _forms_cache[tag] = records
    return records


def complex_algebra_label(tag):
    """Standard name of the complex simple Lie algebra of type tag, e.g. 'sl(3,C)'."""
    fam, rk = _parse_type(tag)
    if fam is None:
        return None
    if rk is None:
        return f"{tag}(C)"
    if _family_dim(fam, rk) is None:
        return None
    if fam == "A":
        return f"sl({rk + 1},C)"
    if fam == "B":
        return f"so({2 * rk + 1},C)"
    if fam == "C":
        return f"sp({2 * rk},C)"
    return f"so({2 * rk},C)"


def _realifications(dim, rk):
    if dim % 2 or (rk is not None and rk % 2):
        return ()
    half_rank = None if rk is None else rk // 2
    records = []
    for tag in complex_type_candidates(dim // 2, half_rank):
        base = complex_algebra_label(tag)
        if base is None:
            continue
        base_dim, base_rank = _type_dimension_rank(tag)
        label = f"{base}_R"
        records.append(
            RealFormRecord(
                label=label,
                aliases=_extra_names.get(label, ()),
                complex_type=f"{tag}+{tag}",
                dimension=2 * base_dim,
                rank=2 * base_rank,
                signature=0,
                maximal_compact_dimension=base_dim,
                is_realification=True,
            )
        )
    return tuple(records)


def real_form_candidates(dimension, rank=None, signature=None, absolutely_simple=None):
    """
    Real forms of a simple Lie algebra matching the given invariants.
    """
    dim = int(dimension)
    if dim is None:
        return (), False
    rk = None if rank is None else int(rank)
    if rank is not None and rk is None:
        return (), False
    sig = None if signature is None else int(signature)
    if signature is not None and sig is None:
        return (), False
    records = []
    if absolutely_simple is not False:
        for tag in complex_type_candidates(dim, rk):
            records += [
                record
                for record in real_forms_of_complex_type(tag)
                if sig is None or record.signature == sig
            ]
    if absolutely_simple is not True and sig in (None, 0):
        records.extend(_realifications(dim, rk))
    return tuple(records), len(records) == 1


def records_at_dimension(dimension):
    """Every real form of the given dimension, over all admissible ranks."""
    dim = int(dimension)
    if dim is None:
        return ()
    records = []
    for tag in complex_type_candidates(dim):
        records.extend(real_forms_of_complex_type(tag))
    records.extend(_realifications(dim, None))
    return tuple(records)


def real_form_by_label(label, dimension):
    for record in records_at_dimension(dimension):
        if label == record.label or label in record.aliases:
            return record
    return None


@dataclass(frozen=True, slots=True)
class RealFormReport:
    """
    Outcome of a real-form identification.
    """

    candidates: tuple
    certain: bool
    dimension: int
    rank: int | None
    signature: int | None = None
    absolutely_simple: bool | None = None
    rank_is_probabilistic: bool = False
    source: str = "computed"
    complex_type: str | None = None
    lengths: tuple | None = None
    dual_coxeter_number: int | None = None

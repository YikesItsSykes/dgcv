from __future__ import annotations

import itertools
from numbers import Number
from typing import Any, Dict, Tuple

from ....._aux._backends._symbolic_router import _scalar_is_zero
from ....._aux._backends._types_and_constants import half, imag_unit
from ....._aux._vmf.vmf import vmf_lookup
from ....combinatorics.combinatorics import permSign

format_combinations = {
    ("standard", "standard"): "standard",
    ("complex", "complex"): "complex",
    ("standard", "complex"): "complex",
    ("complex", "standard"): "complex",
    ("real", "real"): "real",
    ("standard", "real"): "real",
    ("real", "standard"): "real",
    ("complex", "real"): "mixed",
    ("real", "complex"): "mixed",
}
format_filter = {
    "ss": "standard",
    "ch": "complex",
    "ca": "complex",
    "rr": "real",
    "ri": "real",
    "standard": "standard",
    "complex": "complex",
    "real": "real",
}


def _is_scalar_coeff_dict(d: Dict[Any, Any]) -> bool:
    return isinstance(d, dict) and (
        not d or (tuple() in d and all(k == tuple() for k in d))
    )


def _infer_variable_spaces_from_coeff_dict(coeff_dict: dict, seed: dict) -> dict:
    out = dict(seed) if isinstance(seed, dict) else {}
    for key in coeff_dict.keys():
        if not isinstance(key, tuple):
            continue
        kl = len(key)
        if kl == 0 or kl % 3 != 0:
            continue
        deg = kl // 3
        coord_ids = key[2 * deg :]
        for idx, cid in enumerate(coord_ids):
            if cid is None:
                atom = key[idx]
                try:
                    if None not in out:
                        out[None] = {atom: atom}
                    else:
                        out[None][atom] = atom
                except Exception:
                    raise TypeError(
                        "tensor_field_class recieved unsupported init data type. coeff dict keys must be tuples of hashable elements"
                    )
                continue

            if not isinstance(cid, str):
                continue
            if cid in out:
                continue
            info = vmf_lookup(cid, path=True, relatives=True, flattened_relatives=True)
            if info.get("type") != "coordinate":
                raise KeyError(
                    f"tensorField: coord_id '{cid}' is not registered in the VMF."
                )
            flat = info.get("flattened_relatives", None)
            if not isinstance(flat, tuple) or len(flat) == 0:
                raise KeyError(
                    f"tensorField: coord_id '{cid}' did not yield a usable variable space."
                )
            out[cid] = flat
    return out


def _missing_system_msg(system_label: str) -> str:
    return (
        f"tensor_field_class: coordinate system '{system_label}' is not available in cached `variable_spaces`. "
        "Re-initialize with `variable_spaces={...}`."
    )


def _format_combinator(formats, seed=None):
    fc = format_combinations
    ff = format_filter
    if len(formats) == 0:
        return seed if seed else "open"
    if seed:
        formatting = ff.get(seed, formats[0])
    else:
        formatting = formats[0]
    for format2 in formats:
        f1 = ff.get(formatting, "mixed")
        f2 = ff.get(format2, "mixed")
        formatting = fc.get((f1, f2), "mixed")
    return formatting


def _perm_parity(order):
    n = len(order)
    seen = [False] * n
    sign = 1
    for i in range(n):
        if seen[i]:
            continue
        j = i
        cycle_len = 0
        while not seen[j]:
            seen[j] = True
            j = order[j]
            cycle_len += 1
        if cycle_len % 2 == 0:
            sign = -sign
    return sign


def _sort_slot_key(slot):
    idx, valence, syslbl = slot
    idx_key = (0, int(idx), "") if isinstance(idx, Number) else (1, 0, str(idx))
    return (str(syslbl), idx_key, int(valence))


def _slot_format(idx, sys, _variable_dict):
    sys_data = _variable_dict.get(sys)
    if sys_data is None:
        return None
    if sys_data.get("type") != "complex":
        return "standard"
    b0, b1, b2 = sys_data["breaks"]
    return "complex" if idx < b1 else "real"


def _process_coeffs_dict_new(
    data: Dict[Tuple[Any, ...], Any],
    shape: str,
    variable_spaces=None,
    formatting: bool = False,
) -> Tuple[Dict[Tuple[Any, ...], Any], str]:
    if shape not in ("general", "symmetric", "skew", "all"):
        shape = "general"

    if not isinstance(data, dict):
        raise TypeError("`coeff_dict` must be a dictionary.")

    def _parse_key(k, find_format=True, seed=None, inference_dict=None):
        deg = len(k) // 3
        if find_format:
            kf = _profile_key_full_inference(
                k, _variable_dict=vst, inference_dict=inference_dict
            )
            out_format = _format_combinator(kf, seed=seed)
            return (deg, k, out_format)
        return (deg, k)

    def _slotify(k, deg):
        return tuple(zip(k[:deg], k[deg : 2 * deg], k[2 * deg :]))

    def _unslotify(slots):
        idxs, valence_tuple, syslbls = zip(*slots)
        return tuple(idxs) + tuple(valence_tuple) + tuple(syslbls)

    canon: Dict[Tuple[Any, ...], Any] = {}
    vst = _variable_spaces_types_algo(variable_spaces) if formatting else None
    merged = False
    key_cache = {}

    def _slot_key(slot):
        out = key_cache.get(slot)
        if out is None:
            out = _sort_slot_key(slot)
            key_cache[slot] = out
        return out
    if formatting:
        out_format = "open"
    for k, v in data.items():
        if _scalar_is_zero(v):
            continue
        if formatting is False or out_format == "mixed":
            deg, kk = _parse_key(k, find_format=False, inference_dict=variable_spaces)
        else:
            deg, kk, out_format = _parse_key(
                k, seed=out_format, inference_dict=variable_spaces
            )
        if deg <= 1 or shape == "general":
            if kk in canon:
                canon[kk] = canon[kk] + v
                merged = True
            else:
                canon[kk] = v
            continue

        slots = _slotify(kk, deg)

        if shape == "skew" and len(set(slots)) < len(slots):
            continue

        if shape == "symmetric":
            sorted_slots = tuple(sorted(slots, key=_sort_slot_key))
            nk = _unslotify(sorted_slots)
            vv = v
        else:
            keys = [_slot_key(s) for s in slots]
            order = sorted(range(len(slots)), key=keys.__getitem__)
            sign = _perm_parity(order)
            nk = _unslotify(tuple(slots[i] for i in order))
            vv = sign * v

        if nk in canon:
            canon[nk] = canon[nk] + vv
            merged = True
        else:
            canon[nk] = vv

    if merged:
        canon = {k: v for k, v in canon.items() if not _scalar_is_zero(v)}
    if not canon:
        if formatting:
            return {tuple(): 0}, "all", "open"
        return {tuple(): 0}, "all"

    if formatting:
        return canon, shape, out_format
    return canon, shape


def _expand_special_to_general(
    data: Dict[Tuple[Any, ...], Any], shape: str
) -> Dict[Tuple[Any, ...], Any]:
    if shape not in ("general", "symmetric", "skew", "all"):
        raise ValueError("Invalid data_shape.")

    if not isinstance(data, dict):
        raise TypeError("`coeff_dict` must be a dictionary.")

    if shape in ("general", "all"):
        return {k: v for k, v in data.items() if not _scalar_is_zero(v)}

    nz = {k: v for k, v in data.items() if not _scalar_is_zero(v)}
    if not nz:
        return {}

    out: Dict[Tuple[Any, ...], Any] = {}

    first_key = next(iter(nz))
    if not isinstance(first_key, tuple) or len(first_key) % 3 != 0:
        raise ValueError("Invalid coeff_dict key format.")

    def _slotify(k):
        deg_l = len(k) // 3
        idxs = k[:deg_l]
        valence_tuple = k[deg_l : 2 * deg_l]
        syslbls = k[2 * deg_l :]
        return tuple((idxs[i], valence_tuple[i], syslbls[i]) for i in range(deg_l))

    def unslotify(slots):
        idxs = tuple(s[0] for s in slots)
        valence_tuple = tuple(s[1] for s in slots)
        syslbls = tuple(s[2] for s in slots)
        return idxs + valence_tuple + syslbls

    for k, v in nz.items():
        slots = _slotify(k)
        deg = len(k) // 3

        for perm in itertools.permutations(range(deg)):
            perm_slots = tuple(slots[i] for i in perm)
            nk = unslotify(perm_slots)

            if shape == "symmetric":
                vv = v
            else:
                vv = permSign(perm) * v

            out[nk] = out.get(nk, 0) + vv

    return {k: v for k, v in out.items() if not _scalar_is_zero(v)}


def _variable_spaces_types_algo(vs={}):
    out = {}
    if not isinstance(vs, dict):
        vs = {}
    for system, coordinates in vs.items():
        info = vmf_lookup(system)
        sys_type = info.get("sub_type", "standard")
        if sys_type == "complex":
            L = len(coordinates)
            fourth = L // 4
            breaks = [fourth, 2 * fourth, 3 * fourth]
            out[system] = {"type": sys_type, "breaks": breaks}
        else:
            out[system] = {"type": sys_type}
    return out


def _slot_allowed(plan, syslbl, idx):
    if not plan:
        return True

    if syslbl in plan.get("skip_systems", ()):
        return False

    si = plan.get("skip_indices", {}).get(syslbl)
    if si and idx in si:
        return False

    scoped = plan.get("systems", {}).get(syslbl, None)
    if scoped is None:
        return True
    return idx in scoped


def _to_real_algo(plan=None, cd={}, vst={}):
    h = half()
    new_dict = {}

    for k_seed, v_seed in cd.items():
        current_contribution = {k_seed: v_seed}
        degree = len(k_seed) // 3
        valences = k_seed[degree : 2 * degree]
        systems = k_seed[2 * degree :]

        for idx in range(degree):
            new_contribution = {}

            for k, v in current_contribution.items():
                idxs = k[:degree]

                br1, br2, br3 = (
                    idxs[idx],
                    valences[idx],
                    systems[idx],
                )  # break point data
                sys_data = vst[br3]

                if sys_data["type"] != "complex":
                    new_contribution[k] = new_contribution.get(k, 0) + v
                    continue

                if br1 >= sys_data["breaks"][1]:
                    new_contribution[k] = new_contribution.get(k, 0) + v
                    continue

                if not _slot_allowed(plan, br3, br1):
                    new_contribution[k] = new_contribution.get(k, 0) + v
                    continue

                lead = list(idxs[:idx])
                tail = list(idxs[idx + 1 :])
                term_tail = list(valences) + list(systems)

                if br1 >= sys_data["breaks"][0]:  # antiholomorphic variable
                    real_idx = br1 + sys_data["breaks"][0]
                    im_idx = br1 + sys_data["breaks"][1]

                    if br2 == 1:  # contravariant slot
                        real_term = lead + [real_idx] + tail + term_tail
                        im_term = lead + [im_idx] + tail + term_tail
                        new_contribution[tuple(real_term)] = (
                            new_contribution.get(tuple(real_term), 0) + h * v
                        )
                        new_contribution[tuple(im_term)] = (
                            new_contribution.get(tuple(im_term), 0)
                            + h * imag_unit() * v
                        )
                    else:  # covariant slot
                        real_term = lead + [real_idx] + tail + term_tail
                        im_term = lead + [im_idx] + tail + term_tail
                        new_contribution[tuple(real_term)] = (
                            new_contribution.get(tuple(real_term), 0) + v
                        )
                        new_contribution[tuple(im_term)] = (
                            new_contribution.get(tuple(im_term), 0) - imag_unit() * v
                        )

                else:  # holomorphic variable
                    real_idx = br1 + sys_data["breaks"][1]
                    im_idx = br1 + sys_data["breaks"][2]

                    if br2 == 1:  # contravariant slot
                        real_term = lead + [real_idx] + tail + term_tail
                        im_term = lead + [im_idx] + tail + term_tail
                        new_contribution[tuple(real_term)] = (
                            new_contribution.get(tuple(real_term), 0) + h * v
                        )
                        new_contribution[tuple(im_term)] = (
                            new_contribution.get(tuple(im_term), 0)
                            - h * imag_unit() * v
                        )
                    else:  # covariant slot
                        real_term = lead + [real_idx] + tail + term_tail
                        im_term = lead + [im_idx] + tail + term_tail
                        new_contribution[tuple(real_term)] = (
                            new_contribution.get(tuple(real_term), 0) + v
                        )
                        new_contribution[tuple(im_term)] = (
                            new_contribution.get(tuple(im_term), 0) + imag_unit() * v
                        )

            current_contribution = new_contribution

        for key, val in current_contribution.items():
            if not _scalar_is_zero(val):
                new_dict[key] = new_dict.get(key, 0) + val

    if not new_dict:
        new_dict = {tuple(): 0}

    return new_dict


def _to_complex_algo(plan=None, cd={}, vst={}):
    h = half()
    new_dict = {}

    for k_seed, v_seed in cd.items():
        current_contribution = {k_seed: v_seed}
        degree = len(k_seed) // 3
        valences = k_seed[degree : 2 * degree]
        systems = k_seed[2 * degree :]

        for idx in range(degree):
            new_contribution = {}

            for k, v in current_contribution.items():
                idxs = k[:degree]

                vb = valences[idx]
                syslbl = systems[idx]
                sys_data = vst[syslbl]

                if sys_data["type"] != "complex":
                    new_contribution[k] = new_contribution.get(k, 0) + v
                    continue

                # determine which real/imag block we are in
                br0, br1, br2 = sys_data["breaks"]

                if idxs[idx] < br1:
                    new_contribution[k] = (
                        new_contribution.get(k, 0) + v
                    )  # holomorphic / antiholomorphic already
                    continue

                if not _slot_allowed(plan, syslbl, idxs[idx]):
                    new_contribution[k] = new_contribution.get(k, 0) + v
                    continue

                lead = list(idxs[:idx])
                tail = list(idxs[idx + 1 :])
                rest = list(valences) + list(systems)

                j = idxs[idx]

                # real variable
                if j < br2:
                    holo_idx = j - br1
                    anti_idx = holo_idx + br0

                    if vb == 1:  # contravariant
                        holo_term = lead + [holo_idx] + tail + rest
                        anti_term = lead + [anti_idx] + tail + rest
                        new_contribution[tuple(holo_term)] = (
                            new_contribution.get(tuple(holo_term), 0) + v
                        )
                        new_contribution[tuple(anti_term)] = (
                            new_contribution.get(tuple(anti_term), 0) + v
                        )
                    else:
                        holo_term = lead + [holo_idx] + tail + rest
                        anti_term = lead + [anti_idx] + tail + rest
                        new_contribution[tuple(holo_term)] = (
                            new_contribution.get(tuple(holo_term), 0) + v * h
                        )
                        new_contribution[tuple(anti_term)] = (
                            new_contribution.get(tuple(anti_term), 0) + v * h
                        )

                # imaginary variable
                else:
                    holo_idx = j - br2
                    anti_idx = holo_idx + br0

                    if vb == 1:  # contravariant
                        holo_term = lead + [holo_idx] + tail + rest
                        anti_term = lead + [anti_idx] + tail + rest
                        new_contribution[tuple(holo_term)] = (
                            new_contribution.get(tuple(holo_term), 0) + imag_unit() * v
                        )
                        new_contribution[tuple(anti_term)] = (
                            new_contribution.get(tuple(anti_term), 0) - imag_unit() * v
                        )
                    else:
                        holo_term = lead + [holo_idx] + tail + rest
                        anti_term = lead + [anti_idx] + tail + rest
                        new_contribution[tuple(holo_term)] = (
                            new_contribution.get(tuple(holo_term), 0)
                            - imag_unit() * v * h
                        )
                        new_contribution[tuple(anti_term)] = (
                            new_contribution.get(tuple(anti_term), 0)
                            + imag_unit() * v * h
                        )

            current_contribution = new_contribution

        for key, val in current_contribution.items():
            if not _scalar_is_zero(val):
                new_dict[key] = new_dict.get(key, 0) + val

    if not new_dict:
        new_dict = {tuple(): 0}

    return new_dict


def _profile_key_full_inference(k, _variable_dict=None, inference_dict={}):
    degree = len(k) // 3
    idxs = k[:degree]
    systems = k[2 * degree :]

    out = []
    if _variable_dict is None:
        _variable_dict = _variable_spaces_types_algo(inference_dict)

    for idx, sys in zip(idxs, systems):
        sys_data = _variable_dict.get(sys)
        if sys_data is None:
            raise KeyError(
                "At least one system label in a coeff_dict key is not registered in the VMF."
            )

        if sys_data is None or sys_data.get("type") != "complex":
            out.append("ss")
            continue

        b0, b1, b2 = sys_data["breaks"]

        if idx < b0:
            out.append("ch")
        elif idx < b1:
            out.append("ca")
        elif idx < b2:
            out.append("rr")
        else:
            out.append("ri")

    return tuple(out)

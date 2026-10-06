from __future__ import annotations

from fractions import Fraction
from math import lcm


def _exceptional_cartan_matrix(series_type, rank):
    if series_type == "G":
        return ((2, -3), (-1, 2))
    if series_type == "F":
        return ((2, -1, 0, 0), (-1, 2, -1, 0), (0, -2, 2, -1), (0, 0, -1, 2))
    edges = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (2, 4)]
    cartan = [[2 if i == j else 0 for j in range(rank)] for i in range(rank)]
    for a, b in edges:
        if a <= rank and b <= rank:
            cartan[a - 1][b - 1] = cartan[b - 1][a - 1] = -1
    return tuple(map(tuple, cartan))


def _chevalley_structure_data(cartan):
    rank = len(cartan)
    d = [Fraction(1)] + [None] * (rank - 1)
    while None in d:
        for i in range(rank):
            for j in range(rank):
                if d[i] is not None and d[j] is None and cartan[i][j] != 0:
                    d[j] = d[i] * cartan[i][j] / cartan[j][i]
    scale = lcm(*(x.denominator for x in d))
    d = [int(x * scale) for x in d]

    def ip(a, b):
        return sum(
            a[i] * b[j] * d[i] * cartan[i][j] for i in range(rank) for j in range(rank)
        )

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def neg(a):
        return tuple(-x for x in a)

    positive = [tuple(int(i == j) for j in range(rank)) for i in range(rank)]
    known = set(positive)
    layer = list(positive)
    while layer:
        new = []
        for r in layer:
            for i in range(rank):
                p = 0
                while tuple(r[k] - (p + 1) * (k == i) for k in range(rank)) in known:
                    p += 1
                if p - sum(r[j] * cartan[i][j] for j in range(rank)) > 0:
                    s = tuple(r[k] + (k == i) for k in range(rank))
                    if s not in known:
                        known.add(s)
                        new.append(s)
        positive += new
        layer = new
    positive.sort(key=lambda r: (sum(r), neg(r)))
    order = {r: n for n, r in enumerate(positive)}
    roots = known | {neg(r) for r in positive}

    def string_length(r, s):
        p = 0
        while tuple(y - (p + 1) * x for x, y in zip(r, s)) in roots:
            p += 1
        return p + 1

    extraspecial = {}
    for xi in positive:
        for a in positive:
            b = sub(xi, a)
            if b in order and order[a] < order[b]:
                extraspecial[xi] = (a, b)
                break

    signs = {}

    def sign(r, s):
        if (r, s) in signs:
            return signs[(r, s)]
        t = add(r, s)
        if sum(r) > 0 and sum(s) > 0:
            a, b = extraspecial[t]
            if (r, s) == (a, b):
                val = 1
            elif (r, s) == (b, a):
                val = -1
            else:
                terms = []
                for u, v, k in ((s, r, 1), (r, s, -1)):
                    diff = sub(u, a)
                    if diff in roots:
                        terms.append((k * N(u, neg(a)) * N(v, neg(b)), ip(diff, diff)))
                    else:
                        terms.append((0, 1))
                total = terms[0][0] * terms[1][1] + terms[1][0] * terms[0][1]
                val = (1 if total > 0 else -1) * sign(a, b)
        elif sum(r) < 0 and sum(s) < 0:
            val = -sign(neg(r), neg(s))
        else:
            u = neg(t)
            if (sum(u) > 0) == (sum(r) > 0):
                val = sign(u, r)
            else:
                val = sign(s, u)
        signs[(r, s)] = val
        return val

    def N(r, s):
        if add(r, s) not in roots:
            return 0
        return sign(r, s) * string_length(r, s)

    all_roots = positive + [neg(r) for r in positive]
    index = {r: rank + n for n, r in enumerate(all_roots)}
    structure_data = {}
    for r in all_roots:
        for i in range(rank):
            c = sum(r[j] * cartan[i][j] for j in range(rank))
            if c != 0:
                structure_data[(i, index[r], index[r])] = c
    for r in positive:
        norm = ip(r, r)
        for i in range(rank):
            if r[i] != 0:
                structure_data[(index[r], index[neg(r)], i)] = r[i] * 2 * d[i] // norm
    for r in all_roots:
        for s in all_roots:
            if index[r] < index[s] and add(r, s) in roots:
                structure_data[(index[r], index[s], index[add(r, s)])] = N(r, s)
    weights = [(0,) * rank] * rank + all_roots
    return structure_data, [tuple(w[i] for w in weights) for i in range(rank)]

"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.pde

module: dgcv.eds.pde._report

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from ..._aux._backends._display import latex as _routed_latex
from ..._aux._vmf._safeguards import retrieve_passkey
from ...core.base import dgcv_class

_passkey = retrieve_passkey()


class _Report(dgcv_class):
    _dgcv_class_check = _passkey
    _dgcv_category = "eds_report"
    _rows = ()

    def _row_values(self):
        return [(name, getattr(self, attr)) for name, attr in self._rows]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{attr}={value!r}"
                for (_, attr), (_, value) in zip(self._rows, self._row_values())
            )
            + ")"
        )

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _tex(value):
        if isinstance(value, bool):
            return r"\text{" + ("yes" if value else "no") + "}"
        if isinstance(value, str):
            return r"\text{" + value + "}"
        if isinstance(value, (list, tuple)):
            return r"\left(" + ", ".join(_Report._tex(v) for v in value) + r"\right)"
        if isinstance(value, dict):
            return (
                r"\left\{"
                + r",\ ".join(
                    _routed_latex(k) + r"\mapsto " + _Report._tex(v)
                    for k, v in value.items()
                )
                + r"\right\}"
            )
        if value is None:
            return r"\text{--}"
        return _routed_latex(value)

    def _latex(self, printer=None, raw=True, **kwargs):
        rows = r" \\ ".join(
            name + " &= " + self._tex(value) for name, value in self._row_values()
        )
        return r"\begin{aligned} " + rows + r" \end{aligned}"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

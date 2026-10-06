from __future__ import annotations

import numbers
from typing import List, Optional

from ..._aux._backends._symbolic_router import _scalar_is_zero, clear_denominators
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import retrieve_passkey
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ..algebras import algebra_class


class simple_Lie_algebra(algebra_class):
    def __init__(
        self,
        structure_data,
        grading=None,
        base_field=None,
        process_matrix_rep=False,
        preferred_representation=None,
        _label=None,
        _basis_labels=None,
        _calledFromCreator=None,
        _callLock=None,
        _print_warning=None,
        _child_print_warning=None,
        _exclude_from_VMF=None,
        _simple_data=None,
        _basis_labels_parent=None,
    ):
        if _calledFromCreator != retrieve_passkey():
            raise RuntimeError(
                "`simple_Lie_algebra` class instances can only be initialized by internal `dgcv` functions indirectly. To instantiate a simple Lie algebra, use dgcv `creator` functions"
            ) from None
        t_message = "True by construction: instantiated from `simple_Lie_algebra` class constructor"
        super().__init__(
            structure_data,
            grading=grading,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            _label=_label,
            _basis_labels=_basis_labels,
            _calledFromCreator=_calledFromCreator,
            _callLock=_callLock,
            _print_warning=_print_warning,
            _child_print_warning=_child_print_warning,
            _exclude_from_VMF=_exclude_from_VMF,
            _basis_labels_parent=_basis_labels_parent,
            _markers={
                "simple": True,
                "_educed_properties": {
                    "is_simple": t_message,
                    "is_Lie_algebra": t_message,
                    "is_semisimple": t_message,
                    "special_type": "simple",
                    "is_skew": t_message,
                    "satisfies_Jacobi_ID": t_message,
                },
            },
        )

        self.roots = []
        self.simpleRoots = []
        self.rootSpaces = {(0,) * len(self.grading): []}

        def isSimpleRoot(vec):
            if vec.count(0) == len(vec) - 1 and vec.count(1) == 1:
                return True
            else:
                return False

        for elem in self.basis:
            root = tuple(elem.check_element_weight())
            if root in self.rootSpaces:
                self.rootSpaces[root].append(elem)
            else:
                self.rootSpaces[root] = [elem]
                self.roots.append(root)
                if isSimpleRoot(root):
                    self.simpleRoots.append(root)
        self.simpleRootSpaces = {
            root: self.rootSpaces[root] for root in self.simpleRoots
        }
        seriesLabel, rank = _simple_data["type"]
        self.rank = rank
        self.Cartan_subalgebra = self.basis[0:rank]
        self.simpleLieType = f"{seriesLabel}{rank}"  # example: "A3", "D4", ... etch

    def root_space_summary(self):
        def pluralize(idx):
            if idx != 1:
                return "s"
            else:
                return ""

        def rootString(idx):
            if idx == 1:
                return "(r_1)"
            if idx == 2:
                return "(r_1, r_2)"
            if idx == 3:
                return "(r_1, r_2, r_3)"
            else:
                return f"(r_1, ..., r_{idx})"

        print(
            f"This simple algebra {self.simpleLieType} has {self.rank} root{pluralize(self.rank)} {rootString(self.rank)}, which are dual to the Cartan subalgebra basis {self.Cartan_subalgebra}. These roots correspond to vertices in the Dynkin diagram as follows:\n"
        )

        if self.simpleLieType[0] == "D":
            n = self.rank
            if n == 2:
                print(
                    "Dynkin diagram for D2 is just two disconnected vertices corresponding to a direct sum of two u(2) copies."
                )
            else:
                lines = []
                horiz = "   "
                if n > 7:
                    mid_nodes = ["r_1 r_2", f"r_{n - 4}", f"r_{n - 3}", f"r_{n - 2}"]
                    latter_rules = [
                        " ◯───◯─" + " ┅ ─",
                        "─" * (len(mid_nodes[1])),
                        "─" * (len(mid_nodes[2])),
                        "─" * (len(mid_nodes[3])),
                        "",
                    ]
                    horiz += "◯".join(latter_rules)
                    top_labels = f"{' ' * 4}{mid_nodes[0]}{' ' * 3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
                    fork_pos = 16 + len(latter_rules[1]) + len(latter_rules[2])
                elif n > 1:
                    horiz += "───".join("◯" for _ in range(n - 2))
                    top_labels = "   " + " ".join([f"r_{i + 1}" for i in range(n - 2)])
                    horiz += "───◯"
                    fork_pos = 4 * (n - 2) - 1
                else:
                    horiz += "───".join("◯" for _ in range(n - 2))
                    top_labels = "   " + " ".join([f"r_{i + 1}" for i in range(n - 2)])
                    horiz += "───◯"
                    fork_pos = 4 * (n - 2) - 1

                top_labels += " " + f"r_{n - 1}"

                # Final node
                final_line = " " * fork_pos + "│"
                final_node = " " * fork_pos + f"◯ r_{n}"

                # bounding box
                width_bound = len(top_labels)
                title = "│" + self.simpleLieType.center(width_bound) + " │"
                border_top = "┌" + "─" * width_bound + "─┐"
                head_sep = "╞" + "═" * width_bound + "═╡"
                top_labels = "│" + top_labels + " │"
                horiz = "│" + horiz.ljust(width_bound) + " │"
                final_line = "│" + final_line.ljust(width_bound) + " │"
                final_node = "│" + final_node.ljust(width_bound) + " │"
                border_bottom = "└" + "─" * width_bound + "─┘"

                lines.append(border_top)
                lines.append(title)
                lines.append(head_sep)
                lines.append(top_labels)
                lines.append(horiz)
                lines.append(final_line)
                lines.append(final_node)
                lines.append(border_bottom)

                print("\n".join(lines))
        elif self.simpleLieType[0] == "B":
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n - 3}", f"r_{n - 2}", f"r_{n - 1}"]
                latter_rules = [
                    " ◯───◯─" + " ⋯ ─",
                    "─" * (len(mid_nodes[1])),
                    "─" * (len(mid_nodes[2])),
                    "═" * (len(mid_nodes[3]) - 2) + "═>",
                    "",
                ]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' ' * 4}{mid_nodes[0]}{' ' * 3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n - 1))
                top_labels = "   " + " ".join([f"r_{i + 1}" for i in range(n - 1)])
                horiz += "══>◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound = len(top_labels) + 1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌" + "─" * width_bound + "─┐"
            head_sep = "╞" + "═" * width_bound + "═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom = "└" + "─" * width_bound + "─┘"

            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))
        elif self.simpleLieType[0] == "C":
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n - 3}", f"r_{n - 2}", f"r_{n - 1}"]
                latter_rules = [
                    " ◯───◯─" + " ⋯ ─",
                    "─" * (len(mid_nodes[1])),
                    "─" * (len(mid_nodes[2])),
                    "═" * (len(mid_nodes[3]) - 2) + "<═",
                    "",
                ]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' ' * 4}{mid_nodes[0]}{' ' * 3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n - 1))
                top_labels = "   " + " ".join([f"r_{i + 1}" for i in range(n - 1)])
                horiz += "══>◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound = len(top_labels) + 1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌" + "─" * width_bound + "─┐"
            head_sep = "╞" + "═" * width_bound + "═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom = "└" + "─" * width_bound + "─┘"

            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))
        elif self.simpleLieType[0] == "A":
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n - 3}", f"r_{n - 2}", f"r_{n - 1}"]
                latter_rules = [
                    " ◯───◯─" + " ⋯ ─",
                    "─" * (len(mid_nodes[1])),
                    "─" * (len(mid_nodes[2])),
                    "─" * (len(mid_nodes[3])),
                    "",
                ]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' ' * 4}{mid_nodes[0]}{' ' * 3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n - 1))
                top_labels = "   " + " ".join([f"r_{i + 1}" for i in range(n - 1)])
                horiz += "───◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound = len(top_labels) + 1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌" + "─" * width_bound + "─┐"
            head_sep = "╞" + "═" * width_bound + "═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom = "└" + "─" * width_bound + "─┘"

            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))
        elif self.simpleLieType[0] in {"E", "F", "G"}:
            n = self.rank
            branch = False
            if self.simpleLieType[0] == "G":
                chain, links = [1, 2], ["<≡≡"]
            elif self.simpleLieType[0] == "F":
                chain, links = [1, 2, 3, 4], ["───", "══>", "───"]
            else:
                chain = [1] + list(range(3, n + 1))
                links, branch = ["───"] * (n - 2), True
            lines = []
            top_labels = "   " + " ".join(f"r_{i}" for i in chain)
            horiz = "   " + "◯".join([""] + links + [""])

            width_bound = len(top_labels) + 1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌" + "─" * width_bound + "─┐"
            head_sep = "╞" + "═" * width_bound + "═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom = "└" + "─" * width_bound + "─┘"

            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            if branch:
                lines.append("│" + (" " * 11 + "│").ljust(width_bound) + " │")
                lines.append("│" + (" " * 11 + "◯ r_2").ljust(width_bound) + " │")
            lines.append(border_bottom)

            print("\n".join(lines))

    def parabolic_grading(self, roots=None):
        if roots is None:
            roots = []
        if isinstance(roots, numbers.Integral):
            roots = [roots]
        elif not isinstance(roots, (list, tuple)):
            raise TypeError(
                f"The `roots` parameter in `simple_Lie_algebra.parabolic_grading(roots)` should be either `None`, an `int`, or a list of integers in the range (1,...,{self.rank}) representing indices of simple roots as enumerated in the algebras Dynkin diagram (see `simple_Lie_algebra.root_space_summary()` for a summary of this indexing)."
            ) from None
        gradingVector = [
            sum([self.grading[idx - 1][j] for idx in roots])
            for j in range(self.dimension)
        ]
        gradingVector = clear_denominators(gradingVector)
        return gradingVector

    def parabolic_subalgebra(
        self,
        roots: Optional[List[int]] = None,
        label: Optional[str] = None,
        basis_labels: Optional[str | List[str]] = None,
        register_in_vmf: Optional[bool] = False,
        return_created_object: bool = False,
        use_non_positive_weights: bool = False,
        format_as_subalgebra_class: bool = False,
        **kwargs,
    ):
        return_created_object = kwargs.get("return_created_obj", return_created_object)
        if roots is None:
            roots = []
        if isinstance(roots, numbers.Integral):
            roots = [roots]
        if not isinstance(roots, (list, tuple)) or not all(
            root - 1 in range(self.rank) for root in roots
        ):
            dgcv_warning(
                f"Unsuported `roots` parameter: The `roots` parameter in `simple_Lie_algebra.parabolic_subalgebra(roots)` should be either `None`, an `int`, or a list of integers in the range (1,...,{self.rank}) representing indices of simple roots as enumerated in the algebras Dynkin diagram (see `simple_Lie_algebra.root_space_summary()` for a summary of this indexing)."
            )
            return
        marked = set(roots)
        newGrading = [
            sum([self.grading[idx - 1][j] for idx in marked])
            for j in range(self.dimension)
        ]
        if format_as_subalgebra_class is True:
            parabolic = []
        subIndices, filtered_grading, new_dim, index_map = [], [], 0, dict()
        if not isinstance(use_non_positive_weights, bool):
            use_non_positive_weights = False
        # With H_i dual to simple roots, self.grading stores the simple-root coefficients n_i.
        # Sigma = marked nodes. Standard parabolic keeps Sigma-height ≥ 0; opposite keeps Sigma-height ≤ 0.
        sign = -1 if use_non_positive_weights else 1
        for count, weight in enumerate(newGrading):
            if sign * weight >= 0:
                index_map[count] = new_dim
                new_dim += 1
                if format_as_subalgebra_class is True:
                    parabolic.append(self.basis[count])
                subIndices.append(count)
                filtered_grading.append(weight)
        filtered_grading = clear_denominators(filtered_grading)

        def truncateBySubInd(li):
            return [li[j] for j in subIndices]

        def restrict_structure_data(data):
            new_data = dict()
            inner_shape = (new_dim, 1)
            for (i, j, k), v in data.items():
                if i in subIndices and j in subIndices:
                    if k in subIndices:
                        outer_key = (index_map[i], index_map[j])
                        if outer_key in new_data:
                            new_data[outer_key][index_map[k]] = v
                        else:
                            new_data[outer_key] = matrix_dgcv(
                                {index_map[k]: v}, shape=inner_shape
                            )
                    elif v is not None and _scalar_is_zero(v):
                        raise TypeError(
                            "The basis provided to the `simple_Lie_algebra.parabolic_subalgebra` method does not span a subalgebra. Could be likely a bug in `dgcv`."
                        )
            return array_dgcv(
                new_data,
                shape=(new_dim, new_dim),
                null_return=freeze_matrix(matrix_dgcv.zeros(new_dim, 1)),
            )

        if format_as_subalgebra_class is True:
            ignoredList = []
            if label is not None:
                ignoredList.append("label")
            if basis_labels is not None:
                ignoredList.append("basis_labels")
            if register_in_vmf is True:
                ignoredList.append("register_in_vmf")
            if len(ignoredList) == 1:
                dgcv_warning(
                    f"A parameter value was supplied for `{ignoredList[0]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so this parameter value was ignored. A subalgebra_class instance was returned instead."
                )
            elif len(ignoredList) == 2:
                dgcv_warning(
                    f"Parameter values were supplied for `{ignoredList[0]}` and `{ignoredList[1]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so these parameter values were ignored. A subalgebra_class instance was returned instead."
                )
            elif len(ignoredList) == 3:
                dgcv_warning(
                    f"Parameter values were supplied for `{ignoredList[0]}`, `{ignoredList[1]}`, and `{ignoredList[2]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so these parameter values were ignored. `A subalgebra_class instance was returned instead.`"
                )
            from ..subspaces.subalgebras import subalgebra_class

            return subalgebra_class(
                parabolic,
                self,
                grading=[filtered_grading],
                # _compressed_structure_data=structureData,
                # _internal_lock=retrieve_passkey(),
            )

        structureData = restrict_structure_data(self.structureDataDict)
        if register_in_vmf is True:
            if label is None:
                label = self.label + "_parabolic"
            if basis_labels is None:
                basis_labels = label
            elif (
                isinstance(basis_labels, (list, tuple))
                and not all(isinstance(elem, str) for elem in basis_labels)
            ) or not isinstance(basis_labels, str):
                raise TypeError(
                    "If supplying the optional parameter `basis_labels` to `simple_Lie_algebra.parabolic_subalgebra` then it should be either a string or list of strings"
                ) from None
        if not (register_in_vmf or return_created_object):
            register_in_vmf = False
            return_created_object = True
        from ..creators import createAlgebra

        return createAlgebra(
            structureData,
            label=label,
            basis_labels=basis_labels,
            grading=filtered_grading,
            return_created_object=return_created_object,
            forgo_vmf_registry=register_in_vmf is False,
        )

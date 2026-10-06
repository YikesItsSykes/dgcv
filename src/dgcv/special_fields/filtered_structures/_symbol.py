from __future__ import annotations

from ...core.base import dgcv_class
from ._brackets import _symbol_brackets
from ._core import _symbol_core
from ._export import _symbol_export
from ._generators import _symbol_generators
from ._generic import _symbol_generic_constants
from ._printing import _symbol_printing
from ._prolongation import _symbol_prolongation
from ._prolongation_stages import _symbol_prolongation_stages
from ._prolongation_step import _symbol_prolongation_step


class Tanaka_symbol(
    _symbol_core,
    _symbol_prolongation,
    _symbol_prolongation_step,
    _symbol_prolongation_stages,
    _symbol_brackets,
    _symbol_generators,
    _symbol_generic_constants,
    _symbol_export,
    _symbol_printing,
    dgcv_class,
):
    """
    Graded Lie algebra data and related structures prepared for Tanaka prolongation.

    Parameters
    ----------
    GLA : dgcv algebra class (algebra_class, subalgebra_class)
        Graded Lie algebra containing the negative part.
    nonnegParts : list or dict, default []
        Weighted homogeneous elements of nonnegative degree, or a dict keying
        them by degree. dict formatting is only intended for fast init with
        pre-validated data; use the list formatting if unsure.
    assume_FGLA : bool, default False
        Permit assuming the negative part is fundamental (i.e., generated
        by -1 component). Superseded by the generator computation that
        `precompute_generators` performs by default, which validates the
        assumption and is overridden by it when the symbol is not fundamental.
    subspace : subalgebra, optional
        Negative part to use. Defaults to the negative part of `GLA`.
    distinguished_subspaces : list of list, optional
        Subspaces the prolongation must preserve.
    prolongation_label_prefix : str, optional
        Prefix for labels of computed prolongation levels.
    assume_linear_independence : bool, default False
        Skip basis extraction on distinguished subspaces. Only set True if
        supplied distinguished subspace element are known to be linearly
        independant
    assume_NNP_linear_indep : bool, default False
        Skip basis extraction on `nonnegParts`.
    index_threshold : int, optional
        Lowest degree the level indexing recognizes.
    precompute_generators : bool, default True
        Compute a generating set of the negative part at initialization and
        test the derivation rule only on pairs (generator, basis element),
        which suffices for a derivation of a generated algebra. This is the
        validated form of `assume_FGLA` and applies to symbols whose
        generators sit in several weights. Set False to test every pair of
        negative basis elements instead.
    compress_equation_systems : bool or None, default None
        Additionally eliminate the unknowns for the images of generated
        elements: only the images of the generators are solved for, the rest
        follow from the derivation rule. Implies `precompute_generators`.
        Fastest on symbols with constant structure constants; on parametric
        symbols the equations become quadratic in the structure constants.
        None selects it exactly when generators are precomputed and the
        symbol has no parameters; an explicit bool is honored.

    Methods
    -------
    prolong
        Compute prolongation levels, optionally returning a new symbol.
    summary
        Display the levels and their labels.
    export_algebra_data
        Return structure data for abstract algebra isomorphic to the symbol
        with its computed prolongation levels.
    """

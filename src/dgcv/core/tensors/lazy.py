from ..._aux._vmf._safeguards import retrieve_passkey
from ..dgcv_core.spaces.spaces import card_root
from .promotion import _promotion_targets


def _lazy_class(base):
    class lazy_tensorProduct(base):
        _span_class = base

        def __init__(self, thunk, card, weight, grading=None, _hom_id=None, _hom_decomp=None):
            self._thunk = thunk
            self._materialized = None
            targets = _promotion_targets([card], False)
            vs_card = card_root(card) if card in targets else card
            degree = weight + 2
            self.shape = "general" if degree >= 2 else "all"
            self.vs_id = (vs_card,)
            self.vector_space = vs_card.space
            self.max_degree = degree
            self.min_degree = degree
            self.homogeneous = True
            self._card_map = None
            self._homogeneous_dicts = None
            self._weights = None
            self._homogeneous_components = None
            self._free_symbols = None
            self._leading_valence = None
            self._trailing_valence = None
            self._dgcv_class_check = retrieve_passkey()
            self._dgcv_category = "tensorProduct"
            self._terms = None
            self._properties = {"_weight": (None if grading is None else tuple(grading), weight)}
            if _hom_id:
                self._properties["_hom_id"] = _hom_id
            if _hom_decomp:
                self._properties["_hom_decomp"] = _hom_decomp

        @property
        def coeff_dict(self):
            if self._materialized is None:
                eager = self._thunk()
                self._materialized = eager.coeff_dict
                self.vs_id = eager.vs_id
                self.vector_space = eager.vector_space
                self.max_degree = eager.max_degree
                self.min_degree = eager.min_degree
                self.homogeneous = eager.homogeneous
                self.shape = eager.shape
                self._thunk = None
            return self._materialized

        @coeff_dict.setter
        def coeff_dict(self, value):
            self._materialized = value
            self._thunk = None

        @property
        def is_materialized(self):
            return self._materialized is not None

        def __getstate__(self):
            self.coeff_dict
            state = dict(self.__dict__)
            state["_thunk"] = None
            return state

    lazy_tensorProduct.__module__ = base.__module__
    lazy_tensorProduct.__qualname__ = "lazy_tensorProduct"
    return lazy_tensorProduct

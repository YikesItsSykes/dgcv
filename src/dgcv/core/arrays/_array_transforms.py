from ..._aux._backends._symbolic_router import get_free_symbols, simplify


class _array_transforms:
    @property
    def free_symbols(self):
        out = set()
        for v in self._data.values():
            if v is None:
                continue
            fs = get_free_symbols(v)
            if fs is not None:
                out |= set(fs)
        return out

    @property
    def __dgcv_zero_obstr__(self):
        eqns = [v for v in self._data.values() if v is not None]
        return eqns, list(self.free_symbols)

    def apply(self, func, *, in_place=False, skip_none=True, default=None, **kwargs):
        if default is None:
            default = self.null_return
        if in_place:
            target = self
        else:
            target = self.__class__.__new__(self.__class__)
            target._data = {}
            target._data_unspooled_cache = None
            target.shape = tuple(self.shape)
            target.ndim = self.ndim
            target.null_return = self.null_return

            if hasattr(self, "_dgcv_class_check"):
                target._dgcv_class_check = self._dgcv_class_check
            if hasattr(self, "_dgcv_category"):
                target._dgcv_category = self._dgcv_category
            if hasattr(self, "_dgcv_categories"):
                target._dgcv_categories = set(self._dgcv_categories)
        if skip_none:
            for k, v in self._data.items():
                if v is not None:
                    target._data[k] = func(v, **kwargs)
        else:
            n = 1
            for s in self.shape:
                n *= s
            for k in range(n):
                v = self._data.get(k, default)
                target._data[k] = func(v, **kwargs)

        return target

    def __dgcv_apply__(self, func, **kwargs):
        return self.apply(func, **kwargs)

    def subs(self, *args, **kwargs):
        def f(x):
            if x is None:
                return None
            m = getattr(x, "subs", None)
            return m(*args, **kwargs) if callable(m) else x

        return self.apply(f, in_place=False, skip_none=True)

    def __dgcv_simplify__(self, *args, **kwargs):
        return self.apply(simplify, in_place=False, skip_none=True, **kwargs)

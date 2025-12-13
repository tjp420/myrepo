"""Stub for core.lazy_model_loader."""


def load_model(name: str, *args, **kwargs):
    return None


def get_lazy_model_loader():
    class Loader:
        def load(self, name, *a, **k):
            return load_model(name, *a, **k)

    return Loader()

import importlib


def load_class(fullname):
    package, name = fullname.rsplit('.', maxsplit=1)
    package = importlib.import_module(package)
    return getattr(package, name)

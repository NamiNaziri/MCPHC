import gzip
import importlib
import os
import pickle
import sys
from types import ModuleType
import inspect


# Get dependencies local to a specified root in import order
def find_dependencies(module, root):
    def find(module, seen):
        seen.add(module)
        # get around bug in inspect.getmembers (apparently fixed in python 3.11)
        if hasattr(module, "__bases__") and module.__bases__ is not Iterable:
            return [module]
        all = [inspect.getmodule(m[1]) for m in inspect.getmembers(module)]
        accepted = [
            m for m in all if hasattr(m, "__file__") and m.__file__.startswith(root)
        ]
        return [x for m in accepted if m not in seen for x in find(m, seen)] + [module]

    return find(module, set())


def dump(sourceable, root=os.path.abspath(os.getcwd())):
    # Create an associative list of names to module source code
    modules = find_dependencies(importlib.import_module(sourceable.__module__), root)
    sources = [(module.__name__, inspect.getsource(module)) for module in modules]
    return {
        "sources": sources,
        "sourceable": gzip.compress(pickle.dumps(sourceable)),
        # "cls_name": f"{sourceable.__class__.__module__}.{sourceable.__class__.__name__}",
        # "cls_args": sourceable.args,
        # "cls_kwargs": sourceable.kwargs,
    }


def load(d: dict):
    sys_modules = {k: v for k, v in sys.modules.items()}
    # Monkey patch sys modules to look like they did when pickle was made
    for name, source in d["sources"]:
        sys.modules[name] = ModuleType(name)
        exec(source, sys.modules[name].__dict__)
    sourceable = pickle.loads(gzip.decompress(d["sourceable"]))
    # Revert sys modules to original (in-place, important)
    sys.modules.update(sys_modules)
    # remove the modules with source from the pickle (and only these!)
    for name, _ in d["sources"]:
        if name not in sys_modules:
            sys.modules.pop(name)
    return sourceable

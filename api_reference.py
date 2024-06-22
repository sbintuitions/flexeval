"""Generate the code reference pages and navigation."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent


def _get_all_modules(module_import_path: str) -> list[str]:
    module = importlib.import_module(module_import_path)
    module_classes = inspect.getmembers(module, inspect.isclass)
    module_classes += inspect.getmembers(module, inspect.isfunction)
    return [cls[1].__module__ + "." + cls[0] for cls in module_classes]


# collect modules to document from `core` directory
# key: module name, value: list of full class paths
all_core_modules: dict[str, list[str]] = {}
for base_py_file_path in sorted((root / "flexeval" / "core").rglob("base.py")):
    module_directory = base_py_file_path.relative_to(root).parent
    module_import_path = str(module_directory).replace("/", ".")

    all_modules = _get_all_modules(module_import_path)
    base_class_name = next(module for module in all_modules if "base" in module).split(".")[-1]
    all_core_modules[base_class_name] = _get_all_modules(module_import_path)

# collect `EvalSetup` modules from `flexeval/core/eval_setups.py`
all_core_modules["EvalSetup"] = [
    module for module in _get_all_modules("flexeval.core.eval_setups") if module.startswith("flexeval.core.eval_setups")
]

# collect utils
all_core_modules["utils"] = _get_all_modules("flexeval.utils")

for name, module_classes in sorted(all_core_modules.items()):
    # sort classes so that base classes come first
    module_classes.sort(key=lambda x: ("base" not in x and "EvalSetup" not in x, x))
    with mkdocs_gen_files.open(Path("api_reference") / f"{name}.md", "w") as fd:
        for class_path in module_classes:
            class_name = class_path.split(".")[-1]
            fd.write(f"::: {class_path}\n")

with mkdocs_gen_files.open(Path("api_reference") / "index.md", "w") as fd:
    for name in sorted(all_core_modules.keys()):
        fd.write(f"  - [{name}]({name}.md)\n")

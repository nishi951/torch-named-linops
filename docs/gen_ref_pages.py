"""Generate the code reference pages."""

from pathlib import Path
import sys
import mkdocs_gen_files
import inspect

# Add src to path so we can import torchlinops
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torchlinops  # noqa: E402
import torchlinops.nameddim  # noqa: E402
import torchlinops.utils  # noqa: E402
import torchlinops.alg  # noqa: E402


nav = mkdocs_gen_files.Nav()


def fully_qualified_name(module, obj):
    return f"{module.__name__}.{obj.__name__}"


def generate_pages(module, output_subdir, nav):
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]

    for name in names:
        obj = getattr(module, name)

        # Skip modules
        if inspect.ismodule(obj):
            continue
        if (not inspect.isfunction(obj)) and (not inspect.isclass(obj)):
            continue

        full_name = fully_qualified_name(module, obj)

        # Basic check to skip unrelated imports
        if hasattr(obj, "__module__") and obj.__module__:
            if not obj.__module__.startswith(f"torchlinops.{output_subdir}"):
                continue

        # linops/some_linop
        module_path = Path(output_subdir) / Path(*name.split("."))

        # reference/linops/some_linop.md
        doc_path = module_path.with_suffix(".md")

        # parts determines sidebar hierarchy
        parts = tuple(full_name.split(".")[1:])
        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(Path("reference") / doc_path, "w") as f:
            f.write(f"::: {full_name}\n")


# Generate
generate_pages(torchlinops.nameddim, "nameddim", nav)
generate_pages(torchlinops.linops, "linops", nav)
generate_pages(torchlinops.alg, "alg", nav)
generate_pages(torchlinops.utils, "utils", nav)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

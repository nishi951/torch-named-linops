"""Generate the code reference pages."""

from pathlib import Path
import sys
import mkdocs_gen_files
import inspect
import pkgutil

# Add src to path so we can import torchlinops
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torchlinops  # noqa: E402
import torchlinops.nameddim  # noqa: E402
import torchlinops.utils  # noqa: E402
import torchlinops.alg  # noqa: E402


nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    if parts[-1] == "__main__":
        continue
    elif any(("test" in part) for part in parts):
        continue
    elif any(part.startswith("_") for part in parts):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

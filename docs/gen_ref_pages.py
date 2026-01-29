"""Generate the code reference pages."""

from pathlib import Path
import sys
import mkdocs_gen_files
import inspect

# Add src to path so we can import torchlinops
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torchlinops
import torchlinops.utils
import torchlinops.alg

def generate_pages(module, output_subdir, full_name_prefix):
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    
    for name in names:
        obj = getattr(module, name)
        
        # Skip modules
        if inspect.ismodule(obj):
            continue
            
        # Basic check to skip unrelated imports
        if hasattr(obj, "__module__") and obj.__module__:
             if not obj.__module__.startswith("torchlinops"):
                 continue

        full_name = f"{full_name_prefix}.{name}"
        filename = f"{full_name}.md"
        file_path = Path("reference", "generated", output_subdir, filename)
        
        with mkdocs_gen_files.open(file_path, "w") as f:
            f.write(f"::: {full_name}\n")

# Generate
generate_pages(torchlinops, "linops", "torchlinops")
generate_pages(torchlinops.alg, "algorithms", "torchlinops.alg")
generate_pages(torchlinops.utils, "utils", "torchlinops.utils")

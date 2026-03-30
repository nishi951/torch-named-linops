#!/usr/bin/env python
"""
Build tutorials from marimo notebooks.

This script converts marimo notebook files (.py) to Markdown (.md) by:
1. Exporting session snapshot (JSON with outputs)
2. Exporting markdown structure (code blocks)
3. Combining them to produce final markdown with embedded outputs

Usage:
    uv run python scripts/build_tutorials.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configuration
TUTORIALS_DIR = Path("tutorials")
OUTPUT_DIR = Path("docs/tutorials")
MARIMO_OUTPUT_DIR = Path("tutorials/__marimo__")


def run_marimo_export_session(notebook_path: Path) -> Path:
    """Run marimo export session to generate JSON snapshot.

    Returns path to the generated JSON file.
    """
    cmd = [
        "uv",
        "run",
        "marimo",
        "export",
        "session",
        "--force-overwrite",
        str(notebook_path),
    ]
    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: marimo export session failed: {result.stderr}")

    # marimo writes to __marimo__/session/<notebook>.py.json
    json_path = TUTORIALS_DIR / "__marimo__" / "session" / f"{notebook_path.name}.json"
    return json_path


def run_marimo_export_md(notebook_path: Path, output_path: Path) -> bool:
    """Run marimo export md to generate markdown.

    Returns True on success.
    """
    cmd = [
        "uv",
        "run",
        "marimo",
        "export",
        "md",
        str(notebook_path),
        "-o",
        str(output_path),
    ]
    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: marimo export md failed: {result.stderr}")
        return False
    return True


def load_session_json(json_path: Path) -> dict:
    """Load the session JSON file."""
    with open(json_path) as f:
        return json.load(f)


def is_markdown_cell(cell: dict) -> bool:
    """Check if a cell is a markdown cell (mo.md output)."""
    outputs = cell.get("outputs", [])
    for output in outputs:
        if output.get("type") == "data":
            data = output.get("data", {})
            if "text/markdown" in data:
                return True
    return False


def extract_outputs(session_data: dict) -> dict[int, dict]:
    """Extract outputs from session data, indexed by code block position.

    Filters out markdown cells (mo.md calls) and maps remaining code cells
    to their position in the markdown code block sequence.

    Returns dict mapping code block index to {console, outputs}.
    """
    code_cell_idx = 0
    cells_data = {}
    for cell in session_data.get("cells", []):
        # Skip markdown cells - they don't produce code blocks
        if is_markdown_cell(cell):
            continue

        cells_data[code_cell_idx] = {
            "console": cell.get("console", []),
            "outputs": cell.get("outputs", []),
        }
        code_cell_idx += 1
    return cells_data


def format_console_output(console: list[dict]) -> str:
    """Format console output (stdout/stderr/media) for markdown."""
    parts = []
    for item in console:
        item_type = item.get("type", "stream")

        if item_type == "streamMedia":
            # Handle image/plot outputs from matplotlib, etc.
            mimetype = item.get("mimetype", "")
            data = item.get("data", "")

            if mimetype == "application/vnd.marimo+mimebundle":
                try:
                    mime_data = json.loads(data)
                    # Check for image types in order of preference
                    for img_type in [
                        "image/svg+xml",
                        "image/png",
                        "image/jpeg",
                        "image/gif",
                    ]:
                        if img_type in mime_data:
                            img_data = mime_data[img_type]
                            # Embed as markdown image
                            if img_data.startswith("data:"):
                                parts.append(f"![output]({img_data})")
                            else:
                                parts.append(
                                    f"![output](data:{img_type};base64,{img_data})"
                                )
                            break
                except json.JSONDecodeError:
                    pass
        elif item_type == "stream":
            name = item.get("name", "stdout")
            text = item.get("text", "")
            if not text.strip():
                continue
            text = text.rstrip()

            if name == "stderr":
                # Format stderr as warning
                parts.append(f'!!! warning "Warnings/Errors"')
                parts.append("```")
                parts.append(text)
                parts.append("```")
            else:
                # Format stdout as output
                parts.append("**Output:**")
                parts.append("```")
                parts.append(text)
                parts.append("```")

    return "\n\n".join(parts) if parts else ""


def format_cell_output(cell_data: dict) -> str:
    """Format a single cell's output for markdown."""
    parts = []

    # Add console output (stdout/stderr/media)
    console = cell_data.get("console", [])
    if console:
        console_md = format_console_output(console)
        if console_md:
            parts.append(console_md)

    # Add data outputs (images, HTML, etc.)
    for output in cell_data.get("outputs", []):
        output_type = output.get("type", "")
        data = output.get("data", {})

        if output_type == "data":
            # Check for image outputs
            for img_type in ["image/svg+xml", "image/png", "image/jpeg", "image/gif"]:
                if img_type in data:
                    img_data = data[img_type]
                    # Embed as markdown image
                    if img_data.startswith("data:"):
                        parts.append(f"![output]({img_data})")
                    else:
                        parts.append(f"![output](data:{img_type};base64,{img_data})")
                    break

    return "\n\n".join(parts) if parts else ""


def inject_outputs_into_markdown(md_content: str, cells_data: dict[int, dict]) -> str:
    """Inject outputs into markdown content after each code block.

    The markdown has code blocks with {.marimo} class.
    We match them by position to the cells_data.
    """
    # Find all code blocks with .marimo class
    # Pattern: ```python {.marimo}\n...\n```
    code_block_pattern = re.compile(r"```python\s*\{\.marimo\}\n(.*?)```", re.DOTALL)

    # Split content into parts: text, code block, text, code block, etc.
    parts = []
    last_end = 0

    for idx, match in enumerate(code_block_pattern.finditer(md_content)):
        # Add text before this code block
        parts.append(md_content[last_end : match.start()])

        # Add the code block itself
        parts.append(match.group(0))

        # Add the corresponding output if available
        if idx in cells_data:
            output_md = format_cell_output(cells_data[idx])
            if output_md:
                parts.append("\n\n")
                parts.append(output_md)

        last_end = match.end()

    # Add remaining text
    parts.append(md_content[last_end:])

    return "".join(parts)


def process_tutorial(marimo_path: Path, output_dir: Path) -> str:
    """Process a single marimo notebook.

    Returns the title extracted from the notebook.
    """
    print(f"Processing: {marimo_path}")

    # Step 1: Export session snapshot
    json_path = run_marimo_export_session(marimo_path)

    # Step 2: Export markdown
    md_temp_path = output_dir / f"_{marimo_path.stem}_temp.md"
    run_marimo_export_md(marimo_path, md_temp_path)

    # Step 3: Load and process
    session_data = load_session_json(json_path)
    cells_data = extract_outputs(session_data)

    # Read markdown
    md_content = md_temp_path.read_text()

    # Extract title from markdown
    title_match = re.search(r"^#\s+(.+)$", md_content, re.MULTILINE)
    title = (
        title_match.group(1)
        if title_match
        else marimo_path.stem.replace("_", " ").title()
    )

    # Inject outputs
    final_md = inject_outputs_into_markdown(md_content, cells_data)

    # Clean up the {.marimo} class from code blocks
    final_md = re.sub(r"```python\s*\{\.marimo\}", "```python", final_md)

    # Remove frontmatter
    final_md = re.sub(r"^---\n.*?\n---\n", "", final_md, flags=re.DOTALL)

    # Write final markdown
    output_path = output_dir / f"{marimo_path.stem}.md"
    output_path.write_text(final_md)
    print(f"  Written: {output_path}")

    # Clean up temp file
    if md_temp_path.exists():
        md_temp_path.unlink()

    return title


def process_all_tutorials() -> None:
    """Process all marimo notebooks in the tutorials directory."""
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure __marimo__ output dir exists
    (TUTORIALS_DIR / "__marimo__" / "session").mkdir(parents=True, exist_ok=True)

    # Find all tutorials (excluding README and __marimo__ directory)
    tutorial_files = []
    for f in TUTORIALS_DIR.glob("*.py"):
        if f.name == "README.md" or f.name.startswith("_"):
            continue
        # Skip files that are clearly not marimo notebooks
        content = f.read_text()
        if "import marimo" in content or "@app.cell" in content:
            tutorial_files.append(f)

    if not tutorial_files:
        print(f"No marimo notebooks found in {TUTORIALS_DIR}")
        print("Make sure tutorials have 'import marimo' and '@app.cell' decorators.")
        return

    print(f"Found {len(tutorial_files)} marimo notebooks")

    # Sort for consistent order
    tutorial_files.sort()

    # Process each
    tutorials_metadata = []
    for tutorial_path in tutorial_files:
        title = process_tutorial(tutorial_path, output_dir)
        base_name = tutorial_path.stem
        tutorials_metadata.append((base_name, title))

    # Generate index
    generate_index(tutorials_metadata)

    print("\nDone building tutorials!")


def generate_index(tutorials: list[tuple[str, str]]) -> None:
    """Generate the tutorials index page."""
    lines = [
        "# Tutorials",
        "",
        "This section contains interactive tutorials for learning `torchlinops`.",
        "",
        "Each tutorial demonstrates key concepts with runnable code examples.",
        "",
        "## Available Tutorials",
        "",
    ]

    for stem, title in tutorials:
        lines.append(f"- [{title}](./{stem}.md)")

    lines.append("")

    index_path = OUTPUT_DIR / "index.md"
    index_path.write_text("\n".join(lines))
    print(f"Generated: {index_path}")


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build tutorials from marimo notebooks"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tutorials-dir",
        "-t",
        type=Path,
        default=TUTORIALS_DIR,
        help=f"Source directory (default: {TUTORIALS_DIR})",
    )

    args = parser.parse_args()

    globals()["OUTPUT_DIR"] = args.output_dir
    globals()["TUTORIALS_DIR"] = args.tutorials_dir

    process_all_tutorials()


if __name__ == "__main__":
    main()

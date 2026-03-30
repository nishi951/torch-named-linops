#!/usr/bin/env python
"""
Build tutorials from Python files.

This script converts Python tutorial files (.py) to Markdown (.md) by:
1. Parsing cells marked with `# %%`
2. Extracting markdown from comment lines after `# %%`
3. Executing code cells and capturing stdout output
4. Generating .md files in docs/tutorials/

Similar to mkdocs-gallery but compatible with Zensical's static build model.
"""

from __future__ import annotations

import io
import re
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configuration
TUTORIALS_DIR = Path("tutorials")
OUTPUT_DIR = Path("docs/tutorials")


def parse_tutorial_file(filepath: Path) -> list[dict]:
    """Parse a Python tutorial file into cells.

    The format is:
    - Module-level docstring at the top (markdown)
    - `# %%` markers separate cells
    - After `# %%`, lines starting with `#` are markdown
    - Non-comment lines after markdown are code

    Returns a list of dicts with keys 'type' and 'content'.
    """
    content = filepath.read_text()

    cells = []

    # First, check for module-level docstring
    docstring_match = re.match(r'^("""|\'\'\')\s*\n(.*?)\n\1', content, re.DOTALL)

    if docstring_match:
        docstring = docstring_match.group(2)
        # Dedent and clean
        docstring = dedent(docstring).strip()
        if docstring:
            cells.append({"type": "markdown", "content": docstring})
        # Remove docstring from content for further processing
        content = content[docstring_match.end() :]

    # Split by # %% markers
    parts = re.split(r"^# %%\s*\n", content, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Parse this cell
        cell = parse_cell(part)
        if cell:
            cells.extend(cell)

    return cells


def parse_cell(content: str) -> list[dict]:
    """Parse a single cell into markdown and code blocks.

    Lines starting with # are markdown (with # stripped).
    Other lines are code.
    """
    lines = content.split("\n")

    cells = []
    markdown_lines = []
    code_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a comment line (potential markdown)
        if line.strip().startswith("#"):
            # If we have accumulated code, emit it first
            if code_lines:
                cells.append({"type": "code", "content": "\n".join(code_lines).strip()})
                code_lines = []

            # Extract markdown content (strip leading # and optional space)
            stripped = line.strip()
            if stripped == "#":
                # Empty comment = blank line
                markdown_lines.append("")
            else:
                # Remove leading # and one optional space
                md_line = re.sub(r"^#\s?", "", stripped)
                markdown_lines.append(md_line)
        else:
            # If we have accumulated markdown, emit it first
            if markdown_lines:
                cells.append(
                    {"type": "markdown", "content": "\n".join(markdown_lines).strip()}
                )
                markdown_lines = []

            # This is a code line
            code_lines.append(line)

        i += 1

    # Emit any remaining content
    if markdown_lines:
        cells.append({"type": "markdown", "content": "\n".join(markdown_lines).strip()})
    if code_lines:
        cells.append({"type": "code", "content": "\n".join(code_lines).strip()})

    return cells


def execute_code_cell(code: str, namespace: dict) -> tuple[str, str]:
    """Execute a code cell and capture stdout/stderr.

    Returns (stdout_output, stderr_output).
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                exec(compile(code, "<tutorial>", "exec"), namespace)
            except Exception as e:
                stderr_capture.write(f"Error: {type(e).__name__}: {e}\n")

    return stdout_capture.getvalue(), stderr_capture.getvalue()


def process_code_cell(code: str, namespace: dict) -> str:
    """Execute code and return formatted markdown."""
    stdout_output, stderr_output = execute_code_cell(code, namespace)

    parts = []

    # Code block
    parts.append("```python")
    parts.append(code)
    parts.append("```")
    parts.append("")

    # Stdout output block
    if stdout_output.strip():
        parts.append("**Output:**")
        parts.append("```")
        parts.append(stdout_output.rstrip())
        parts.append("```")
        parts.append("")

    # Stderr/output warnings
    if stderr_output.strip():
        parts.append('!!! warning "Warnings/Errors"')
        parts.append("```")
        parts.append(stderr_output.rstrip())
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def build_tutorial(tutorial_path: Path, output_path: Path) -> str:
    """Build a single tutorial file.

    Converts a .py tutorial to .md format.
    Returns the title extracted from the first cell.
    """
    print(f"Processing: {tutorial_path}")

    # Parse cells
    cells = parse_tutorial_file(tutorial_path)

    if not cells:
        print(f"  Warning: No cells found in {tutorial_path}")
        return tutorial_path.stem

    # Process each cell
    namespace: dict = {"__name__": "__main__"}
    # Import common packages into namespace
    try:
        namespace["torch"] = __import__("torch")
    except ImportError:
        pass

    markdown_parts = []
    title = None

    for cell in cells:
        try:
            if cell["type"] == "markdown":
                md = cell["content"]
                # Extract title from first markdown cell
                if title is None:
                    title_match = re.match(r"^#\s+(.+)$", md, re.MULTILINE)
                    if title_match:
                        title = title_match.group(1).strip()
                markdown_parts.append(md)
            else:
                md = process_code_cell(cell["content"], namespace)
                markdown_parts.append(md)
        except Exception as e:
            print(f"  Error processing cell: {e}")
            # Include the cell anyway with error
            if cell["type"] == "code":
                markdown_parts.append(f"```python\n{cell['content']}\n```")
                markdown_parts.append(f"!!! error\n    Error executing this cell: {e}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(markdown_parts))
    print(f"  Written: {output_path}")

    return title or tutorial_path.stem.replace("_", " ").title()


def build_all_tutorials() -> None:
    """Build all tutorials in the tutorials directory."""
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .py files in tutorials directory
    tutorial_files = sorted(TUTORIALS_DIR.glob("*.py"))

    if not tutorial_files:
        print(f"No tutorial files found in {TUTORIALS_DIR}")
        return

    print(f"Found {len(tutorial_files)} tutorial files")

    # Build each tutorial
    tutorials_metadata = []
    for tutorial_path in tutorial_files:
        if tutorial_path.name.startswith("_"):
            continue

        output_path = output_dir / f"{tutorial_path.stem}.md"
        title = build_tutorial(tutorial_path, output_path)
        tutorials_metadata.append((tutorial_path.stem, title))

    # Generate index page
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
    """Entry point for the build script."""
    import argparse

    parser = argparse.ArgumentParser(description="Build tutorials from Python files")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for generated markdown files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tutorials-dir",
        "-t",
        type=Path,
        default=TUTORIALS_DIR,
        help=f"Source directory containing .py tutorial files (default: {TUTORIALS_DIR})",
    )

    args = parser.parse_args()

    globals()["OUTPUT_DIR"] = args.output_dir
    globals()["TUTORIALS_DIR"] = args.tutorials_dir

    build_all_tutorials()


if __name__ == "__main__":
    main()

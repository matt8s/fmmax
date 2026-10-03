"""Export the stored tutorial notebooks as Markdown pages.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import os
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter


def export_notebooks(notebook_dir: str, output_dir: str | None = None) -> None:
    """Export every notebook in a directory to Markdown.

    The exporter uses the outputs already stored in each notebook; it does not run
    notebook cells.

    Args:
        notebook_dir: Directory containing the notebooks.
        output_dir: Directory that receives the Markdown and image files. When
            omitted, files are written beside the source notebooks.
    """
    # Create the output directory when needed.
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export every notebook in the source directory.
    for filename in sorted(os.listdir(notebook_dir)):
        if filename.endswith(".ipynb"):
            # Read the notebook without executing it.
            with open(os.path.join(notebook_dir, filename)) as f:
                notebook = nbformat.read(f, as_version=4)

            exporter = MarkdownExporter()

            # Keep each notebook's stored outputs in its own directory so files
            # with common nbconvert names such as `output_1_0.png` cannot collide.
            notebook_stem = filename[:-6]
            body, resources = exporter.from_notebook_node(
                notebook, resources={"output_files_dir": notebook_stem}
            )

            output_filename = filename[:-6] + ".md"
            output_filepath = os.path.join(output_dir or notebook_dir, output_filename)
            print(output_filepath)
            with open(output_filepath, "w") as f:
                f.write(body)

            for image_filename, image_data in resources.get("outputs", {}).items():
                image_path = Path(output_dir or notebook_dir) / image_filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                with image_path.open("wb") as f:
                    f.write(image_data)


if __name__ == "__main__":
    notebook_dir = "../notebooks"

    output_dir = "./docs/Tutorials"

    export_notebooks(notebook_dir, output_dir)

"""Generate API reference pages from the public FMMAX docstrings.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import ast
import copy
import glob
import os
from typing import Any, Dict

from docstring_parser import parse


def write_to_file(output_dir: str, filename: str, data: str, mode: str = "w") -> None:
    """Write generated Markdown to a file.

    Args:
        output_dir: Directory that receives the file.
        filename: Name of the generated file.
        data: Markdown content to write.
        mode: File mode, such as `"w"` or `"a"`.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), mode) as f:
        f.write(data)


def process_file(output_dir: str, filename: str, toc: Dict[str, Any]) -> None:
    """Generate API pages for the public definitions in one Python module.

    Args:
        output_dir: Directory that receives generated pages.
        filename: Python source file to process.
        toc: Table-of-contents data updated by this function.
    """
    with open(filename, "r") as f:
        module = ast.parse(f.read())
    submodule_name = os.path.splitext(os.path.basename(filename))[0]
    toc[submodule_name] = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            process_class(output_dir, node, toc[submodule_name], submodule_name)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not node.name.startswith("_"):
            process_function(output_dir, node, toc[submodule_name], submodule_name)


def write_header(header: str) -> str:
    """Return Docusaurus front matter for a generated document.

    Args:
        header: Document identifier.

    Returns:
        Docusaurus front matter.
    """
    return f"---\nid: {header}\n---\n\n"


def process_class(
    output_dir: str, node: ast.ClassDef, toc: Dict[str, Any], submodule_name: str
) -> None:
    """Generate a reference page for a public class.

    Args:
        output_dir: Directory that receives the generated page.
        node: Class syntax tree node.
        toc: Table-of-contents data updated by this function.
        submodule_name: Module containing the class.
    """
    docstring = ast.get_docstring(node)
    init_docstring = None
    for sub_node in node.body:
        if (
            isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and sub_node.name == "__init__"
        ):
            init_docstring = ast.get_docstring(sub_node)
            break
    public_methods = [
        sub_node
        for sub_node in node.body
        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not sub_node.name.startswith("_")
        and ast.get_docstring(sub_node)
    ]
    if not (docstring or init_docstring or public_methods):
        return

    filename = f"{submodule_name}.{node.name}.md"
    markdown = docstring_to_markdown(docstring)
    if init_docstring:
        markdown += "\n" + docstring_to_markdown(init_docstring)
    class_id = f"{submodule_name}.{node.name}"
    write_to_file(
        output_dir,
        filename,
        write_header(class_id) + f"### Class `{class_id}`\n{markdown}",
    )
    description = get_first_line(docstring or init_docstring or f"Class {class_id}.")
    toc[node.name] = {"__doc__": {"link": filename, "desc": description}}

    for sub_node in public_methods:
        process_function(
            output_dir,
            sub_node,
            toc[node.name],
            f"{submodule_name}.{node.name}",
            True,
        )


def process_function(
    output_dir: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    toc: Dict[str, Any],
    parent_name: str,
    is_method: bool = False,
) -> None:
    """Append a public function or method to its API reference page.

    Args:
        output_dir: Directory that receives the generated page.
        node: Function syntax tree node.
        toc: Table-of-contents data updated by this function.
        parent_name: Module or class containing the function.
        is_method: Whether the function is a class method.
    """
    docstring = ast.get_docstring(node)
    if docstring:
        filename = f"{parent_name}.md" if is_method else f"{parent_name}.{node.name}.md"
        markdown = docstring_to_markdown(docstring)
        is_property = any(
            isinstance(decorator, ast.Name) and decorator.id == "property"
            for decorator in node.decorator_list
        )

        mode = "a" if is_method else "w"
        with open(os.path.join(output_dir, filename), mode) as f:
            if not is_method:
                f.write(write_header(f"{parent_name}.{node.name}"))
            signature_args = copy.deepcopy(node.args)
            is_static = any(
                isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                for decorator in node.decorator_list
            )
            if is_method and not is_static:
                positional_args = signature_args.posonlyargs + signature_args.args
                if positional_args and positional_args[0].arg in ("self", "cls"):
                    if signature_args.posonlyargs:
                        signature_args.posonlyargs.pop(0)
                    else:
                        signature_args.args.pop(0)
            signature = ast.unparse(signature_args)
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            qualified_name = f"{parent_name}.{node.name}"
            f.write(f"\n### `{qualified_name}`\n")
            if is_property:
                f.write("\nProperty.\n")
            else:
                f.write(f"\n```python\n{qualified_name}({signature}){returns}\n```\n")
            f.write(markdown)
        toc[node.name] = {"link": filename, "desc": get_first_line(docstring)}


def get_first_line(docstring: str | None) -> str:
    """Return the first line of a docstring.

    Args:
        docstring: Docstring to summarize.

    Returns:
        First line of the docstring.
    """
    return docstring.split("\n")[0] if docstring else ""


def docstring_to_markdown(docstring: str | None) -> str:
    """Convert one parsed docstring to Markdown.

    Args:
        docstring: Docstring to convert.

    Returns:
        Generated Markdown.
    """
    if docstring is None:
        return ""

    doc = parse(docstring)
    if doc.short_description is None:
        return docstring

    markdown = doc.short_description + "\n\n"

    if doc.long_description:
        markdown += doc.long_description + "\n"

    if doc.params:
        markdown += "\n#### Args:\n"
        for param in doc.params:
            markdown += f"- **{param.arg_name}**: {param.description}\n"

    if doc.returns:
        markdown += "\n#### Returns:\n"
        return_type = f"**{doc.returns.type_name}**: " if doc.returns.type_name else ""
        markdown += f"- {return_type}{doc.returns.description}\n"
    return markdown


def traverse_directory(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """Generate API pages for every Python module below a directory.

    Args:
        input_dir: Root Python source directory.
        output_dir: Directory that receives generated pages.

    Returns:
        Nested table-of-contents data.
    """
    # Check if the output dir exists, create it if it doesn't.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remove any existing doc files.
    files = glob.glob(os.path.join(output_dir, "*.md"))
    for f in files:
        os.remove(f)

    # Walk through the file tree and process each file.
    toc: Dict[str, Any] = {}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".py"):
                process_file(output_dir, os.path.join(root, file), toc)
    return toc


def write_toc(output_dir: str, toc: Dict[str, Any]) -> None:
    """Write a Markdown table of contents for generated API pages.

    Args:
        output_dir: Directory that receives the table of contents.
        toc: Nested table-of-contents data.
    """
    with open(os.path.join(output_dir, "TOC.md"), "w") as f:
        f.write("# API Reference\n")
        for submodule_name, classes in sorted(toc.items()):
            for class_name, functions in sorted(classes.items()):
                if class_name == "__doc__":
                    continue
                else:
                    f.write(f"\n## {submodule_name}.{class_name}\n")
                    f.write(
                        f"- [{class_name}]({functions['__doc__']['link']}): {functions['__doc__']['desc']}\n"
                    )
                    for function_name, function in sorted(functions.items()):
                        if function_name != "__doc__":
                            f.write(
                                f"  - [{function_name}]({function['link']}): {function['desc']}\n"
                            )


if __name__ == "__main__":
    input_dir = "../src/fmmax"
    output_dir = "./docs/API"
    toc = traverse_directory(input_dir, output_dir)

    # Docusaurus builds the API sidebar from the generated pages.
    # write_toc(output_dir, toc)

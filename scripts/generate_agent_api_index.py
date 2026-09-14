"""Generate the packaged tf_gnns agent-skill API index."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tf_gnns"
OUT = PACKAGE / "agent_skill" / "api_index.json"


def _references_files() -> list[Path]:
    return sorted((PACKAGE / "agent_skill" / "references").glob("*.md"))


def _public_names() -> list[str]:
    names: set[str] = set()
    # Root package __all__
    module = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    names.update(ast.literal_eval(node.value))
    # Submodule __all__ (e.g. tf_gnns.models exports GCNv2), but only keep
    # names that are referenced in the reference docs so internal runtime
    # helpers (e.g. maybe_patch_ld_library_path_for_tensorflow) do not leak in.
    referenced_docs = "\n".join(p.read_text(encoding="utf-8") for p in _references_files())
    for subinit in sorted(PACKAGE.rglob("__init__.py")):
        if subinit == PACKAGE / "__init__.py":
            continue
        try:
            submod = ast.parse(subinit.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in submod.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        for name in ast.literal_eval(node.value):
                            if name.startswith("_"):
                                continue
                            pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
                            if re.search(pattern, referenced_docs):
                                names.add(name)
    # Filter out private names (underscore-prefixed)
    return sorted(n for n in names if not n.startswith("_"))


def _signature(name: str, node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        init = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
        if init is None:
            return f"{name}(...)"
        args = init.args
    elif isinstance(node, ast.FunctionDef):
        args = node.args
    else:
        return name

    parts = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg == "self":
            continue
        text = arg.arg
        if default is not None:
            text += f"={ast.unparse(default)}"
        parts.append(text)
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if default is not None:
            text += f"={ast.unparse(default)}"
        parts.append(text)
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return f"{name}({', '.join(parts)})"


def _summary(docstring: str | None) -> str:
    if not docstring:
        return "No docstring summary."
    for line in docstring.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return "No docstring summary."


def _returns(docstring: str | None) -> str:
    if not docstring or "Returns:" not in docstring:
        return ""
    lines = docstring.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "Returns:":
            collected = []
            for follow in lines[index + 1 :]:
                stripped = follow.strip()
                if not stripped:
                    continue
                if stripped.endswith(":") and not follow.startswith((" ", "\t")):
                    break
                collected.append(stripped)
                if len(" ".join(collected)) > 140:
                    break
            return " ".join(collected)
    return ""


def _topic(name: str, module: str) -> str:
    lower = name.lower()
    if lower in {"node", "edge", "graph", "graphtuple"} or "graph_tuple" in lower:
        return "graphs"
    if "gcn" in lower:
        return "gcn"
    if "backend" in module:
        return "backends"
    if "graphnet" in lower or "graphindep" in lower or "mlp" in lower or "agg" in lower:
        return "graphnets"
    return "quickstart"


def _examples() -> dict[str, str]:
    examples: dict[str, str] = {}
    for path in (PACKAGE / "agent_skill" / "references").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        code = "\n".join(re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL))
        for name in _public_names():
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
            if re.search(pattern, code):
                examples.setdefault(name, path.stem)
    return examples


def build_index() -> dict[str, object]:
    public = set(_public_names())
    examples = _examples()
    entries = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "agent_skill" in path.parts or path.name == "agent_docs.py":
            continue
        module_name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)) or node.name not in public:
                continue
            docstring = ast.get_docstring(node)
            entries.append(
                {
                    "name": node.name,
                    "import": f"from {module_name} import {node.name}",
                    "signature": _signature(node.name, node),
                    "summary": _summary(docstring),
                    "returns": _returns(docstring),
                    "source": f"{path.relative_to(ROOT)}:{node.lineno}",
                    "topic": _topic(node.name, module_name),
                    "example": examples.get(node.name, ""),
                }
            )
    entries.sort(key=lambda entry: entry["name"].lower())
    return {"generated_by": "scripts/generate_agent_api_index.py", "entries": entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if api_index.json is stale")
    args = parser.parse_args(argv)
    index = build_index()
    rendered = json.dumps(index, indent=2, sort_keys=True) + "\n"
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != rendered:
            raise SystemExit("api_index.json is stale; run: python scripts/generate_agent_api_index.py")
        return 0
    OUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

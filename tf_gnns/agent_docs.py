"""Command-line retrieval for the packaged tf_gnns agent skill."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


SKILL_PACKAGE = "tf_gnns.agent_skill"
SKILL_NAME = "tf-gnns-user-docs"


@dataclass(frozen=True)
class Topic:
    name: str
    title: str
    when: str
    path: str


TOPICS = [
    Topic("quickstart", "Quickstart", "creating a small GraphTuple and running one model", "references/quickstart.md"),
    Topic("graphs", "Graph Data", "building Node, Edge, Graph, and GraphTuple inputs", "references/graphs.md"),
    Topic("graphnets", "GraphNet Blocks", "using GraphNet factories, MPNN layers, globals, and aggregators", "references/graphnets.md"),
    Topic("gcn", "Sparse GCN", "using SparseGCNConv, SparseGCN, or GCNv2 on tensor dictionaries", "references/gcn.md"),
    Topic("backends", "Backends", "choosing TensorFlow, Torch, or JAX Keras backends", "references/backends.md"),
]


def _resource_text(path: str) -> str:
    return _skill_root().joinpath(path).read_text(encoding="utf-8")


def _skill_root():
    try:
        return resources.files(SKILL_PACKAGE)
    except ModuleNotFoundError:
        return Path(__file__).with_name("agent_skill")


def _copy_skill(destination: Path) -> None:
    try:
        root = resources.files(SKILL_PACKAGE)
    except ModuleNotFoundError:
        shutil.copytree(Path(__file__).with_name("agent_skill"), destination)
        return
    with resources.as_file(root) as skill_root:
        shutil.copytree(skill_root, destination)


def _api_index() -> list[dict[str, object]]:
    return json.loads(_resource_text("api_index.json"))["entries"]


def _topic_by_name(name: str) -> Topic:
    for topic in TOPICS:
        if topic.name == name:
            return topic
    raise SystemExit(f"Unknown topic '{name}'. Run: python -m tf_gnns.agent_docs list")


def _extract_examples(text: str) -> str:
    blocks = re.findall(r"```python\n.*?```", text, flags=re.DOTALL)
    return "\n\n".join(blocks)


def _extract_rules(text: str) -> str:
    lines = []
    in_rules = False
    for line in text.splitlines():
        if line == "## Rules":
            in_rules = True
            continue
        if in_rules and line.startswith("## "):
            break
        if in_rules:
            lines.append(line)
    return "\n".join(lines).strip()


def list_topics(_args: argparse.Namespace) -> int:
    for topic in TOPICS:
        print(f"{topic.name:10} {topic.title} - read when {topic.when}")
    return 0


def get_topic(args: argparse.Namespace) -> int:
    topic = _topic_by_name(args.topic)
    text = _resource_text(topic.path)
    if args.examples:
        text = _extract_examples(text)
    elif args.rules:
        text = _extract_rules(text)
    print(text.rstrip())
    return 0


def _topic_texts() -> list[tuple[Topic, str]]:
    return [(topic, _resource_text(topic.path)) for topic in TOPICS]


def _api_card(entry: dict[str, object]) -> str:
    return (
        f"API {entry['name']}\n"
        f"  import: {entry['import']}\n"
        f"  signature: {entry['signature']}\n"
        f"  summary: {entry['summary']}\n"
        f"  returns: {entry.get('returns') or 'not documented'}\n"
        f"  topic: {entry['topic']}\n"
        f"  covering_example: {entry.get('example') or 'none'}"
    )


def search(args: argparse.Namespace) -> int:
    regex = re.compile(args.regex, flags=re.IGNORECASE)
    hits: list[str] = []

    for entry in _api_index():
        haystack = "\n".join(str(entry.get(key, "")) for key in ("name", "signature", "summary", "returns", "topic"))
        if regex.search(haystack):
            hits.append(_api_card(entry))

    for topic, text in _topic_texts():
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not regex.search(line):
                continue
            if args.examples_only:
                before = "\n".join(lines[max(0, i - args.context) : i + args.context + 1])
                if "```python" not in before:
                    continue
            start = max(0, i - args.context)
            end = min(len(lines), i + args.context + 1)
            snippet = "\n".join(f"{n + 1}: {lines[n]}" for n in range(start, end))
            hits.append(f"{topic.name}:{i + 1}\n{snippet}")

    per_page = max(1, args.limit)
    pages = max(1, (len(hits) + per_page - 1) // per_page)
    page = min(max(1, args.page), pages)
    start = (page - 1) * per_page
    selected = hits[start : start + per_page]
    print(f"page {page}/{pages} hits {len(hits)}")
    for hit in selected:
        print("\n---")
        print(hit)
    return 0


def install(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    destination = root / ".opencode" / "skills" / SKILL_NAME
    if destination.exists():
        if not args.force:
            raise SystemExit(f"Refusing to overwrite {destination}; pass --force")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_skill(destination)
    print(destination)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tf_gnns.agent_docs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available topics")
    list_parser.set_defaults(func=list_topics)

    get_parser = subparsers.add_parser("get", help="Read a topic")
    get_parser.add_argument("topic")
    mode = get_parser.add_mutually_exclusive_group()
    mode.add_argument("--examples", action="store_true")
    mode.add_argument("--rules", action="store_true")
    get_parser.set_defaults(func=get_topic)

    search_parser = subparsers.add_parser("search", help="Search topics and API cards")
    search_parser.add_argument("regex")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--page", type=int, default=1)
    search_parser.add_argument("--context", type=int, default=2)
    search_parser.add_argument("--examples-only", action="store_true")
    search_parser.set_defaults(func=search)

    install_parser = subparsers.add_parser("install-opencode-skill", help="Install the packaged skill into a project")
    install_parser.add_argument("--project-root", default=".")
    install_parser.add_argument("--force", action="store_true")
    install_parser.set_defaults(func=install)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

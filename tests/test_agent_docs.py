import json
import importlib.util
import sys
from pathlib import Path

import pytest

from scripts.generate_agent_api_index import build_index, main as generate_index_main


def _load_agent_docs():
    path = Path("tf_gnns/agent_docs.py").resolve()
    spec = importlib.util.spec_from_file_location("agent_docs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agent_docs = _load_agent_docs()


def test_topics_and_skill_resources_present(capsys):
    assert agent_docs.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "quickstart" in output
    assert "graphnets" in output

    assert agent_docs.main(["get", "quickstart", "--examples"]) == 0
    examples = capsys.readouterr().out
    assert "```python" in examples
    assert "GraphNetMPNN_MLP" in examples


def test_search_prints_api_cards_and_pages(capsys):
    assert agent_docs.main(["search", "SparseGCN", "--limit", "2"]) == 0
    output = capsys.readouterr().out
    assert "page 1/" in output
    assert "API SparseGCN" in output
    assert "signature:" in output


def test_installer_refuses_overwrite_without_force(tmp_path):
    assert agent_docs.main(["install-opencode-skill", "--project-root", str(tmp_path)]) == 0
    installed = tmp_path / ".opencode" / "skills" / agent_docs.SKILL_NAME
    assert (installed / "SKILL.md").exists()

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        agent_docs.main(["install-opencode-skill", "--project-root", str(tmp_path)])

    assert agent_docs.main(["install-opencode-skill", "--project-root", str(tmp_path), "--force"]) == 0


def test_api_index_is_fresh_and_has_examples():
    generate_index_main(["--check"])
    index = json.loads(Path("tf_gnns/agent_skill/api_index.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in index["entries"]}
    assert {"GraphTuple", "make_mlp_graphnet_functions", "SparseGCN"}.issubset(names)
    covered = {entry["name"] for entry in index["entries"] if entry["example"]}
    assert {"GraphTuple", "GraphNetMPNN_MLP", "SparseGCN"}.issubset(covered)
    assert build_index() == index

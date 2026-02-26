from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from crabpath.cli import main


def _write_texts(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "note.md::0": "alpha",
                "note.md::1": "beta",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_embed_command_creates_index(tmp_path) -> None:
    texts_path = tmp_path / "texts.json"
    _write_texts(texts_path)
    output_path = tmp_path / "index.json"
    embed_script = tmp_path / "embed.py"
    embed_script.write_text(
        """import json
import sys


for line in sys.stdin:
    payload = json.loads(line)
    text = payload['text']
    print(json.dumps({'id': payload['id'], 'embedding': [float(len(text)), 0.0]}))
""",
        encoding="utf-8",
    )

    code = main(
        [
            "embed",
            "--texts",
            str(texts_path),
            "--output",
            str(output_path),
            "--command",
            f"{sys.executable} {embed_script}",
        ]
    )
    assert code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["note.md::0"] == [5.0, 0.0]
    assert payload["note.md::1"] == [4.0, 0.0]


def test_embed_provider_openai(tmp_path, monkeypatch) -> None:
    texts_path = tmp_path / "texts.json"
    _write_texts(texts_path)
    output_path = tmp_path / "index.json"

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "openai.py").write_text(
        """class _Data:\n    def __init__(self, embedding):\n        self.embedding = embedding\n\n\nclass _Resp:\n    def __init__(self, embedding):\n        self.data = [_Data(embedding)]\n\n\nclass _Embeddings:\n    def create(self, model, input):\n        text = input[0]\n        return _Resp([float(len(text)), 1.0])\n\n\nclass OpenAI:\n    def __init__(self):\n        self.embeddings = _Embeddings()\n""",
        encoding="utf-8",
    )
    env_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), env_path]) if env_path else str(stub_dir))

    code = main(["embed", "--texts", str(texts_path), "--output", str(output_path), "--provider", "openai"])
    assert code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["note.md::0"] == [5.0, 1.0]
    assert payload["note.md::1"] == [4.0, 1.0]


def test_init_with_embed_provider(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.md").write_text(
        "## Alpha\nalpha content\n\n## Beta\nbeta content",
        encoding="utf-8",
    )

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "openai.py").write_text(
        """class _Data:\n    def __init__(self, embedding):\n        self.embedding = embedding\n\n\nclass _Resp:\n    def __init__(self, embedding):\n        self.data = [_Data(embedding)]\n\n\nclass _Embeddings:\n    def create(self, model, input):\n        text = input[0]\n        return _Resp([float(len(text)), 1.0])\n\n\nclass OpenAI:\n    def __init__(self):\n        self.embeddings = _Embeddings()\n""",
        encoding="utf-8",
    )
    env_path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), env_path]) if env_path else str(stub_dir))

    output_dir = tmp_path / "crabpath-data"
    code = main(
        [
            "init",
            "--workspace",
            str(workspace),
            "--output",
            str(output_dir),
            "--embed-provider",
            "openai",
        ]
    )
    assert code == 0

    graph_path = output_dir / "graph.json"
    texts_path = output_dir / "texts.json"
    index_path = output_dir / "index.json"
    assert graph_path.exists()
    assert texts_path.exists()
    assert index_path.exists()

    texts_payload = json.loads(texts_path.read_text(encoding="utf-8"))
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(texts_payload) == len(index_payload)

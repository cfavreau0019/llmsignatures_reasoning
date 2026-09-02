import os

import reasoning_trajectories.env as envmod


def _reset():
    envmod._LOADED = False


def test_no_env_file_is_noop(tmp_path, monkeypatch):
    _reset()
    monkeypatch.chdir(tmp_path)
    # also anchor the package-dir search inside the empty tree
    monkeypatch.setattr(envmod, "__file__", str(tmp_path / "pkg" / "env.py"))
    assert envmod.load_env() is None


def test_fallback_parser_loads_without_override(tmp_path, monkeypatch):
    _reset()
    (tmp_path / ".env").write_text(
        "# comment\n"
        "NDIF_API_KEY=abc123\n"
        'export HF_TOKEN="hf_xyz"\n'
        "ALREADY_SET=from_file\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALREADY_SET", "from_env")
    monkeypatch.delenv("NDIF_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # both python-dotenv and the built-in fallback honour override=False
    used = envmod.load_env()

    assert used is not None and used.name == ".env"
    assert os.environ["NDIF_API_KEY"] == "abc123"
    assert os.environ["HF_TOKEN"] == "hf_xyz"
    assert os.environ["ALREADY_SET"] == "from_env"  # real env wins


def test_explicit_path(tmp_path, monkeypatch):
    _reset()
    p = tmp_path / "custom.env"
    p.write_text("FOO_BAR=baz\n")
    monkeypatch.delenv("FOO_BAR", raising=False)
    assert envmod.load_env(p) == p
    assert os.environ["FOO_BAR"] == "baz"

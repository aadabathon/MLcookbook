"""The offline history catalog and explicit reading resolution stay portable."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from mlcookbook.research import paper_specs, resolve_library, resolve_paper, timeline_entries


def test_timeline_has_chronological_unique_sourced_records_and_all_readings():
    entries = timeline_entries()
    papers = paper_specs()
    ids = [entry["id"] for entry in entries]
    assert len(entries) >= 35
    assert len(ids) == len(set(ids))
    assert [entry["year"] for entry in entries] == sorted(entry["year"] for entry in entries)
    assert entries[0]["year"] == -350 and entries[-1]["year"] == 2025
    assert {entry["era"] for entry in entries} == {
        "Foundations", "Early AI", "Statistical learning", "Deep learning", "Language models",
    }
    known_papers = {paper["id"] for paper in papers}
    assert len(known_papers) == len(papers) == 6
    linked = set()
    for entry in entries:
        assert set(entry) == {"id", "year", "date", "era", "title", "people", "summary", "significance",
                              "connections", "sources", "paper_ids"}
        assert all(isinstance(entry[key], str) and entry[key] for key in (
            "id", "date", "era", "title", "people", "summary", "significance"))
        assert entry["connections"] and entry["sources"]
        assert set(entry["paper_ids"]) <= known_papers
        linked.update(entry["paper_ids"])
        for source in entry["sources"]:
            url = urlsplit(source["url"])
            assert source["title"] and url.scheme == "https" and url.netloc
            assert not url.username and not url.password
    assert linked == known_papers
    for paper in papers:
        assert Path(paper["filename"]).name == paper["filename"]
        assert Path(paper["filename"]).suffix == ".pdf"
        assert paper["kind"] in {"Paper", "Explainer"}
        assert urlsplit(paper["url"]).scheme == "https"
    json.dumps({"timeline": entries, "papers": papers}, allow_nan=False)


def test_catalog_dates_separate_field_birth_and_later_reading_exports():
    entries = {entry["id"]: entry for entry in timeline_entries()}
    assert entries["dartmouth"]["year"] == 1956
    assert "1955" in entries["dartmouth"]["summary"]
    assert entries["illustrated-transformer"]["year"] == 2018
    assert "Jay Alammar" == entries["illustrated-transformer"]["people"]
    assert "export" in entries["illustrated-transformer"]["significance"]
    assert "explanatory" in entries["matrix-calculus"]["significance"]
    assert "endpoint" in entries["qwen3"]["significance"]


def test_catalog_callers_cannot_mutate_future_views():
    entries = timeline_entries()
    papers = paper_specs()
    entries[0]["title"] = "Changed"
    entries[0]["sources"][0]["url"] = "https://example.invalid"
    entries[0]["connections"].clear()
    papers[0]["filename"] = "escape.pdf"
    assert timeline_entries()[0]["title"] != "Changed"
    assert timeline_entries()[0]["sources"][0]["url"] != "https://example.invalid"
    assert timeline_entries()[0]["connections"]
    assert paper_specs()[0]["filename"] != "escape.pdf"


def test_library_resolves_only_exact_catalog_names_and_does_not_read_content(tmp_path):
    specs = paper_specs()
    chosen = specs[0]
    path = tmp_path / chosen["filename"]
    path.write_bytes(b"PDF content is opened by the OS, not parsed by the catalog")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / specs[1]["filename"]).write_bytes(b"not recursively discovered")
    (tmp_path / "unlisted.pdf").write_bytes(b"not in this catalog")
    library = resolve_library(tmp_path)
    assert len(library) == 6
    assert {paper["id"] for paper in library if paper["available"]} == {chosen["id"]}
    assert resolve_paper(chosen["id"], tmp_path) == path.resolve()
    assert path.read_bytes().startswith(b"PDF content")
    assert all(set(paper) == set(chosen) | {"path", "available"} for paper in library)
    json.dumps(library, allow_nan=False)


@pytest.mark.parametrize("unknown", ["../unlisted.pdf", "transformer/../gpt3", "", "Qwen3", None, 1])
def test_open_rejects_non_catalog_identities(unknown, tmp_path):
    with pytest.raises(ValueError, match="catalog"):
        resolve_paper(unknown, tmp_path)


def test_missing_folder_and_reading_are_descriptive_and_never_created(tmp_path):
    absent = tmp_path / "missing"
    assert not any(paper["available"] for paper in resolve_library(absent))
    with pytest.raises(FileNotFoundError, match="unavailable"):
        resolve_paper("qwen3", absent)
    assert not absent.exists()


def test_default_library_is_relative_to_users_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = tmp_path / "learning/books/papers/ai-ML"
    reading = paper_specs()[0]
    expected.mkdir(parents=True)
    path = expected / reading["filename"]
    path.write_bytes(b"pdf")
    assert resolve_paper(reading["id"]) == path.resolve()


def test_a_pdf_directory_is_not_an_available_reading(tmp_path):
    (tmp_path / paper_specs()[0]["filename"]).mkdir()
    assert not resolve_library(tmp_path)[0]["available"]


def test_catalog_symlink_cannot_escape_selected_folder(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"external")
    target = root / paper_specs()[0]["filename"]
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires a Windows privilege")
    assert not resolve_library(root)[0]["available"]
    with pytest.raises(FileNotFoundError):
        resolve_paper("transformer", root)


def test_importing_catalog_does_not_load_scientific_or_gui_runtime():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    check = subprocess.run([sys.executable, "-c", "import sys; import mlcookbook.research; "
                            "assert not ({'numpy','pandas','torch','sklearn','PySide6'} & set(sys.modules))"],
                           env=env, capture_output=True, text=True, timeout=15)
    assert check.returncode == 0, check.stderr

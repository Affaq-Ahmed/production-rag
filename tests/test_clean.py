from pathlib import Path

import pytest

from rag.ingest.clean import clean
from rag.ingest.load import load


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("a<!-- overview -->b<!-- multi\nline -->c", "abc"),
        (
            'Data is stored in {{< glossary_tooltip text="etcd" term_id="etcd" >}} by default.',
            "Data is stored in etcd by default.",
        ),
        ('{{<glossary_tooltip term_id="etcd" text="etcd">}}', "etcd"),
        ('{{% heading "prerequisites" %}}', "Before you begin"),
        (
            '{{< feature-state for_k8s_version="v1.30" state="stable" >}}',
            "Feature state: stable since v1.30",
        ),
        ('{{% code_sample file="pods/simple-pod.yaml" %}}', "(Example file: pods/simple-pod.yaml)"),
        ('{{% tab name="Linux" %}}\nRun kubectl.\n{{% /tab %}}', "Linux: Run kubectl."),
        ("{{< unknown >}}keep{{< /unknown >}}{{% other %}}", "keep"),
        ("See [docs](https://example.com/a(b)).", "See docs."),
        ("a![alt](image.png)b", "ab"),
        ("a\n\n\n\nb", "a\n\nb"),
    ],
)
def test_clean_rules(source: str, expected: str) -> None:
    assert clean(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("## Pod phase {#pod-phase}", "## Pod phase"),
        ("#### Container restarts {#restart-policy}  \nText.", "#### Container restarts\nText."),
        ("### Versioned {#v1.0}", "### Versioned"),
    ],
)
def test_heading_anchor_removed(source: str, expected: str) -> None:
    assert clean(source) == expected


def test_anchor_text_outside_heading_or_in_code_is_kept() -> None:
    source = "Use {#id} syntax for anchors.\n\n```\n## not a heading {#kept}\n```"
    assert clean(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('## {{% heading "prerequisites" %}}', "## Before you begin"),
        ('## {{% heading "cleanup" %}}', "## Clean up"),
        ('### {{% heading "objectives" %}}', "### Objectives"),
        ('## {{% heading "glossary" %}}', "## glossary"),
    ],
)
def test_heading_shortcode_supplies_text_not_a_second_marker(source: str, expected: str) -> None:
    assert clean(source) == expected


@pytest.mark.parametrize(
    "heading", ['## {{% heading "whatsnext" %}}', "## What's next", "## What's Next"]
)
def test_whats_next_section_is_dropped_up_to_the_next_heading_at_its_level(heading: str) -> None:
    source = (
        "## Usage\nKeep this.\n\n"
        f"{heading}\n* Read about Pods\n\n### Deeper\n* More links\n\n"
        "## Appendix\nKeep this too."
    )
    assert clean(source) == "## Usage\nKeep this.\n\n## Appendix\nKeep this too."


def test_whats_next_in_prose_is_kept() -> None:
    source = "What's next contains a link to a more detailed example."
    assert clean(source) == source


def test_whats_next_drop_tracks_code_fences() -> None:
    source = (
        "Intro.\n\n"
        '## {{% heading "whatsnext" %}}\n```\n## not a heading\n```\n* link\n\n'
        '## Appendix\n```\n## {{% heading "whatsnext" %}}\n```'
    )
    # The fenced "## not a heading" must not end the dropped section; after it ends, a fenced
    # copy of the shortcode line is code, not a heading, so it is kept verbatim.
    assert clean(source) == ('Intro.\n\n## Appendix\n```\n## {{% heading "whatsnext" %}}\n```')


@pytest.mark.parametrize("kind", ["note", "caution", "warning"])
def test_admonitions(kind: str) -> None:
    assert clean(f"{{{{< {kind} >}}}}\nInner text.\n{{{{< /{kind} >}}}}") == (
        f"{kind.capitalize()}: Inner text."
    )


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_code_blocks_are_preserved(fence: str) -> None:
    source = (
        f"{fence}yaml\n# <!-- comment -->\n{{{{< example >}}}}\n"
        f"[link](target)\n\n\n\nkubectl get pods\n{fence}"
    )
    assert clean(source) == source


@pytest.fixture
def corpus_dir() -> Path:
    corpus = Path(__file__).resolve().parents[1] / "data/k8s-docs/content/en/docs"
    if not corpus.is_dir():
        pytest.skip("Kubernetes corpus has not been fetched")
    return corpus


def test_no_shortcodes_survive_on_real_page(corpus_dir: Path) -> None:
    doc = load(corpus_dir, corpus_dir / "concepts/workloads/pods/pod-lifecycle.md")
    assert doc is not None
    assert "{{" not in clean(doc.body)


def test_no_doubled_heading_marker_or_whats_next_on_real_page(corpus_dir: Path) -> None:
    doc = load(corpus_dir, corpus_dir / "tasks/administer-cluster/dns-custom-nameservers.md")
    assert doc is not None
    assert "{{% heading" in doc.body
    cleaned = clean(doc.body)
    assert "## ##" not in cleaned
    assert "What's next" not in cleaned and "Whatsnext" not in cleaned

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
        ('{{% heading "prerequisites" %}}', "## Prerequisites"),
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

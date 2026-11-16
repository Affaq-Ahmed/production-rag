from rag.embed import embedding_input, query_input


def test_structural_embedding_input() -> None:
    path = "Pod Lifecycle \u203a Container probes \u203a Types of probe"
    text = "A readinessProbe indicates whether the container is ready to respond to requests."
    assert embedding_input(path, text) == path + "\n\n" + text


def test_fixed_embedding_input() -> None:
    assert embedding_input("Pod Lifecycle", "Chunk text.") == "Pod Lifecycle\n\nChunk text."


def test_embedding_input_preserves_code_and_whitespace() -> None:
    text = "```yaml\n  readinessProbe: {}\n```\n"
    assert embedding_input("Probes", text) == "Probes\n\n" + text


def test_query_input_is_the_bare_query() -> None:
    assert query_input("my pod keeps restarting") == "my pod keeps restarting"


def test_query_input_has_no_heading_prefix() -> None:
    assert query_input("  my pod keeps restarting\n") == "my pod keeps restarting"
    assert query_input("q") != embedding_input("", "q")

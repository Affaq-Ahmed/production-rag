import pytest
import tiktoken

from rag.embed import batches

ENCODING = tiktoken.get_encoding("cl100k_base")


def test_input_limit() -> None:
    items = [(i, "short") for i in range(250)]
    result = list(batches(items))
    assert [len(batch) for batch in result] == [100, 100, 50]
    assert [item for batch in result for item in batch] == items


def test_token_limit() -> None:
    items = [(i, " word" * 6) for i in range(5)]
    result = list(batches(items, max_tokens=12))
    assert [len(batch) for batch in result] == [2, 2, 1]
    assert all(sum(len(ENCODING.encode(text)) for _, text in batch) <= 12 for batch in result)
    assert [item for batch in result for item in batch] == items


def test_oversized_item_is_alone() -> None:
    items = [(0, "short"), (1, " word" * 30), (2, "short")]
    assert list(batches(items, max_tokens=10)) == [[items[0]], [items[1]], [items[2]]]


def test_empty() -> None:
    assert list(batches([])) == []


def test_both_limits() -> None:
    items = [(i, " word" * (6 if i < 3 else 2)) for i in range(8)]
    result = list(batches(items, max_inputs=3, max_tokens=12))
    assert [len(batch) for batch in result] == [2, 3, 3]
    assert all(len(batch) <= 3 for batch in result)
    assert all(sum(len(ENCODING.encode(text)) for _, text in batch) <= 12 for batch in result)


@pytest.mark.parametrize(("inputs", "tokens"), [(0, 10), (10, 0), (-1, 10), (10, -1)])
def test_invalid_limits(inputs: int, tokens: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        list(batches([], max_inputs=inputs, max_tokens=tokens))


def test_literal_special_token_text() -> None:
    items = [(1, "<|endoftext|>")]
    assert list(batches(items)) == [items]

import httpx
import pytest

from medcopilot.rag.embedding_api import ApiEmbeddingFunction


@pytest.mark.parametrize(
    "data",
    [
        [None],
        ["not an object"],
        [{"index": True, "embedding": [1.0]}],
        [{"index": 0}],
        [{"index": 0, "embedding": 1.0}],
        [{"index": 0, "embedding": "1.0"}],
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
    ],
)
def test_malformed_indexed_embedding_rows_fail_without_echoing_response(data):
    response = httpx.Response(200, json={"data": data, "api_key": "private-response-value"})
    with pytest.raises(RuntimeError) as caught:
        ApiEmbeddingFunction._parse(response, len(data))
    assert "private-response-value" not in str(caught.value)


def test_indexed_embeddings_reordered_without_changing_values():
    response = httpx.Response(200, json={"data": [
        {"index": 1, "embedding": [2.0, 3.0]},
        {"index": 0, "embedding": [0.0, 1.0]},
    ]})
    assert ApiEmbeddingFunction._parse(response, 2) == [[0.0, 1.0], [2.0, 3.0]]

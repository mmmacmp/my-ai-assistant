from my_ai_assistant_offline.application.rag.splitters import get_splitter


def test_splitter_treats_special_token_literal_as_normal_text() -> None:
    text = 'Tokenizer vocabulary contains "<|endoftext|>" as an example.'

    splitter = get_splitter(
        chunk_size=64,
        summarization_type=None,
    )

    chunks = splitter.split_text(text)

    assert chunks == [text]

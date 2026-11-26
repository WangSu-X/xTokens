from unittest.mock import Mock

from x_tokens.entrypoints.serve.openai.protocol import ChatCompletionRequest
from x_tokens.entrypoints.serve.services.request_processor import process_chat_request


def test_chat_uses_the_model_template_and_returns_token_ids() -> None:
    tokenizer = Mock(chat_template="template")
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    request = ChatCompletionRequest(
        model="qwen", messages=[{"role": "user", "content": "你好"}]
    )
    prompt, _ = process_chat_request(request, tokenizer=tokenizer)

    assert prompt == (1, 2, 3)
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "你好"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )

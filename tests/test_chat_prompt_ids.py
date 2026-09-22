# SPDX-License-Identifier: MIT
"""chat_prompt_ids must return a flat list[int] regardless of what the installed
transformers returns from apply_chat_template(tokenize=True): transformers >=5
returns a BatchEncoding (dict with "input_ids"); older versions returned list[int].
Regression for the "too many dimensions 'str'" engine crash on serve."""

from superl8serve.api.request_helpers import chat_prompt_ids
from superl8serve.api.schemas import ChatMessage


class _DictLikeBatchEncoding(dict):
    """Mimics transformers.BatchEncoding: a dict with an .input_ids attribute."""

    @property
    def input_ids(self):
        return self["input_ids"]


def _msgs():
    return [ChatMessage(role="user", content="hi")]


def test_chat_prompt_ids_unwraps_batchencoding():
    class Tok:
        def apply_chat_template(self, conv, **kw):
            assert kw.get("tokenize") is True
            return _DictLikeBatchEncoding(input_ids=[1, 2, 3], attention_mask=[1, 1, 1])

    out = chat_prompt_ids(Tok(), _msgs())
    assert out == [1, 2, 3]
    assert all(isinstance(t, int) for t in out)


def test_chat_prompt_ids_plain_list_passthrough():
    class Tok:
        def apply_chat_template(self, conv, **kw):
            return [4, 5, 6]

    assert chat_prompt_ids(Tok(), _msgs()) == [4, 5, 6]


def test_chat_prompt_ids_unwraps_nested_list():
    class Tok:
        def apply_chat_template(self, conv, **kw):
            return [[7, 8, 9]]  # some versions nest per-conversation

    assert chat_prompt_ids(Tok(), _msgs()) == [7, 8, 9]

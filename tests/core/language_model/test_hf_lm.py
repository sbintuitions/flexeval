import functools
from typing import Callable

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from flexeval.core.language_model.base import LMOutput
from flexeval.core.language_model.hf_lm import (
    HuggingFaceLM,
    LanguageModel,
    decode_for_lm_continuation,
    get_prefix_and_completion_from_chat,
    tokenize_text_for_lm_continuation,
    tokenize_text_for_lm_prefix,
)


@pytest.mark.parametrize(
    "tokenizer_name",
    ["rinna/japanese-gpt2-xsmall", "line-corporation/japanese-large-lm-1.7b", "tokyotech-llm/Swallow-7b-instruct-hf"],
)
def test_output_type_and_shape_from_text_for_lm_prefix(tokenizer_name: str) -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")

    for text_list in [["これは prefix です。", "こんにちは、本文です。", ""], ["", ""]]:
        model_inputs = tokenize_text_for_lm_prefix(text_list, tokenizer)

        # check the output type and shape
        assert model_inputs.input_ids.shape == model_inputs.attention_mask.shape
        assert ((model_inputs.input_ids != tokenizer.pad_token_id) == model_inputs.attention_mask).all()

        assert model_inputs.input_ids.dtype == torch.long
        assert model_inputs.attention_mask.dtype == torch.long


@pytest.mark.parametrize(
    ("tokenizer_name", "add_special_tokens", "has_bos_tokens"),
    [
        # These tokenizers do not prepend bos tokens regardless of the add_special_tokens flag
        ("rinna/japanese-gpt2-xsmall", True, False),
        ("rinna/japanese-gpt2-xsmall", False, False),
        ("line-corporation/japanese-large-lm-1.7b", True, False),
        ("line-corporation/japanese-large-lm-1.7b", False, False),
        # These tokenizers prepend bos tokens when add_special_tokens is True
        ("tokyotech-llm/Swallow-7b-instruct-hf", True, True),
        ("tokyotech-llm/Swallow-7b-instruct-hf", False, False),
    ],
)
def test_if_tokenizer_add_bos_tokens_in_an_expected_way(
    tokenizer_name: str,
    add_special_tokens: bool,
    has_bos_tokens: bool,
) -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    for text_list in [["これは prefix です。", "こんにちは、本文です。", ""], ["", ""]]:
        model_inputs = tokenize_text_for_lm_prefix(text_list, tokenizer, add_special_tokens=add_special_tokens)
        for input_ids in model_inputs.input_ids:
            assert (tokenizer.bos_token_id in input_ids) == has_bos_tokens


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "line-corporation/japanese-large-lm-1.7b",
        "rinna/japanese-gpt-1b",
        "sbintuitions/sarashina2-7b",
        # We cannot get the tokenizer from CI because we need permission to access the model.
        # Leave this for manual testing.
        # "meta-llama/Meta-Llama-3-8B",
    ],
)
def test_tokenize_text_for_lm_continuation(tokenizer_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    # Set pad_token for tokenizers such as "meta-llama/Meta-Llama-3-8B"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # normal test cases
    # The character 'm' forms a weird token when it follows certain multi-byte characters in Llama3 tokenizer.
    text_list = ["は続き", "is continuation.", "m"]
    batch_encoding = tokenize_text_for_lm_continuation(text_list, tokenizer)
    for i, tokens in enumerate(batch_encoding.input_ids):
        first_token = tokenizer.convert_ids_to_tokens([tokens[0]])[0]
        assert not first_token.startswith("▁")  # check if the prefix of sentencepiece is not added
        assert tokenizer.decode(tokens, skip_special_tokens=True) == text_list[i]

    # Test with conditional operations
    # This is mainly for tokenizers with add_prefix_space=True,
    # which adds a space to the beginning of the text but not to the continuation.
    text_list = ["これは文頭", "これは続き"]
    as_continuation = [False, True]
    batch_encoding = tokenize_text_for_lm_continuation(text_list, tokenizer, as_continuation=as_continuation)
    for i, (tokens, as_cont) in enumerate(zip(batch_encoding.input_ids, as_continuation)):
        first_token = tokenizer.convert_ids_to_tokens([tokens[0]])[0]
        starts_with_prefix = (not as_cont) and tokenizer.add_prefix_space
        assert first_token.startswith("▁") == starts_with_prefix
        assert tokenizer.decode(tokens, skip_special_tokens=True) == text_list[i]


@pytest.mark.parametrize(
    "tokenizer_name",
    ["sbintuitions/sarashina2-7b", "llm-jp/llm-jp-3-3.7b", "meta-llama/Meta-Llama-3-8B", "Qwen/Qwen2.5-0.5B"],
)
@pytest.mark.parametrize(
    "text", ["def foo():\n", "    return 1", "こんにちは世界", "<|im_start|>Hello<|end_of_text|>Yes"]
)
def test_decode_for_lm_continuation(tokenizer_name: str, text: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    # First we need to check if the tokenizer does not change the text
    assert tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"]) == text

    # Simulate generated tokens at various text boundaries
    for i in range(1, len(text) - 1):
        prefix = text[:i]
        continuation = text[i:]
        prefix_tokens = tokenize_text_for_lm_prefix([prefix], tokenizer).input_ids[0].tolist()
        continuation_tokens = tokenize_text_for_lm_continuation([continuation], tokenizer).input_ids[0].tolist()
        prefix = tokenizer.decode(prefix_tokens, skip_special_tokens=False)
        # The point is, just decoding the continuation_tokens as follows sometimes can not restore the original text.
        # `continuation = tokenizer.decode(continuation_tokens, skip_special_tokens=True)`
        continuation = decode_for_lm_continuation(continuation_tokens, prefix_tokens, tokenizer)
        assert prefix + continuation == text


@pytest.fixture(scope="module")
def lm_init_func(model: str = "sbintuitions/tiny-lm") -> Callable[..., HuggingFaceLM]:
    # use float32 because half precision is not supported in some hardware
    return functools.partial(
        HuggingFaceLM,
        model=model,
        model_kwargs={"torch_dtype": "float32"},
        tokenizer_kwargs={"use_fast": False},
    )


@pytest.fixture(scope="module")
def lm() -> HuggingFaceLM:
    # use float32 because half precision is not supported in some hardware
    return HuggingFaceLM(
        model="sbintuitions/tiny-lm",
        model_kwargs={"torch_dtype": "float32"},
        tokenizer_kwargs={"use_fast": False},
    )


def test_batch_complete_text(lm: HuggingFaceLM) -> None:
    completions = lm.batch_complete_text(["こんにちは、", "おはよう、"])
    assert len(completions) == 2
    assert isinstance(completions[0], LMOutput)
    assert isinstance(completions[0].text, str)
    assert isinstance(completions[0].finish_reason, str)


def test_complete_text(lm: HuggingFaceLM) -> None:
    completion = lm.complete_text("こんにちは、")
    assert isinstance(completion, LMOutput)
    assert isinstance(completion.text, str)
    assert isinstance(completion.finish_reason, str)

    completions = lm.batch_complete_text(["こんにちは、", "おはよう、"])
    assert len(completions) == 2
    assert isinstance(completions[0], LMOutput)
    assert isinstance(completions[0].text, str)
    assert isinstance(completions[0].finish_reason, str)


def test_batch_complete_text_is_not_affected_by_batch(lm: LanguageModel) -> None:
    single_batch_input = ["こんにちは。今日もいい天気。"]
    multi_batch_inputs = ["こんにちは。今日もいい天気。", "Lorem ipsum"]

    gen_kwargs = {"do_sample": False, "stop_sequences": ["。"], "max_length": 100}
    completions_without_batch = lm.batch_complete_text(single_batch_input, **gen_kwargs)
    completions_with_batch = lm.batch_complete_text(multi_batch_inputs, **gen_kwargs)
    assert completions_without_batch[0].text == completions_with_batch[0].text
    assert completions_without_batch[0].finish_reason == completions_with_batch[0].finish_reason


def test_max_tokens(lm: LanguageModel) -> None:
    # assume that the lm will repeat 0
    completion = lm.batch_complete_text(["0 0 0 0 0 0 0 0 0 0"], max_new_tokens=1)[0]
    assert len(completion.text.strip()) == 1
    assert completion.finish_reason == "length"


def test_stop_sequences(lm: LanguageModel) -> None:
    # assume that the lm will repeat "10"
    completion = lm.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["1"], max_new_tokens=10)[0]
    assert completion.text.strip() == ""
    assert completion.finish_reason == "stop"

    completion = lm.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["0"], max_new_tokens=10)[0]
    assert completion.text.strip() == "1"
    assert completion.finish_reason == "stop"


def test_compute_log_probs(lm: LanguageModel) -> None:
    log_prob = lm.compute_log_probs("こんにちは")
    assert isinstance(log_prob, float)

    log_probs = lm.batch_compute_log_probs(["こんにちは", "こんばんは"])
    assert len(log_probs) == 2
    assert isinstance(log_probs[0], float)


def test_batch_compute_log_probs_produces_reasonable_comparisons(lm: LanguageModel) -> None:
    # test if the shorter sentence has higher log prob
    log_probs = lm.batch_compute_log_probs(["これは正しい日本語です。", "これは正しい日本語です。そして…"])
    assert log_probs[0] > log_probs[1]

    # test if the more natural short phrase has higher log prob
    log_probs = lm.batch_compute_log_probs(["こんにちは", "コニチハ"])
    assert log_probs[0] > log_probs[1]

    # test if the grammatical sentence has higher log prob
    log_probs = lm.batch_compute_log_probs(["これは正しい日本語です。", "は正いしこれで日語本す。"])
    assert log_probs[0] > log_probs[1]

    # test if the right prefix reduces the log prob
    log_probs = lm.batch_compute_log_probs(["富士山", "富士山"], prefix_list=["日本で一番高い山は", "Yes, we are"])
    assert log_probs[0] > log_probs[1]


def test_batch_compute_log_probs_is_not_affected_by_batch(lm: LanguageModel) -> None:
    # test if the shorter sentence has higher log prob
    log_probs_without_batch = lm.batch_compute_log_probs(["これは正しい日本語です。"])

    log_probs_with_batch = lm.batch_compute_log_probs(
        ["これは正しい日本語です。", "これは正しい日本語です。padding を作るために余計な文を入れます。"],
    )

    assert round(log_probs_without_batch[0], 4) == round(log_probs_with_batch[0], 4)


def test_if_random_seed_fixes_the_lm_outputs(lm_init_func: Callable[..., HuggingFaceLM]) -> None:
    # first check if the outputs are different without fixing the seed
    completions = set()
    for i in range(3):
        lm = lm_init_func(random_seed=i)
        completion = lm.batch_complete_text(["<s>"], do_sample=True)[0]
        completions.add(completion.text)
    assert len(completions) > 1

    # then check if the outputs are the same with fixing the seed
    completions = set()
    for _ in range(3):
        lm = lm_init_func(random_seed=42)
        completion = lm.batch_complete_text(["<s>"], do_sample=True)[0]
        completions.add(completion.text)
    assert len(completions) == 1

    # note that the randomness starts in __init__
    # so if you sample outputs from the same instance, the outputs will be different
    lm = lm_init_func(random_seed=42)
    completions = set()
    for _ in range(3):
        completion = lm.batch_complete_text(["<s>"], do_sample=True)[0]
        completions.add(completion.text)
    assert len(completions) > 1


@pytest.fixture(scope="module")
def chat_lm(model_name: str = "sbintuitions/tiny-lm-chat") -> HuggingFaceLM:
    return HuggingFaceLM(model=model_name, model_kwargs={"torch_dtype": "float32"})


def test_batch_generate_chat_response(chat_lm: LanguageModel) -> None:
    responses = chat_lm.batch_generate_chat_response([[{"role": "user", "content": "こんにちは。"}]], max_length=40)
    assert len(responses) == 1
    assert isinstance(responses[0], LMOutput)
    assert isinstance(responses[0].text, str)
    assert isinstance(responses[0].finish_reason, str)


def test_generate_chat_response(chat_lm: LanguageModel) -> None:
    response = chat_lm.generate_chat_response([{"role": "user", "content": "こんにちは。"}], max_length=40)
    assert isinstance(response, LMOutput)
    assert isinstance(response.text, str)
    assert isinstance(response.finish_reason, str)

    responses = chat_lm.generate_chat_response(
        [
            [{"role": "user", "content": "こんにちは。"}],
            [{"role": "user", "content": "こんばんわ"}],
        ],
        max_length=40,
    )
    assert len(responses) == 2
    assert isinstance(responses[0], LMOutput)
    assert isinstance(responses[0].text, str)
    assert isinstance(responses[0].finish_reason, str)


def test_if_custom_chat_template_is_given(lm_init_func: Callable[..., HuggingFaceLM]) -> None:
    # To verify that the template specified in `custom_chat_template` is passed to `tokenizer.apply_chat_template()`,
    # prepare a template where the model is expected to output "0 0..." for any input.
    custom_chat_template = "0 0 0 0 0 0 0 0 0 0 0"
    lm = lm_init_func(
        random_seed=42,
        custom_chat_template=custom_chat_template,
    )
    responses = lm.batch_generate_chat_response([[{"role": "user", "content": "こんにちは。"}]], max_length=40)
    assert len(responses) == 1
    assert responses[0].text.strip().startswith("0 0")


def test_if_stop_sequences_work_as_expected(chat_lm: HuggingFaceLM) -> None:
    test_inputs = [[{"role": "user", "content": "こんにちは"}]]

    # check if the response does not have eos_token by default
    response = chat_lm.batch_generate_chat_response(test_inputs, max_new_tokens=50)[0]
    assert response.text
    assert response.finish_reason == "stop"

    # check if ignore_eos=True works
    response = chat_lm.batch_generate_chat_response(test_inputs, max_new_tokens=50, ignore_eos=True)[0]
    assert response.text
    assert response.finish_reason == "length"


def test_if_gen_kwargs_work_as_expected() -> None:
    lm = HuggingFaceLM(model="sbintuitions/tiny-lm", default_gen_kwargs={"max_new_tokens": 1})
    # check if the default gen_kwargs is used and the max_new_tokens is 1
    text = lm.complete_text("000000")
    assert len(text.text) == 1

    # check if the gen_kwargs will be overwritten by the given gen_kwargs
    text = lm.complete_text("000000", max_new_tokens=10)
    assert len(text.text) > 1


def test_get_prefix_and_completion_from_chat() -> None:
    tokenizer = AutoTokenizer.from_pretrained("sbintuitions/tiny-lm-chat", padding_side="left")
    prefix, completion = get_prefix_and_completion_from_chat(
        [{"role": "user", "content": "Hello."}], {"role": "assistant", "content": "Hi."}, tokenizer=tokenizer
    )
    assert prefix == "<|user|>Hello.</s><|assistant|>"
    assert completion == "Hi.</s>"

    prefix, completion = get_prefix_and_completion_from_chat(
        [{"role": "user", "content": "Hello."}],
        {"role": "assistant", "content": "Hi."},
        tokenizer=tokenizer,
        custom_chat_template="CUSTOM_TEMPLATE",
    )
    assert prefix == "CUSTOM_TEMPLATE"
    assert completion == ""


def test_batch_compute_chat_log_probs(chat_lm: HuggingFaceLM) -> None:
    log_probs_natural = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Hello, how are you?"}]],
        [{"role": "assistant", "content": "Good."}],
    )
    log_probs_unnatural_lang = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Hello, how are you?"}]],
        [{"role": "assistant", "content": "!?本日は晴天ナリ."}],
    )
    log_probs_unnatural_ord = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Good."}]],
        [{"role": "assistant", "content": "Hello, how are you?"}],
    )

    assert len(log_probs_natural) == 1
    assert isinstance(log_probs_natural[0], float)
    assert len(log_probs_unnatural_lang) == 1
    assert isinstance(log_probs_unnatural_lang[0], float)
    assert len(log_probs_unnatural_ord) == 1
    assert isinstance(log_probs_unnatural_ord[0], float)
    assert log_probs_natural[0] > log_probs_unnatural_lang[0]
    assert log_probs_natural[0] > log_probs_unnatural_ord[0]


def test_compute_chat_log_probs(chat_lm: HuggingFaceLM) -> None:
    prompt = [{"role": "user", "content": "Hello, how are you?"}]
    response = {"role": "assistant", "content": "Good."}
    log_prob = chat_lm.compute_chat_log_probs(prompt, response)
    assert isinstance(log_prob, float)
    batch_log_prob = chat_lm.batch_compute_chat_log_probs([prompt], [response])
    assert log_prob == batch_log_prob[0]

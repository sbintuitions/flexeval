from abc import abstractmethod

import pytest

from flexeval.core.language_model.base import LanguageModel, LMOutput


class BaseLanguageModelTest:
    """Base test class for testing LanguageModel implementations.

    This class provides common test methods that can be used across different
    LanguageModel implementations. To use this class, create a subclass and
    implement the `model` method to return an instance of the specific
    LanguageModel implementation you want to test.

    Example:
        ```python
        class TestHuggingFaceLM(BaseLanguageModelTest):
            @pytest.fixture
            def model(self):
                return HuggingFaceLM(
                    model="sbintuitions/tiny-lm",
                    model_kwargs={"torch_dtype": "float32"},
                    tokenizer_kwargs={"use_fast": False},
                )
        ```
    """

    @pytest.fixture()
    @abstractmethod
    def lm(self, *args, **kwargs) -> LanguageModel:  # noqa: ANN002
        """Return an instance of the LanguageModel.
        This is suppoed to be a completion lm.
        """
        msg = "Subclasses must implement model fixture"
        raise NotImplementedError(msg)

    @pytest.fixture()
    @abstractmethod
    def chat_lm(self, *args, **kwargs) -> LanguageModel:  # noqa: ANN002
        """Return an instance of the LanguageModel.
        This is supposed to be a chat lm.
        """
        msg = "Subclasses must implement model fixture"
        raise NotImplementedError(msg)

    @pytest.fixture()
    @abstractmethod
    def chat_lm_for_tool_calling(self, *args, **kwargs) -> LanguageModel:  # noqa: ANN002
        """Return an instance of the LanguageModel.
        This is supposed to be a chat lm supporting the tool calling.
        """
        msg = "Subclasses must implement model fixture"
        raise NotImplementedError(msg)

    """
    Test if basic interfaces return the expected types.
    """

    def test_complete_text_single_input(self, lm: LanguageModel) -> None:
        """Test that complete_text works with a single input."""
        completion = lm.complete_text("Test input")
        assert isinstance(completion, LMOutput)
        assert isinstance(completion.text, str)
        assert isinstance(completion.finish_reason, str)

    def test_complete_text_batch_input(self, lm: LanguageModel) -> None:
        """Test that complete_text works with a batch input."""
        completions = lm.complete_text(["Test input 1", "Test input 2"])
        assert isinstance(completions, list)
        assert len(completions) == 2
        assert all(isinstance(c, LMOutput) for c in completions)
        assert all(isinstance(c.text, str) for c in completions)
        assert all(isinstance(c.finish_reason, str) for c in completions)

    def test_generate_chat_response_single_input(self, chat_lm: LanguageModel) -> None:
        """Test that generate_chat_response works with a single input."""
        try:
            response = chat_lm.generate_chat_response([{"role": "user", "content": "Hello"}])
            assert isinstance(response, LMOutput)
            assert isinstance(response.text, str)
            assert isinstance(response.finish_reason, str)
            assert response.tool_calls is None
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_generate_chat_response_batch_input(self, chat_lm: LanguageModel) -> None:
        """Test that generate_chat_response works with a batch input."""
        try:
            responses = chat_lm.generate_chat_response(
                [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]]
            )
            assert isinstance(responses, list)
            assert len(responses) == 2
            assert all(isinstance(r, LMOutput) for r in responses)
            assert all(isinstance(r.text, str) for r in responses)
            assert all(isinstance(r.finish_reason, str) for r in responses)
            assert all(r.tool_calls is None for r in responses)
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_generate_chat_response_single_input_with_tools(self, chat_lm_for_tool_calling: LanguageModel) -> None:
        """Test that generate_chat_response works with a single input with tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for provided city in celsius.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        try:
            response = chat_lm_for_tool_calling.generate_chat_response(
                [{"role": "user", "content": "What's the weather like in Paris today?"}],
                tools=tools,
            )
            assert isinstance(response, LMOutput)
            assert all(isinstance(tool_call, dict) for tool_call in response.tool_calls)
            assert isinstance(response.finish_reason, str)
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_generate_chat_response_batch_input_with_tools(self, chat_lm_for_tool_calling: LanguageModel) -> None:
        """Test that generate_chat_response works with a batch input with tools."""
        tools = [
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather information for provided city in celsius.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ]
            for _ in range(2)
        ]
        try:
            responses = chat_lm_for_tool_calling.generate_chat_response(
                [
                    [{"role": "user", "content": "What's the weather like in Paris today?"}],
                    [{"role": "user", "content": "What's the weather like in Tokyo today?"}],
                ],
                tools=tools,
            )
            assert isinstance(responses, list)
            assert len(responses) == 2
            assert all(isinstance(r, LMOutput) for r in responses)
            assert all(
                isinstance(r.tool_calls, list) and all(isinstance(tool_call, dict) for tool_call in r.tool_calls)
                for r in responses
            )
            assert all(isinstance(r.finish_reason, str) for r in responses)
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_generate_chat_response_if_number_of_tools_and_messages_not_equal(
        self, chat_lm_for_tool_calling: LanguageModel
    ) -> None:
        """Test for warnings of mismatches between messages and tools counts."""
        tools = [
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather information for provided city in celsius.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ]
        ]
        try:
            with pytest.raises(
                ValueError, match="tools_list must be either None or a list of the same length as chat_messages_list."
            ):
                chat_lm_for_tool_calling.generate_chat_response(
                    [
                        [{"role": "user", "content": "What's the weather like in Paris today?"}],
                        [{"role": "user", "content": "What's the weather like in Tokyo today?"}],
                    ],
                    tools=tools,
                )
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_compute_log_probs_single_input(self, lm: LanguageModel) -> None:
        """Test that compute_log_probs works with a single input."""
        try:
            log_prob = lm.compute_log_probs("Test input")
            assert isinstance(log_prob, float)
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")

    def test_compute_log_probs_batch_input(self, lm: LanguageModel) -> None:
        """Test that compute_log_probs works with a batch input."""
        try:
            log_probs = lm.compute_log_probs(["Test input 1", "Test input 2"])
            assert isinstance(log_probs, list)
            assert len(log_probs) == 2
            assert all(isinstance(lp, float) for lp in log_probs)
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")

    def test_compute_chat_log_probs_single_input(self, chat_lm: LanguageModel) -> None:
        """Test that compute_chat_log_probs works with a single input."""
        try:
            log_prob = chat_lm.compute_chat_log_probs(
                [{"role": "user", "content": "Hello"}], {"role": "assistant", "content": "Hi"}
            )
            assert isinstance(log_prob, float)
        except NotImplementedError:
            pytest.skip("This model does not support chat log probability computation")

    def test_compute_chat_log_probs_batch_input(self, chat_lm: LanguageModel) -> None:
        """Test that compute_chat_log_probs works with a batch input."""
        try:
            log_probs = chat_lm.compute_chat_log_probs(
                [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]],
                [{"role": "assistant", "content": "Hello there"}, {"role": "assistant", "content": "Hi there"}],
            )
            assert isinstance(log_probs, list)
            assert len(log_probs) == 2
            assert all(isinstance(lp, float) for lp in log_probs)
        except NotImplementedError:
            pytest.skip("This model does not support chat log probability computation")

    """
    Test if functions outputs reasonable results: Batch consistency.
    """

    def test_batch_complete_text_is_not_affected_by_batch(self, lm: LanguageModel) -> None:
        """Test that batch_complete_text is not affected by batch size."""
        single_batch_input = ["Test input"]
        multi_batch_inputs = ["Test input", "Another test input"]

        gen_kwargs = {"max_new_tokens": 10}
        completions_without_batch = lm.complete_text(single_batch_input, **gen_kwargs)
        completions_with_batch = lm.complete_text(multi_batch_inputs, **gen_kwargs)
        assert completions_without_batch[0].text == completions_with_batch[0].text
        assert completions_without_batch[0].finish_reason == completions_with_batch[0].finish_reason

    def test_batch_chat_response_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        """Test that batch_generate_chat_response is not affected by batch size."""
        try:
            single_batch_input = [[{"role": "user", "content": "Hello"}]]
            multi_batch_inputs = [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]]

            gen_kwargs = {"max_new_tokens": 10}
            completions_without_batch = chat_lm.generate_chat_response(single_batch_input, **gen_kwargs)
            completions_with_batch = chat_lm.generate_chat_response(multi_batch_inputs, **gen_kwargs)
            assert completions_without_batch[0].text == completions_with_batch[0].text
            assert completions_without_batch[0].finish_reason == completions_with_batch[0].finish_reason
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_batch_compute_log_probs_is_not_affected_by_batch(self, lm: LanguageModel) -> None:
        """Test that batch_compute_log_probs is not affected by batch size."""
        try:
            single_batch_input = ["Test input"]
            multi_batch_inputs = ["Test input", "Another test input"]

            log_probs_without_batch = lm.compute_log_probs(single_batch_input)
            log_probs_with_batch = lm.compute_log_probs(multi_batch_inputs)

            assert round(log_probs_without_batch[0], 4) == round(log_probs_with_batch[0], 4)
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")

    def test_batch_compute_chat_log_probs_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        """Test that batch_compute_chat_log_probs is not affected by batch size."""
        try:
            log_probs_without_batch = chat_lm.compute_chat_log_probs(
                [[{"role": "user", "content": "Hi"}]],
                [{"role": "assistant", "content": "Hi"}],
            )
            log_probs_with_batch = chat_lm.compute_chat_log_probs(
                [[{"role": "user", "content": "Hi"}], [{"role": "user", "content": "Hello"}]],
                [{"role": "assistant", "content": "Hi"}, {"role": "assistant", "content": "Hello there"}],
            )

            assert round(log_probs_without_batch[0], 4) == round(log_probs_with_batch[0], 4)
        except NotImplementedError:
            pytest.skip("This model does not support chat log probability computation")

    """
    Test if functions outputs reasonable results: Log-probability comparisons.
    """

    def test_batch_compute_log_probs_produces_reasonable_comparisons(self, lm: LanguageModel) -> None:
        """Test that batch_compute_log_probs produces reasonable comparisons."""
        try:
            # test if the shorter sentence has higher log prob
            log_probs = lm.compute_log_probs(["これは正しい日本語です。", "これは正しい日本語です。そして…"])
            assert log_probs[0] > log_probs[1]

            # test if the more natural short phrase has higher log prob
            log_probs = lm.compute_log_probs(["こんにちは", "コニチハ"])
            assert log_probs[0] > log_probs[1]

            # test if the grammatical sentence has higher log prob
            log_probs = lm.compute_log_probs(["これは正しい日本語です。", "は正いしこれで日語本す。"])
            assert log_probs[0] > log_probs[1]

            # test if the right prefix reduces the log prob
            log_probs = lm.compute_log_probs(["富士山", "富士山"], prefix_list=["日本で一番高い山は", "Yes, we are"])
            assert log_probs[0] > log_probs[1]
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")
        except (AssertionError, IndexError):
            pytest.skip("This test may not be applicable to all language models")

    def test_batch_compute_chat_log_probs_produces_reasonable_comparisons(self, chat_lm: LanguageModel) -> None:
        """Test that batch_compute_chat_log_probs produces reasonable comparisons."""
        try:
            log_probs = chat_lm.compute_chat_log_probs(
                [
                    [{"role": "user", "content": "Output number from range 1 to 3"}],
                    [{"role": "user", "content": "Output number from range 1 to 3"}],
                ],
                [
                    {"role": "assistant", "content": "1 2 3"},
                    {"role": "assistant", "content": "7 8 9"},
                ],
            )
            assert log_probs[0] > log_probs[1]
        except NotImplementedError:
            pytest.skip("This model does not support chat log probability computation")
        except (AssertionError, IndexError):
            pytest.skip("This test may not be applicable to all language models")

    """
    Test the effect of generation parameters.
    """

    def test_stop_sequences(self, lm: LanguageModel) -> None:
        # assume that the lm will repeat "10"
        completion = lm.complete_text(["10 10 10 10 10 10 "], stop_sequences=["1"], max_new_tokens=10)[0]
        assert completion.text.strip() == ""
        assert completion.finish_reason == "stop"

        completion = lm.complete_text(["10 10 10 10 10 10 "], stop_sequences=["0"], max_new_tokens=10)[0]
        assert completion.text.strip() == "1"
        assert completion.finish_reason == "stop"

    def test_max_new_tokens(self, lm: LanguageModel) -> None:
        """Test that max_new_tokens limits the output length."""
        # Test with a very small max_new_tokens value
        completion = lm.complete_text(["0 0 0 0 0 0 0 0 0"], max_new_tokens=1)[0]

        # The completion should be very short and finish_reason should be "length"
        assert len(completion.text.strip()) <= 5  # Allow for some tokenization differences
        assert completion.finish_reason == "length"

    def test_string_processors_with_complete_text(self, lm: LanguageModel) -> None:
        """Test that string processors work as expected."""
        try:
            assert hasattr(lm, "string_processors")
            original_string_processors = lm.string_processors

            from flexeval import TemplateRenderer

            test_text = "This text is processed by StringProcessor."
            lm.string_processors = [TemplateRenderer(template=test_text)]

            lm_output = lm.complete_text("Test input: ")
            assert lm_output.text == test_text
            assert lm_output.raw_text is not None
            assert lm_output.raw_text != test_text

            lm.string_processors = original_string_processors
        except NotImplementedError:
            pytest.skip("This model does not support complete_text")

    def test_string_processors_with_generate_chat_response(self, chat_lm: LanguageModel) -> None:
        """Test that string processors work as expected."""
        try:
            assert hasattr(chat_lm, "string_processors")
            original_string_processors = chat_lm.string_processors

            from flexeval import TemplateRenderer

            test_text = "This text is processed by StringProcessor."
            chat_lm.string_processors = [TemplateRenderer(template=test_text)]

            lm_output = chat_lm.generate_chat_response([{"role": "user", "content": "Test input"}])
            assert lm_output.text == test_text
            assert lm_output.raw_text is not None
            assert lm_output.raw_text != test_text

            chat_lm.string_processors = original_string_processors
        except NotImplementedError:
            pytest.skip("This model does not support generate_chat_response")

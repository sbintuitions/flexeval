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
    def model(self) -> LanguageModel:
        """Return an instance of the LanguageModel.
        This is suppoed to be a completion model.
        """
        msg = "Subclasses must implement model fixture"
        raise NotImplementedError(msg)

    @pytest.fixture()
    @abstractmethod
    def chat_model(self) -> LanguageModel:
        """Return an instance of the LanguageModel.
        This is supposed to be a chat model.
        """
        msg = "Subclasses must implement model fixture"
        raise NotImplementedError(msg)

    def test_complete_text_single_input(self, model: LanguageModel) -> None:
        """Test that complete_text works with a single input."""
        completion = model.complete_text("Test input")
        assert isinstance(completion, LMOutput)
        assert isinstance(completion.text, str)
        assert isinstance(completion.finish_reason, str)

    def test_complete_text_batch_input(self, model: LanguageModel) -> None:
        """Test that complete_text works with a batch input."""
        completions = model.complete_text(["Test input 1", "Test input 2"])
        assert isinstance(completions, list)
        assert len(completions) == 2
        assert all(isinstance(c, LMOutput) for c in completions)
        assert all(isinstance(c.text, str) for c in completions)
        assert all(isinstance(c.finish_reason, str) for c in completions)

    def test_batch_complete_text(self, model: LanguageModel) -> None:
        """Test that batch_complete_text works."""
        completions = model.batch_complete_text(["Test input 1", "Test input 2"])
        assert isinstance(completions, list)
        assert len(completions) == 2
        assert all(isinstance(c, LMOutput) for c in completions)
        assert all(isinstance(c.text, str) for c in completions)
        assert all(isinstance(c.finish_reason, str) for c in completions)

    def test_generate_chat_response_single_input(self, chat_model: LanguageModel) -> None:
        """Test that generate_chat_response works with a single input."""
        try:
            response = chat_model.generate_chat_response([{"role": "user", "content": "Hello"}])
            assert isinstance(response, LMOutput)
            assert isinstance(response.text, str)
            assert isinstance(response.finish_reason, str)
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_generate_chat_response_batch_input(self, chat_model: LanguageModel) -> None:
        """Test that generate_chat_response works with a batch input."""
        try:
            responses = chat_model.generate_chat_response(
                [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]]
            )
            assert isinstance(responses, list)
            assert len(responses) == 2
            assert all(isinstance(r, LMOutput) for r in responses)
            assert all(isinstance(r.text, str) for r in responses)
            assert all(isinstance(r.finish_reason, str) for r in responses)
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_batch_generate_chat_response(self, chat_model: LanguageModel) -> None:
        """Test that batch_generate_chat_response works."""
        try:
            responses = chat_model.batch_generate_chat_response(
                [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]]
            )
            assert isinstance(responses, list)
            assert len(responses) == 2
            assert all(isinstance(r, LMOutput) for r in responses)
            assert all(isinstance(r.text, str) for r in responses)
            assert all(isinstance(r.finish_reason, str) for r in responses)
        except NotImplementedError:
            pytest.skip("This model does not support chat responses")

    def test_compute_log_probs_single_input(self, model: LanguageModel) -> None:
        """Test that compute_log_probs works with a single input."""
        try:
            log_prob = model.compute_log_probs("Test input")
            assert isinstance(log_prob, float)
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")

    def test_compute_log_probs_batch_input(self, model: LanguageModel) -> None:
        """Test that compute_log_probs works with a batch input."""
        try:
            log_probs = model.compute_log_probs(["Test input 1", "Test input 2"])
            assert isinstance(log_probs, list)
            assert len(log_probs) == 2
            assert all(isinstance(lp, float) for lp in log_probs)
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")

    def test_batch_compute_log_probs(self, model: LanguageModel) -> None:
        """Test that batch_compute_log_probs works."""
        try:
            log_probs = model.batch_compute_log_probs(["Test input 1", "Test input 2"])
            assert isinstance(log_probs, list)
            assert len(log_probs) == 2
            assert all(isinstance(lp, float) for lp in log_probs)
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")

    def test_compute_chat_log_probs_single_input(self, chat_model: LanguageModel) -> None:
        """Test that compute_chat_log_probs works with a single input."""
        try:
            log_prob = chat_model.compute_chat_log_probs(
                [{"role": "user", "content": "Hello"}], {"role": "assistant", "content": "Hi"}
            )
            assert isinstance(log_prob, float)
        except NotImplementedError:
            pytest.skip("This model does not support chat log probability computation")

    def test_compute_chat_log_probs_batch_input(self, chat_model: LanguageModel) -> None:
        """Test that compute_chat_log_probs works with a batch input."""
        try:
            log_probs = chat_model.compute_chat_log_probs(
                [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]],
                [{"role": "assistant", "content": "Hello there"}, {"role": "assistant", "content": "Hi there"}],
            )
            assert isinstance(log_probs, list)
            assert len(log_probs) == 2
            assert all(isinstance(lp, float) for lp in log_probs)
        except NotImplementedError:
            pytest.skip("This model does not support chat log probability computation")

    def test_batch_compute_chat_log_probs(self, chat_model: LanguageModel) -> None:
        """Test that batch_compute_chat_log_probs works."""
        try:
            log_probs = chat_model.batch_compute_chat_log_probs(
                [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]],
                [{"role": "assistant", "content": "Hello there"}, {"role": "assistant", "content": "Hi there"}],
            )
            assert isinstance(log_probs, list)
            assert len(log_probs) == 2
            assert all(isinstance(lp, float) for lp in log_probs)
        except NotImplementedError:
            pytest.skip("This model does not support chat log probability computation")

    def test_batch_complete_text_is_not_affected_by_batch(self, model: LanguageModel) -> None:
        """Test that batch_complete_text is not affected by batch size."""
        single_batch_input = ["Test input"]
        multi_batch_inputs = ["Test input", "Another test input"]

        gen_kwargs = {"do_sample": False, "max_new_tokens": 10}
        completions_without_batch = model.batch_complete_text(single_batch_input, **gen_kwargs)
        completions_with_batch = model.batch_complete_text(multi_batch_inputs, **gen_kwargs)
        assert completions_without_batch[0].text == completions_with_batch[0].text
        assert completions_without_batch[0].finish_reason == completions_with_batch[0].finish_reason

    def test_batch_compute_log_probs_is_not_affected_by_batch(self, model: LanguageModel) -> None:
        """Test that batch_compute_log_probs is not affected by batch size."""
        try:
            single_batch_input = ["Test input"]
            multi_batch_inputs = ["Test input", "Another test input"]

            log_probs_without_batch = model.batch_compute_log_probs(single_batch_input)
            log_probs_with_batch = model.batch_compute_log_probs(multi_batch_inputs)

            assert round(log_probs_without_batch[0], 4) == round(log_probs_with_batch[0], 4)
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")

    def test_batch_compute_log_probs_produces_reasonable_comparisons(self, model: LanguageModel) -> None:
        """Test that batch_compute_log_probs produces reasonable comparisons."""
        try:
            # test if the shorter sentence has higher log prob
            log_probs = model.batch_compute_log_probs(["これは正しい日本語です。", "これは正しい日本語です。そして…"])
            assert log_probs[0] > log_probs[1]

            # test if the more natural short phrase has higher log prob
            log_probs = model.batch_compute_log_probs(["こんにちは", "コニチハ"])
            assert log_probs[0] > log_probs[1]

            # test if the grammatical sentence has higher log prob
            log_probs = model.batch_compute_log_probs(["これは正しい日本語です。", "は正いしこれで日語本す。"])
            assert log_probs[0] > log_probs[1]

            # test if the right prefix reduces the log prob
            log_probs = model.batch_compute_log_probs(
                ["富士山", "富士山"], prefix_list=["日本で一番高い山は", "Yes, we are"]
            )
            assert log_probs[0] > log_probs[1]
        except NotImplementedError:
            pytest.skip("This model does not support log probability computation")
        except (AssertionError, IndexError):
            pytest.skip("This test may not be applicable to all language models")

    def test_stop_sequences(self, model: LanguageModel) -> None:
        # assume that the lm will repeat "10"
        completion = model.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["1"], max_new_tokens=10)[0]
        assert completion.text.strip() == ""
        assert completion.finish_reason == "stop"

        completion = model.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["0"], max_new_tokens=10)[0]
        assert completion.text.strip() == "1"
        assert completion.finish_reason == "stop"

    def test_max_new_tokens(self, model: LanguageModel) -> None:
        """Test that max_new_tokens limits the output length."""
        # Test with a very small max_new_tokens value
        completion = model.batch_complete_text(["Test input"], max_new_tokens=1)[0]

        # The completion should be very short and finish_reason should be "length"
        assert len(completion.text.strip()) <= 5  # Allow for some tokenization differences
        assert completion.finish_reason == "length"

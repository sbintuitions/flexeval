from __future__ import annotations

import os
from unittest import mock

import pytest

from flexeval.core.language_model.base import LanguageModel
from flexeval.core.language_model.vllm_serve_lm import VLLMServeLM, VLLMServerManager
from tests.conftest import is_vllm_enabled
from tests.core.language_model.base import BaseLanguageModelTest


class DummyStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._iter = iter(lines)

    def readline(self) -> str:
        try:
            return next(self._iter)
        except StopIteration:
            return ""


@pytest.fixture(scope="module")
def chat_lm() -> VLLMServeLM:
    openai_api_key = os.environ.pop("OPENAI_API_KEY", None)
    llm = VLLMServeLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "tokenizer_mode": "slow",
        },
    )
    yield llm
    llm.manager.stop()
    if openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = openai_api_key


def test_port_is_auto_assigned() -> None:
    with mock.patch("socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
        manager = VLLMServerManager(model="dummy", model_kwargs={})
        assert manager.port == 12345


def test_start_invokes_subprocess_and_waits_for_http() -> None:
    dummy_stdout = DummyStream(["booting...\n", "running...\n", ""])
    dummy_stderr = DummyStream([""])

    mock_popen = mock.Mock()
    mock_popen.stdout = dummy_stdout
    mock_popen.stderr = dummy_stderr
    mock_popen.poll.return_value = None
    mock_popen.wait.return_value = 0
    mock_popen.terminate.return_value = None

    with mock.patch("subprocess.Popen", return_value=mock_popen), mock.patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200

        manager = VLLMServerManager(model="dummy")
        host, port = manager.start()
        manager.stop()
        assert host == "localhost"
        assert isinstance(port, int)
        mock_get.assert_called_with(f"http://localhost:{port}/v1/models", timeout=1)


def test_stop_terminates_process() -> None:
    mock_process = mock.Mock()
    mock_process.poll.return_value = None
    mock_process.wait.return_value = 0

    manager = VLLMServerManager(model="dummy")
    manager.process = mock_process
    manager.stop()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
class TestVLLMServeLM(BaseLanguageModelTest):
    @pytest.fixture()
    def lm(self, chat_lm: VLLMServeLM) -> VLLMServeLM:
        return chat_lm

    @pytest.fixture()
    def chat_lm(self, chat_lm: VLLMServeLM) -> VLLMServeLM:
        return chat_lm

    @pytest.mark.skip(reason="Even with temperature 0.0, the output is not deterministic via API.")
    def test_batch_complete_text_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        pass

    @pytest.mark.skip(reason="Even with temperature 0.0, the output is not deterministic via API.")
    def test_batch_chat_response_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        pass

    @pytest.mark.skip(reason="A larger, more high-performance model is required to test the behavior of Tool Calling.")
    def test_generate_chat_response_single_input_with_tools(self, chat_lm_for_tool_calling: LanguageModel) -> None:
        pass

    @pytest.mark.skip(reason="A larger, more high-performance model is required to test the behavior of Tool Calling.")
    def test_generate_chat_response_batch_input_with_tools(self, chat_lm_for_tool_calling: LanguageModel) -> None:
        pass

    @pytest.mark.skip(reason="A larger, more high-performance model is required to test the behavior of Tool Calling.")
    def test_generate_chat_response_if_number_of_tools_and_messages_not_equal(
        self, chat_lm_for_tool_calling: LanguageModel
    ) -> None:
        pass

@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_set_random_seed(chat_lm: VLLMServeLM):
    chat_lm.set_random_seed(42)
    assert chat_lm.default_gen_kwargs["seed"] == 42

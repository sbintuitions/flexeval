from __future__ import annotations

import atexit
import logging
import socket
import subprocess
import threading
import time
from typing import IO, Any, Callable

import requests
import torch
from loguru import logger

from flexeval.core.language_model.base import LMOutput
from flexeval.core.string_processor import StringProcessor

from .openai_api import OpenAIChatAPI


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class VLLMServerManager:
    """
    Manages the lifecycle of a `vllm serve` process.

    This class launches the vLLM OpenAI-compatible Web API server in the background, streams its stdout/stderr,
    and waits until the server is ready to accept requests.

    Args:
        model: The name or path of the model to serve.
        model_kwargs: Additional keyword arguments to pass as command-line options to `vllm serve`.
            Each key-value pair is converted to a corresponding CLI argument.
        timeout: Maximum time in seconds to wait for the server to become available.
    """

    def __init__(self, model: str, model_kwargs: dict[str, Any] | None = None, timeout: int = 3600) -> None:
        self.model = model
        self.model_kwargs = model_kwargs or {}
        self.host = "localhost"
        self.port = find_free_port()
        self.base_url = f"http://{self.host}:{self.port}/v1"
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._log_threads: list[threading.Thread] = []

    def start(self) -> tuple[str, int]:  # noqa: C901
        """
        Start the `vllm serve` process and wait until it is ready.
        """
        if self.is_ready():
            logger.warning("vLLM server is already running. Skipping start.")
            return self.host, self.port

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--disable-uvicorn-access-log",
        ]

        for key, value in self.model_kwargs.items():
            key_formatted = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(key_formatted)
            else:
                cmd.extend([key_formatted, str(value)])

        logger.info("Starting vLLM server with command: {}", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        atexit.register(self.stop)

        def stream_logger(stream: IO, level: str, label: str) -> None:
            for line in iter(stream.readline, ""):
                if ("logger.py" in line or "engine.py" in line) and label == "vllm/stdout":
                    continue
                logger.log(level, f"[{label}] {line.rstrip()}")

        for stream, level, label in [
            (self.process.stdout, "INFO", "vllm/stdout"),
            (self.process.stderr, "INFO", "vllm/stderr"),
        ]:
            t = threading.Thread(target=stream_logger, args=(stream, level, label), daemon=True)
            t.start()
            self._log_threads.append(t)

        logger.info("Waiting for server to become available")

        interval_second = 1
        for _ in range(self.timeout // interval_second):
            if self.is_ready():
                logger.info(f"vLLM server is ready at {self.base_url}")
                return self.host, self.port
            time.sleep(interval_second)

        self.stop()
        msg = f"vLLM server did not start within {self.timeout} seconds"
        raise RuntimeError(msg)

    def stop(self) -> None:
        """
        stop `vllm serve` process and clean up resources.
        """
        if self.process and self.process.poll() is None:
            logger.info("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
                logger.info("vLLM server stopped cleanly.")
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not stop in time. Killing...")
                self.process.kill()
        self.process = None
        self._log_threads.clear()

    def is_ready(self) -> bool:
        """
        Check if the vLLM server is ready to accept requests.
        """
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            response = requests.get(f"{self.base_url}/models", timeout=1)
            return response.status_code == 200  # noqa: TRY300
        except requests.RequestException:
            return False


class VLLMServeLM(OpenAIChatAPI):
    """
    The LanguageModel class that uses vLLM via `vllm serve`.
    This class starts a OpenAI-compatible vLLM API server in the background and communicates with it via HTTP.

    We also have the `VLLM` class that uses the vLLM Python API,
    but there are slight differences in the features available to each other.
    Using this class, for example, you can take advantage of vLLM's tool-calling support
    through options such as `--tool-call-parser` and `--enable-auto-tool-choice`.
    https://docs.vllm.ai/en/stable/features/tool_calling.html

    Args:
        model: The name or path of the model to serve.
        api_headers: A dictionary of headers to use when making requests to the OpenAI API.
        model_kwargs: Additional keyword arguments to pass as command-line options to `vllm serve`.
            Each key-value pair is converted to a corresponding CLI argument.
            (e.g. `{"tool_call_parser": "hermes", "enable_auto_tool_choice": True}`).
            Do not include the prefix "--".
            See also: https://docs.vllm.ai/en/latest/cli/index.html#options
        booting_timeout: Maximum time in seconds to wait for the server to become available.
        default_gen_kwargs: Default generation kwargs to use when calling the API.
        developer_message: Instructions to the model that are prioritized ahead of user messages.
            Previously called the system prompt.
        string_processors: A single or a list of StringProcessor objects to process the model's output.
        model_limit_new_tokens: An upper limit on the number of tokens the model can generate.
            For example, if a too-large `max_new_tokens` is given to generate_chat_response(), this value will cap it.
        tools: Default tools to use in chat responses when no tools are explicitly provided.
        max_parallel_requests: Maximum number of parallel requests to send to the OpenAI API.
    """

    def __init__(
        self,
        model: str,
        api_headers: dict[str, str] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        booting_timeout: int = 3600,
        default_gen_kwargs: dict[str, Any] | None = None,
        developer_message: str | None = None,
        string_processors: StringProcessor | list[StringProcessor] | None = None,
        model_limit_new_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_parallel_requests: int | None = None,
    ) -> None:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        model_kwargs = model_kwargs or {}
        if "tensor_parallel_size" not in model_kwargs:
            model_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        self.manager = VLLMServerManager(model=model, model_kwargs=model_kwargs, timeout=booting_timeout)
        if api_headers is None:
            api_headers = {}
        api_headers["base_url"] = self.manager.base_url
        api_headers["api_key"] = "EMPTY"  # OpenAI client requires an api_key, but vLLM does not use it.
        super().__init__(
            model=model,
            api_headers=api_headers,
            default_gen_kwargs=default_gen_kwargs,
            developer_message=developer_message,
            string_processors=string_processors,
            model_limit_new_tokens=model_limit_new_tokens,
            tools=tools,
            max_parallel_requests=max_parallel_requests,
        )

    @staticmethod
    def load_model(method: Callable) -> Callable:
        """Decorator to load the model lazily."""

        def wrapper(self: VLLMServeLM, *args: tuple, **kwargs: dict) -> Callable:
            if not self.manager.is_ready():
                self.manager.start()
            return method(self, *args, **kwargs)

        return wrapper

    @load_model
    def _batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        return super()._batch_complete_text(text_list, stop_sequences, max_new_tokens, **kwargs)

    @load_model
    def _batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, Any]]],
        tools_list: list[list[dict[str, Any]] | None] | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        return super()._batch_generate_chat_response(
            chat_messages_list,
            tools_list=tools_list,
            **kwargs,
        )

    def _batch_compute_chat_log_probs(
        self,
        prompt_list: list[list[dict[str, Any]]],
        response_list: list[dict[str, Any]],
        temperature: float = 0,
        seed: int = 42,
        top_logprobs: int = 20,
    ) -> list[float | None]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"

    def cleanup_resources(self) -> None:
        if hasattr(self, "manager"):
            self.manager.stop()

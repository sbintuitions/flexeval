from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .chat_dataset import ChatDataset
from .evaluate_chat_response import evaluate_chat_response
from .evaluate_generation import evaluate_generation
from .evaluate_multiple_choice import evaluate_multiple_choice
from .evaluate_perplexity import evaluate_perplexity
from .few_shot_generator import FewShotGenerator
from .generation_dataset import GenerationDataset
from .language_model import LanguageModel
from .metric import Metric
from .metric.tokenizer import Tokenizer
from .multiple_choice_dataset import MultipleChoiceDataset
from .prompt_template import PromptTemplate
from .text_dataset import TextDataset


class EvalSetup(ABC):
    """Abstract class to give evaluation functions a common interface."""

    @abstractmethod
    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        pass


@dataclass
class ChatResponse(EvalSetup):
    """
    Evaluation setup for chat response generation.
    In this setup, the model receives context in a chat format like:
    ```json
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
    ```
    """

    eval_dataset: ChatDataset
    gen_kwargs: dict[str, Any]
    few_shot_generator: FewShotGenerator | None = None
    metrics: list[Metric] | Metric | None = None
    batch_size: int = 4
    max_instances: int | None = None

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        metrics = self.metrics or []
        if isinstance(metrics, Metric):
            metrics = [metrics]

        return evaluate_chat_response(
            language_model=language_model,
            gen_kwargs=self.gen_kwargs,
            eval_dataset=self.eval_dataset,
            metrics=metrics,
            batch_size=self.batch_size,
            max_instances=self.max_instances,
            few_shot_generator=self.few_shot_generator,
        )


@dataclass
class Generation(EvalSetup):
    """
    Evaluation setup for text generation.
    The model receives a prompt in a plain text format and generates its continuation.
    """

    eval_dataset: GenerationDataset
    prompt_template: PromptTemplate
    gen_kwargs: dict[str, Any]
    few_shot_generator: FewShotGenerator | None = None
    metrics: list[Metric] | Metric | None = None
    batch_size: int = 4
    max_instances: int | None = None

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        metrics = self.metrics or []
        if isinstance(metrics, Metric):
            metrics = [metrics]

        return evaluate_generation(
            language_model=language_model,
            gen_kwargs=self.gen_kwargs,
            eval_dataset=self.eval_dataset,
            prompt_template=self.prompt_template,
            few_shot_generator=self.few_shot_generator,
            metrics=metrics,
            batch_size=self.batch_size,
            max_instances=self.max_instances,
        )


@dataclass
class MultipleChoice(EvalSetup):
    """
    Evaluation setup for multiple choice questions.
    The model receives a prompt and a list of choices and selects the answer with the highest probability.
    """

    eval_dataset: MultipleChoiceDataset
    prompt_template: PromptTemplate
    few_shot_generator: FewShotGenerator | None = None
    batch_size: int = 4
    max_instances: int | None = None

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        return evaluate_multiple_choice(
            language_model=language_model,
            eval_dataset=self.eval_dataset,
            prompt_template=self.prompt_template,
            few_shot_generator=self.few_shot_generator,
            batch_size=self.batch_size,
            max_instances=self.max_instances,
        )


@dataclass
class Perplexity(EvalSetup):
    """
    Evaluation setup for perplexity.
    The model receives plain text and computes the perplexity of the text.
    """

    eval_dataset: TextDataset
    batch_size: int = 4
    tokenizer: Tokenizer | None = None
    max_instances: int | None = None

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        metrics = evaluate_perplexity(
            language_model=language_model,
            eval_dataset=self.eval_dataset,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            max_instances=self.max_instances,
        )
        return metrics, None

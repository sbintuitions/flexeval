from abc import ABC, abstractmethod


class StringProcessor(ABC):
    """An interface class used to process the model's output before evaluation.
    Typically used in `Metric`.
    """

    @abstractmethod
    def __call__(self, text: str) -> str:
        """
        Process the input text.

        Args:
            text: The text to process.
        """
        raise NotImplementedError

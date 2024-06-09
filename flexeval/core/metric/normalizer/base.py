from abc import ABC, abstractmethod


class Normalizer(ABC):
    """Base class for text normalizers.
    Normalizers are used to preprocess the model's output before evaluation.
    Typically used in `Metric`.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize the input text.

        Args:
            text: The text to normalize.
        """
        raise NotImplementedError

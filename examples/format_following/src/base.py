from __future__ import annotations

from abc import ABC, abstractmethod


class ResponseConstraint(ABC):
    """
    Base class for response constraints.
    """

    @abstractmethod
    def check(self, response: str) -> bool:
        """
        Check if the response satisfies the constraint.
        """

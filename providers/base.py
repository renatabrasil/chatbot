from abc import abstractmethod, ABC
from typing import List

from core.types import Message, ModelMetrics


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[Message]) -> tuple[str, ModelMetrics]:
        raise NotImplementedError
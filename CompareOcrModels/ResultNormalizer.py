from abc import ABC, abstractmethod

class ResultNormalizer(ABC):
    @abstractmethod
    def normalize(self, result: dict) -> dict:
        pass

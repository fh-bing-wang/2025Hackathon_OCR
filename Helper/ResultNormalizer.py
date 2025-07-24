from abc import ABC, abstractmethod

class ResultNormalizer(ABC):
    @staticmethod
    @abstractmethod
    def normalize(result: dict) -> dict:
        pass
        # {
        #     "text": "",
        #     "confidence": 0.0,
        #     "bounding_box": [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        # }[]

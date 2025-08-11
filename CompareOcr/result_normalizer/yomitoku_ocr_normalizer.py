class YomitokuOcrNormalizer():
    @staticmethod
    def normalize(result: dict) -> dict:
        """
        Normalize the OCR result from PaddleOCR format to a standard format.
        
        Args:
            result (dict): The OCR result from PaddleOCR.
        
        Returns:
            dict: Normalized OCR result.
        """
        normalized_result = []

        paragraphs = result.get('paragraphs', [])



        rec_texts = result.get('rec_texts', [])
        rec_scores = result.get('rec_scores', [])
        rec_boxes = result.get('rec_boxes', [])

        if len(rec_texts) != len(rec_scores) or len(rec_texts) != len(rec_boxes):
            raise ValueError("Inconsistent lengths of rec_texts, rec_scores, and rec_boxes")

        for i in range(len(paragraphs)):
            normalized_result.append({
                "text": paragraphs[i].get("contents", ""),
                "confidence": 0.0,
                "bounding_box": paragraphs[i].get("box", [])
            })

        return normalized_result

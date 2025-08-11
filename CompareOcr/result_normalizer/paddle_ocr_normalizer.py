class PaddleOcrNormalizer():
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

        rec_texts = result.get('rec_texts', [])
        rec_scores = result.get('rec_scores', [])
        rec_boxes = result.get('rec_boxes', [])

        if len(rec_texts) != len(rec_scores) or len(rec_texts) != len(rec_boxes):
            raise ValueError("Inconsistent lengths of rec_texts, rec_scores, and rec_boxes")

        for i in range(len(rec_texts)):
            normalized_result.append({
                "text": rec_texts[i],
                "confidence": rec_scores[i],
                "bounding_box": rec_boxes[i]
            })

        return normalized_result

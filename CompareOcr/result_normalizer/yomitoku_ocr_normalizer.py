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
        # normalized_result = []

        words = result.get('words', [])

        rec_texts = [p.get("content", "") for p in words]
        rec_scores = [p.get("det_score", "") for p in words]
        rec_boxes = [YomitokuOcrNormalizer.to_axis_aligned(p.get("points", [])) for p in words]

        if (len(rec_texts) != len(rec_boxes)):
            return {"error": f"Yomitoku result lengths: rec_texts={len(rec_texts)}, rec_boxes={len(rec_boxes)}"}

        return {
            "text": rec_texts,
            "rec_confidence": rec_scores,
            "rec_boxes": rec_boxes
        }

    @staticmethod
    def to_axis_aligned(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

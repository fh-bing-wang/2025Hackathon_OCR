from collections import defaultdict

class GoogleOcrNormalizer():
    @staticmethod
    def normalize(result: list[dict]) -> dict:
        """
        Normalize the OCR result from GoogleOCR format to a standard format.
        
        Args:
            result (dict): The OCR result from GoogleOCR.
        
        Returns:
            dict: Normalized OCR result.
        """
        

    @staticmethod
    def merge_ocr_blocks(items):
        grouped = defaultdict(list)

        # Group by level, page_num, block_num
        for item in items:
            key = (item['level'], item['page_num'], item['block_num'])
            grouped[key].append(item)

        rec_texts = []
        rec_scores = []
        rec_boxes = []

        for key, group_items in grouped.items():
            # Sort by page -> par -> line -> word to keep reading order
            group_items.sort(key=lambda x: (x['page_num'], x['par_num'], x['line_num'], x['word_num']))

            # Concatenate text
            concatenated_text = ''.join([x['text'] for x in group_items])

            # Average confidence (ignore 0s if you want, or include all)
            confidences = [x['confidence'] for x in group_items]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            # Combine bounding boxes
            x_min = min(x['bounding_box']['left'] for x in group_items)
            y_min = min(x['bounding_box']['top'] for x in group_items)
            x_max = max(x['bounding_box']['left'] + x['bounding_box']['width'] for x in group_items)
            y_max = max(x['bounding_box']['top'] + x['bounding_box']['height'] for x in group_items)
            combined_box = (x_min, y_min, x_max, y_max)

            rec_texts.append(concatenated_text)
            rec_scores.append(avg_conf)
            rec_boxes.append(combined_box)

        return {
            "text": rec_texts,
            "rec_scores": rec_scores,
            "rec_boxes": rec_boxes
        }

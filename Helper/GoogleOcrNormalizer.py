

class GoogleOcrNormalizer():
    @staticmethod
    def normalize(result: dict) -> dict:
        """
        Normalize the OCR result from Google OCR format to a standard format.

        Args:
            result (dict): The OCR result from Google OCR.

        Returns:
            dict: Normalized OCR result.
        """
        normalized_result = []
        # = {
        #     "text": "",
        #     "confidence": 0.0,
        #     "bounding_box": [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        # }

        text = result.get('text', '')
        pages = result.get('pages', [])

        if (len(pages) == 0):
            raise ValueError("No pages found in the OCR result")
        page = pages[0]  # Assuming we only want the first page
        blocks = page.get('blocks', [])

        for block in blocks:
            block_layout = block.get('layout', {})
            confidence = block_layout.get('confidence', 0.0)
            text_anchor = block_layout.get('textAnchor', {})

            segments = text_anchor.get('textSegments', [])
            segmentLen = len(segments)
            if segmentLen > 1:
                print("more than 1 segment:", segments)
            if segmentLen > 0:
                segment = segments[0]
                print("segment:", segment)
                startIndex = int(segment.get('startIndex', 0))
                endIndex = int(segment.get('endIndex', 0))
                print("startIndex:", startIndex, "endIndex:", endIndex)
                segment_text = text[startIndex:endIndex]
            else:
                raise Exception("No text segments found in the block layout")
            
            bounding_poly = block_layout.get('boundingPoly', {})
            vertices = bounding_poly.get('vertices', [])
            if len(vertices) != 4:
                raise ValueError("Bounding box must have exactly 4 vertices")
            top_left_x, top_left_y = vertices[0]['x'], vertices[0]['y']
            bottom_right_x, bottom_right_y = vertices[2]['x'], vertices[2]['y']

            normalized_result.append({
                "text": segment_text,
                "confidence": confidence,
                "bounding_box": [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            })

        return normalized_result

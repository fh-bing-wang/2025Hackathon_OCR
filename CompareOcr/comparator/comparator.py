from ast import List
from collections import defaultdict
from typing import Tuple
import unicodedata

import re


class Comparator():
    SYMBOL_MAP = {
        "【": "[", "】": "]",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‛": "'",
        "–": "-", "—": "-", "―": "-",
        "…": "...",
    }
    
    @staticmethod
    def compare(model1: str, res1: dict, model2: str, res2: dict, iou_threshold: float) -> dict:
        texts1 = res1.get("text", [])
        texts2 = res2.get("text", [])

        boxes1 = res1.get("rec_boxes", [])
        boxes2 = res2.get("rec_boxes", [])

        if not (len(texts1) == len(boxes1) and len(texts2) == len(boxes2)):
            return {"error": "Inconsistent result lengths"}

        matched = defaultdict(list)
        used_indices = set()

        for idx1, box1 in enumerate(boxes1):
            # find overlaps in list2
            print(f"comparing {box1} {texts1[idx1]} with:")
            for idx2, box2 in enumerate(boxes2):

                if idx2 in used_indices:
                    continue

                print(f"comparing {box2} {texts2[idx2]}")
                text2 = texts2[idx2]

                overlap = Comparator.iou(box1, box2)
                print(f"overlap: {overlap}")
                if overlap > iou_threshold:
                    matched[idx1].append((box2, text2))
                    used_indices.add(idx2)
        
        print("compared matched: ")
        print(matched)

        print(f"used index : {used_indices}")

        res = []
        for idx1, box1 in enumerate(boxes1):
            matches = sorted(matched[idx1], key=lambda item: item[0][0])
            print(f"matches: {matches}")
            combined_texts = "".join(match[1] for match in matches)
            combined_boxes = Comparator.merge_boxes([match[0] for match in matches])
            res.append({
                f"{model1}_text": texts1[idx1],
                f"{model1}_box": box1,
                f"{model2}_text": combined_texts,
                f"{model2}_box": combined_boxes
            })


        # leftovers in list2
        leftovers = [{f"{model2}_text": texts2[i], f"{model2}_box": box2[i]} for i, box2 in enumerate(boxes2) if i not in used_indices]

        return res, leftovers


    @staticmethod
    def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # Intersection
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0  # no overlap

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Union
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def merge_boxes(boxes: list[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        if not boxes:
            return (0, 0, 0, 0)

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)

        return (x1, y1, x2, y2)

    @staticmethod
    def combine_texts(res: dict):
        texts = res.get("text", [])
        boxes = res.get("rec_boxes", [])
        return Comparator.combine_texts_by_rows(boxes, texts)

    @staticmethod
    def combine_texts_by_rows(
        boxes: list[tuple[int, int, int, int]],
        texts: list[str],
        y_threshold: int = 10
    ) -> str:
        """
        Combine OCR texts row by row (top-to-bottom, then left-to-right within each row).

        Args:
            boxes: List of bounding boxes (x_min, y_min, x_max, y_max).
            texts: List of recognized texts corresponding to boxes.
            y_threshold: Vertical tolerance in pixels for grouping boxes into the same row.

        Returns:
            A single string of combined text.
        """

        if len(boxes) != len(texts):
            raise ValueError("Number of boxes must match number of texts")

        # Attach texts to their boxes
        items = [(box, text) for box, text in zip(boxes, texts)]

        # Sort primarily by top (y_min), then by left (x_min)
        items.sort(key=lambda item: (item[0][1], item[0][0]))

        # Group into rows
        rows = []
        current_row = []
        current_y = None

        for (box, text) in items:
            y_min = box[1]
            if current_y is None:
                current_row.append((box, text))
                current_y = y_min
            else:
                # Check if the box belongs to the current row
                if abs(y_min - current_y) <= y_threshold:
                    current_row.append((box, text))
                else:
                    # Save the finished row
                    rows.append(current_row)
                    # Start a new row
                    current_row = [(box, text)]
                    current_y = y_min

        if current_row:
            rows.append(current_row)

        # Sort each row left-to-right
        for row in rows:
            row.sort(key=lambda item: item[0][0])  # sort by x_min

        # Join texts row by row
        combined_lines = ["".join(text for _, text in row) for row in rows]

        # Join rows with newline
        combined_text = "".join(combined_lines)

        return Comparator.normalize_text(combined_text)

    @staticmethod
    def normalize_symbols(s: str) -> str:
        for k, v in Comparator.SYMBOL_MAP.items():
            s = s.replace(k, v)
        return s

    @staticmethod
    def normalize_unicode(s: str) -> str:
        # NFKC maps full-width to half-width, squashes compatibility chars
        return unicodedata.normalize("NFKC", s)

    @staticmethod
    def normalize_text(s: str) -> str:
        s = Comparator.normalize_symbols(s)
        s = Comparator.normalize_unicode(s)
        return re.sub(r"\s+", "", s)

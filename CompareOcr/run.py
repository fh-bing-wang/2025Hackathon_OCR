from paddleocr import PaddleOCR
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Process a file by name.")
parser.add_argument("file_name", type=Path, help="Path to the file")
args = parser.parse_args()

file_name = args.file_name

ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # text detection + text recognition

input_file = f"../TestFiles/{file_name}"
result = ocr.predict(input_file)
for res in result:
    res.print()
    res.save_to_img("../TestResults/PaddleOcr")
    res.save_to_json("../TestResults/PaddleOcr")

from PIL import Image
from paddleocr import PaddleOCR
import argparse
from pathlib import Path
import io
import numpy as np

parser = argparse.ArgumentParser(description="Process a file by name.")
parser.add_argument("file_name", type=Path, help="Path to the file")
args = parser.parse_args()

file_name = args.file_name

ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # text detection + text recognition

input_file = f"../TestFiles/{file_name}"
with open(input_file, "rb") as f:
    binary_data = f.read()
    image = Image.open(io.BytesIO(binary_data))
    image_np = np.array(image)
    result = ocr.predict(image_np)
    print(f"Output length: {len(result)}")

    for res in result:
        res.print()
        res.save_to_img("../TestResults/PaddleOcr")
        res.save_to_json("../TestResults/PaddleOcr")

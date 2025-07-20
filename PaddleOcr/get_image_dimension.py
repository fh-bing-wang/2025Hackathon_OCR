import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description="Process a file by name.")
parser.add_argument("file_name", type=Path, help="Path to the file")
args = parser.parse_args()

file_name = args.file_name

input_file = f"../TestFiles/{file_name}"
image = cv2.imread(input_file)
height, width = image.shape[:2]
print(f"Image size: {width}x{height}")

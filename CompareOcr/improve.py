import cv2
import sys
import os

import numpy as np

def enhance_image(input_path, output_path=None):
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Could not open image: {input_path}")
        return

    # Denoise (color)
    # denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(img, -1, kernel)

    # Improve contrast using CLAHE (on L-channel in LAB color space)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2,a,b))
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if not ext:
            ext = ".png"
        output_path = f"{base}_enhanced{ext}"

    cv2.imwrite(output_path, contrast)
    print(f"✅ Enhanced image saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enhance.py <input_image> [output_image]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    enhance_image(input_file, output_file)

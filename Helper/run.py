import json
from CompareOcr.result_normalizer.paddle_ocr_normalizer import PaddleOcrNormalizer
from GoogleOcrNormalizer import GoogleOcrNormalizer

def main():
    # googleResultInput = "../GoogleResults/05_text.json"
    # with open(googleResultInput, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # googleResult = GoogleOcrNormalizer.normalize(data)

    # for result in googleResult:
    #     print(f"Google OCR Result: {result}")

    paddleResultInput = "../TestResults/PaddleOcr/03_pathological_report_res.json"
    with open(paddleResultInput, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paddleResult = PaddleOcrNormalizer.normalize(data)

    for result in paddleResult:
        print(f"Paddle OCR Result: {result}")

if __name__ == "__main__":
    main()

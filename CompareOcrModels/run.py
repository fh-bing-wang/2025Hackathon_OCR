import json
from GoogleOcrNormalizer import GoogleOcrNormalizer

def main():
    googleResultInput = "../GoogleResults/05_text.json"
    with open(googleResultInput, 'r', encoding='utf-8') as f:
        data = json.load(f)
    googleResult = GoogleOcrNormalizer.normalize(data)
    print(googleResult)

    for result in googleResult:
        print(f"Google OCR Result: {result}")

if __name__ == "__main__":
    main()

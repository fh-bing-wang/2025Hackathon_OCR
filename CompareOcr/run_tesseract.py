# from PIL import Image
# from tesserocr import PyTessBaseAPI, RIL

# image = Image.open('00_breast_examine.png')
# with PyTessBaseAPI() as api:
#     api.SetImage(image)
#     boxes = api.GetComponentImages(RIL.TEXTLINE, True)
#     print('Found {} textline image components.'.format(len(boxes)))
#     # for i, (im, box, u1, u2) in enumerate(boxes):
#     for i, box in enumerate(boxes):

#         # im is a PIL image object
#         # box is a dict with x, y, w and h keys
#         api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
#         ocrResult = api.GetUTF8Text()
#         conf = api.MeanTextConf()
#         print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
#               "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))

import json
from PIL import Image
import pytesseract

data = pytesseract.image_to_data(Image.open('00_breast_examine.png'), output_type=pytesseract.Output.DICT,  lang='jpn+eng')

# Process the data and structure it into a list of dictionaries for JSON
text_data = []
n_boxes = len(data['level'])
for i in range(n_boxes):
    if data['text'][i].strip():  # Only include if text is not empty
        box_info = {
            'text': data['text'][i],
            'confidence': float(data['conf'][i]),
            'bounding_box': {
                'left': int(data['left'][i]),
                'top': int(data['top'][i]),
                'width': int(data['width'][i]),
                'height': int(data['height'][i])
            },
            'level': int(data['level'][i]),
            'page_num': int(data['page_num'][i]),
            'block_num': int(data['block_num'][i]),
            'par_num': int(data['par_num'][i]),
            'line_num': int(data['line_num'][i]),
            'word_num': int(data['word_num'][i])
        }
        text_data.append(box_info)

    # Convert the list of dictionaries to a JSON string
    

    # You can then save this JSON string to a file
    with open('tesseract_output.json', 'w', encoding='utf-8') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=2)

    print("JSON output saved to 'tesseract_output.json'.")


from paddleocr import TextRecognition

model = TextRecognition()
file_path = "../TestFiles/00_breast_examine.png"

output = model.predict(input=file_path)
for res in output:
    res.print()
    res.save_to_img(save_path="./output2/")
    res.save_to_json(save_path="./output2/res.json")

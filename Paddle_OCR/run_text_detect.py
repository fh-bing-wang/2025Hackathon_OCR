from paddleocr import TextDetection

model = TextDetection()
file_path = "../TestFiles/00_breast_examine.png"
output = model.predict(file_path)

print(f"Output length: {len(output)}:")

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")

from paddleocr import PPStructureV3

file_path = "../TestFiles/00_breast_examine.png"

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)
output = pipeline.predict(
    input=file_path)
for res in output:
    res.print()
    res.save_to_json(save_path="output3")
    res.save_to_markdown(save_path="output3")

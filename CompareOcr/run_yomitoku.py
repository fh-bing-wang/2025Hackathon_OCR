import cv2

from yomitoku import DocumentAnalyzer

from yomitoku.data.functions import load_image

file_name = "03_pathological_report.jpg"
input_file = f"../TestFiles/{file_name}"

analyzer = DocumentAnalyzer(visualize=True, device="cuda")

# PDFファイルを読み込み

imgs = load_image(input_file)

for i, img in enumerate(imgs):

    results, ocr_vis, layout_vis = analyzer(img)

    # HTML形式で解析結果をエクスポート

    results.to_html(f"output_{i}.html", img=img)

    # 可視化画像を保存

    cv2.imwrite(f"output_ocr_{i}.jpg", ocr_vis)

    cv2.imwrite(f"output_layout_{i}.jpg", layout_vis)

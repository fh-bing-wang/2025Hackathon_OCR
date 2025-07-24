# Activate virtual env
source ./compare_models_env/bin/activate
source ./paddle_env/bin/activate


# Doc for installation of PaddleOCR 
https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html

# Install PaddleOCR
python3 -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/


# Installed nvidia-utils-535 for NVIDIA-SMI
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Use PaddleOcr Cli
paddleocr ocr -i ../TestFiles/00_breast_examine.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --output out_00.json

# Run script
python3 run.py 00_breast_examine.png
python3 run.py 02_pathological_report.jpg

# 
Origin: top-left
"rec_texts"
"R",
"N",
"病理検査報告書",

Rec polys:
top left; top right; bottom right; bottom left
[[101,61],[171,61],[171,131],[101,131]],
[[1287,59],[1354,59],[1354,124],[1287,124]],
[[481,73],[764,73],[764,117],[481, 117]],

Rec boxes:
top left; bottom right
[101, 61, 171, 131],
[1287, 59, 1354, 124],
[481,73,764,117],

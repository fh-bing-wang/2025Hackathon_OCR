# activate virtual env
source ./compare_ocr_env/bin/activate

# start endpoint
python3 start_api_server.py

# call endpoint
curl http://127.0.0.1:8000/

# serve the .html file
python3 -m http.server 8888

port forward 8888

# Access
http://bing-gpu-01.dev.flatiron.co.jp:8888/html_pages/ocr_viewer.html

# CUDA not available
Ubuntu system sees an NVIDIA GPU, but the NVIDIA driver isn’t installed or isn’t loaded properly
```
$ lspci | grep -i nvidia
00:1e.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
$ nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

```
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
```
```
nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.03              Driver Version: 575.64.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:1E.0 Off |                    0 |
| N/A   41C    P0             26W /   70W |       0MiB /  15360MiB |      9%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
```

## Installation of Tesseract OCR
```
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-jpn
sudo apt install libtesseract-dev
```

```
tesseract 03_pathological_report.jpg res \
    -l jpn+eng \
    --oem 1 \
    --psm 6 \
    json
```

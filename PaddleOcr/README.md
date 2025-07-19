# Activate virtual env
source ./paddle_env/bin/activate

# Doc for installation of PaddleOCR 
https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html

# Install PaddleOCR
python3 -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/


# Installed nvidia-utils-535 for NVIDIA-SMI
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535

